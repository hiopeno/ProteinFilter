from __future__ import annotations

from dataclasses import dataclass
import ast
import gzip
import json
from pathlib import Path
from typing import Optional, Tuple

import lmdb
import numpy as np
import pandas as pd

from lightscorer.features import (
    distance_matrix_from_coords,
    normalize_distance_matrix,
    resize_matrix,
    synthetic_distance_matrix,
)


@dataclass
class MockDataConfig:
    train_size: int = 2000
    val_size: int = 400
    test_size: int = 400
    matrix_size: int = 128
    seed: int = 42


@dataclass
class RealDataConfig:
    manifest_path: Path
    raw_lmdb_dir: Path
    matrix_size: int = 128
    clip_max: float = 30.0
    max_samples_per_split: Optional[int] = None
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    max_test_samples: Optional[int] = None
    feature_dtype: str = "float16"
    seed: int = 42


def _make_split(size: int, rng: np.random.Generator, matrix_size: int) -> Tuple[np.ndarray, np.ndarray]:
    y = rng.binomial(1, 0.25, size=size).astype(np.int64)
    x = []
    for label in y:
        n_res = int(rng.integers(60, 220))
        mat = synthetic_distance_matrix(n_res=n_res, good=bool(label), rng=rng)
        mat = resize_matrix(mat, out_size=matrix_size)
        x.append(mat)
    return np.stack(x, axis=0).astype(np.float32), y


def load_mock_data(config: MockDataConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(config.seed)
    x_train, y_train = _make_split(config.train_size, rng, config.matrix_size)
    x_val, y_val = _make_split(config.val_size, rng, config.matrix_size)
    x_test, y_test = _make_split(config.test_size, rng, config.matrix_size)
    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
    }


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    expected = {"split", "target_id", "decoy_id", "label"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"manifest missing required columns: {sorted(missing)}")
    if "sample_id" not in df.columns:
        df["sample_id"] = df["target_id"].astype(str) + "/" + df["decoy_id"].astype(str)
    return df


def split_manifest_frames(manifest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        split: part.reset_index(drop=True)
        for split, part in manifest.groupby("split", sort=False)
    }


def load_real_manifest_splits(manifest_path: Path) -> dict[str, pd.DataFrame]:
    manifest = load_manifest(manifest_path)
    frames = split_manifest_frames(manifest)
    for required in ("train", "val", "test"):
        if required not in frames:
            raise ValueError(f"manifest missing split: {required}")
    return frames


def _sample_id_from_payload_id(raw_id: object) -> str:
    if isinstance(raw_id, (list, tuple)) and len(raw_id) >= 2:
        return f"{raw_id[0]}/{raw_id[1]}"
    if isinstance(raw_id, str):
        txt = raw_id.strip()
        if txt.startswith("(") and txt.endswith(")"):
            try:
                parsed = ast.literal_eval(txt)
                if isinstance(parsed, (tuple, list)) and len(parsed) >= 2:
                    return f"{parsed[0]}/{parsed[1]}"
            except (ValueError, SyntaxError):
                pass
        if "/" in txt:
            return txt
        if "," in txt:
            lhs, rhs = txt.split(",", maxsplit=1)
            return f"{lhs.strip()}/{rhs.strip()}"
    raise ValueError(f"Unsupported payload id format: {raw_id!r}")


def _build_index_for_ids(
    raw_lmdb_dir: Path, required_ids: set[str]
) -> dict[str, bytes]:
    env = lmdb.open(
        str(raw_lmdb_dir),
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=1,
        subdir=True,
    )
    found: dict[str, bytes] = {}
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            payload = json.loads(gzip.decompress(value).decode("utf-8"))
            sample_id = _sample_id_from_payload_id(payload.get("id"))
            if sample_id in required_ids:
                found[sample_id] = bytes(key)
                if len(found) >= len(required_ids):
                    break
    return found


def _atoms_to_ca_coords(atoms_payload: dict) -> np.ndarray:
    columns = atoms_payload.get("columns", [])
    rows = atoms_payload.get("data", [])
    if not columns or not rows:
        raise ValueError("atoms payload is empty")
    col_idx = {name: i for i, name in enumerate(columns)}
    for req in ("name", "x", "y", "z"):
        if req not in col_idx:
            raise ValueError(f"atoms payload missing column: {req}")
    name_i = col_idx["name"]
    x_i, y_i, z_i = col_idx["x"], col_idx["y"], col_idx["z"]
    coords = []
    for row in rows:
        atom_name = str(row[name_i]).strip()
        if atom_name != "CA":
            continue
        coords.append([float(row[x_i]), float(row[y_i]), float(row[z_i])])
    if not coords:
        raise ValueError("No CA atoms found in atoms payload")
    return np.asarray(coords, dtype=np.float32)


def _feature_from_payload(atoms_payload: dict, matrix_size: int, clip_max: float) -> np.ndarray:
    coords = _atoms_to_ca_coords(atoms_payload)
    dist = distance_matrix_from_coords(coords)
    dist = normalize_distance_matrix(dist, clip_max=clip_max)
    return resize_matrix(dist, out_size=matrix_size)


def _resolve_split_caps(config: RealDataConfig) -> dict[str, Optional[int]]:
    base = config.max_samples_per_split
    return {
        "train": config.max_train_samples if config.max_train_samples is not None else base,
        "val": config.max_val_samples if config.max_val_samples is not None else base,
        "test": config.max_test_samples if config.max_test_samples is not None else base,
    }


def _resolve_feature_dtype(dtype_name: str):
    name = dtype_name.lower().strip()
    if name == "float16":
        return np.float16
    if name == "float32":
        return np.float32
    raise ValueError("feature_dtype must be one of: float16, float32")


def load_real_data(config: RealDataConfig) -> dict[str, np.ndarray]:
    manifest = load_manifest(config.manifest_path)
    split_caps = _resolve_split_caps(config)
    sampled_parts = []
    for split, part in manifest.groupby("split", sort=False):
        cap = split_caps.get(split)
        if cap is not None:
            n = min(len(part), int(cap))
            sampled_parts.append(part.sample(n=n, random_state=config.seed))
        else:
            sampled_parts.append(part)
    manifest = pd.concat(sampled_parts, ignore_index=True)

    needed_ids = set(manifest["sample_id"].astype(str).tolist())
    sample_to_key = _build_index_for_ids(config.raw_lmdb_dir, needed_ids)
    missing_ids = needed_ids - set(sample_to_key.keys())
    if missing_ids:
        raise ValueError(f"Missing {len(missing_ids)} sample_ids in raw LMDB")

    env = lmdb.open(
        str(config.raw_lmdb_dir),
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=1,
        subdir=True,
    )
    feature_dtype = _resolve_feature_dtype(config.feature_dtype)
    split_frames = {
        split: part.reset_index(drop=True)
        for split, part in manifest.groupby("split", sort=False)
    }
    for required in ("train", "val", "test"):
        if required not in split_frames:
            split_frames[required] = manifest.iloc[0:0].copy()

    out: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        n = len(split_frames[split])
        out[f"x_{split}"] = np.empty(
            (n, config.matrix_size, config.matrix_size), dtype=feature_dtype
        )
        out[f"y_{split}"] = np.empty((n,), dtype=np.int64)

    with env.begin() as txn:
        for split in ("train", "val", "test"):
            frame = split_frames[split]
            for idx, row in enumerate(frame.itertuples(index=False)):
                sample_id = str(row.sample_id)
                key = sample_to_key[sample_id]
                raw = txn.get(key)
                if raw is None:
                    raise ValueError(f"LMDB key not found for sample_id: {sample_id}")
                payload = json.loads(gzip.decompress(raw).decode("utf-8"))
                feature = _feature_from_payload(
                    payload["atoms"], matrix_size=config.matrix_size, clip_max=config.clip_max
                ).astype(feature_dtype, copy=False)
                out[f"x_{split}"][idx] = feature
                out[f"y_{split}"][idx] = int(row.label)
    return out
