from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence
import ast
import gzip
import json

import lmdb
import pandas as pd


@dataclass
class ManifestBuildConfig:
    output_path: Path
    data_root: Optional[Path] = None
    raw_lmdb_dir: Optional[Path] = None
    score_file: Optional[Path] = None
    tm_threshold: float = 0.5
    label_policy: str = "tm_threshold"
    split_seed: int = 42
    split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1)
    max_entries: Optional[int] = None


def _read_split_targets(path: Path, split: str) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            target_id, decoy_id = [x.strip() for x in line.split(",", maxsplit=1)]
            rows.append({"split": split, "target_id": target_id, "decoy_id": decoy_id})
    return pd.DataFrame(rows)


def _read_scores(score_file: Path) -> pd.DataFrame:
    scores = pd.read_csv(score_file)
    expected = {"target_id", "decoy_id", "tm"}
    missing = expected - set(scores.columns)
    if missing:
        raise ValueError(f"score_file missing required columns: {sorted(missing)}")
    return scores[list(expected)].copy()


def _assign_label(df: pd.DataFrame, policy: str, tm_threshold: float) -> pd.Series:
    if policy == "native_vs_decoy":
        return (df["target_id"] == df["decoy_id"]).astype(int)
    if policy == "tm_threshold":
        if "tm" not in df.columns:
            raise ValueError("tm_threshold policy requires tm column from score_file")
        return (df["tm"] >= tm_threshold).astype(int)
    raise ValueError(f"Unsupported label policy: {policy}")


def _parse_sample_id(raw_id: object) -> tuple[str, str]:
    if isinstance(raw_id, (list, tuple)) and len(raw_id) >= 2:
        return str(raw_id[0]), str(raw_id[1])
    if isinstance(raw_id, str):
        tuple_like = raw_id.strip()
        if tuple_like.startswith("(") and tuple_like.endswith(")"):
            try:
                parsed = ast.literal_eval(tuple_like)
                if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
                    return str(parsed[0]), str(parsed[1])
            except (SyntaxError, ValueError):
                pass
        if "/" in raw_id:
            lhs, rhs = raw_id.split("/", maxsplit=1)
            return lhs.strip(), rhs.strip()
        if "," in raw_id:
            lhs, rhs = raw_id.split(",", maxsplit=1)
            return lhs.strip(), rhs.strip()
        raise ValueError(f"Unsupported id string format: {raw_id}")
    raise ValueError(f"Unsupported id type: {type(raw_id).__name__}")


def _read_raw_lmdb_entries(raw_lmdb_dir: Path, max_entries: Optional[int] = None) -> pd.DataFrame:
    env = lmdb.open(
        str(raw_lmdb_dir),
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=1,
        subdir=True,
    )
    rows = []
    with env.begin() as txn:
        cursor = txn.cursor()
        for idx, (_, value) in enumerate(cursor):
            payload = json.loads(gzip.decompress(value).decode("utf-8"))
            target_id, decoy_id = _parse_sample_id(payload.get("id"))
            score = payload.get("scores", {}) or {}
            rows.append(
                {
                    "target_id": target_id,
                    "decoy_id": decoy_id,
                    "tm": score.get("tm"),
                    "gdt_ts": score.get("gdt_ts"),
                    "gdt_ha": score.get("gdt_ha"),
                    "rmsd": score.get("rmsd"),
                }
            )
            if max_entries is not None and (idx + 1) >= max_entries:
                break
    return pd.DataFrame(rows)


def _validate_split_ratio(split_ratio: Sequence[float]) -> tuple[float, float, float]:
    if len(split_ratio) != 3:
        raise ValueError("split_ratio must contain 3 values: train,val,test")
    train_ratio, val_ratio, test_ratio = [float(x) for x in split_ratio]
    ratio_sum = train_ratio + val_ratio + test_ratio
    if ratio_sum <= 0:
        raise ValueError("split_ratio sum must be > 0")
    train_ratio /= ratio_sum
    val_ratio /= ratio_sum
    test_ratio /= ratio_sum
    return train_ratio, val_ratio, test_ratio


def _split_by_target(
    df: pd.DataFrame, split_seed: int, split_ratio: Sequence[float]
) -> pd.Series:
    train_ratio, val_ratio, _ = _validate_split_ratio(split_ratio)
    targets = (
        df["target_id"].dropna().astype(str).sort_values().drop_duplicates().tolist()
    )
    rng = pd.Series(targets).sample(frac=1.0, random_state=split_seed).tolist()

    n_targets = len(rng)
    n_train = int(n_targets * train_ratio)
    n_val = int(n_targets * val_ratio)
    train_targets = set(rng[:n_train])
    val_targets = set(rng[n_train : n_train + n_val])
    test_targets = set(rng[n_train + n_val :])

    split = pd.Series(index=df.index, dtype="object")
    split[df["target_id"].isin(train_targets)] = "train"
    split[df["target_id"].isin(val_targets)] = "val"
    split[df["target_id"].isin(test_targets)] = "test"
    if split.isna().any():
        raise ValueError("Some samples did not receive split assignment")
    return split


def _assert_no_target_leakage(frames: Iterable[pd.DataFrame]) -> None:
    split_targets = {}
    for frame in frames:
        if frame.empty:
            continue
        split = frame["split"].iloc[0]
        split_targets[split] = set(frame["target_id"].unique().tolist())

    names = list(split_targets.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = split_targets[names[i]] & split_targets[names[j]]
            if overlap:
                raise ValueError(
                    f"Target leakage between {names[i]} and {names[j]}: {len(overlap)} overlap"
                )


def summarize_manifest(manifest: pd.DataFrame) -> dict[str, float]:
    total = len(manifest)
    pos_ratio = float(manifest["label"].mean()) if total else 0.0
    tm_missing_ratio = (
        float(manifest["tm"].isna().mean()) if "tm" in manifest.columns else 1.0
    )
    split_count = manifest["split"].value_counts().to_dict()
    target_count = (
        manifest.groupby("split")["target_id"].nunique().to_dict()
        if "split" in manifest.columns
        else {}
    )
    return {
        "n_samples": float(total),
        "positive_ratio": pos_ratio,
        "tm_missing_ratio": tm_missing_ratio,
        **{f"samples_{k}": float(v) for k, v in split_count.items()},
        **{f"targets_{k}": float(v) for k, v in target_count.items()},
    }


def build_manifest(config: ManifestBuildConfig) -> pd.DataFrame:
    if config.raw_lmdb_dir is not None:
        manifest = _read_raw_lmdb_entries(
            config.raw_lmdb_dir, max_entries=config.max_entries
        )
        manifest["split"] = _split_by_target(
            manifest, split_seed=config.split_seed, split_ratio=config.split_ratio
        )
        frame_by_split = [manifest[manifest["split"] == s] for s in ("train", "val", "test")]
        _assert_no_target_leakage(frame_by_split)
    elif config.data_root is not None:
        split_dir = config.data_root / "targets"
        train = _read_split_targets(split_dir / "train.txt", "train")
        val = _read_split_targets(split_dir / "val.txt", "val")
        test = _read_split_targets(split_dir / "test.txt", "test")
        _assert_no_target_leakage([train, val, test])
        manifest = pd.concat([train, val, test], ignore_index=True)
    else:
        raise ValueError("Either data_root or raw_lmdb_dir must be provided")

    if config.score_file is not None:
        scores = _read_scores(config.score_file)
        manifest = manifest.merge(scores, on=["target_id", "decoy_id"], how="left")

    manifest["label"] = _assign_label(
        manifest, policy=config.label_policy, tm_threshold=config.tm_threshold
    )
    manifest["sample_id"] = manifest["target_id"] + "/" + manifest["decoy_id"]

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(config.output_path, index=False)
    return manifest
