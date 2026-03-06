from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist
from skimage.transform import resize


def parse_ca_coords_from_pdb(pdb_path: Path) -> np.ndarray:
    coords = []
    with pdb_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            coords.append([x, y, z])
    if not coords:
        raise ValueError(f"No CA atoms found in: {pdb_path}")
    return np.asarray(coords, dtype=np.float32)


def distance_matrix_from_coords(coords: np.ndarray) -> np.ndarray:
    dist = cdist(coords, coords, metric="euclidean")
    return dist.astype(np.float32)


def normalize_distance_matrix(dist: np.ndarray, clip_max: float = 30.0) -> np.ndarray:
    clipped = np.clip(dist, 0.0, clip_max)
    return clipped / clip_max


def resize_matrix(dist: np.ndarray, out_size: int = 128) -> np.ndarray:
    out = resize(
        dist,
        (out_size, out_size),
        anti_aliasing=True,
        preserve_range=True,
        mode="reflect",
    )
    return out.astype(np.float32)


def build_feature_from_pdb(
    pdb_path: Path, out_size: int = 128, clip_max: float = 30.0
) -> np.ndarray:
    coords = parse_ca_coords_from_pdb(pdb_path)
    dist = distance_matrix_from_coords(coords)
    dist = normalize_distance_matrix(dist, clip_max=clip_max)
    dist = resize_matrix(dist, out_size=out_size)
    return dist


def synthetic_distance_matrix(
    n_res: int, good: bool, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    coords = np.cumsum(rng.normal(0, 1.0 if good else 2.0, size=(n_res, 3)), axis=0)
    if good:
        theta = np.linspace(0, 6 * np.pi, n_res)
        helix = np.stack([np.cos(theta), np.sin(theta), theta / (2 * np.pi)], axis=1)
        coords = 0.6 * coords + 0.4 * helix
    dist = distance_matrix_from_coords(coords.astype(np.float32))
    return normalize_distance_matrix(dist)
