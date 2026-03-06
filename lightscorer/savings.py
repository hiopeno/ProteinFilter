from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_savings_curve(
    y_score: np.ndarray,
    thresholds: np.ndarray,
    n_candidates: int = 10_000,
    af2_seconds_per_sample: float = 18.0,
    lightscorer_ms_per_sample: float = 5.0,
) -> pd.DataFrame:
    rows = []
    base_total_seconds = n_candidates * af2_seconds_per_sample
    ls_total_seconds = n_candidates * (lightscorer_ms_per_sample / 1000.0)
    for th in thresholds:
        keep_ratio = float((y_score >= th).mean())
        af2_after_seconds = n_candidates * keep_ratio * af2_seconds_per_sample
        total_seconds = ls_total_seconds + af2_after_seconds
        rows.append(
            {
                "threshold": float(th),
                "keep_ratio": keep_ratio,
                "reject_ratio": 1.0 - keep_ratio,
                "baseline_hours": base_total_seconds / 3600.0,
                "pipeline_hours": total_seconds / 3600.0,
                "hours_saved": (base_total_seconds - total_seconds) / 3600.0,
                "speedup": base_total_seconds / max(total_seconds, 1e-9),
            }
        )
    return pd.DataFrame(rows)
