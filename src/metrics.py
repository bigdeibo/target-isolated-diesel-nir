"""Regression metrics calculated from complete out-of-fold predictions."""
from __future__ import annotations

import numpy as np


def rmsep(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def regression_metrics(y_true, y_pred) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    error = rmsep(y, prediction)
    return {
        "n": len(y),
        "rmsep": error,
        "r_squared": float(1 - np.sum((y - prediction) ** 2) / np.sum((y - y.mean()) ** 2)),
        "mae": float(np.mean(np.abs(prediction - y))),
        "rpd": float(np.std(y, ddof=1) / error),
        "bias": float(np.mean(prediction - y)),
    }


def relative_improvement(reference_rmsep: float, candidate_rmsep: float) -> float:
    return 100.0 * (reference_rmsep - candidate_rmsep) / reference_rmsep
