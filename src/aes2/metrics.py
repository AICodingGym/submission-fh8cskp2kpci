from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score

SCORE_MIN = 1
SCORE_MAX = 6


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def apply_thresholds(predictions: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    thresholds = np.asarray(thresholds, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    labels = np.digitize(predictions, thresholds) + SCORE_MIN
    return np.clip(labels, SCORE_MIN, SCORE_MAX).astype(int)


def optimize_thresholds(
    y_true: np.ndarray,
    raw_predictions: np.ndarray,
    initial_thresholds: tuple[float, float, float, float, float] = (
        1.5,
        2.5,
        3.5,
        4.5,
        5.5,
    ),
) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=int)
    raw_predictions = np.asarray(raw_predictions, dtype=float)

    def objective(thresholds: np.ndarray) -> float:
        if np.any(np.diff(thresholds) <= 0):
            return 1.0
        y_pred = apply_thresholds(raw_predictions, thresholds)
        return -quadratic_weighted_kappa(y_true, y_pred)

    result = minimize(
        objective,
        np.asarray(initial_thresholds, dtype=float),
        method="Nelder-Mead",
        options={"maxiter": 1000},
    )
    return np.sort(result.x)
