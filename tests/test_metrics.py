import numpy as np

from aes2.metrics import apply_thresholds, quadratic_weighted_kappa


def test_apply_thresholds_outputs_integer_scores() -> None:
    raw = np.array([0.9, 1.6, 2.6, 3.6, 4.6, 5.7, 9.0])
    pred = apply_thresholds(raw, np.array([1.5, 2.5, 3.5, 4.5, 5.5]))
    assert pred.tolist() == [1, 2, 3, 4, 5, 6, 6]


def test_quadratic_weighted_kappa_perfect_score() -> None:
    y = np.array([1, 2, 3, 4, 5, 6])
    assert quadratic_weighted_kappa(y, y) == 1.0
