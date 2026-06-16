from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from aes2.data import load_sample_submission, load_test, load_train
from aes2.features import build_tfidf_features
from aes2.metrics import (
    apply_thresholds,
    optimize_thresholds,
    quadratic_weighted_kappa,
)
from aes2.paths import MODEL_DIR, SUBMISSION_DIR, ensure_output_dirs

RANDOM_STATE = 42
N_SPLITS = 5


def build_model() -> Pipeline:
    return Pipeline(
        [
            ("features", build_tfidf_features()),
            ("model", Ridge(alpha=10.0, random_state=RANDOM_STATE)),
        ]
    )


def main() -> None:
    ensure_output_dirs()
    train = load_train()
    test = load_test()
    sample_submission = load_sample_submission()

    x = train["full_text"].astype(str).tolist()
    y = train["score"].astype(int).to_numpy()
    test_x = test["full_text"].astype(str).tolist()

    oof_raw = np.zeros(len(train), dtype=float)
    test_raw = np.zeros(len(test), dtype=float)
    fold_scores = []

    splitter = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(x, y), start=1):
        model = build_model()
        x_train = [x[i] for i in train_idx]
        y_train = y[train_idx]
        x_valid = [x[i] for i in valid_idx]
        y_valid = y[valid_idx]

        model.fit(x_train, y_train)
        valid_raw = model.predict(x_valid)
        oof_raw[valid_idx] = valid_raw
        test_raw += model.predict(test_x) / N_SPLITS

        fold_thresholds = optimize_thresholds(y_valid, valid_raw)
        valid_pred = apply_thresholds(valid_raw, fold_thresholds)
        fold_qwk = quadratic_weighted_kappa(y_valid, valid_pred)
        fold_scores.append(fold_qwk)
        print(f"fold={fold} qwk={fold_qwk:.5f} thresholds={fold_thresholds}")

    thresholds = optimize_thresholds(y, oof_raw)
    oof_pred = apply_thresholds(oof_raw, thresholds)
    cv_qwk = quadratic_weighted_kappa(y, oof_pred)
    print(f"cv_qwk={cv_qwk:.5f}")
    print(f"fold_mean={np.mean(fold_scores):.5f} fold_std={np.std(fold_scores):.5f}")
    print(f"global_thresholds={thresholds}")

    final_model = build_model()
    final_model.fit(x, y)
    joblib.dump(
        {"model": final_model, "thresholds": thresholds, "cv_qwk": cv_qwk},
        MODEL_DIR / "tfidf_ridge.joblib",
    )

    final_raw = final_model.predict(test_x)
    final_pred = apply_thresholds(final_raw, thresholds)
    submission = pd.DataFrame(
        {
            "essay_id": sample_submission["essay_id"],
            "score": final_pred,
        }
    )
    submission_path = SUBMISSION_DIR / "tfidf_ridge_submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"wrote {submission_path}")


if __name__ == "__main__":
    main()
