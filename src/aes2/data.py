from pathlib import Path

import pandas as pd

from aes2.paths import DATA_DIR


def load_train(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    train = pd.read_csv(data_dir / "train.csv")
    expected = {"essay_id", "full_text", "score"}
    missing = expected.difference(train.columns)
    if missing:
        raise ValueError(f"train.csv is missing columns: {sorted(missing)}")
    return train


def load_test(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    test = pd.read_csv(data_dir / "test.csv")
    expected = {"essay_id", "full_text"}
    missing = expected.difference(test.columns)
    if missing:
        raise ValueError(f"test.csv is missing columns: {sorted(missing)}")
    return test


def load_sample_submission(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    expected = {"essay_id", "score"}
    missing = expected.difference(sample.columns)
    if missing:
        raise ValueError(f"sample_submission.csv is missing columns: {sorted(missing)}")
    return sample
