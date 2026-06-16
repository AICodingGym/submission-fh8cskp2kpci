from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


class TextStatsTransformer(BaseEstimator, TransformerMixin):
    def fit(self, texts: list[str], y: Optional[np.ndarray] = None):
        return self

    def transform(self, texts: list[str]) -> sparse.csr_matrix:
        rows = []
        for text in texts:
            text = str(text)
            words = text.split()
            word_count = len(words)
            char_count = len(text)
            sentence_count = max(1, text.count(".") + text.count("!") + text.count("?"))
            avg_word_len = char_count / max(1, word_count)
            punctuation_count = sum(1 for ch in text if ch in ".,;:!?\"'()")
            paragraph_count = max(1, text.count("\n") + 1)
            rows.append(
                [
                    word_count,
                    char_count,
                    sentence_count,
                    avg_word_len,
                    punctuation_count,
                    paragraph_count,
                ]
            )
        return sparse.csr_matrix(np.asarray(rows, dtype=np.float32))


def build_tfidf_features() -> FeatureUnion:
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    return FeatureUnion(
        [
            ("word_tfidf", word_tfidf),
            ("char_tfidf", char_tfidf),
            ("text_stats", TextStatsTransformer()),
        ]
    )
