"""
Text feature extraction using TF-IDF, count vectorization, or hashing.
Handles free-text columns detected automatically or specified by user.
"""
from typing import Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfVectorizer,
)

from src.utils.logger import get_logger

log = get_logger(__name__)


class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts features from text columns via TF-IDF, count, or hashing.

    Example:
        >>> extractor = TextFeatureExtractor(method="tfidf", max_features=500)
        >>> X_out = extractor.fit_transform(X_train)
    """

    def __init__(
        self,
        text_cols: Optional[list[str]] = None,
        auto_detect: bool = True,
        method: str = "tfidf",       # tfidf | count | hashing
        max_features: int = 500,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        drop_original: bool = True,
        avg_length_threshold: int = 20,
    ) -> None:
        self.text_cols = text_cols
        self.auto_detect = auto_detect
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.drop_original = drop_original
        self.avg_length_threshold = avg_length_threshold

        self._detected_cols: list[str] = []
        self._vectorizers: dict[str, BaseEstimator] = {}

    def fit(self, X: pd.DataFrame, y=None) -> "TextFeatureExtractor":
        if self.text_cols:
            self._detected_cols = [c for c in self.text_cols if c in X.columns]
        elif self.auto_detect:
            self._detected_cols = self._auto_detect_text(X)

        for col in self._detected_cols:
            vec = self._build_vectorizer(col)
            text_data = X[col].fillna("").astype(str)
            vec.fit(text_data)
            self._vectorizers[col] = vec
            log.debug(f"Fitted {self.method} vectorizer on column '{col}'")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        frames = [X]

        for col, vec in self._vectorizers.items():
            if col not in X.columns:
                continue
            try:
                text_data = X[col].fillna("").astype(str)
                mat = vec.transform(text_data)
                if hasattr(mat, "toarray"):
                    mat = mat.toarray()

                if hasattr(vec, "get_feature_names_out"):
                    feat_names = [f"{col}__{f}" for f in vec.get_feature_names_out()]
                else:
                    feat_names = [f"{col}__f{i}" for i in range(mat.shape[1])]

                feat_df = pd.DataFrame(mat, columns=feat_names, index=X.index)
                frames.append(feat_df)

                if self.drop_original:
                    X = X.drop(columns=[col])
                    frames[0] = X
            except Exception as e:
                log.warning(f"Text feature extraction failed for '{col}': {e}")

        return pd.concat(frames, axis=1) if len(frames) > 1 else X

    def _build_vectorizer(self, col: str):
        kwargs = dict(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
        )
        if self.method == "tfidf":
            return TfidfVectorizer(**kwargs)
        elif self.method == "count":
            return CountVectorizer(**kwargs)
        elif self.method == "hashing":
            return HashingVectorizer(
                ngram_range=self.ngram_range,
                n_features=self.max_features,
                alternate_sign=False,
            )
        else:
            log.warning(f"Unknown text method '{self.method}', using tfidf")
            return TfidfVectorizer(**kwargs)

    def _auto_detect_text(self, X: pd.DataFrame) -> list[str]:
        detected = []
        for col in X.select_dtypes(include="object").columns:
            avg_len = X[col].dropna().astype(str).str.len().mean()
            if avg_len > self.avg_length_threshold:
                detected.append(col)
        if detected:
            log.info(f"Auto-detected text columns: {detected}")
        return detected
