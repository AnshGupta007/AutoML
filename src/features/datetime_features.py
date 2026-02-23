"""
Datetime feature extraction.
Extracts temporal components and cyclical encodings from datetime columns.
"""
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils.logger import get_logger

log = get_logger(__name__)

DEFAULT_EXTRACTIONS = [
    "year", "month", "day", "hour", "minute",
    "dayofweek", "quarter", "weekofyear", "is_weekend",
    "dayofyear", "days_in_month",
]


class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts rich features from datetime columns.

    Supports standard component extraction (year, month, day, hour…)
    plus cyclical sin/cos encoding for periodic features.

    Example:
        >>> extractor = DatetimeFeatureExtractor(auto_detect=True)
        >>> X_transformed = extractor.fit_transform(X)
    """

    def __init__(
        self,
        datetime_cols: Optional[list[str]] = None,
        auto_detect: bool = True,
        extractions: list[str] = DEFAULT_EXTRACTIONS,
        cyclical_encoding: bool = True,
        drop_original: bool = True,
    ) -> None:
        self.datetime_cols = datetime_cols
        self.auto_detect = auto_detect
        self.extractions = extractions
        self.cyclical_encoding = cyclical_encoding
        self.drop_original = drop_original
        self._detected_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> "DatetimeFeatureExtractor":
        if self.datetime_cols:
            self._detected_cols = [c for c in self.datetime_cols if c in X.columns]
        elif self.auto_detect:
            self._detected_cols = self._auto_detect_datetime(X)
        log.debug(f"Datetime columns detected: {self._detected_cols}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        new_features: list[pd.DataFrame] = []

        for col in self._detected_cols:
            if col not in X.columns:
                continue
            try:
                dt = pd.to_datetime(X[col], infer_datetime_format=True, errors="coerce")
                features = self._extract_features(dt, col)
                new_features.append(features)
                if self.drop_original:
                    X = X.drop(columns=[col])
            except Exception as e:
                log.warning(f"Failed to extract datetime features from '{col}': {e}")

        if new_features:
            X = pd.concat([X] + new_features, axis=1)

        return X

    def _extract_features(self, dt: pd.Series, col: str) -> pd.DataFrame:
        features: dict[str, pd.Series] = {}
        prefix = f"{col}_"

        extraction_map = {
            "year": dt.dt.year,
            "month": dt.dt.month,
            "day": dt.dt.day,
            "hour": dt.dt.hour,
            "minute": dt.dt.minute,
            "second": dt.dt.second,
            "dayofweek": dt.dt.dayofweek,
            "quarter": dt.dt.quarter,
            "weekofyear": dt.dt.isocalendar().week.astype(int),
            "dayofyear": dt.dt.dayofyear,
            "days_in_month": dt.dt.days_in_month,
            "is_weekend": (dt.dt.dayofweek >= 5).astype(int),
            "is_month_start": dt.dt.is_month_start.astype(int),
            "is_month_end": dt.dt.is_month_end.astype(int),
            "is_quarter_start": dt.dt.is_quarter_start.astype(int),
            "is_quarter_end": dt.dt.is_quarter_end.astype(int),
        }

        for feat in self.extractions:
            if feat in extraction_map:
                features[f"{prefix}{feat}"] = extraction_map[feat]

        if self.cyclical_encoding:
            cyclical_map = {
                "month": 12,
                "dayofweek": 7,
                "hour": 24,
                "minute": 60,
                "dayofyear": 365,
                "weekofyear": 52,
            }
            for feat in self.extractions:
                if feat in cyclical_map and feat in extraction_map:
                    period = cyclical_map[feat]
                    values = extraction_map[feat].astype(float)
                    features[f"{prefix}{feat}_sin"] = np.sin(2 * np.pi * values / period)
                    features[f"{prefix}{feat}_cos"] = np.cos(2 * np.pi * values / period)

        return pd.DataFrame(features, index=dt.index)

    @staticmethod
    def _auto_detect_datetime(X: pd.DataFrame) -> list[str]:
        detected = []
        for col in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                detected.append(col)
            elif X[col].dtype == object:
                sample = X[col].dropna().head(50)
                try:
                    parsed = pd.to_datetime(sample, infer_datetime_format=True, errors="coerce")
                    if parsed.notna().mean() >= 0.8:
                        detected.append(col)
                except Exception:
                    pass
        return detected
