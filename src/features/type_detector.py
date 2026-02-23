"""
Automatic column type detection.
Infers semantic types (numeric, categorical, datetime, text, boolean, id)
beyond simple dtype inspection.
"""
from typing import Optional

import pandas as pd
import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)

DETECTED_TYPES = [
    "numeric_continuous",
    "numeric_discrete",
    "categorical_low",
    "categorical_high",
    "boolean",
    "datetime",
    "text",
    "id",
    "constant",
    "unknown",
]


class TypeDetector:
    """Detects semantic column types from a DataFrame.

    Goes beyond dtype to infer whether a numeric column is discrete/continuous,
    whether an object column is categorical or text, and detects datetime/ID columns.

    Example:
        >>> detector = TypeDetector()
        >>> types = detector.detect(df)
        >>> print(types["age"])  # "numeric_continuous"
    """

    def __init__(
        self,
        categorical_threshold: int = 50,
        text_avg_length_threshold: int = 20,
        id_uniqueness_threshold: float = 0.95,
        discrete_skew_threshold: float = 0.01,
    ) -> None:
        self.categorical_threshold = categorical_threshold
        self.text_avg_length_threshold = text_avg_length_threshold
        self.id_uniqueness_threshold = id_uniqueness_threshold
        self.discrete_skew_threshold = discrete_skew_threshold

    def detect(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> dict[str, str]:
        """Detect types for all columns in the DataFrame.

        Args:
            df: Input DataFrame.
            target_column: Target column to skip from detection.

        Returns:
            Dict mapping column name → detected type string.
        """
        types = {}
        for col in df.columns:
            if col == target_column:
                types[col] = "target"
                continue
            types[col] = self._detect_column(df[col])

        log.info(
            f"Type detection complete: "
            + ", ".join(f"{t}={sum(1 for v in types.values() if v == t)}"
                        for t in DETECTED_TYPES if any(v == t for v in types.values()))
        )
        return types

    def _detect_column(self, series: pd.Series) -> str:
        """Detect type of a single Series."""
        if series.nunique() <= 1:
            return "constant"

        # Boolean
        if series.dtype == bool or set(series.dropna().unique()) <= {0, 1, True, False}:
            return "boolean"

        # Datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        if series.dtype == object:
            if self._is_datetime_string(series):
                return "datetime"

        # Numeric
        if pd.api.types.is_numeric_dtype(series):
            return self._classify_numeric(series)

        # Object/string types
        if series.dtype == object or hasattr(series, "cat"):
            return self._classify_object(series)

        return "unknown"

    def _classify_numeric(self, series: pd.Series) -> str:
        """Classify numeric column as continuous, discrete, boolean, or id."""
        n_unique = series.nunique()
        n_total = len(series.dropna())

        # ID-like column
        uniqueness = n_unique / n_total if n_total > 0 else 0
        if uniqueness >= self.id_uniqueness_threshold and n_unique > 100:
            return "id"

        # Discrete: only integer values
        non_null = series.dropna()
        if (non_null == non_null.astype(int)).all() and n_unique <= self.categorical_threshold:
            return "numeric_discrete"

        return "numeric_continuous"

    def _classify_object(self, series: pd.Series) -> str:
        """Classify object column as categorical (low/high cardinality), text, or id."""
        n_unique = series.nunique()
        n_total = len(series.dropna())

        # ID-like
        uniqueness = n_unique / n_total if n_total > 0 else 0
        if uniqueness >= self.id_uniqueness_threshold and n_unique > 100:
            return "id"

        # Check average string length for text detection
        try:
            avg_len = series.dropna().astype(str).str.len().mean()
            if avg_len > self.text_avg_length_threshold and n_unique > self.categorical_threshold:
                return "text"
        except Exception:
            pass

        if n_unique <= self.categorical_threshold:
            return "categorical_low"
        return "categorical_high"

    @staticmethod
    def _is_datetime_string(series: pd.Series) -> bool:
        """Heuristic check if an object column contains datetime strings."""
        sample = series.dropna().head(50)
        if len(sample) == 0:
            return False
        try:
            parsed = pd.to_datetime(sample, infer_datetime_format=True, errors="coerce")
            return parsed.notna().mean() >= 0.80
        except Exception:
            return False

    def get_columns_by_type(
        self, types: dict[str, str], include: list[str]
    ) -> list[str]:
        """Return column names matching any of the given types.

        Args:
            types: Output of .detect().
            include: List of type strings to include.

        Returns:
            List of column names.
        """
        return [col for col, t in types.items() if t in include]
