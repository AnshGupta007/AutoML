"""
Missing value imputation strategies.
Supports mean/median/mode, KNN, iterative (MICE), and constant imputation.
Provides a unified sklearn-compatible transformer interface.
"""
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
# IterativeImputer is still experimental in sklearn 1.x — the flag must be imported first.
# This try/except makes the code forward-compatible for when it graduates to stable.
try:
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
except ImportError:
    pass  # Already stable in future sklearn versions
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

from src.utils.logger import get_logger

log = get_logger(__name__)

NUMERIC_STRATEGIES = ["mean", "median", "knn", "iterative", "constant", "zero"]
CATEGORICAL_STRATEGIES = ["most_frequent", "constant"]


class SmartImputer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible imputer that handles numeric and categorical columns separately.

    Automatically fits the appropriate strategy per column type.

    Example:
        >>> imputer = SmartImputer(numeric_strategy="median", categorical_strategy="most_frequent")
        >>> X_imputed = imputer.fit_transform(X_train)
        >>> X_test_imputed = imputer.transform(X_test)
    """

    def __init__(
        self,
        numeric_strategy: str = "median",
        categorical_strategy: str = "most_frequent",
        fill_value: Union[str, int, float] = "MISSING",
        knn_neighbors: int = 5,
        iterative_max_iter: int = 10,
        add_indicator: bool = False,
    ) -> None:
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.fill_value = fill_value
        self.knn_neighbors = knn_neighbors
        self.iterative_max_iter = iterative_max_iter
        self.add_indicator = add_indicator

        self._numeric_cols: list[str] = []
        self._cat_cols: list[str] = []
        self._numeric_imputer = None
        self._cat_imputer = None

    def fit(self, X: pd.DataFrame, y=None) -> "SmartImputer":
        """Fit imputers on the training data.

        Args:
            X: Feature DataFrame (numeric + categorical).
            y: Ignored.

        Returns:
            self
        """
        X = self._validate_input(X)
        self._numeric_cols = X.select_dtypes(include="number").columns.tolist()
        self._cat_cols = X.select_dtypes(exclude="number").columns.tolist()

        if self._numeric_cols:
            self._numeric_imputer = self._build_numeric_imputer()
            self._numeric_imputer.fit(X[self._numeric_cols])
            log.debug(
                f"Fitted numeric imputer ({self.numeric_strategy}) "
                f"on {len(self._numeric_cols)} columns"
            )

        if self._cat_cols:
            self._cat_imputer = SimpleImputer(
                strategy=self.categorical_strategy,
                fill_value=str(self.fill_value),
            )
            self._cat_imputer.fit(X[self._cat_cols])
            log.debug(
                f"Fitted categorical imputer ({self.categorical_strategy}) "
                f"on {len(self._cat_cols)} columns"
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values.

        Args:
            X: Feature DataFrame.

        Returns:
            Imputed DataFrame with same columns.
        """
        X = self._validate_input(X).copy()

        if self._numeric_cols and self._numeric_imputer is not None:
            # Only impute columns that exist in X
            cols = [c for c in self._numeric_cols if c in X.columns]
            if cols:
                X[cols] = self._numeric_imputer.transform(X[cols])

        if self._cat_cols and self._cat_imputer is not None:
            cols = [c for c in self._cat_cols if c in X.columns]
            if cols:
                X[cols] = self._cat_imputer.transform(X[cols])

        null_remaining = X.isnull().sum().sum()
        if null_remaining > 0:
            log.warning(f"After imputation, {null_remaining} nulls remain.")

        return X

    def get_missing_stats(self, X: pd.DataFrame) -> pd.DataFrame:
        """Get per-column missing value statistics.

        Args:
            X: DataFrame to analyze.

        Returns:
            DataFrame with missing count, percent, and dtype per column.
        """
        stats = pd.DataFrame({
            "missing_count": X.isnull().sum(),
            "missing_pct": (X.isnull().mean() * 100).round(2),
            "dtype": X.dtypes.astype(str),
        })
        return stats[stats["missing_count"] > 0].sort_values("missing_pct", ascending=False)

    def _build_numeric_imputer(self):
        if self.numeric_strategy == "knn":
            return KNNImputer(n_neighbors=self.knn_neighbors, add_indicator=self.add_indicator)
        elif self.numeric_strategy == "iterative":
            return IterativeImputer(
                max_iter=self.iterative_max_iter,
                random_state=42,
                add_indicator=self.add_indicator,
            )
        elif self.numeric_strategy == "zero":
            return SimpleImputer(strategy="constant", fill_value=0)
        elif self.numeric_strategy == "constant":
            return SimpleImputer(
                strategy="constant",
                fill_value=self.fill_value,
                add_indicator=self.add_indicator,
            )
        elif self.numeric_strategy in ("mean", "median"):
            return SimpleImputer(
                strategy=self.numeric_strategy,
                add_indicator=self.add_indicator,
            )
        else:
            raise ValueError(f"Unknown numeric_strategy: '{self.numeric_strategy}'")

    @staticmethod
    def _validate_input(X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(X).__name__}")
        return X
