"""
Numerical feature scaling strategies.
Wraps sklearn scalers behind a uniform interface.
"""
from typing import Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from src.utils.logger import get_logger

log = get_logger(__name__)

SCALING_STRATEGIES = ["standard", "minmax", "robust", "maxabs", "power", "quantile", "none"]


class SmartScaler(BaseEstimator, TransformerMixin):
    """Scales numeric features with configurable strategy.

    Only scales numeric columns; non-numeric columns are passed through unchanged.

    Example:
        >>> scaler = SmartScaler(strategy="robust")
        >>> X_scaled = scaler.fit_transform(X_train)
        >>> X_test_scaled = scaler.transform(X_test)
    """

    def __init__(
        self,
        strategy: str = "robust",
        quantile_range: tuple[float, float] = (25.0, 75.0),
        cols: Optional[list[str]] = None,
    ) -> None:
        self.strategy = strategy
        self.quantile_range = quantile_range
        self.cols = cols

        self._scaler = None
        self._numeric_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> "SmartScaler":
        """Fit scaler on numeric columns."""
        if self.strategy == "none":
            return self

        X = self._validate(X)
        if self.cols:
            self._numeric_cols = [c for c in self.cols if c in X.columns]
        else:
            self._numeric_cols = X.select_dtypes(include="number").columns.tolist()

        if not self._numeric_cols:
            log.debug("No numeric columns to scale.")
            return self

        self._scaler = self._build_scaler()
        self._scaler.fit(X[self._numeric_cols])
        log.debug(
            f"Fitted {self.strategy} scaler on {len(self._numeric_cols)} columns"
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale numeric columns."""
        if self.strategy == "none" or self._scaler is None:
            return X

        X = self._validate(X).copy()
        cols = [c for c in self._numeric_cols if c in X.columns]
        if cols:
            X[cols] = self._scaler.transform(X[cols])
        return X

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse scale (where supported)."""
        if self._scaler is None:
            return X
        X = X.copy()
        cols = [c for c in self._numeric_cols if c in X.columns]
        if cols and hasattr(self._scaler, "inverse_transform"):
            X[cols] = self._scaler.inverse_transform(X[cols])
        return X

    def _build_scaler(self):
        if self.strategy == "standard":
            return StandardScaler()
        elif self.strategy == "minmax":
            return MinMaxScaler()
        elif self.strategy == "robust":
            return RobustScaler(quantile_range=self.quantile_range)
        elif self.strategy == "maxabs":
            return MaxAbsScaler()
        elif self.strategy == "power":
            return PowerTransformer(method="yeo-johnson")
        elif self.strategy == "quantile":
            return QuantileTransformer(output_distribution="normal", random_state=42)
        else:
            raise ValueError(
                f"Unknown scaling strategy '{self.strategy}'. "
                f"Choose from: {SCALING_STRATEGIES}"
            )

    @staticmethod
    def _validate(X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(X).__name__}")
        return X
