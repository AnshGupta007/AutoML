"""
Categorical encoding strategies.
Wraps sklearn and category-encoders for one-hot, ordinal, target,
weight-of-evidence, binary, and frequency encoding.
"""
from typing import Optional, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder

from src.utils.logger import get_logger

log = get_logger(__name__)

ENCODING_STRATEGIES = ["onehot", "ordinal", "target", "woe", "binary", "frequency", "hashing"]


class SmartEncoder(BaseEstimator, TransformerMixin):
    """Encodes categorical columns using configurable strategies.

    Handles high-cardinality columns separately with a dedicated strategy.
    Supports unknown categories gracefully at inference time.

    Example:
        >>> encoder = SmartEncoder(default_strategy="target", high_cardinality_strategy="target")
        >>> X_enc = encoder.fit_transform(X_train, y_train)
        >>> X_test_enc = encoder.transform(X_test)
    """

    def __init__(
        self,
        default_strategy: str = "target",
        high_cardinality_strategy: str = "target",
        high_cardinality_threshold: int = 20,
        handle_unknown: str = "value",
        cols: Optional[list[str]] = None,
    ) -> None:
        self.default_strategy = default_strategy
        self.high_cardinality_strategy = high_cardinality_strategy
        self.high_cardinality_threshold = high_cardinality_threshold
        self.handle_unknown = handle_unknown
        self.cols = cols

        self._encoders: dict = {}
        self._cat_cols: list[str] = []
        self._high_card_cols: list[str] = []
        self._low_card_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> "SmartEncoder":
        """Fit encoders on categorical columns.

        Args:
            X: Feature DataFrame.
            y: Target series (required for target/WoE encoding).

        Returns:
            self
        """
        X = X.copy()
        if self.cols:
            self._cat_cols = [c for c in self.cols if c in X.columns]
        else:
            self._cat_cols = X.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

        self._high_card_cols = [
            c for c in self._cat_cols
            if X[c].nunique() > self.high_cardinality_threshold
        ]
        self._low_card_cols = [
            c for c in self._cat_cols
            if c not in self._high_card_cols
        ]

        # Fit low-cardinality encoder
        if self._low_card_cols:
            enc = self._build_encoder(self.default_strategy, self._low_card_cols, y)
            if enc is not None:
                enc.fit(X[self._low_card_cols], y)
                self._encoders["low"] = enc
                log.debug(
                    f"Fitted {self.default_strategy} encoder on "
                    f"{len(self._low_card_cols)} low-cardinality columns"
                )

        # Fit high-cardinality encoder
        if self._high_card_cols:
            enc = self._build_encoder(
                self.high_cardinality_strategy, self._high_card_cols, y
            )
            if enc is not None:
                enc.fit(X[self._high_card_cols], y)
                self._encoders["high"] = enc
                log.debug(
                    f"Fitted {self.high_cardinality_strategy} encoder on "
                    f"{len(self._high_card_cols)} high-cardinality columns"
                )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical columns.

        Args:
            X: Feature DataFrame.

        Returns:
            Encoded DataFrame.
        """
        X = X.copy()

        if "low" in self._encoders and self._low_card_cols:
            cols = [c for c in self._low_card_cols if c in X.columns]
            if cols:
                try:
                    X = self._apply_encoder(X, self._encoders["low"], cols)
                except Exception as e:
                    log.warning(f"Low-card encoding failed: {e}")

        if "high" in self._encoders and self._high_card_cols:
            cols = [c for c in self._high_card_cols if c in X.columns]
            if cols:
                try:
                    X = self._apply_encoder(X, self._encoders["high"], cols)
                except Exception as e:
                    log.warning(f"High-card encoding failed: {e}")

        return X

    def _build_encoder(self, strategy: str, cols: list[str], y):
        """Build the appropriate encoder for a strategy."""
        try:
            if strategy == "onehot":
                from sklearn.preprocessing import OneHotEncoder
                return OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                )
            elif strategy == "ordinal":
                return OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                )
            elif strategy == "target":
                import category_encoders as ce
                return ce.TargetEncoder(cols=cols, handle_unknown="value")
            elif strategy == "woe":
                import category_encoders as ce
                return ce.WOEEncoder(cols=cols)
            elif strategy == "binary":
                import category_encoders as ce
                return ce.BinaryEncoder(cols=cols)
            elif strategy == "frequency":
                return _FrequencyEncoder(cols=cols)
            elif strategy == "hashing":
                import category_encoders as ce
                return ce.HashingEncoder(cols=cols, n_components=16)
            else:
                log.warning(f"Unknown encoding strategy '{strategy}'; using ordinal")
                return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        except ImportError as e:
            log.warning(f"Encoder '{strategy}' not available: {e}. Using ordinal.")
            return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    @staticmethod
    def _apply_encoder(X: pd.DataFrame, encoder, cols: list[str]) -> pd.DataFrame:
        """Apply encoder transform and merge back into DataFrame."""
        transformed = encoder.transform(X[cols])
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        if isinstance(transformed, pd.DataFrame):
            X = X.drop(columns=cols)
            X = pd.concat([X, transformed], axis=1)
        else:
            # Array output (e.g. OneHotEncoder) — create column names
            if len(transformed.shape) == 2 and transformed.shape[1] != len(cols):
                new_cols = [f"{c}_{i}" for c in cols for i in range(
                    transformed.shape[1] // len(cols) or 1
                )][:transformed.shape[1]]
                X = X.drop(columns=cols)
                X[new_cols] = transformed
            else:
                X[cols] = transformed

        return X


class _FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Simple frequency (count) encoder."""

    def __init__(self, cols: Optional[list[str]] = None) -> None:
        self.cols = cols
        self._freq_maps: dict[str, dict] = {}

    def fit(self, X: pd.DataFrame, y=None) -> "_FrequencyEncoder":
        cols = self.cols or X.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in cols:
            self._freq_maps[col] = X[col].value_counts(normalize=True).to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, freq_map in self._freq_maps.items():
            if col in X.columns:
                X[col] = X[col].map(freq_map).fillna(0.0)
        return X
