"""
Automated feature generation.
Creates polynomial interactions, ratio features, and statistical aggregations.
"""
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures

from src.utils.logger import get_logger

log = get_logger(__name__)


class FeatureGenerator(BaseEstimator, TransformerMixin):
    """Generates new features via polynomial expansion, interactions, and ratios.

    Selectively generates features up to `max_generated_features` to avoid
    curse of dimensionality.

    Example:
        >>> gen = FeatureGenerator(polynomial_degree=2, interaction_features=True)
        >>> X_expanded = gen.fit_transform(X_train)
    """

    def __init__(
        self,
        polynomial_degree: int = 2,
        interaction_features: bool = True,
        ratio_features: bool = True,
        max_generated_features: int = 100,
        include_bias: bool = False,
    ) -> None:
        self.polynomial_degree = polynomial_degree
        self.interaction_features = interaction_features
        self.ratio_features = ratio_features
        self.max_generated_features = max_generated_features
        self.include_bias = include_bias

        self._poly: Optional[PolynomialFeatures] = None
        self._numeric_cols: list[str] = []
        self._ratio_pairs: list[tuple[str, str]] = []
        self._generated_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureGenerator":
        X = self._validate(X)
        self._numeric_cols = X.select_dtypes(include="number").columns.tolist()

        budget = self.max_generated_features
        self._generated_cols = []

        if self.interaction_features and len(self._numeric_cols) >= 2:
            # Determine how many interaction pairs we can create
            pairs = list(combinations(self._numeric_cols, 2))[:budget // 2]
            self._ratio_pairs = pairs[:budget // 4] if self.ratio_features else []

        if self.polynomial_degree >= 2 and self._numeric_cols:
            # Limit features for polynomial expansion
            n_cols = min(len(self._numeric_cols), 10)
            cols_to_poly = self._numeric_cols[:n_cols]
            self._poly = PolynomialFeatures(
                degree=self.polynomial_degree,
                include_bias=self.include_bias,
                interaction_only=not self.interaction_features,
            )
            sample = X[cols_to_poly].fillna(0)
            self._poly.fit(sample)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._validate(X).copy()
        new_frames = []

        # Polynomial + interaction features
        if self._poly is not None:
            n_cols = min(len(self._numeric_cols), 10)
            cols_to_poly = [c for c in self._numeric_cols[:n_cols] if c in X.columns]
            if cols_to_poly:
                sample = X[cols_to_poly].fillna(0)
                poly_arr = self._poly.transform(sample)
                feat_names = self._poly.get_feature_names_out(cols_to_poly)
                # Drop original feature names to avoid duplicates
                orig_set = set(cols_to_poly)
                new_feat_mask = [f not in orig_set for f in feat_names]
                poly_df = pd.DataFrame(
                    poly_arr[:, new_feat_mask],
                    columns=[f"poly__{f}" for f in feat_names if f not in orig_set],
                    index=X.index,
                )
                new_frames.append(poly_df)

        # Ratio features
        if self.ratio_features and self._ratio_pairs:
            ratio_data = {}
            for col1, col2 in self._ratio_pairs:
                if col1 in X.columns and col2 in X.columns:
                    denom = X[col2].replace(0, np.nan)
                    ratio_data[f"ratio__{col1}__{col2}"] = (X[col1] / denom).fillna(0)

            if ratio_data:
                new_frames.append(pd.DataFrame(ratio_data, index=X.index))

        if new_frames:
            X = pd.concat([X] + new_frames, axis=1)
            # Cap to max features
            if len(X.columns) > self.max_generated_features + len(self._numeric_cols):
                orig_cols = X.columns[:len(self._numeric_cols)].tolist()
                extra = [c for c in X.columns if c not in orig_cols]
                keep = extra[:self.max_generated_features]
                X = X[orig_cols + keep]

        log.debug(f"Feature generation: {X.shape[1]} total features")
        return X

    @staticmethod
    def _validate(X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(X).__name__}")
        return X
