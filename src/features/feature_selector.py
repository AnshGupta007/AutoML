"""
Feature selection methods.
Supports mutual information, chi-squared, RFECV, L1 (Lasso),
and variance-based selection. Returns reduced feature sets.
"""
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    VarianceThreshold,
    chi2,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)

from src.utils.logger import get_logger

log = get_logger(__name__)


class SmartFeatureSelector(BaseEstimator, TransformerMixin):
    """Selects the most informative features using multiple strategies.

    Supports:
    - ``mutual_info``: Mutual information.
    - ``f_test``: ANOVA F-test (classification) or F-regression.
    - ``lasso``: L1 regularization model.
    - ``variance``: Variance threshold.
    - ``correlation``: Remove highly correlated features.

    Example:
        >>> selector = SmartFeatureSelector(method="mutual_info", max_features=30)
        >>> X_selected = selector.fit_transform(X_train, y_train)
    """

    def __init__(
        self,
        method: str = "mutual_info",
        max_features: Optional[int] = 50,
        min_importance: float = 0.001,
        task_type: str = "auto",
        variance_threshold: float = 0.0,
        correlation_threshold: float = 0.95,
        random_state: int = 42,
    ) -> None:
        self.method = method
        self.max_features = max_features
        self.min_importance = min_importance
        self.task_type = task_type
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.random_state = random_state

        self._selector = None
        self._selected_cols: list[str] = []
        self._importances: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y=None) -> "SmartFeatureSelector":
        X = self._validate(X)

        # Step 1: remove variance-zero columns
        vt = VarianceThreshold(threshold=self.variance_threshold)
        vt.fit(X.select_dtypes(include="number").fillna(0))
        zero_var_cols = [
            c for c, s in zip(X.select_dtypes(include="number").columns, vt.get_support())
            if not s
        ]
        if zero_var_cols:
            log.warning(f"Dropping {len(zero_var_cols)} zero-variance columns")
            X = X.drop(columns=zero_var_cols)

        # Step 2: remove highly correlated columns
        X = self._remove_correlated(X)

        numeric_X = X.select_dtypes(include="number").fillna(0)
        self._numeric_cols_at_fit = numeric_X.columns.tolist()
        self._all_input_cols = X.columns.tolist()

        if y is None or len(numeric_X.columns) == 0:
            self._selected_cols = list(X.columns)
            return self

        # Detect task type
        task = self.task_type
        if task == "auto":
            task = "classification" if y.nunique() <= 20 else "regression"

        if self.method in ("mutual_info", "f_test"):
            self._fit_univariate(numeric_X, y, task)
        elif self.method == "lasso":
            self._fit_lasso(numeric_X, y, task)
        else:
            self._selected_cols = list(X.columns)

        log.info(
            f"Feature selection: {len(self._selected_cols)} features selected "
            f"(from {len(X.columns)}) via '{self.method}'"
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._validate(X)
        cols = [c for c in self._selected_cols if c in X.columns]
        return X[cols]

    @property
    def feature_importances(self) -> pd.Series:
        return (
            pd.Series(self._importances)
            .sort_values(ascending=False)
        )

    def _fit_univariate(self, X: pd.DataFrame, y: pd.Series, task: str) -> None:
        if self.method == "mutual_info":
            score_fn = (
                mutual_info_classif if task == "classification" else mutual_info_regression
            )
        else:
            score_fn = f_classif if task == "classification" else f_regression

        k = min(self.max_features or len(X.columns), len(X.columns))
        selector = SelectKBest(score_func=score_fn, k=k)
        selector.fit(X.fillna(0), y)
        scores = selector.scores_

        self._importances = {col: float(s) for col, s in zip(X.columns, scores)}
        # Select by k best and min importance
        sorted_cols = sorted(self._importances, key=self._importances.get, reverse=True)
        self._selected_cols = [
            c for c in sorted_cols
            if self._importances[c] >= self.min_importance
        ][:self.max_features or len(sorted_cols)]

    def _fit_lasso(self, X: pd.DataFrame, y: pd.Series, task: str) -> None:
        from sklearn.linear_model import Lasso, LogisticRegression
        if task == "classification":
            model = LogisticRegression(
                C=1.0, penalty="l1", solver="saga",
                max_iter=500, random_state=self.random_state,
            )
        else:
            model = Lasso(alpha=0.01, max_iter=1000, random_state=self.random_state)

        sfm = SelectFromModel(model, max_features=self.max_features, threshold=self.min_importance)
        sfm.fit(X.fillna(0), y)
        mask = sfm.get_support()
        self._selected_cols = [c for c, s in zip(X.columns, mask) if s]

    def _remove_correlated(self, X: pd.DataFrame) -> pd.DataFrame:
        numeric = X.select_dtypes(include="number")
        if numeric.shape[1] < 2:
            return X
        corr_matrix = numeric.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [
            col for col in upper.columns
            if any(upper[col] > self.correlation_threshold)
        ]
        if to_drop:
            log.debug(
                f"Dropping {len(to_drop)} highly correlated columns "
                f"(threshold={self.correlation_threshold})"
            )
            X = X.drop(columns=to_drop)
        return X

    @staticmethod
    def _validate(X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(X).__name__}")
        return X
