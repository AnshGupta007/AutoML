"""
Linear model wrapper (Logistic Regression / Ridge / Lasso / ElasticNet).
"""
from typing import Optional

import numpy as np
import pandas as pd

from src.models.base_model import BaseModel
from src.utils.logger import get_logger

log = get_logger(__name__)


class LinearModel(BaseModel):
    """Sklearn linear model wrapper — auto-selects Logistic/Ridge/Lasso.

    Example:
        >>> model = LinearModel(task_type="classification")
        >>> model.fit(X_train, y_train)
    """

    def __init__(
        self,
        task_type: str = "auto",
        model_type: str = "auto",
        C: float = 1.0,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        max_iter: int = 1000,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs,
    ) -> None:
        super().__init__(name="linear", task_type=task_type, random_state=random_state)
        self.model_type = model_type
        self.params = {
            "C": C,
            "alpha": alpha,
            "l1_ratio": l1_ratio,
            "max_iter": max_iter,
            "random_state": random_state,
            "n_jobs": n_jobs,
        }

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs,
    ) -> "LinearModel":
        task = self._resolve_task(y_train)
        self._model = self._build_model(task)
        self._model.fit(X_train, y_train)
        self._feature_names = list(X_train.columns)
        self._is_fitted = True
        self.task_type = task
        log.info(f"Linear model ({self.model_type} / {task}) fitted.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        if self.task_type != "classification":
            raise NotImplementedError
        return self._model.predict_proba(X)

    def get_feature_importance(self) -> Optional[pd.Series]:
        self._check_fitted()
        if hasattr(self._model, "coef_"):
            coef = self._model.coef_
            if len(coef.shape) > 1:
                coef = np.abs(coef).mean(axis=0)
            return pd.Series(np.abs(coef), index=self._feature_names).sort_values(ascending=False)
        return None

    def _build_model(self, task: str):
        from sklearn.linear_model import (
            ElasticNet,
            Lasso,
            LogisticRegression,
            Ridge,
        )
        params = self.params.copy()

        if task == "classification":
            mt = self.model_type if self.model_type != "auto" else "logistic"
            if mt == "logistic":
                return LogisticRegression(
                    C=params["C"],
                    max_iter=params["max_iter"],
                    random_state=params["random_state"],
                    n_jobs=params["n_jobs"],
                    solver="lbfgs",
                    multi_class="auto",
                )
        else:
            mt = self.model_type if self.model_type != "auto" else "ridge"
            if mt == "ridge":
                return Ridge(alpha=params["alpha"], random_state=params["random_state"])
            elif mt == "lasso":
                return Lasso(alpha=params["alpha"], max_iter=params["max_iter"],
                             random_state=params["random_state"])
            elif mt == "elasticnet":
                return ElasticNet(
                    alpha=params["alpha"],
                    l1_ratio=params["l1_ratio"],
                    max_iter=params["max_iter"],
                    random_state=params["random_state"],
                )
        # fallback
        return Ridge(alpha=params["alpha"])

    def _resolve_task(self, y: pd.Series) -> str:
        if self.task_type != "auto":
            return self.task_type
        return "classification" if y.nunique() <= 20 else "regression"
