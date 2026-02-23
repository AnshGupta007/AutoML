"""
LightGBM model wrapper.
"""
from typing import Optional

import numpy as np
import pandas as pd

from src.models.base_model import BaseModel
from src.utils.logger import get_logger

log = get_logger(__name__)


class LightGBMModel(BaseModel):
    """LightGBM model wrapper for classification and regression.

    Example:
        >>> model = LightGBMModel(task_type="regression", n_estimators=1000)
        >>> model.fit(X_train, y_train, X_val, y_val)
    """

    def __init__(
        self,
        task_type: str = "auto",
        n_estimators: int = 1000,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        max_depth: int = -1,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        early_stopping_rounds: int = 50,
        n_jobs: int = -1,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(name="lightgbm", task_type=task_type, random_state=random_state)
        self.params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "n_jobs": n_jobs,
            "random_state": random_state,
            "verbose": -1,
            **kwargs,
        }
        self.early_stopping_rounds = early_stopping_rounds

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs,
    ) -> "LightGBMModel":
        import lightgbm as lgb

        task = self._resolve_task(y_train)
        model_params = self.params.copy()

        callbacks = [lgb.early_stopping(self.early_stopping_rounds, verbose=False)]

        if task == "classification":
            n_classes = y_train.nunique()
            if n_classes == 2:
                model_params["objective"] = "binary"
                self._model = lgb.LGBMClassifier(**model_params)
            else:
                model_params["objective"] = "multiclass"
                model_params["num_class"] = n_classes
                self._model = lgb.LGBMClassifier(**model_params)
        else:
            model_params["objective"] = "regression"
            self._model = lgb.LGBMRegressor(**model_params)

        fit_kwargs: dict = {"callbacks": callbacks}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]

        self._model.fit(X_train, y_train, **fit_kwargs)
        self._feature_names = list(X_train.columns)
        self._is_fitted = True
        self.task_type = task
        log.info(f"LightGBM ({task}) fitted.")
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
        return pd.Series(
            self._model.feature_importances_,
            index=self._feature_names,
        ).sort_values(ascending=False)

    def _resolve_task(self, y: pd.Series) -> str:
        if self.task_type != "auto":
            return self.task_type
        return "classification" if y.nunique() <= 20 else "regression"
