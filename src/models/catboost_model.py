"""
CatBoost model wrapper.
"""
from typing import Optional

import numpy as np
import pandas as pd

from src.models.base_model import BaseModel
from src.utils.logger import get_logger

log = get_logger(__name__)


class CatBoostModel(BaseModel):
    """CatBoost model wrapper.

    CatBoost natively handles categorical features, so the encoder
    step can be skipped when using this model.

    Example:
        >>> model = CatBoostModel(task_type="classification", iterations=500)
        >>> model.fit(X_train, y_train, X_val, y_val)
    """

    def __init__(
        self,
        task_type: str = "auto",
        iterations: int = 1000,
        learning_rate: float = 0.05,
        depth: int = 6,
        l2_leaf_reg: float = 3.0,
        early_stopping_rounds: int = 50,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(name="catboost", task_type=task_type, random_state=random_state)
        self.params = {
            "iterations": iterations,
            "learning_rate": learning_rate,
            "depth": depth,
            "l2_leaf_reg": l2_leaf_reg,
            "random_seed": random_state,
            "verbose": 0,
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
    ) -> "CatBoostModel":
        from catboost import CatBoostClassifier, CatBoostRegressor, Pool

        task = self._resolve_task(y_train)
        params = self.params.copy()
        params["early_stopping_rounds"] = self.early_stopping_rounds

        train_pool = Pool(X_train, y_train)
        eval_pool = Pool(X_val, y_val) if X_val is not None else None

        if task == "classification":
            self._model = CatBoostClassifier(**params)
        else:
            self._model = CatBoostRegressor(**params)

        self._model.fit(train_pool, eval_set=eval_pool, verbose=0)
        self._feature_names = list(X_train.columns)
        self._is_fitted = True
        self.task_type = task
        log.info(f"CatBoost ({task}) fitted.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self._model.predict(X).flatten()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        if self.task_type != "classification":
            raise NotImplementedError
        return self._model.predict_proba(X)

    def get_feature_importance(self) -> Optional[pd.Series]:
        self._check_fitted()
        return pd.Series(
            self._model.get_feature_importance(),
            index=self._feature_names,
        ).sort_values(ascending=False)

    def _resolve_task(self, y: pd.Series) -> str:
        if self.task_type != "auto":
            return self.task_type
        return "classification" if y.nunique() <= 20 else "regression"
