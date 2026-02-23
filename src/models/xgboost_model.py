"""
XGBoost model wrapper.
"""
from typing import Optional

import numpy as np
import pandas as pd

from src.models.base_model import BaseModel
from src.utils.logger import get_logger

log = get_logger(__name__)


class XGBoostModel(BaseModel):
    """XGBoost model wrapper supporting classification and regression.

    Example:
        >>> model = XGBoostModel(task_type="classification", n_estimators=500)
        >>> model.fit(X_train, y_train, X_val, y_val)
        >>> preds = model.predict(X_test)
    """

    def __init__(
        self,
        task_type: str = "auto",
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        min_child_weight: int = 1,
        gamma: float = 0.0,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        early_stopping_rounds: int = 50,
        n_jobs: int = -1,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(name="xgboost", task_type=task_type, random_state=random_state)
        self.params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "min_child_weight": min_child_weight,
            "gamma": gamma,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "n_jobs": n_jobs,
            "random_state": random_state,
            "tree_method": "hist",
            "verbosity": 0,
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
    ) -> "XGBoostModel":
        import xgboost as xgb

        task = self._resolve_task(y_train)
        model_params = self.params.copy()

        if task == "classification":
            n_classes = y_train.nunique()
            if n_classes == 2:
                model_params["objective"] = "binary:logistic"
                model_params["eval_metric"] = "logloss"
            else:
                model_params["objective"] = "multi:softprob"
                model_params["num_class"] = n_classes
                model_params["eval_metric"] = "mlogloss"
            self._model = xgb.XGBClassifier(**model_params)
        else:
            model_params["objective"] = "reg:squarederror"
            model_params["eval_metric"] = "rmse"
            self._model = xgb.XGBRegressor(**model_params)

        fit_kwargs: dict = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["early_stopping_rounds"] = self.early_stopping_rounds
            fit_kwargs["verbose"] = False

        self._model.fit(X_train, y_train, **fit_kwargs)
        self._feature_names = list(X_train.columns)
        self._is_fitted = True
        self.task_type = task
        log.info(f"XGBoost ({task}) fitted. Best iteration: {getattr(self._model, 'best_iteration', 'N/A')}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        if self.task_type != "classification":
            raise NotImplementedError("predict_proba only for classification")
        return self._model.predict_proba(X)

    def get_feature_importance(self) -> Optional[pd.Series]:
        self._check_fitted()
        imp = self._model.feature_importances_
        return pd.Series(imp, index=self._feature_names).sort_values(ascending=False)

    def _resolve_task(self, y: pd.Series) -> str:
        if self.task_type != "auto":
            return self.task_type
        return "classification" if y.nunique() <= 20 else "regression"
