"""
Model selection — automatically evaluates candidate models and picks the best.
"""
from typing import Optional

import numpy as np
import pandas as pd

from src.models.base_model import BaseModel
from src.models.catboost_model import CatBoostModel
from src.models.lightgbm_model import LightGBMModel
from src.models.linear_model import LinearModel
from src.models.xgboost_model import XGBoostModel
from src.utils.logger import get_logger
from src.utils.timer import Timer

log = get_logger(__name__)

MODEL_REGISTRY = {
    "xgboost": XGBoostModel,
    "lightgbm": LightGBMModel,
    "catboost": CatBoostModel,
    "linear": LinearModel,
}


class ModelSelector:
    """Evaluates multiple candidate models and returns the ranked results.

    Performs a quick cross-validation on each candidate and ranks by metric.

    Example:
        >>> selector = ModelSelector(candidates=["xgboost", "lightgbm"])
        >>> results = selector.select(X_train, y_train, task_type="classification")
        >>> best_model = results[0]["model"]
    """

    def __init__(
        self,
        candidates: list[str] = None,
        metric: str = "auto",
        cv_folds: int = 3,
        task_type: str = "auto",
        time_budget_per_model: int = 300,
        n_best: int = 3,
        random_state: int = 42,
    ) -> None:
        self.candidates = candidates or ["xgboost", "lightgbm", "catboost", "linear"]
        self.metric = metric
        self.cv_folds = cv_folds
        self.task_type = task_type
        self.time_budget_per_model = time_budget_per_model
        self.n_best = n_best
        self.random_state = random_state
        self.results: list[dict] = []

    def select(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> list[dict]:
        """Evaluate all candidate models and return ranked results.

        Args:
            X_train, y_train: Training data.
            X_val, y_val: Optional validation data (faster than CV if provided).

        Returns:
            List of dicts sorted by score descending, each with
            keys: model_name, score, model, fit_time.
        """
        task = self._resolve_task(y_train)
        metric = self.metric
        if metric == "auto":
            metric = "roc_auc" if task == "classification" else "rmse"

        self.results = []
        for name in self.candidates:
            if name not in MODEL_REGISTRY:
                log.warning(f"Unknown model '{name}', skipping.")
                continue

            model_cls = MODEL_REGISTRY[name]
            model = model_cls(task_type=task, random_state=self.random_state)
            log.info(f"Evaluating model: {name}")

            try:
                if X_val is not None and y_val is not None:
                    with Timer(f"{name} fit") as t:
                        model.fit(X_train, y_train, X_val, y_val)
                    score = self._score(model, X_val, y_val, metric, task)
                    fit_time = t.elapsed
                else:
                    score, fit_time = self._cv_score(model_cls, X_train, y_train, metric, task)

                # Normalize lower-is-better metrics
                if metric in ("rmse", "mae"):
                    display_score = -score
                    score = score
                else:
                    display_score = score

                self.results.append({
                    "model_name": name,
                    "score": display_score,
                    "raw_score": score,
                    "metric": metric,
                    "model": model,
                    "fit_time": fit_time,
                })
                log.info(f"  {name}: {metric}={display_score:.4f} (t={fit_time:.1f}s)")
            except Exception as e:
                log.warning(f"  {name} failed: {e}")

        # Sort by score descending
        self.results.sort(key=lambda r: r["score"], reverse=True)
        log.info(
            f"Best model: {self.results[0]['model_name']} "
            f"({metric}={self.results[0]['score']:.4f})"
        )
        return self.results[:self.n_best]

    def _cv_score(self, model_cls, X, y, metric, task):
        from sklearn.model_selection import KFold, StratifiedKFold
        cv = (
            StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            if task == "classification"
            else KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        )
        scores, times = [], []
        for tr_idx, val_idx in cv.split(X, y):
            model = model_cls(task_type=task, random_state=self.random_state)
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
            with Timer(f"cv_fit") as t:
                model.fit(X_tr, y_tr)
            scores.append(self._score(model, X_val, y_val, metric, task))
            times.append(t.elapsed)
        return float(np.mean(scores)), float(np.sum(times))

    @staticmethod
    def _score(model, X_val, y_val, metric, task) -> float:
        from sklearn.metrics import (
            accuracy_score, roc_auc_score, f1_score,
            mean_squared_error, r2_score, mean_absolute_error,
        )
        y_pred = model.predict(X_val)
        if metric == "accuracy":
            return accuracy_score(y_val, y_pred)
        elif metric == "f1":
            return f1_score(y_val, y_pred, average="weighted", zero_division=0)
        elif metric == "roc_auc":
            try:
                proba = model.predict_proba(X_val)
                if proba.shape[1] == 2:
                    return roc_auc_score(y_val, proba[:, 1])
                return roc_auc_score(y_val, proba, multi_class="ovr", average="weighted")
            except Exception:
                return accuracy_score(y_val, y_pred)
        elif metric == "rmse":
            return -np.sqrt(mean_squared_error(y_val, y_pred))
        elif metric == "mae":
            return -mean_absolute_error(y_val, y_pred)
        elif metric == "r2":
            return r2_score(y_val, y_pred)
        return accuracy_score(y_val, y_pred) if task == "classification" else r2_score(y_val, y_pred)

    def _resolve_task(self, y: pd.Series) -> str:
        if self.task_type != "auto":
            return self.task_type
        return "classification" if y.nunique() <= 20 else "regression"
