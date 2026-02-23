"""
Optuna-based hyperparameter optimization.
Tunes any BaseModel subclass using configurable search spaces.
"""
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


class HyperparameterTuner:
    """Tunes model hyperparameters using Optuna.

    Supports any model that inherits BaseModel and provides
    a search space definition.

    Example:
        >>> tuner = HyperparameterTuner(n_trials=50, timeout=1800)
        >>> best_params = tuner.tune(model_class, X_train, y_train)
    """

    def __init__(
        self,
        n_trials: int = 50,
        timeout: int = 3600,
        metric: str = "auto",
        direction: str = "maximize",
        cv_folds: int = 3,
        random_state: int = 42,
        n_jobs: int = 1,
    ) -> None:
        self.n_trials = n_trials
        self.timeout = timeout
        self.metric = metric
        self.direction = direction
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.best_params: dict = {}
        self.study = None

    def tune(
        self,
        model_class,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        search_space: dict,
        task_type: str = "classification",
    ) -> dict:
        """Run hyperparameter search.

        Args:
            model_class: Model class (not instance) to tune.
            X_train: Training features.
            y_train: Training target.
            search_space: Dict mapping param name → (low, high) or list of choices.
            task_type: 'classification' or 'regression'.

        Returns:
            Best params dict.
        """
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        metric = self.metric
        if metric == "auto":
            metric = "roc_auc" if task_type == "classification" else "rmse"

        direction = "minimize" if metric in ("rmse", "mae", "mse") else "maximize"

        def objective(trial: optuna.Trial) -> float:
            params = self._sample_params(trial, search_space)
            model = model_class(task_type=task_type, **params)
            return self._cross_val_score(model, X_train, y_train, metric, task_type)

        self.study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            pruner=optuna.pruners.HyperbandPruner(),
        )

        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=False,
        )

        self.best_params = self.study.best_params
        log.info(
            f"HPO complete: best {metric}={self.study.best_value:.4f} | "
            f"params={self.best_params}"
        )
        return self.best_params

    def _cross_val_score(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        metric: str,
        task_type: str,
    ) -> float:
        """Evaluate model with cross-validation."""
        from sklearn.model_selection import StratifiedKFold, KFold

        if task_type == "classification":
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                                 random_state=self.random_state)
        else:
            cv = KFold(n_splits=self.cv_folds, shuffle=True,
                       random_state=self.random_state)

        scores = []
        for train_idx, val_idx in cv.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            try:
                model.fit(X_tr, y_tr)
                score = self._compute_metric(model, X_val, y_val, metric, task_type)
                scores.append(score)
            except Exception as e:
                log.debug(f"CV fold failed: {e}")
                return -999.0 if metric != "rmse" else 999.0

        return float(np.mean(scores))

    @staticmethod
    def _compute_metric(model, X_val, y_val, metric: str, task_type: str) -> float:
        from sklearn.metrics import (
            accuracy_score, f1_score, roc_auc_score,
            mean_squared_error, mean_absolute_error, r2_score,
        )
        y_pred = model.predict(X_val)

        if metric == "accuracy":
            return accuracy_score(y_val, y_pred)
        elif metric == "f1":
            return f1_score(y_val, y_pred, average="weighted", zero_division=0)
        elif metric == "roc_auc":
            try:
                y_proba = model.predict_proba(X_val)
                if y_proba.shape[1] == 2:
                    return roc_auc_score(y_val, y_proba[:, 1])
                return roc_auc_score(y_val, y_proba, multi_class="ovr", average="weighted")
            except Exception:
                return accuracy_score(y_val, y_pred)
        elif metric == "rmse":
            return -np.sqrt(mean_squared_error(y_val, y_pred))
        elif metric == "mae":
            return -mean_absolute_error(y_val, y_pred)
        elif metric == "r2":
            return r2_score(y_val, y_pred)
        return accuracy_score(y_val, y_pred) if task_type == "classification" else r2_score(y_val, y_pred)

    @staticmethod
    def _sample_params(trial, search_space: dict) -> dict:
        """Sample parameters from search space."""
        import optuna
        params = {}
        for name, spec in search_space.items():
            if isinstance(spec, list):
                # Categorical
                params[name] = trial.suggest_categorical(name, spec)
            elif isinstance(spec, tuple) and len(spec) == 2:
                low, high = spec
                if isinstance(low, int) and isinstance(high, int):
                    params[name] = trial.suggest_int(name, low, high)
                else:
                    params[name] = trial.suggest_float(name, low, high, log=(low > 0 and high / low > 10))
        return params

    def plot_optimization_history(self):
        """Plot Optuna optimization history (requires plotly)."""
        if self.study is None:
            log.warning("No study found. Run tune() first.")
            return None
        try:
            from optuna.visualization import plot_optimization_history
            return plot_optimization_history(self.study)
        except Exception as e:
            log.warning(f"Could not plot history: {e}")
            return None
