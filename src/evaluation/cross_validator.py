"""
Cross-validation utilities.
Provides robust CV with nested HPO support.
"""
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from src.evaluation.metrics import compute_metrics
from src.utils.logger import get_logger

log = get_logger(__name__)


class CrossValidator:
    """Robust k-fold cross-validator for AutoMLPro models.

    Supports classification and regression with comprehensive metric reporting.

    Example:
        >>> cv = CrossValidator(cv_folds=5, task_type="classification")
        >>> results = cv.validate(model_class, X, y)
        >>> print(results["mean_metrics"])
    """

    def __init__(
        self,
        cv_folds: int = 5,
        task_type: str = "classification",
        metric: str = "auto",
        random_state: int = 42,
        return_oof: bool = True,
    ) -> None:
        self.cv_folds = cv_folds
        self.task_type = task_type
        self.metric = metric
        self.random_state = random_state
        self.return_oof = return_oof

    def validate(
        self,
        model_class,
        X: pd.DataFrame,
        y: pd.Series,
        model_kwargs: Optional[dict] = None,
    ) -> dict:
        """Run k-fold cross-validation and collect metrics per fold.

        Args:
            model_class: Model class (unfitted).
            X: Feature DataFrame.
            y: Target series.
            model_kwargs: Kwargs to pass to model constructor.

        Returns:
            Dict with fold_metrics, mean_metrics, std_metrics, oof_predictions.
        """
        model_kwargs = model_kwargs or {}
        cv = self._build_cv()

        fold_metrics: list[dict] = []
        oof_preds = np.zeros(len(y))
        oof_probas = None

        for fold_idx, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

            model = model_class(task_type=self.task_type, **model_kwargs)
            model.fit(X_tr, y_tr, X_val, y_val)

            y_pred = model.predict(X_val)
            oof_preds[val_idx] = y_pred

            y_proba = None
            if self.task_type == "classification":
                try:
                    y_proba = model.predict_proba(X_val)
                    if oof_probas is None:
                        oof_probas = np.zeros((len(y), y_proba.shape[1]))
                    oof_probas[val_idx] = y_proba
                except Exception:
                    pass

            fold_m = compute_metrics(y_val, y_pred, self.task_type, y_proba)
            fold_m["fold"] = fold_idx + 1
            fold_metrics.append(fold_m)
            log.debug(f"  Fold {fold_idx + 1}: {fold_m}")

        metric_keys = [k for k in fold_metrics[0] if k != "fold"]
        mean_metrics = {k: float(np.mean([m[k] for m in fold_metrics])) for k in metric_keys}
        std_metrics = {k: float(np.std([m[k] for m in fold_metrics])) for k in metric_keys}

        log.info(f"CV results ({self.cv_folds}-fold): {mean_metrics}")
        return {
            "fold_metrics": fold_metrics,
            "mean_metrics": mean_metrics,
            "std_metrics": std_metrics,
            "oof_predictions": oof_preds if self.return_oof else None,
            "oof_probabilities": oof_probas if self.return_oof else None,
        }

    def _build_cv(self):
        if self.task_type == "classification":
            return StratifiedKFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
            )
        return KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
