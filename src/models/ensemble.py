"""
Ensemble module: stacking and blending strategies.
Combines predictions from multiple base models via a meta-learner.
"""
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from src.models.base_model import BaseModel
from src.utils.logger import get_logger

log = get_logger(__name__)


class StackingEnsemble(BaseModel):
    """Stacking ensemble with cross-validated out-of-fold predictions.

    Trains base models and combines their OOF predictions to train a meta-learner.

    Example:
        >>> ensemble = StackingEnsemble(base_models=[xgb, lgb, cat], meta_learner=lr)
        >>> ensemble.fit(X_train, y_train)
        >>> preds = ensemble.predict(X_test)
    """

    def __init__(
        self,
        base_models: list[BaseModel],
        meta_learner=None,
        task_type: str = "classification",
        cv_folds: int = 5,
        passthrough: bool = False,
        random_state: int = 42,
    ) -> None:
        super().__init__(name="stacking_ensemble", task_type=task_type, random_state=random_state)
        self.base_models = base_models
        self.meta_learner = meta_learner
        self.cv_folds = cv_folds
        self.passthrough = passthrough

        self._fitted_base_models: list[BaseModel] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs,
    ) -> "StackingEnsemble":
        log.info(
            f"Training stacking ensemble with {len(self.base_models)} base models, "
            f"cv_folds={self.cv_folds}"
        )

        if self.task_type == "classification":
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        else:
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        n = len(X_train)
        n_models = len(self.base_models)
        oof_preds = np.zeros((n, n_models))

        for model_idx, model in enumerate(self.base_models):
            log.info(f"  OOF training: {model.name}")
            for fold_idx, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
                X_fold_train = X_train.iloc[tr_idx]
                y_fold_train = y_train.iloc[tr_idx]
                X_fold_val = X_train.iloc[val_idx]

                import copy
                fold_model = copy.deepcopy(model)
                fold_model.fit(X_fold_train, y_fold_train)

                if self.task_type == "classification":
                    try:
                        oof_preds[val_idx, model_idx] = fold_model.predict_proba(X_fold_val)[:, 1]
                    except Exception:
                        oof_preds[val_idx, model_idx] = fold_model.predict(X_fold_val)
                else:
                    oof_preds[val_idx, model_idx] = fold_model.predict(X_fold_val)

            # Refit on full data
            full_model = copy.deepcopy(model)
            full_model.fit(X_train, y_train)
            self._fitted_base_models.append(full_model)

        # Build meta-features
        meta_X = pd.DataFrame(
            oof_preds,
            columns=[f"model_{m.name}" for m in self.base_models]
        )
        if self.passthrough:
            meta_X = pd.concat([meta_X, X_train.reset_index(drop=True)], axis=1)

        # Train meta-learner
        if self.meta_learner is None:
            if self.task_type == "classification":
                from sklearn.linear_model import LogisticRegression
                self.meta_learner = LogisticRegression(max_iter=500, random_state=self.random_state)
            else:
                from sklearn.linear_model import Ridge
                self.meta_learner = Ridge()

        self.meta_learner.fit(meta_X, y_train)
        self._is_fitted = True
        log.info("Stacking ensemble fitted.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        meta_X = self._build_meta_features(X)
        return self.meta_learner.predict(meta_X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        if self.task_type != "classification":
            raise NotImplementedError
        meta_X = self._build_meta_features(X)
        if hasattr(self.meta_learner, "predict_proba"):
            return self.meta_learner.predict_proba(meta_X)
        raise NotImplementedError("Meta-learner does not support predict_proba")

    def _build_meta_features(self, X: pd.DataFrame) -> pd.DataFrame:
        preds = []
        for model in self._fitted_base_models:
            if self.task_type == "classification":
                try:
                    p = model.predict_proba(X)[:, 1]
                except Exception:
                    p = model.predict(X).astype(float)
            else:
                p = model.predict(X)
            preds.append(p)

        meta_X = pd.DataFrame(
            np.column_stack(preds) if preds else np.zeros((len(X), 1)),
            columns=[f"model_{m.name}" for m in self._fitted_base_models],
        )
        if self.passthrough:
            meta_X = pd.concat([meta_X, X.reset_index(drop=True)], axis=1)
        return meta_X


class BlendingEnsemble(BaseModel):
    """Simple averaging / weighted blending ensemble.

    Averages predictions from multiple trained models.
    Weights can be provided or optimized automatically.
    """

    def __init__(
        self,
        base_models: list[BaseModel],
        weights: Optional[list[float]] = None,
        task_type: str = "classification",
        random_state: int = 42,
    ) -> None:
        super().__init__(name="blending_ensemble", task_type=task_type, random_state=random_state)
        self.base_models_config = base_models
        self.weights = weights

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs,
    ) -> "BlendingEnsemble":
        for model in self.base_models_config:
            model.fit(X_train, y_train, X_val, y_val)
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        preds = np.column_stack([m.predict(X) for m in self.base_models_config])
        weights = self.weights or [1 / len(self.base_models_config)] * len(self.base_models_config)
        return np.average(preds, axis=1, weights=weights)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        probas = [m.predict_proba(X) for m in self.base_models_config]
        weights = self.weights or [1 / len(probas)] * len(probas)
        return np.average(probas, axis=0, weights=weights)
