"""
Abstract base model class.
All model wrappers must inherit from this and implement the interface.
"""
import abc
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


class BaseModel(abc.ABC):
    """Abstract base class for all AutoMLPro model wrappers.

    Defines the interface that every model must implement.
    Provides common utilities like model naming, scoring, and serialization.

    Subclasses must implement:
        - fit(X_train, y_train, X_val=None, y_val=None)
        - predict(X)
        - predict_proba(X)   [for classifiers]
        - get_feature_importance()
    """

    def __init__(
        self,
        name: str,
        task_type: str = "auto",
        random_state: int = 42,
        **kwargs,
    ) -> None:
        self.name = name
        self.task_type = task_type
        self.random_state = random_state
        self.params: dict = kwargs
        self._model = None
        self._is_fitted = False
        self._feature_names: list[str] = []
        self.training_history: dict = {}

    @abc.abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs,
    ) -> "BaseModel":
        """Fit (train) the model on training data."""
        ...

    @abc.abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions for regression or class labels for classification."""
        ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate class probability estimates (classification only).

        Override in classifier subclasses.

        Returns:
            Array of shape (n_samples, n_classes).
        """
        raise NotImplementedError(
            f"Model '{self.name}' does not support predict_proba."
        )

    def get_feature_importance(self) -> Optional[pd.Series]:
        """Return feature importance as a Series indexed by feature name.

        Override in subclasses that support feature importance.

        Returns:
            pd.Series sorted by importance descending, or None.
        """
        return None

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Compute default score (accuracy or R²) on the given data.

        Args:
            X: Feature DataFrame.
            y: True target series.

        Returns:
            Scalar score.
        """
        from sklearn.metrics import accuracy_score, r2_score

        y_pred = self.predict(X)
        if self.task_type == "classification":
            return float(accuracy_score(y, y_pred))
        return float(r2_score(y, y_pred))

    def set_params(self, **params) -> "BaseModel":
        """Update model parameters (for HPO integration)."""
        self.params.update(params)
        return self

    def get_params(self) -> dict:
        """Return current model parameters."""
        return self.params.copy()

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def model(self):
        """Access underlying sklearn/xgb/lgb/cat model object."""
        return self._model

    def _check_fitted(self) -> None:
        from src.utils.exceptions import ModelNotFittedError
        if not self._is_fitted:
            raise ModelNotFittedError(self.name)

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return f"{self.__class__.__name__}(name={self.name!r}, task={self.task_type!r}, status={status})"
