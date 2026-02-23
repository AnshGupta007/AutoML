"""
Model serializer — saves/loads models in joblib and ONNX formats.
"""
import json
from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


class ModelSerializer:
    """Serializes and deserializes fitted models in multiple formats.

    Supports: joblib (universal), ONNX (cross-runtime inference).

    Example:
        >>> ser = ModelSerializer()
        >>> ser.save_joblib(model, "models/my_model.joblib")
        >>> model = ser.load_joblib("models/my_model.joblib")
    """

    @staticmethod
    def save_joblib(model, path: Union[str, Path], compress: int = 3) -> Path:
        """Save model with joblib compression."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path, compress=compress)
        log.info(f"Model saved (joblib) to {path}")
        return path

    @staticmethod
    def load_joblib(path: Union[str, Path]):
        """Load a joblib-serialized model."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found at {path}")
        model = joblib.load(path)
        log.info(f"Model loaded from {path}")
        return model

    @staticmethod
    def save_onnx(
        model,
        path: Union[str, Path],
        feature_names: list[str],
        n_features: int,
        task_type: str = "classification",
    ) -> Optional[Path]:
        """Export model to ONNX format (if supported).

        Args:
            model: Fitted sklearn-compatible model.
            path: Target .onnx file path.
            feature_names: List of input feature names.
            n_features: Number of input features.
            task_type: 'classification' or 'regression'.

        Returns:
            Path if successful, None otherwise.
        """
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType

            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            underlying = getattr(model, "_model", model)
            initial_types = [("float_input", FloatTensorType([None, n_features]))]
            onnx_model = convert_sklearn(underlying, initial_types=initial_types)

            with open(path, "wb") as f:
                f.write(onnx_model.SerializeToString())

            log.info(f"Model exported to ONNX at {path}")
            return path
        except (ImportError, Exception) as e:
            log.warning(f"ONNX export failed: {e}")
            return None

    @staticmethod
    def save_metadata(
        meta: dict,
        path: Union[str, Path],
    ) -> Path:
        """Save model metadata as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        return path
