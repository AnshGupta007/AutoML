"""
Error analysis — identifies where the model performs worst.
Segments predictions by error magnitude and feature bins.
"""
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


class ErrorAnalyzer:
    """Identifies systematic error patterns in model predictions.

    Example:
        >>> analyzer = ErrorAnalyzer(task_type="classification")
        >>> report = analyzer.analyze(y_true, y_pred, X_test)
    """

    def __init__(self, task_type: str = "classification", top_n: int = 20) -> None:
        self.task_type = task_type
        self.top_n = top_n

    def analyze(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        X_test: Optional[pd.DataFrame] = None,
    ) -> dict:
        """Analyse errors and return a summary.

        Args:
            y_true: Ground truth.
            y_pred: Predicted values.
            X_test: Features (enriches error segmentation).

        Returns:
            Dict with error summary statistics.
        """
        results: dict = {}
        errors = pd.Series(y_true.values != y_pred, name="is_error") \
            if self.task_type == "classification" \
            else pd.Series(np.abs(y_true.values - y_pred), name="abs_error")

        results["overall_error_rate"] = float(errors.mean()) \
            if self.task_type == "classification" \
            else float(errors.mean())

        results["worst_samples"] = (
            y_true.to_frame("y_true")
            .assign(y_pred=y_pred, error=errors.values)
            .nlargest(self.top_n, "error")
            .to_dict(orient="records")
        )

        if X_test is not None:
            # Per-feature error bucketing for numeric columns
            feature_errors = {}
            for col in X_test.select_dtypes(include="number").columns[:10]:
                try:
                    buckets = pd.qcut(X_test[col], q=5, duplicates="drop")
                    err_by_bucket = errors.groupby(buckets).mean().to_dict()
                    feature_errors[col] = {str(k): round(v, 4) for k, v in err_by_bucket.items()}
                except Exception:
                    pass
            results["feature_error_rates"] = feature_errors

        return results
