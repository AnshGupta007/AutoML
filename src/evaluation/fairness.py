"""
Fairness evaluation using Fairlearn.
Computes disparity metrics across sensitive attributes.
"""
from typing import Optional

import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


class FairnessEvaluator:
    """Computes and reports fairness metrics across demographic groups.

    Wraps Fairlearn's MetricFrame to compute group-level disparities.

    Example:
        >>> evaluator = FairnessEvaluator(sensitive_features=["gender"])
        >>> report = evaluator.evaluate(y_true, y_pred, df_test)
    """

    def __init__(
        self,
        sensitive_features: list[str],
        task_type: str = "classification",
    ) -> None:
        self.sensitive_features = sensitive_features
        self.task_type = task_type

    def evaluate(
        self,
        y_true: pd.Series,
        y_pred,
        X_test: pd.DataFrame,
    ) -> dict:
        """Compute fairness metrics per sensitive feature.

        Args:
            y_true: Ground truth labels.
            y_pred: Model predictions.
            X_test: Test DataFrame containing sensitive columns.

        Returns:
            Dict mapping feature name → metrics dict.
        """
        results = {}
        for feature in self.sensitive_features:
            if feature not in X_test.columns:
                log.warning(f"Sensitive feature '{feature}' not in test data, skipping.")
                continue
            try:
                results[feature] = self._compute_group_metrics(
                    y_true, y_pred, X_test[feature]
                )
            except Exception as e:
                log.warning(f"Fairness evaluation failed for '{feature}': {e}")

        return results

    def _compute_group_metrics(self, y_true, y_pred, sensitive_col: pd.Series) -> dict:
        try:
            from fairlearn.metrics import MetricFrame
            from sklearn.metrics import accuracy_score, selection_rate

            mf = MetricFrame(
                metrics={
                    "accuracy": accuracy_score,
                    "selection_rate": selection_rate,
                },
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_col,
            )
            return {
                "group_metrics": mf.by_group.to_dict(),
                "difference": mf.difference().to_dict(),
                "ratio": mf.ratio().to_dict(),
            }
        except ImportError:
            # Manual fallback
            groups = sensitive_col.unique()
            group_accuracy = {}
            for g in groups:
                mask = sensitive_col == g
                yt = y_true[mask]
                yp = y_pred[mask] if hasattr(y_pred, "__getitem__") else pd.Series(y_pred)[mask]
                from sklearn.metrics import accuracy_score
                group_accuracy[str(g)] = float(accuracy_score(yt, yp))
            values = list(group_accuracy.values())
            return {
                "group_accuracy": group_accuracy,
                "disparate_impact_ratio": min(values) / max(values) if max(values) > 0 else None,
                "accuracy_difference": max(values) - min(values),
            }
