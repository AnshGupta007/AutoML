"""
Comprehensive evaluation metrics for classification and regression.
"""
from typing import Optional

import numpy as np
import pandas as pd


def compute_classification_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    average: str = "weighted",
) -> dict:
    """Compute a full suite of classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities (optional).
        average: Averaging strategy for multi-class metrics.

    Returns:
        Dict of metric_name → float.
    """
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        cohen_kappa_score,
        f1_score,
        log_loss,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    metrics: dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }

    if y_proba is not None:
        try:
            if y_proba.shape[1] == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
                metrics["average_precision"] = float(
                    average_precision_score(y_true, y_proba[:, 1])
                )
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average=average)
                )
            metrics["log_loss"] = float(log_loss(y_true, y_proba))
        except Exception:
            pass

    return metrics


def compute_regression_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict:
    """Compute a full suite of regression metrics.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        Dict of metric_name → float.
    """
    from sklearn.metrics import (
        explained_variance_score,
        max_error,
        mean_absolute_error,
        mean_absolute_percentage_error,
        mean_squared_error,
        median_absolute_error,
        r2_score,
    )

    mse = mean_squared_error(y_true, y_pred)
    return {
        "rmse": float(np.sqrt(mse)),
        "mse": float(mse),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "median_ae": float(median_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "explained_variance": float(explained_variance_score(y_true, y_pred)),
        "max_error": float(max_error(y_true, y_pred)),
    }


def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    task_type: str,
    y_proba: Optional[np.ndarray] = None,
) -> dict:
    """Unified metric computation dispatcher.

    Args:
        y_true: Ground truth labels/values.
        y_pred: Model predictions.
        task_type: 'classification' or 'regression'.
        y_proba: Probability estimates (classification only).

    Returns:
        Dict of metric_name → float value.
    """
    if task_type == "classification":
        return compute_classification_metrics(y_true, y_pred, y_proba)
    return compute_regression_metrics(y_true, y_pred)


def get_confusion_matrix(y_true, y_pred) -> pd.DataFrame:
    """Return confusion matrix as a labeled DataFrame."""
    from sklearn.metrics import confusion_matrix
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=labels, columns=labels)
