"""
Model performance monitoring.
Tracks prediction accuracy metrics over time using windowed evaluation.
"""
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_metrics
from src.utils.logger import get_logger

log = get_logger(__name__)


class ModelPerformanceMonitor:
    """Tracks model performance metrics over rolling windows.

    Stores prediction log and computes metric degradation over time.

    Example:
        >>> monitor = ModelPerformanceMonitor(window_size=1000)
        >>> monitor.log_predictions(y_true_batch, y_pred_batch)
        >>> metrics = monitor.get_current_metrics()
    """

    def __init__(
        self,
        task_type: str = "classification",
        window_size: int = 1000,
        alert_threshold: float = 0.05,
    ) -> None:
        self.task_type = task_type
        self.window_size = window_size
        self.alert_threshold = alert_threshold

        self._y_true_window: deque = deque(maxlen=window_size)
        self._y_pred_window: deque = deque(maxlen=window_size)
        self._baseline_metrics: Optional[dict] = None
        self._history: list[dict] = []

    def set_baseline(self, y_true, y_pred) -> None:
        """Set baseline metrics from validation data."""
        self._baseline_metrics = compute_metrics(
            pd.Series(y_true),
            np.array(y_pred),
            self.task_type,
        )
        log.info(f"Baseline metrics set: {self._baseline_metrics}")

    def log_predictions(self, y_true, y_pred) -> None:
        """Add a new batch of predictions to the monitoring window."""
        if hasattr(y_true, "__iter__"):
            self._y_true_window.extend(y_true)
            self._y_pred_window.extend(y_pred)
        else:
            self._y_true_window.append(y_true)
            self._y_pred_window.append(y_pred)

    def get_current_metrics(self) -> dict:
        """Compute metrics over the current rolling window."""
        if not self._y_true_window:
            return {}

        y_true = pd.Series(list(self._y_true_window))
        y_pred = np.array(list(self._y_pred_window))
        metrics = compute_metrics(y_true, y_pred, self.task_type)

        entry = {"timestamp": datetime.utcnow().isoformat(), **metrics}
        self._history.append(entry)
        return metrics

    def check_degradation(self) -> dict:
        """Check if performance has degraded vs baseline.

        Returns:
            Dict with degraded flag and per-metric deltas.
        """
        if self._baseline_metrics is None:
            return {"degraded": False, "reason": "No baseline set"}

        current = self.get_current_metrics()
        if not current:
            return {"degraded": False, "reason": "No prediction data"}

        deltas = {}
        degraded = False

        for metric, baseline_val in self._baseline_metrics.items():
            if metric not in current:
                continue
            delta = current[metric] - baseline_val
            deltas[metric] = round(delta, 4)

            # Lower-is-better metrics: regression
            if metric in ("rmse", "mae", "mse", "mape"):
                if delta > self.alert_threshold * baseline_val:
                    degraded = True
            else:
                # Higher-is-better: accuracy, AUC, etc.
                if delta < -self.alert_threshold:
                    degraded = True

        return {
            "degraded": degraded,
            "current_metrics": current,
            "baseline_metrics": self._baseline_metrics,
            "deltas": deltas,
        }

    def get_history(self) -> pd.DataFrame:
        """Return monitoring history as a DataFrame."""
        if not self._history:
            return pd.DataFrame()
        return pd.DataFrame(self._history)

    def save_history(self, path: Union[str, Path]) -> None:
        """Persist monitoring history to CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.get_history().to_csv(path, index=False)
        log.info(f"Monitoring history saved to {path}")
