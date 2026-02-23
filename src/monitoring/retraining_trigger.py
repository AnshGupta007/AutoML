"""
Automatic retraining trigger.
Monitors drift and performance signals and schedules retraining when needed.
"""
from datetime import datetime, timedelta
from typing import Optional

from src.monitoring.alerting import Alert, AlertManager
from src.monitoring.data_drift import DataDriftDetector
from src.monitoring.model_performance import ModelPerformanceMonitor
from src.utils.logger import get_logger

log = get_logger(__name__)


class RetrainingTrigger:
    """Evaluates monitoring signals to decide if retraining is needed.

    Combines data drift, performance degradation, and time-based signals.

    Example:
        >>> trigger = RetrainingTrigger()
        >>> decision = trigger.evaluate(drift_report, perf_report)
        >>> if decision["should_retrain"]:
        ...     run_retraining()
    """

    def __init__(
        self,
        drift_threshold: float = 0.3,
        performance_degradation_threshold: float = 0.05,
        max_days_since_training: int = 30,
        alert_manager: Optional[AlertManager] = None,
    ) -> None:
        self.drift_threshold = drift_threshold
        self.performance_degradation_threshold = performance_degradation_threshold
        self.max_days_since_training = max_days_since_training
        self.alert_manager = alert_manager or AlertManager(channels=["log"])
        self._last_trained: Optional[datetime] = None

    def set_last_trained(self, dt: datetime) -> None:
        """Record when the model was last trained."""
        self._last_trained = dt

    def evaluate(
        self,
        drift_report: Optional[dict] = None,
        performance_report: Optional[dict] = None,
    ) -> dict:
        """Evaluate all signals and decide on retraining.

        Args:
            drift_report: Output from DataDriftDetector.detect().
            performance_report: Output from ModelPerformanceMonitor.check_degradation().

        Returns:
            Dict with should_retrain bool and reasons list.
        """
        reasons: list[str] = []
        should_retrain = False

        # Drift check
        if drift_report:
            drift_share = drift_report.get("drift_share", 0.0)
            if drift_share >= self.drift_threshold:
                reasons.append(
                    f"Data drift: {drift_share:.1%} of features drifted (threshold={self.drift_threshold:.1%})"
                )
                should_retrain = True

        # Performance check
        if performance_report and performance_report.get("degraded"):
            deltas = performance_report.get("deltas", {})
            reasons.append(f"Performance degraded: {deltas}")
            should_retrain = True

        # Time-based check
        if self._last_trained is not None:
            days_since = (datetime.utcnow() - self._last_trained).days
            if days_since >= self.max_days_since_training:
                reasons.append(
                    f"Scheduled retrain: {days_since} days since last training "
                    f"(max={self.max_days_since_training})"
                )
                should_retrain = True

        if should_retrain:
            self.alert_manager.send(Alert(
                level="WARNING",
                title="Retraining Required",
                message=" | ".join(reasons),
                metadata={
                    "drift_share": drift_report.get("drift_share") if drift_report else None,
                    "triggered_at": datetime.utcnow().isoformat(),
                },
            ))

        return {
            "should_retrain": should_retrain,
            "reasons": reasons,
            "evaluated_at": datetime.utcnow().isoformat(),
        }
