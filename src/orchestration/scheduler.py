"""
Scheduler stub — wraps APScheduler to run periodic monitoring checks.
"""
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.monitoring.alerting import AlertManager
from src.monitoring.data_drift import DataDriftDetector
from src.monitoring.model_performance import ModelPerformanceMonitor
from src.monitoring.retraining_trigger import RetrainingTrigger
from src.utils.logger import get_logger

log = get_logger(__name__)


def build_scheduler(
    reference_path: str,
    models_dir: str = "models",
    interval_minutes: int = 60,
):
    """Build and return an APScheduler that runs monitoring on a schedule.

    Args:
        reference_path: CSV path to training reference data.
        models_dir: Directory containing model artifacts.
        interval_minutes: How often to run checks.

    Returns:
        Configured APScheduler BlockingScheduler instance.
    """
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
    except ImportError:
        log.warning("APScheduler not installed. Install it with: pip install apscheduler")
        return None

    reference_df = pd.read_csv(reference_path)
    drift_detector = DataDriftDetector(
        output_dir=str(Path(models_dir).parent / "reports" / "monitoring")
    )
    drift_detector.set_reference(reference_df)
    trigger = RetrainingTrigger(
        alert_manager=AlertManager(channels=["log"])
    )

    def _monitoring_job():
        log.info(f"[Scheduler] Running monitoring check at {datetime.utcnow().isoformat()}")
        current_path = os.getenv("CURRENT_DATA_PATH")
        if not current_path or not Path(current_path).exists():
            log.warning("[Scheduler] CURRENT_DATA_PATH not set or file missing.")
            return
        current_df = pd.read_csv(current_path)
        drift_report = drift_detector.detect(current_df)
        decision = trigger.evaluate(drift_report)
        if decision["should_retrain"]:
            log.warning("[Scheduler] Retraining flag raised.")

    scheduler = BackgroundScheduler()
    scheduler.add_job(_monitoring_job, "interval", minutes=interval_minutes, id="monitoring")
    return scheduler
