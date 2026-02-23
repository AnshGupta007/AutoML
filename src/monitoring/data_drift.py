"""
Data drift detection using Evidently.
Monitors incoming data for statistical drift vs. reference (training) data.
"""
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


class DataDriftDetector:
    """Detects data drift between reference and current production data.

    Uses Evidently for statistical tests (Jensen-Shannon, Wasserstein,
    chi-squared) across all features.

    Example:
        >>> detector = DataDriftDetector()
        >>> detector.set_reference(X_train)
        >>> report = detector.detect(X_current)
        >>> print(report["drift_share"])
    """

    def __init__(
        self,
        drift_threshold: float = 0.3,
        stattest: str = "auto",
        output_dir: Union[str, Path] = "reports/monitoring",
    ) -> None:
        self.drift_threshold = drift_threshold
        self.stattest = stattest
        self.output_dir = Path(output_dir)
        self._reference: Optional[pd.DataFrame] = None

    def set_reference(self, X_reference: pd.DataFrame) -> None:
        """Store reference dataset for drift comparison."""
        self._reference = X_reference.copy()
        log.info(f"Reference data set: {X_reference.shape}")

    def detect(
        self,
        X_current: pd.DataFrame,
        save_html: bool = True,
    ) -> dict:
        """Detect data drift in the current batch.

        Args:
            X_current: Current production data batch.
            save_html: Whether to save an Evidently HTML report.

        Returns:
            Dict with drift results: drift_detected, drift_share, per_column_drift.
        """
        if self._reference is None:
            raise RuntimeError("Call set_reference() first.")

        try:
            return self._evidently_detect(X_current, save_html)
        except ImportError:
            log.warning("Evidently not installed. Using simple KS-based drift detection.")
            return self._fallback_detect(X_current)

    def _evidently_detect(self, X_current: pd.DataFrame, save_html: bool) -> dict:
        from evidently.metrics import DataDriftTable, DatasetDriftMetric
        from evidently.report import Report

        report = Report(metrics=[DatasetDriftMetric(), DataDriftTable()])
        report.run(reference_data=self._reference, current_data=X_current)

        result_dict = report.as_dict()
        drift_metric = result_dict["metrics"][0]["result"]
        col_drift_metric = result_dict["metrics"][1]["result"]

        drift_share = drift_metric.get("share_of_drifted_columns", 0.0)
        drift_detected = drift_share > self.drift_threshold

        per_column = {}
        for col_result in col_drift_metric.get("drift_by_columns", {}).values():
            per_column[col_result.get("column_name", "?")] = {
                "drifted": col_result.get("drift_detected", False),
                "score": col_result.get("stattest_threshold", None),
            }

        if save_html:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            from datetime import datetime
            html_path = self.output_dir / f"drift_{datetime.utcnow():%Y%m%d_%H%M%S}.html"
            report.save_html(str(html_path))
            log.info(f"Drift report saved to {html_path}")

        result = {
            "drift_detected": drift_detected,
            "drift_share": round(drift_share, 4),
            "per_column_drift": per_column,
        }

        if drift_detected:
            log.warning(f"Data drift detected! {drift_share:.1%} of features drifted.")
        else:
            log.info(f"No significant drift detected (share={drift_share:.1%})")

        return result

    def _fallback_detect(self, X_current: pd.DataFrame) -> dict:
        """Simplified KS / chi-squared fallback when Evidently not available."""
        from scipy import stats

        per_column: dict = {}
        drifted_count = 0

        for col in self._reference.columns:
            if col not in X_current.columns:
                continue
            ref_col = self._reference[col].dropna()
            cur_col = X_current[col].dropna()
            try:
                if pd.api.types.is_numeric_dtype(ref_col):
                    stat, p = stats.ks_2samp(ref_col, cur_col)
                    drifted = p < 0.05
                else:
                    # Chi-squared on value counts
                    combined = pd.concat([ref_col, cur_col]).unique()
                    ref_counts = ref_col.value_counts().reindex(combined, fill_value=0)
                    cur_counts = cur_col.value_counts().reindex(combined, fill_value=0)
                    stat, p = stats.chisquare(cur_counts + 1, f_exp=ref_counts + 1)
                    drifted = p < 0.05

                per_column[col] = {"drifted": drifted, "p_value": round(p, 4)}
                if drifted:
                    drifted_count += 1
            except Exception:
                per_column[col] = {"drifted": False, "p_value": None}

        drift_share = drifted_count / max(len(self._reference.columns), 1)
        return {
            "drift_detected": drift_share > self.drift_threshold,
            "drift_share": round(drift_share, 4),
            "per_column_drift": per_column,
        }
