"""
Model calibration evaluation.
Assesses and corrects probability calibration via isotonic/Platt scaling.
"""
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


class CalibrationEvaluator:
    """Evaluates and improves classifier probability calibration.

    Example:
        >>> cal = CalibrationEvaluator()
        >>> cal.fit(model, X_val, y_val, method="isotonic")
        >>> calibrated_proba = cal.predict_proba(X_test)
    """

    def __init__(self, method: str = "isotonic", n_bins: int = 10) -> None:
        self.method = method
        self.n_bins = n_bins
        self._calibrated_model = None
        self._base_model = None

    def evaluate(self, y_true, y_proba, output_dir: Optional[Path] = None) -> dict:
        """Compute calibration metrics.

        Args:
            y_true: True binary labels.
            y_proba: Predicted probabilities for positive class.
            output_dir: If set, save calibration curve plot here.

        Returns:
            Dict with brier_score, ece, calibration curve data.
        """
        from sklearn.calibration import calibration_curve
        from sklearn.metrics import brier_score_loss

        brier = float(brier_score_loss(y_true, y_proba))
        fraction_of_pos, mean_predicted = calibration_curve(
            y_true, y_proba, n_bins=self.n_bins
        )
        ece = float(np.mean(np.abs(fraction_of_pos - mean_predicted)))

        result = {
            "brier_score": brier,
            "expected_calibration_error": ece,
            "fraction_of_positives": fraction_of_pos.tolist(),
            "mean_predicted_value": mean_predicted.tolist(),
        }

        if output_dir is not None:
            self._save_calibration_plot(
                fraction_of_pos, mean_predicted, Path(output_dir)
            )
        return result

    def fit(self, model, X_cal: pd.DataFrame, y_cal: pd.Series) -> "CalibrationEvaluator":
        """Fit a calibrated wrapper around the model.

        Args:
            model: Fitted base classifier.
            X_cal: Calibration set features.
            y_cal: Calibration set labels.

        Returns:
            self
        """
        from sklearn.calibration import CalibratedClassifierCV

        self._base_model = model
        underlying = getattr(model, "_model", model)

        self._calibrated_model = CalibratedClassifierCV(
            estimator=underlying, method=self.method, cv="prefit"
        )
        self._calibrated_model.fit(X_cal, y_cal)
        log.info(f"Calibration fitted ({self.method})")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._calibrated_model is None:
            raise RuntimeError("Call .fit() before predict_proba()")
        return self._calibrated_model.predict_proba(X)

    @staticmethod
    def _save_calibration_plot(frac_pos, mean_pred, output_dir: Path) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            output_dir.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(7, 6))
            plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
            plt.plot(mean_pred, frac_pos, "s-", label="Model")
            plt.xlabel("Mean predicted probability")
            plt.ylabel("Fraction of positives")
            plt.title("Calibration curve")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / "calibration_curve.png", dpi=150)
            plt.close()
        except Exception as e:
            log.warning(f"Could not save calibration plot: {e}")
