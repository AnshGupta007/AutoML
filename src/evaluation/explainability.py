"""
Model explainability using SHAP and LIME.
Provides global feature importance, local explanations, and summary plots.
"""
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


class ModelExplainer:
    """Provides SHAP and LIME explanations for any fitted model.

    Example:
        >>> explainer = ModelExplainer(model, feature_names=X_train.columns)
        >>> explainer.fit(X_train)
        >>> shap_df = explainer.get_shap_values(X_test)
        >>> explainer.plot_summary(X_test, output_dir="reports/explain")
    """

    def __init__(
        self,
        model,
        feature_names: Optional[list[str]] = None,
        task_type: str = "classification",
        max_samples: int = 500,
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self.task_type = task_type
        self.max_samples = max_samples

        self._shap_explainer = None
        self._background_data = None

    def fit(self, X_background: pd.DataFrame) -> "ModelExplainer":
        """Fit SHAP explainer using background (training) data.

        Args:
            X_background: Training data for SHAP background distribution.

        Returns:
            self
        """
        try:
            import shap

            # Sub-sample background for speed
            n = min(self.max_samples, len(X_background))
            self._background_data = X_background.sample(n, random_state=42)

            # Choose explainer type
            underlying = getattr(self.model, "_model", self.model)
            class_name = type(underlying).__name__.lower()

            if any(k in class_name for k in ("xgb", "lgbm", "catboost", "gbm")):
                self._shap_explainer = shap.TreeExplainer(underlying)
            else:
                self._shap_explainer = shap.KernelExplainer(
                    self.model.predict_proba
                    if self.task_type == "classification"
                    else self.model.predict,
                    self._background_data,
                )
            log.info("SHAP explainer fitted.")
        except ImportError:
            log.warning("SHAP not installed. Explainability skipped.")
        return self

    def get_shap_values(
        self, X: pd.DataFrame, check_additivity: bool = False
    ) -> Optional[pd.DataFrame]:
        """Compute SHAP values for a DataFrame.

        Args:
            X: Data to explain.
            check_additivity: Whether to check additivity constraint.

        Returns:
            DataFrame of SHAP values per sample per feature, or None.
        """
        if self._shap_explainer is None:
            log.warning("SHAP explainer not fitted. Call .fit() first.")
            return None

        try:
            import shap

            shap_values = self._shap_explainer.shap_values(X, check_additivity=check_additivity)

            if isinstance(shap_values, list):
                # Multi-class: take class 1 or average
                shap_values = shap_values[1] if len(shap_values) == 2 else np.abs(np.array(shap_values)).mean(axis=0)

            return pd.DataFrame(shap_values, columns=X.columns, index=X.index)
        except Exception as e:
            log.warning(f"SHAP values computation failed: {e}")
            return None

    def get_global_importance(self, X: pd.DataFrame) -> Optional[pd.Series]:
        """Return mean absolute SHAP value per feature (global importance).

        Args:
            X: Data to compute importance over.

        Returns:
            pd.Series sorted descending, or None if SHAP unavailable.
        """
        shap_df = self.get_shap_values(X)
        if shap_df is None:
            return None
        return shap_df.abs().mean().sort_values(ascending=False)

    def explain_instance(
        self, X_instance: pd.DataFrame, method: str = "shap"
    ) -> Optional[dict]:
        """Explain a single prediction.

        Args:
            X_instance: Single row DataFrame.
            method: 'shap' or 'lime'.

        Returns:
            Dict with feature contributions.
        """
        if method == "lime":
            return self._lime_explain(X_instance)
        shap_df = self.get_shap_values(X_instance)
        if shap_df is None:
            return None
        return shap_df.iloc[0].to_dict()

    def _lime_explain(self, X_instance: pd.DataFrame) -> Optional[dict]:
        try:
            from lime.lime_tabular import LimeTabularExplainer

            assert self._background_data is not None
            lime_exp = LimeTabularExplainer(
                self._background_data.values,
                feature_names=list(X_instance.columns),
                mode="classification" if self.task_type == "classification" else "regression",
            )
            predict_fn = (
                self.model.predict_proba
                if self.task_type == "classification"
                else self.model.predict
            )
            exp = lime_exp.explain_instance(X_instance.values[0], predict_fn, num_features=20)
            return dict(exp.as_list())
        except Exception as e:
            log.warning(f"LIME explanation failed: {e}")
            return None

    def plot_summary(
        self,
        X: pd.DataFrame,
        output_dir: Union[str, Path] = "reports/explain",
        plot_type: str = "bar",
    ) -> Optional[Path]:
        """Save SHAP summary plot as PNG.

        Args:
            X: Data to create SHAP summary for.
            output_dir: Directory to save plot.
            plot_type: 'bar' or 'beeswarm'.

        Returns:
            Path to saved plot file, or None.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import shap

            shap_df = self.get_shap_values(X)
            if shap_df is None:
                return None

            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            plot_path = output_dir / f"shap_{plot_type}.png"

            shap_values = shap_df.values
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X, plot_type=plot_type, show=False)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            log.info(f"SHAP summary plot saved to {plot_path}")
            return plot_path
        except Exception as e:
            log.warning(f"Could not save SHAP plot: {e}")
            return None
