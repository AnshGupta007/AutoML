"""
Page 2: Training Results
Displays model metrics, feature importances, and CV fold scores.
"""
import json
from pathlib import Path

import streamlit as st
import pandas as pd

from src.dashboard.components.charts import feature_importance_chart, metric_trend_chart
from src.dashboard.components.widgets import render_metric_row


def _load_metadata(models_dir: str = "models") -> dict:
    meta_path = Path(models_dir) / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}


def render(models_dir: str = "models") -> None:
    st.title("🏆 Training Results")
    st.markdown("Review the best model metrics, hyperparameters, and feature importances.")

    meta = _load_metadata(models_dir)

    if not meta:
        st.warning("⚠️ No model metadata found. Train a model first via the sidebar or CLI.")
        return

    # Key identifiers
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.info(f"**Model:** `{meta.get('model_name', 'N/A')}`")
    col2.info(f"**Task:** `{meta.get('task_type', 'N/A')}`")
    col3.info(f"**Run ID:** `{meta.get('run_id', 'N/A')}`")

    # Metric cards
    metrics = meta.get("metrics", {})
    if metrics:
        st.markdown("---")
        st.subheader("📊 Validation Metrics")
        render_metric_row(metrics)

    # Best hyperparameters
    best_params = meta.get("best_params", {})
    if best_params:
        st.markdown("---")
        with st.expander("⚙️ Best Hyperparameters", expanded=True):
            params_df = pd.DataFrame(
                list(best_params.items()), columns=["Parameter", "Value"]
            )
            st.dataframe(params_df, use_container_width=True)

    # Feature importances
    fi_path = Path(models_dir) / "feature_importances.json"
    if fi_path.exists():
        st.markdown("---")
        st.subheader("📈 Feature Importances")
        fi_series = pd.read_json(fi_path, typ="series").sort_values(ascending=False)
        n_top = st.slider("Number of top features to show", 5, min(50, len(fi_series)), 20)
        st.plotly_chart(feature_importance_chart(fi_series, top_n=n_top), use_container_width=True)

    # MLflow experiment link
    mlflow_uri = meta.get("mlflow_uri", "http://localhost:5000")
    st.markdown("---")
    st.markdown(f"🔗 [View full experiment in MLflow UI]({mlflow_uri})")
