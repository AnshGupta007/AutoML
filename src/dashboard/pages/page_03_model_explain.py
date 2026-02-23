"""
Page 3: Model Explainability
Shows global and local SHAP explanations.
"""
import streamlit as st
import pandas as pd
import numpy as np

from src.dashboard.components.charts import feature_importance_chart
from src.dashboard.components.widgets import (
    file_uploader_csv,
    render_dataframe_section,
    sidebar_model_loader,
)


def render(models_dir: str = "models") -> None:
    st.title("🔍 Model Explainability")
    st.markdown("Understand model decisions using SHAP — globally and per prediction.")

    model, pipeline = sidebar_model_loader(models_dir)
    if model is None:
        st.warning("No model loaded. Train a model first.")
        return

    df = file_uploader_csv(label="Upload data for explanations (CSV)", key="explain_upload")
    if df is None:
        st.info("⬆️ Upload a dataset to compute SHAP explanations.")
        return

    # Apply feature pipeline
    try:
        if pipeline is not None:
            X_transformed = pipeline.transform(df)
        else:
            X_transformed = df.select_dtypes("number")
    except Exception as e:
        st.error(f"Feature transformation failed: {e}")
        return

    # Compute SHAP
    with st.spinner("Computing SHAP values... (may take a moment)"):
        try:
            from src.evaluation.explainability import ModelExplainer
            task_type = getattr(model, "task_type", "classification")
            explainer = ModelExplainer(model, task_type=task_type)
            sample = X_transformed.sample(min(200, len(X_transformed)), random_state=42)
            explainer.fit(sample)

            shap_df = explainer.get_shap_values(X_transformed.head(500))
            global_fi = explainer.get_global_importance(X_transformed.head(500))
        except Exception as e:
            st.error(f"SHAP computation failed: {e}")
            return

    if global_fi is None or shap_df is None:
        st.error("SHAP explanations could not be computed.")
        return

    # Global importance
    st.markdown("---")
    st.subheader("🌍 Global Feature Importance (Mean |SHAP|)")
    n_top = st.slider("Features to show", 5, min(40, len(global_fi)), 15, key="shap_top_n")
    st.plotly_chart(feature_importance_chart(global_fi, top_n=n_top), use_container_width=True)

    # Local explanation
    st.markdown("---")
    st.subheader("🎯 Local Explanation — Single Prediction")
    row_idx = st.number_input("Row index to explain", 0, len(df) - 1, 0, key="local_explain_idx")
    local_vals = shap_df.iloc[row_idx].sort_values(key=abs, ascending=False).head(15)
    local_df = local_vals.reset_index()
    local_df.columns = ["Feature", "SHAP Value"]
    local_df["Direction"] = local_df["SHAP Value"].map(lambda v: "↑ Positive" if v > 0 else "↓ Negative")

    import plotly.express as px
    fig = px.bar(
        local_df, x="SHAP Value", y="Feature", orientation="h",
        color="Direction",
        color_discrete_map={"↑ Positive": "#34d399", "↓ Negative": "#f87171"},
        title=f"SHAP Values for Row {row_idx}",
        template="plotly_dark",
    )
    fig.update_layout(height=450, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    # SHAP value table
    with st.expander("📋 Full SHAP Value Table"):
        st.dataframe(shap_df.head(50), use_container_width=True)
