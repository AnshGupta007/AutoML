"""
Page 1: Data Overview
Displays dataset statistics, missing values, distributions, and correlations.
"""
import streamlit as st
import pandas as pd

from src.dashboard.components.charts import (
    correlation_heatmap,
    histogram_chart,
    missing_value_chart,
)
from src.dashboard.components.widgets import (
    column_selector,
    file_uploader_csv,
    render_dataframe_section,
    render_metric_row,
)


def render() -> None:
    st.title("📊 Data Overview")
    st.markdown("Upload your dataset to explore its structure, statistics, and distributions.")

    df = file_uploader_csv(label="Upload training or raw CSV", key="data_overview_upload")

    if df is None:
        st.info("⬆️ Upload a CSV or Parquet file to get started.")
        return

    # Summary metrics
    n_rows, n_cols = df.shape
    n_missing = int(df.isnull().sum().sum())
    n_duplicates = int(df.duplicated().sum())
    st.markdown("---")
    render_metric_row({
        "Rows": f"{n_rows:,}",
        "Columns": str(n_cols),
        "Missing Cells": f"{n_missing:,}",
        "Duplicate Rows": f"{n_duplicates:,}",
        "Numeric Cols": str(len(df.select_dtypes("number").columns)),
        "Categorical Cols": str(len(df.select_dtypes("object").columns)),
    }, cols_per_row=6)

    st.markdown("---")
    render_dataframe_section(df, title="📋 Data Preview (first 10 rows)", n_rows=10)

    # Descriptive statistics
    with st.expander("📈 Descriptive Statistics"):
        st.dataframe(df.describe(include="all").T, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    # Missing values
    with col1:
        st.subheader("🕳️ Missing Values")
        mv_chart = missing_value_chart(df)
        if mv_chart.data:
            st.plotly_chart(mv_chart, use_container_width=True)
        else:
            st.success("✅ No missing values detected.")

    # Column distribution
    with col2:
        st.subheader("📊 Column Distribution")
        numeric_cols = df.select_dtypes("number").columns.tolist()
        if numeric_cols:
            sel_col = column_selector(df, key="dist_col", label="Select column", dtype_filter="numeric")
            st.plotly_chart(histogram_chart(df[sel_col]), use_container_width=True)
        else:
            st.info("No numeric columns to plot.")

    st.markdown("---")

    # Correlation
    st.subheader("🔗 Feature Correlations")
    numeric_df = df.select_dtypes("number")
    if numeric_df.shape[1] >= 2:
        st.plotly_chart(correlation_heatmap(numeric_df), use_container_width=True)
    else:
        st.info("Need at least 2 numeric columns for correlation matrix.")

    # Data types breakdown
    st.markdown("---")
    with st.expander("🗂️ Column Data Types"):
        dtype_df = pd.DataFrame({
            "Column": df.columns,
            "Dtype": [str(dt) for dt in df.dtypes],
            "Non-Null Count": df.notnull().sum().values,
            "Missing %": (df.isnull().mean() * 100).round(2).values,
            "Unique Values": df.nunique().values,
        })
        st.dataframe(dtype_df, use_container_width=True)

    # -----------------------------------------------------------------------
    # Auto-Train Section
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.subheader("🚀 Auto-Train Model")
    st.markdown("Configure and launch the AutoML pipeline on this dataset.")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        target_col = st.selectbox("Select Target Column", df.columns, index=len(df.columns)-1)
    with col2:
        # Simple task type detection
        default_task = "classification"
        if pd.api.types.is_numeric_dtype(df[target_col]) and df[target_col].nunique() > 10:
            default_task = "regression"
        task_type = st.selectbox("Task Type", ["classification", "regression"], 
                                 index=0 if default_task == "classification" else 1)
    with col3:
        n_trials = st.number_input("HPO Trials", 5, 100, 20)

    if st.button("🏁 Start Training Pipeline", use_container_width=True):
        import os
        from src.orchestration.pipeline import automl_pipeline

        # Guard: data dir exists
        os.makedirs("data", exist_ok=True)
        data_path = "data/uploaded_dataset.csv"
        df.to_csv(data_path, index=False)

        with st.status("🏗️ Training in progress...", expanded=True) as status:
            st.write("🔄 Initializing pipeline...")
            try:
                result = automl_pipeline(
                    data_path=data_path,
                    target_column=target_col,
                    task_type=task_type,
                    n_trials=n_trials,
                    output_dir="reports",
                    enable_hpo=True,
                    enable_ensemble=True
                )
                status.update(label="✅ Training Complete!", state="complete", expanded=False)
                st.success("🎉 Model trained successfully!")

                # Display key metrics
                metrics = result["eval_result"]["metrics"]
                st.markdown("### 📊 Performance Summary")
                render_metric_row(metrics)

                st.info(f"📁 Reports saved to: `{result['report_path']}`")
                st.balloons()
            except Exception as e:
                status.update(label="❌ Training Failed", state="error")
                st.error(f"Error during training: {str(e)}")
