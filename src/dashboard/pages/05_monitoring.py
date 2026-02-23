"""
Page 5: Monitoring
Shows data drift, model performance trends, and alerting history.
"""
import streamlit as st
import pandas as pd

from src.dashboard.components.charts import drift_bar_chart, metric_trend_chart
from src.dashboard.components.widgets import file_uploader_csv, render_metric_row, sidebar_model_loader


def render(models_dir: str = "models") -> None:
    st.title("📡 Monitoring")
    st.markdown("Detect data drift and track model performance over time.")

    model, pipeline = sidebar_model_loader(models_dir)

    tab1, tab2 = st.tabs(["🌊 Data Drift", "📈 Performance Trend"])

    # -----------------------------------------------------------------------
    # Tab 1: Data Drift
    # -----------------------------------------------------------------------
    with tab1:
        st.subheader("Compare reference and current data distributions")
        col1, col2 = st.columns(2)
        with col1:
            ref_df = file_uploader_csv("Upload Reference Data (training)", key="drift_ref")
        with col2:
            cur_df = file_uploader_csv("Upload Current Batch", key="drift_cur")

        if ref_df is not None and cur_df is not None:
            drift_threshold = st.slider(
                "Drift threshold (share of drifted features)",
                0.1, 0.9, 0.3, 0.05, key="drift_thresh"
            )

            if st.button("🔍 Run Drift Detection", key="run_drift"):
                with st.spinner("Detecting drift..."):
                    try:
                        from src.monitoring.data_drift import DataDriftDetector
                        detector = DataDriftDetector(drift_threshold=drift_threshold)
                        detector.set_reference(ref_df)
                        report = detector.detect(cur_df, save_html=False)

                        # Summary
                        st.markdown("---")
                        drift_cols = [
                            ("Drift Detected", "✅ No" if not report["drift_detected"] else "⚠️ YES"),
                            ("Drifted Share", f"{report['drift_share']:.1%}"),
                            ("Threshold", f"{drift_threshold:.1%}"),
                        ]
                        render_metric_row(dict(drift_cols), cols_per_row=3)

                        if report["drift_detected"]:
                            st.error("🚨 Significant data drift detected! Consider retraining.")
                        else:
                            st.success("✅ Data drift is within acceptable limits.")

                        # Per-column chart
                        st.plotly_chart(drift_bar_chart(report), use_container_width=True)

                        # Detailed table
                        with st.expander("📋 Per-Column Drift Details"):
                            per_col = report.get("per_column_drift", {})
                            if per_col:
                                detail_df = pd.DataFrame(per_col).T.reset_index()
                                detail_df.columns = ["Feature"] + list(detail_df.columns[1:])
                                st.dataframe(detail_df, use_container_width=True)

                    except Exception as e:
                        st.error(f"Drift detection failed: {e}")
        else:
            st.info("⬆️ Upload both reference and current datasets to run drift detection.")

    # -----------------------------------------------------------------------
    # Tab 2: Performance Trend
    # -----------------------------------------------------------------------
    with tab2:
        st.subheader("Track model performance metrics over time")

        history_file = file_uploader_csv("Upload monitoring history CSV", key="perf_history")
        if history_file is not None:
            numeric_cols = history_file.select_dtypes("number").columns.tolist()
            if not numeric_cols:
                st.warning("No numeric columns found in history file.")
            else:
                sel_metric = st.selectbox("Metric to plot", numeric_cols, key="perf_metric_sel")
                if "timestamp" in history_file.columns:
                    history_file = history_file.set_index("timestamp")
                st.plotly_chart(
                    metric_trend_chart(history_file, sel_metric, title=f"{sel_metric} over time"),
                    use_container_width=True,
                )

                # Degradation alert
                baseline_val = st.number_input(
                    f"Baseline {sel_metric} value", value=0.0, key="baseline_val"
                )
                current_val = float(history_file[sel_metric].iloc[-1])
                if baseline_val > 0:
                    delta = current_val - baseline_val
                    threshold = 0.05
                    if sel_metric in ("rmse", "mae", "mse", "mape"):
                        degraded = delta > threshold * baseline_val
                    else:
                        degraded = delta < -threshold

                    if degraded:
                        st.error(f"⚠️ Performance degraded: {sel_metric}={current_val:.4f} vs baseline={baseline_val:.4f}")
                    else:
                        st.success(f"✅ Performance stable: {sel_metric}={current_val:.4f}")
        else:
            st.info(
                "Upload a monitoring history CSV or call `monitor.save_history()` from the CLI to generate one."
            )
            st.markdown("**Expected columns:** `timestamp`, metric columns (e.g. `accuracy`, `rmse`)")
