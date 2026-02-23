"""
Page 4: Predictions
Allows interactive single-record and batch file predictions.
"""
import streamlit as st
import pandas as pd

from src.dashboard.components.charts import scatter_predictions
from src.dashboard.components.widgets import (
    file_uploader_csv,
    render_dataframe_section,
    sidebar_model_loader,
)


def render(models_dir: str = "models") -> None:
    st.title("🎯 Predictions")
    st.markdown("Run predictions interactively or upload a batch file.")

    model, pipeline = sidebar_model_loader(models_dir)
    if model is None:
        st.warning("No model loaded. Train a model first.")
        return

    tab1, tab2 = st.tabs(["🔹 Single Prediction", "📂 Batch File Prediction"])

    # -----------------------------------------------------------------------
    # Tab 1: Single prediction
    # -----------------------------------------------------------------------
    with tab1:
        st.subheader("Enter feature values manually")
        feature_names = getattr(model, "_feature_names", [])
        if not feature_names and pipeline is not None:
            feature_names = list(getattr(pipeline, "_feature_names_out", []))

        if not feature_names:
            st.info("Feature names not stored. Upload a file in the batch tab instead.")
        else:
            with st.form("single_predict_form"):
                values = {}
                cols = st.columns(min(3, len(feature_names)))
                for i, feat in enumerate(feature_names[:30]):  # limit to 30 for UX
                    values[feat] = cols[i % len(cols)].number_input(
                        feat, value=0.0, key=f"feat_{feat}"
                    )
                submitted = st.form_submit_button("🚀 Predict")

            if submitted:
                import pandas as pd
                row = pd.DataFrame([values])
                try:
                    if pipeline:
                        row = pipeline.transform(row)
                    pred = model.predict(row)[0]
                    st.success(f"**Prediction: `{pred}`**")
                    try:
                        probas = model.predict_proba(row)[0]
                        proba_df = pd.DataFrame({"Class": range(len(probas)), "Probability": probas})
                        import plotly.express as px
                        fig = px.bar(
                            proba_df, x="Class", y="Probability",
                            template="plotly_dark", color="Probability",
                            color_continuous_scale="Blues",
                            title="Class Probabilities",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        pass
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

    # -----------------------------------------------------------------------
    # Tab 2: Batch prediction
    # -----------------------------------------------------------------------
    with tab2:
        st.subheader("Upload a CSV for batch inference")
        batch_df = file_uploader_csv(label="Upload batch data", key="batch_pred_upload")

        if batch_df is not None:
            render_dataframe_section(batch_df, title="Input Preview", n_rows=5)

            # Optional true labels for evaluation
            target_col = None
            if st.checkbox("Dataset includes true labels?", key="batch_has_labels"):
                target_col = st.selectbox("True label column", batch_df.columns, key="batch_target_col")

            return_proba = st.checkbox("Return class probabilities", key="batch_proba")

            if st.button("🚀 Run Batch Prediction", key="batch_run"):
                with st.spinner("Predicting..."):
                    try:
                        X = batch_df.drop(columns=[target_col]) if target_col else batch_df
                        if pipeline:
                            X = pipeline.transform(X)
                        preds = model.predict(X)
                        result_df = pd.DataFrame({"prediction": preds})

                        if return_proba:
                            try:
                                probas = model.predict_proba(X)
                                for ci in range(probas.shape[1]):
                                    result_df[f"proba_class_{ci}"] = probas[:, ci]
                            except Exception:
                                pass

                        st.success(f"✅ Predictions complete for {len(result_df):,} records.")
                        render_dataframe_section(result_df, "Prediction Results", n_rows=20, expander=False)

                        # Evaluate if labels provided
                        if target_col:
                            y_true = batch_df[target_col].values
                            task_type = getattr(model, "task_type", "classification")
                            from src.evaluation.metrics import compute_metrics
                            metrics = compute_metrics(
                                pd.Series(y_true), preds, task_type
                            )
                            st.subheader("📊 Evaluation vs True Labels")
                            from src.dashboard.components.widgets import render_metric_row
                            render_metric_row(metrics)

                            if task_type == "regression":
                                st.plotly_chart(
                                    scatter_predictions(y_true, preds),
                                    use_container_width=True,
                                )

                        # Download button
                        csv_bytes = result_df.to_csv(index=False).encode()
                        st.download_button(
                            "⬇️ Download Predictions CSV",
                            csv_bytes, "predictions.csv", "text/csv",
                        )
                    except Exception as e:
                        st.error(f"Batch prediction failed: {e}")
