"""
Reusable Streamlit widget helpers for the AutoMLPro dashboard.
"""
import streamlit as st
import pandas as pd
from pathlib import Path


def render_metric_row(metrics: dict, cols_per_row: int = 4) -> None:
    """Render a dict of metrics as styled st.metric cards in a grid."""
    items = list(metrics.items())
    for i in range(0, len(items), cols_per_row):
        row = items[i:i + cols_per_row]
        cols = st.columns(len(row))
        for col, (name, value) in zip(cols, row):
            formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
            col.metric(label=name.replace("_", " ").title(), value=formatted)


def render_dataframe_section(
    df: pd.DataFrame,
    title: str = "Data Preview",
    n_rows: int = 5,
    expander: bool = True,
) -> None:
    """Render a DataFrame with an optional expander."""
    if expander:
        with st.expander(title, expanded=False):
            st.dataframe(df.head(n_rows), use_container_width=True)
            st.caption(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    else:
        st.subheader(title)
        st.dataframe(df.head(n_rows), use_container_width=True)
        st.caption(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")


def column_selector(
    df: pd.DataFrame,
    key: str = "col_select",
    label: str = "Select column",
    dtype_filter: str = "all",
) -> str:
    """Dropdown for selecting a DataFrame column, optionally filtered by dtype."""
    if dtype_filter == "numeric":
        cols = df.select_dtypes(include="number").columns.tolist()
    elif dtype_filter == "categorical":
        cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    else:
        cols = df.columns.tolist()
    return st.selectbox(label, cols, key=key)


def info_badge(text: str, color: str = "blue") -> None:
    """Render a colored badge using st.markdown."""
    color_map = {
        "blue": "#0c4a6e",
        "green": "#064e3b",
        "red": "#7f1d1d",
        "yellow": "#78350f",
    }
    bg = color_map.get(color, "#1e293b")
    st.markdown(
        f'<span style="background:{bg};padding:4px 12px;border-radius:20px;'
        f'font-size:0.78rem;font-weight:600">{text}</span>',
        unsafe_allow_html=True,
    )


def sidebar_model_loader(models_dir: str = "models") -> tuple:
    """Sidebar widget to load model + pipeline from the models directory.

    Returns:
        (model, feature_pipeline) tuple — either may be None if not found.
    """
    import joblib
    st.sidebar.header("🤖 Model")
    model_path = Path(models_dir) / "best_model.joblib"
    pipeline_path = Path(models_dir) / "feature_pipeline.joblib"

    model, pipeline = None, None
    if model_path.exists():
        model = joblib.load(model_path)
        st.sidebar.success(f"✅ Model: `{getattr(model, 'name', 'loaded')}`")
    else:
        st.sidebar.warning("No model found. Run training first.")

    if pipeline_path.exists():
        pipeline = joblib.load(pipeline_path)
    return model, pipeline


def file_uploader_csv(label: str = "Upload CSV", key: str = "upload") -> pd.DataFrame | None:
    """File uploader returning a DataFrame or None."""
    uploaded = st.file_uploader(label, type=["csv", "parquet"], key=key)
    if uploaded is None:
        return None
    if uploaded.name.endswith(".parquet"):
        return pd.read_parquet(uploaded)
    return pd.read_csv(uploaded)
