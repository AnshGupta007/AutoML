"""
Reusable chart components for the Streamlit dashboard.
All functions return Plotly figures for consistent dark-themed styling.
"""
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

DARK_TEMPLATE = "plotly_dark"
PRIMARY_COLOR = "#38bdf8"
ACCENT_COLOR = "#818cf8"
SUCCESS_COLOR = "#34d399"
WARNING_COLOR = "#fbbf24"
DANGER_COLOR = "#f87171"

COLOR_SEQUENCE = [PRIMARY_COLOR, ACCENT_COLOR, SUCCESS_COLOR, WARNING_COLOR, DANGER_COLOR,
                  "#c084fc", "#fb923c", "#a3e635"]


def metric_trend_chart(history_df: pd.DataFrame, metric: str, title: str = "") -> go.Figure:
    """Line chart for a metric over time."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history_df.index,
        y=history_df[metric],
        mode="lines+markers",
        name=metric,
        line=dict(color=PRIMARY_COLOR, width=2),
        marker=dict(size=6),
    ))
    fig.update_layout(
        template=DARK_TEMPLATE,
        title=title or metric,
        xaxis_title="Timestamp",
        yaxis_title=metric,
        height=350,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def feature_importance_chart(fi_series: pd.Series, top_n: int = 20) -> go.Figure:
    """Horizontal bar chart for feature importances."""
    top = fi_series.nlargest(top_n).sort_values()
    fig = go.Figure(go.Bar(
        x=top.values,
        y=top.index.tolist(),
        orientation="h",
        marker_color=PRIMARY_COLOR,
        marker=dict(
            color=top.values,
            colorscale="Blues",
            showscale=False,
        ),
    ))
    fig.update_layout(
        template=DARK_TEMPLATE,
        title=f"Top {top_n} Feature Importances",
        xaxis_title="Importance",
        height=max(300, top_n * 20),
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def confusion_matrix_chart(cm: pd.DataFrame) -> go.Figure:
    """Heatmap for a confusion matrix DataFrame."""
    fig = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Confusion Matrix",
        template=DARK_TEMPLATE,
    )
    fig.update_layout(height=400, margin=dict(l=40, r=20, t=40, b=40))
    return fig


def correlation_heatmap(df: pd.DataFrame, max_cols: int = 20) -> go.Figure:
    """Correlation matrix heatmap."""
    numeric = df.select_dtypes(include="number").iloc[:, :max_cols]
    corr = numeric.corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Feature Correlation Matrix",
        template=DARK_TEMPLATE,
    )
    fig.update_layout(height=500, margin=dict(l=40, r=20, t=40, b=40))
    return fig


def histogram_chart(series: pd.Series, title: str = "") -> go.Figure:
    """Histogram for a numeric series."""
    fig = px.histogram(
        series,
        nbins=40,
        title=title or series.name,
        template=DARK_TEMPLATE,
        color_discrete_sequence=[PRIMARY_COLOR],
    )
    fig.update_layout(height=300, margin=dict(l=40, r=20, t=40, b=40))
    return fig


def roc_curve_chart(fpr: list, tpr: list, auc: float) -> go.Figure:
    """ROC curve with AUC annotation."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        name=f"ROC (AUC={auc:.3f})",
        line=dict(color=PRIMARY_COLOR, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random", line=dict(color=ACCENT_COLOR, dash="dash"),
    ))
    fig.update_layout(
        template=DARK_TEMPLATE,
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=400,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def drift_bar_chart(drift_results: dict) -> go.Figure:
    """Bar chart showing per-column drift score."""
    per_col = drift_results.get("per_column_drift", {})
    if not per_col:
        return go.Figure()

    cols = list(per_col.keys())
    drifted = [per_col[c].get("drifted", False) for c in cols]
    colors = [DANGER_COLOR if d else SUCCESS_COLOR for d in drifted]

    fig = go.Figure(go.Bar(
        x=cols,
        y=[int(d) for d in drifted],
        marker_color=colors,
        text=["DRIFT" if d else "OK" for d in drifted],
        textposition="auto",
    ))
    fig.update_layout(
        template=DARK_TEMPLATE,
        title="Data Drift by Feature",
        yaxis=dict(tickvals=[0, 1], ticktext=["No Drift", "Drifted"]),
        height=350,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def scatter_predictions(y_true, y_pred, title: str = "Predictions vs Actual") -> go.Figure:
    """Scatter plot for regression predictions vs actual values."""
    fig = px.scatter(
        x=y_true, y=y_pred,
        labels={"x": "Actual", "y": "Predicted"},
        title=title,
        template=DARK_TEMPLATE,
        color_discrete_sequence=[PRIMARY_COLOR],
        opacity=0.6,
    )
    # Perfect prediction line
    mn, mx = min(y_true), max(y_true)
    fig.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx],
        mode="lines", name="Perfect",
        line=dict(color=ACCENT_COLOR, dash="dash"),
    ))
    fig.update_layout(height=400, margin=dict(l=40, r=20, t=40, b=40))
    return fig


def missing_value_chart(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart showing missing value percent per column."""
    missing = (df.isnull().mean() * 100).sort_values(ascending=True)
    missing = missing[missing > 0]
    if missing.empty:
        return go.Figure()
    fig = go.Figure(go.Bar(
        x=missing.values,
        y=missing.index.tolist(),
        orientation="h",
        marker_color=[v > 20 and DANGER_COLOR or WARNING_COLOR for v in missing.values],
    ))
    fig.update_layout(
        template=DARK_TEMPLATE,
        title="Missing Value Percentage",
        xaxis_title="Missing %",
        height=max(200, len(missing) * 22),
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig
