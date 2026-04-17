"""Predictive signal quality analysis."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


def detect_regimes(X: pd.DataFrame, method: str = "pca") -> pd.DataFrame:
    """Reduce dimensionality for regime visualization."""

    # Select numeric columns only
    data = X.select_dtypes(include=[np.number]).dropna()

    if method == "pca":
        reducer = PCA(n_components=2)
        components = reducer.fit_transform(data)
        labels = [f"PC{i+1}" for i in range(2)]
    elif method == "umap":
        reducer = umap.UMAP(n_components=2, random_state=42)
        components = reducer.fit_transform(data)
        labels = ["UMAP1", "UMAP2"]
    else:
        raise ValueError(f"Unknown method: {method}")

    result = pd.DataFrame(components, columns=labels, index=data.index)
    return result


def plot_regime_map(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    color_by: str = "time",
    method: str = "umap",
) -> go.Figure:
    """2D regime visualization colored by time or returns."""

    embedding = detect_regimes(X, method)

    if color_by == "time":
        color = embedding.index.astype(int)  # Convert datetime to int for coloring
        color_label = "Time"
        colorscale = "Viridis"
    elif color_by == "return" and y is not None:
        color = y.reindex(embedding.index)
        color_label = "Next Return"
        colorscale = "RdYlGn"
    else:
        color = None
        color_label = None
        colorscale = None

    fig = go.Figure()

    scatter = go.Scatter(
        x=embedding.iloc[:, 0],
        y=embedding.iloc[:, 1],
        mode="markers",
        marker=dict(
            color=color,
            colorscale=colorscale,
            colorbar=dict(title=color_label) if color_label else None,
            size=4,
            opacity=0.6,
        ),
        text=[str(idx) for idx in embedding.index],
        hovertemplate="%{text}<br>X: %{x:.3f}<br>Y: %{y:.3f}",
    )

    fig.add_trace(scatter)

    fig.update_layout(
        title=f"Market Regimes ({method.upper()}) - Colored by {color_by}",
        xaxis_title=embedding.columns[0],
        yaxis_title=embedding.columns[1],
        template="plotly_dark",
        height=600,
    )

    return fig


def plot_lead_lag_heatmap(
    dfs: Dict[str, pd.DataFrame],
    tickers: List[str],
    feature: str = "log_return",
    max_lag: int = 5,
) -> go.Figure:
    """Heatmap of cross-correlations at different lags."""

    correlations = np.zeros((len(tickers), len(tickers)))

    for i, t1 in enumerate(tickers):
        for j, t2 in enumerate(tickers):
            if t1 not in dfs or t2 not in dfs:
                continue

            s1 = dfs[t1][feature] if feature in dfs[t1].columns else dfs[t1].iloc[:, 0]
            s2 = dfs[t2][feature] if feature in dfs[t2].columns else dfs[t2].iloc[:, 0]

            # Align and compute correlation
            aligned = pd.concat([s1, s2.shift(1)], axis=1).dropna()
            if len(aligned) > 10:
                correlations[i, j] = aligned.corr().iloc[0, 1]

    fig = px.imshow(
        correlations,
        x=tickers,
        y=tickers,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        range_color=[-1, 1],
        title=f"Lead-Lag Correlation Matrix (1-bar lag, {feature})",
    )

    fig.update_layout(template="plotly_dark", height=500)
    return fig


def plot_feature_stability(
    X: pd.DataFrame, feature: str, window: int = 60
) -> go.Figure:
    """Plot rolling mean and std to detect feature drift."""

    data = X[feature]
    roll_mean = data.rolling(window).mean()
    roll_std = data.rolling(window).std()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(f"{feature} Rolling Mean", f"{feature} Rolling Std"),
    )

    fig.add_trace(
        go.Scatter(x=X.index, y=roll_mean, name="Mean", line=dict(width=1)),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=X.index, y=roll_std, name="Std", line=dict(width=1, color="orange")
        ),
        row=2,
        col=1,
    )

    # Add stability bands
    overall_mean = data.mean()
    overall_std = data.std()

    fig.add_hline(y=overall_mean, line_dash="dash", line_color="green", row=1, col=1)
    fig.add_hline(y=overall_std, line_dash="dash", line_color="green", row=2, col=1)

    fig.update_layout(
        title=f"Feature Stability Analysis ({window}-bar window)",
        template="plotly_dark",
        height=500,
    )

    return fig


def plot_prediction_diagnostics(
    y_true: np.ndarray, y_pred: np.ndarray, timestamps: pd.DatetimeIndex
) -> Dict[str, go.Figure]:
    """Generate model diagnostic plots."""

    residuals = y_pred - y_true
    direction_acc = np.mean((y_true > 0) == (y_pred > 0))

    figs = {}

    # 1. Time series comparison
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=timestamps, y=y_true, name="Actual", line=dict(width=1)))
    fig.add_trace(
        go.Scatter(x=timestamps, y=y_pred, name="Predicted", line=dict(width=1))
    )
    fig.update_layout(
        title=f"Prediction vs Actual (Dir. Acc: {direction_acc:.2%})",
        template="plotly_dark",
        height=400,
    )
    figs["timeseries"] = fig

    # 2. Residual distribution
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=residuals, nbinsx=50, name="Residuals"))
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    fig.update_layout(title="Residual Distribution", template="plotly_dark", height=400)
    figs["residuals"] = fig

    # 3. Scatter actual vs predicted
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_pred,
            mode="markers",
            marker=dict(size=4, opacity=0.5),
            name="Predictions",
        )
    )

    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(dash="dash", color="red"),
            name="Perfect",
        )
    )

    fig.update_layout(
        title="Predicted vs Actual",
        xaxis_title="Actual",
        yaxis_title="Predicted",
        template="plotly_dark",
        height=400,
    )
    figs["scatter"] = fig

    return figs


def plot_simple_backtest(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: pd.DatetimeIndex,
    threshold: float = 0.0,
) -> go.Figure:
    """Simple long/short strategy backtest."""

    # Strategy: Long if pred > threshold, Short if pred < -threshold
    positions = np.where(y_pred > threshold, 1, np.where(y_pred < -threshold, -1, 0))
    returns = positions * y_true

    # Cumulative returns
    cum_returns = np.cumsum(returns)
    buy_hold = np.cumsum(y_true)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=cum_returns,
            name="Strategy",
            line=dict(width=1.5, color="green"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=buy_hold,
            name="Buy & Hold",
            line=dict(width=1.5, color="gray", dash="dash"),
        )
    )

    # Add zero line
    fig.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.3)

    # Calculate metrics
    sharpe = np.sqrt(252) * returns.mean() / (returns.std() + 1e-10)

    fig.update_layout(
        title=f"Cumulative Returns (Sharpe: {sharpe:.2f})",
        xaxis_title="Time",
        yaxis_title="Cumulative Log Return",
        template="plotly_dark",
        height=500,
    )

    return fig
