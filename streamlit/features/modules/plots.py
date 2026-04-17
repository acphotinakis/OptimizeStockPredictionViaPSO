"""Plotting utilities for feature diagnostics."""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


def plot_feature_timeseries(
    X: pd.DataFrame,
    feature: str,
    y: Optional[pd.Series] = None,
    ohlcv: Optional[pd.DataFrame] = None,
    rolling_windows: List[int] = [20, 60],
) -> go.Figure:
    """Interactive feature time-series with optional price overlay."""

    fig = make_subplots(
        rows=3 if ohlcv is not None else 2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.4, 0.3] if ohlcv is not None else [0.7, 0.3],
        subplot_titles=(
            (f"{feature}", "Next Return", "Price")
            if ohlcv is not None
            else (f"{feature}", "Next Return")
        ),
    )

    # Main feature plot
    fig.add_trace(
        go.Scatter(x=X.index, y=X[feature], name=feature, line=dict(width=1)),
        row=1,
        col=1,
    )

    # Rolling statistics
    for w in rolling_windows:
        if len(X) > w:
            roll_mean = X[feature].rolling(w).mean()
            fig.add_trace(
                go.Scatter(
                    x=X.index,
                    y=roll_mean,
                    name=f"MA{w}",
                    line=dict(width=1, dash="dash"),
                ),
                row=1,
                col=1,
            )

    # Target overlay
    if y is not None:
        fig.add_trace(
            go.Scatter(
                x=y.index,
                y=y,
                name="Next Return",
                line=dict(width=1, color="orange"),
                opacity=0.7,
            ),
            row=2,
            col=1,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    # Price overlay if available
    if ohlcv is not None:
        fig.add_trace(
            go.Scatter(
                x=ohlcv.index,
                y=ohlcv["close"],
                name="Close",
                line=dict(width=1, color="green"),
            ),
            row=3,
            col=1,
        )

    fig.update_layout(
        height=800 if ohlcv is not None else 600,
        showlegend=True,
        hovermode="x unified",
        template="plotly_dark",
    )

    return fig


def plot_multi_feature_overlay(
    X: pd.DataFrame, features: List[str], normalize: bool = True
) -> go.Figure:
    """Overlay multiple features (z-score normalized)."""

    fig = go.Figure()

    for feat in features:
        if feat not in X.columns:
            continue
        data = X[feat]
        if normalize:
            data = (data - data.mean()) / (data.std() + 1e-10)
            label = f"{feat} (z)"
        else:
            label = feat

        fig.add_trace(
            go.Scatter(x=X.index, y=data, name=label, line=dict(width=1), opacity=0.8)
        )

    fig.update_layout(
        title="Multi-Feature Overlay" + (" (Z-Score Normalized)" if normalize else ""),
        xaxis_title="Time",
        yaxis_title="Value",
        hovermode="x unified",
        template="plotly_dark",
        height=500,
    )

    return fig


def plot_distribution_comparison(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    feature: str,
    plot_type: str = "histogram",
) -> go.Figure:
    """Compare distributions across splits."""

    fig = go.Figure()

    splits = [("Train", X_train), ("Val", X_val), ("Test", X_test)]
    colors = ["blue", "orange", "green"]

    for (name, X), color in zip(splits, colors):
        if feature not in X.columns:
            continue
        data = X[feature].dropna()

        if plot_type == "histogram":
            fig.add_trace(
                go.Histogram(
                    x=data,
                    name=f"{name} (μ={data.mean():.3f})",
                    opacity=0.6,
                    nbinsx=50,
                    marker_color=color,
                )
            )
        else:  # KDE using histogram approximation
            fig.add_trace(
                go.Violin(
                    x=data,
                    name=name,
                    box_visible=True,
                    meanline_visible=True,
                    line_color=color,
                )
            )

    fig.update_layout(
        title=f"{feature} Distribution Comparison",
        xaxis_title=feature,
        yaxis_title="Density" if plot_type == "histogram" else "",
        barmode="overlay",
        template="plotly_dark",
        height=400,
    )

    return fig


def plot_rolling_statistics(
    X: pd.DataFrame, feature: str, window: int = 60
) -> go.Figure:
    """Plot rolling mean, std, and z-score."""

    feat_data = X[feature]
    roll_mean = feat_data.rolling(window).mean()
    roll_std = feat_data.rolling(window).std()
    z_score = (feat_data - roll_mean) / (roll_std + 1e-10)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(f"{feature}", f"Rolling Std ({window})", "Z-Score"),
    )

    fig.add_trace(
        go.Scatter(x=X.index, y=feat_data, name=feature, line=dict(width=1)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=X.index, y=roll_mean, name=f"MA{window}", line=dict(width=1, dash="dash")
        ),
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

    fig.add_trace(
        go.Scatter(
            x=X.index, y=z_score, name="Z-Score", line=dict(width=1, color="purple")
        ),
        row=3,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
    fig.add_hline(y=2, line_dash="dot", line_color="red", row=3, col=1)
    fig.add_hline(y=-2, line_dash="dot", line_color="red", row=3, col=1)

    fig.update_layout(height=700, template="plotly_dark", showlegend=True)
    return fig


def plot_scatter_hexbin(
    X: pd.DataFrame, y: pd.Series, feature: str, nbins: int = 30
) -> go.Figure:
    """Hexbin-style density scatter plot (using 2D histogram)."""

    data = pd.DataFrame({feature: X[feature], "next_return": y}).dropna()

    fig = go.Figure()

    # 2D histogram for density
    fig.add_trace(
        go.Histogram2d(
            x=data[feature],
            y=data["next_return"],
            nbinsx=nbins,
            nbinsy=nbins,
            colorscale="Viridis",
            showscale=True,
            name="Density",
        )
    )

    # Add regression line
    z = np.polyfit(data[feature], data["next_return"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(data[feature].min(), data[feature].max(), 100)

    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=p(x_line),
            mode="lines",
            name=f"Trend (slope={z[0]:.4f})",
            line=dict(color="red", width=2),
        )
    )

    fig.update_layout(
        title=f"{feature} vs Next Return",
        xaxis_title=feature,
        yaxis_title="Next Log Return",
        template="plotly_dark",
        height=500,
    )

    return fig


def plot_quantile_returns(
    X: pd.DataFrame, y: pd.Series, feature: str, n_bins: int = 10
) -> go.Figure:
    """Binned feature quantiles vs mean future returns."""

    data = pd.DataFrame({feature: X[feature], "target": y}).dropna()

    # Create quantile bins
    data["bin"] = pd.qcut(data[feature], q=n_bins, labels=False, duplicates="drop")

    # Calculate statistics per bin
    bin_stats = (
        data.groupby("bin")
        .agg({"target": ["mean", "std", "count"], feature: "mean"})
        .reset_index()
    )

    bin_stats.columns = ["bin", "mean_ret", "std_ret", "count", "feat_mean"]

    # Create bar plot with error bars
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=[f"Q{i+1}" for i in bin_stats["bin"]],
            y=bin_stats["mean_ret"],
            error_y=dict(type="data", array=bin_stats["std_ret"], visible=True),
            marker_color=bin_stats["mean_ret"],
            marker_colorscale="RdYlGn",
            text=[f"n={int(c)}" for c in bin_stats["count"]],
            textposition="outside",
            name="Mean Return ± Std",
        )
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        title=f"Future Return by {feature} Quantile",
        xaxis_title=f"{feature} Quantile (Feature Mean Value)",
        yaxis_title="Mean Next Log Return",
        template="plotly_dark",
        height=450,
    )

    return fig


def plot_acf_pacf(series: pd.Series, lags: int = 40) -> go.Figure:
    """Autocorrelation and partial autocorrelation plots."""
    from statsmodels.tsa.stattools import acf, pacf

    # Calculate ACF and PACF
    acf_vals = acf(series.dropna(), nlags=lags, fft=True)
    pacf_vals = pacf(series.dropna(), nlags=lags, method="ols")

    conf_level = 1.96 / np.sqrt(len(series))

    fig = make_subplots(rows=2, cols=1, subplot_titles=("ACF", "PACF"))

    # ACF
    fig.add_trace(
        go.Bar(x=list(range(len(acf_vals))), y=acf_vals, name="ACF"), row=1, col=1
    )
    fig.add_hline(y=conf_level, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=-conf_level, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=0, line_color="gray", row=1, col=1)

    # PACF
    fig.add_trace(
        go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals, name="PACF"), row=2, col=1
    )
    fig.add_hline(y=conf_level, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=-conf_level, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=0, line_color="gray", row=2, col=1)

    fig.update_layout(height=600, template="plotly_dark", showlegend=False)
    return fig
