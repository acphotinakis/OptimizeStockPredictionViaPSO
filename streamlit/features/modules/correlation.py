"""Correlation structure analysis and diagnostics."""

from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from scipy.stats import pearsonr


# def compute_correlation_matrix(
#     X: pd.DataFrame, method: str = "pearson"
# ) -> pd.DataFrame:
#     """Compute correlation matrix with validation."""
#     if method == "spearman":
#         return X.corr(method="spearman")
#     return X.corr(method="pearson")


def compute_correlation_matrix(
    X: pd.DataFrame, method: str = "pearson"
) -> pd.DataFrame:
    """Compute correlation matrix with validation."""

    if method == "spearman":
        corr = X.corr(method="spearman")
    else:
        corr = X.corr(method="pearson")

    return corr if isinstance(corr, pd.DataFrame) else pd.DataFrame(corr)


def plot_correlation_heatmap(
    X: pd.DataFrame,
    features: Optional[List[str]] = None,
    title: str = "Feature Correlation Matrix",
) -> go.Figure:
    """Interactive correlation heatmap."""

    if features is not None:
        X = X[features]

    corr = compute_correlation_matrix(X)

    # Cluster if reasonable size
    if len(corr) < 50:
        # Simple clustering via linkage
        from scipy.cluster.hierarchy import linkage, leaves_list

        linkage_matrix = linkage(1 - np.abs(corr), method="average")
        order = leaves_list(linkage_matrix)
        corr = corr.iloc[order, order]

    fig = px.imshow(
        corr,
        text_auto=".2f",  # type: ignore[arg-type]
        aspect="auto",
        color_continuous_scale="RdBu_r",
        range_color=[-1, 1],
        title=title,
    )
    fig.update_layout(height=max(400, len(corr) * 20), template="plotly_dark")
    return fig


def plot_feature_target_correlation(
    X: pd.DataFrame, y: pd.Series, top_n: int = 30
) -> go.Figure:
    """Rank features by correlation with target."""

    correlations = []
    pvalues = []

    for col in X.columns:
        valid_idx = X[col].notna() & y.notna()
        if valid_idx.sum() < 10:
            corr, pval = 0, 1
        else:
            corr, pval = pearsonr(X.loc[valid_idx, col], y[valid_idx])
        correlations.append(corr)
        pvalues.append(pval)

    corr_df = pd.DataFrame(
        {
            "feature": X.columns,
            "correlation": correlations,
            "abs_corr": np.abs(correlations),
            "pvalue": pvalues,
        }
    ).sort_values("abs_corr", ascending=False)

    # Select top N
    top_df = corr_df.head(top_n)

    # Color by significance
    colors = ["green" if p < 0.05 else "gray" for p in top_df["pvalue"]]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=top_df["correlation"],
            y=top_df["feature"],
            orientation="h",
            marker_color=colors,
            text=[f"{c:.3f}" for c in top_df["correlation"]],
            textposition="outside",
        )
    )

    fig.update_layout(
        title=f"Top {top_n} Features by Target Correlation",
        xaxis_title="Pearson Correlation",
        yaxis=dict(autorange="reversed"),
        template="plotly_dark",
        height=max(400, top_n * 20),
        showlegend=False,
    )

    # Add significance annotation
    fig.add_annotation(
        x=0.95,
        y=0.02,
        xref="paper",
        yref="paper",
        text="Green: p < 0.05",
        showarrow=False,
        font=dict(color="green"),
    )

    return fig


def plot_rolling_correlation(
    X: pd.DataFrame, feature1: str, feature2: str, window: int = 60
) -> go.Figure:
    """Rolling correlation between two features or feature and target."""

    if feature2 in X.columns:
        s1, s2 = X[feature1], X[feature2]
    else:
        s1 = X[feature1]
        s2 = pd.Series(feature2, index=X.index)  # Assume feature2 is target series

    roll_corr = s1.rolling(window).corr(s2)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=X.index, y=roll_corr, name=f"Rolling Corr ({window})", line=dict(width=1)
        )
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_hline(y=0.5, line_dash="dot", line_color="red", opacity=0.5)
    fig.add_hline(y=-0.5, line_dash="dot", line_color="red", opacity=0.5)

    fig.update_layout(
        title=f"Rolling Correlation: {feature1} vs {feature2}",
        xaxis_title="Time",
        yaxis_title="Correlation",
        template="plotly_dark",
        height=400,
    )

    return fig


def plot_vif_analysis(feature_names: List[str], vif_values: List[float]) -> go.Figure:
    """Visualize Variance Inflation Factor distribution."""

    fig = go.Figure()

    # Histogram
    fig.add_trace(
        go.Histogram(x=vif_values, nbinsx=20, name="VIF Distribution", opacity=0.7)
    )

    # Threshold line
    fig.add_vline(
        x=10, line_dash="dash", line_color="red", annotation_text="VIF=10 Threshold"
    )

    # Highlight high VIF features
    high_vif = [(f, v) for f, v in zip(feature_names, vif_values) if v > 10]

    if high_vif:
        fig.add_trace(
            go.Scatter(
                x=[v for _, v in high_vif],
                y=[0] * len(high_vif),
                mode="markers+text",
                name="High VIF Features",
                text=[f for f, _ in high_vif],
                textposition="top center",
                marker=dict(color="red", size=10),
            )
        )

    fig.update_layout(
        title="Variance Inflation Factor Analysis",
        xaxis_title="VIF Value",
        yaxis_title="Count",
        template="plotly_dark",
        height=400,
    )

    return fig


def plot_feature_selection_funnel(stage_counts: dict) -> go.Figure:
    """Funnel chart showing feature survival through selection stages."""

    stages = list(stage_counts.keys())
    counts = list(stage_counts.values())

    fig = go.Figure(
        go.Funnel(
            y=stages,
            x=counts,
            textposition="inside",
            textinfo="value+percent initial",
            marker=dict(color=["blue", "cyan", "green", "orange", "red"]),
        )
    )

    fig.update_layout(
        title="Feature Selection Pipeline Survival", template="plotly_dark", height=500
    )

    return fig


def plot_mi_ranking(mi_scores: dict, threshold: Optional[float] = None) -> go.Figure:
    """Plot Mutual Information scores with cutoff."""

    features = list(mi_scores.keys())
    scores = list(mi_scores.values())

    # Sort by score
    sorted_pairs = sorted(zip(features, scores), key=lambda x: x[1])
    features, scores = zip(*sorted_pairs)

    colors = ["green" if s > (threshold or 0) else "gray" for s in scores]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=scores,
            y=features,
            orientation="h",
            marker_color=colors,
            text=[f"{s:.4f}" for s in scores],
            textposition="outside",
        )
    )

    if threshold:
        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Cutoff: {threshold:.4f}",
        )

    fig.update_layout(
        title="Mutual Information Scores (Feature → Target)",
        xaxis_title="MI Score",
        yaxis=dict(autorange="reversed"),
        template="plotly_dark",
        height=max(400, len(features) * 15),
    )

    return fig
