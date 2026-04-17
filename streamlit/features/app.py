#!/usr/bin/env python3
"""Production-grade Streamlit dashboard for feature engineering diagnostics."""

import sys
from pathlib import Path

# Resolve project root (adjust depth if needed)
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]  # adjust if structure changes

# Ensure only the project root (not file paths) is added
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Debug prints (optional)
print("Current file:", CURRENT_FILE)
print("Project root:", PROJECT_ROOT)
print("sys.path updated:")
print(sys.path)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.loader import load_ticker_data, get_available_tickers
from modules.plots import (
    plot_feature_timeseries,
    plot_multi_feature_overlay,
    plot_distribution_comparison,
    plot_rolling_statistics,
    plot_scatter_hexbin,
    plot_quantile_returns,
    plot_acf_pacf,
)
from modules.correlation import (
    plot_correlation_heatmap,
    plot_feature_target_correlation,
    plot_rolling_correlation,
    plot_vif_analysis,
    plot_feature_selection_funnel,
    plot_mi_ranking,
)
from modules.signal_analysis import (
    plot_regime_map,
    plot_feature_stability,
    plot_prediction_diagnostics,
    plot_simple_backtest,
)


# Page configuration
st.set_page_config(
    page_title="Feature Engineering Diagnostics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .stAlert { margin-top: 1rem; }
    .metric-container { background-color: #1e1e1e; padding: 1rem; border-radius: 0.5rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🔍 Feature Engineering Diagnostics Dashboard")
st.markdown("Production-grade analysis suite for ML feature validation")


# Sidebar Controls
with st.sidebar:
    st.header("Configuration")

    # Data directory
    data_dir = st.text_input("Data Directory", value="data/features")

    # Ticker selection
    available_tickers = get_available_tickers(Path(data_dir))
    if not available_tickers:
        st.error("No tickers found in directory")
        st.stop()

    ticker = st.selectbox("Select Ticker", available_tickers)

    # Load data
    try:
        dataset = load_ticker_data(ticker, data_dir)
        st.success(f"Loaded {dataset.n_features} features")
    except Exception as e:
        st.error(f"Failed to load {ticker}: {e}")
        st.stop()

    # Split selection
    split = st.selectbox("Data Split", ["train", "val", "test"])
    X, y = dataset.get_split(split)

    # Feature selection
    st.subheader("Feature Selection")
    feature_categories = {
        "All": dataset.feature_names,
        "Technical": [
            f
            for f in dataset.feature_names
            if any(
                x in f.lower() for x in ["rsi", "macd", "ema", "bb_", "atr", "stoch"]
            )
        ],
        "Statistical": [
            f
            for f in dataset.feature_names
            if any(
                x in f.lower() for x in ["ret_", "skew", "kurt", "hurst", "autocorr"]
            )
        ],
        "Volume": [
            f
            for f in dataset.feature_names
            if any(x in f.lower() for x in ["volume", "vwap", "rvol"])
        ],
        "Cross-Ticker": [
            f
            for f in dataset.feature_names
            if any(x in f.lower() for x in ["spy", "peer", "beta", "corr"])
        ],
        "Lags": [f for f in dataset.feature_names if "lag" in f.lower()],
    }

    category = st.selectbox("Category", list(feature_categories.keys()))
    available_features = feature_categories[category]

    selected_feature = st.selectbox("Primary Feature", available_features)

    # Rolling window
    window = st.slider("Rolling Window", 10, 120, 60)

    # Analysis mode
    st.subheader("Analysis Mode")
    mode = st.radio(
        "Mode",
        [
            "Time Series",
            "Distribution",
            "Correlation",
            "Signal",
            "Diagnostics",
            "Selection",
        ],
    )


# Main Content Area
if mode == "Time Series":
    st.header(f"⏱️ Time Series Analysis: {selected_feature}")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Main time series plot
        fig = plot_feature_timeseries(
            X,
            selected_feature,
            y,
            dataset.ohlcv if dataset.ohlcv is not None else None,
            [window, window * 2],
        )
        # st.plotly_chart(fig, width="stretch")
        st.plotly_chart(fig, width="stretch")

        # Multi-feature overlay
        st.subheader("Multi-Feature Overlay")
        overlay_features = st.multiselect(
            "Select Features to Overlay (max 5)",
            available_features,
            default=[selected_feature] if available_features else [],
            max_selections=5,
        )
        if overlay_features:
            fig_overlay = plot_multi_feature_overlay(
                X, overlay_features, normalize=True
            )
            # st.plotly_chart(fig_overlay, width="stretch")
            st.plotly_chart(fig_overlay, width="stretch")

    with col2:
        # Feature statistics
        st.subheader("Feature Statistics")
        stats = X[selected_feature].describe()
        st.dataframe(stats, width="stretch")

        # Rolling statistics
        st.subheader("Rolling Statistics")
        fig_roll = plot_rolling_statistics(X, selected_feature, window)
        st.plotly_chart(fig_roll, width="stretch")

        # Stability check
        st.subheader("Stability Metrics")
        recent_mean = X[selected_feature].iloc[-window:].mean()
        recent_std = X[selected_feature].iloc[-window:].std()
        overall_mean = X[selected_feature].mean()

        drift = abs(recent_mean - overall_mean) / (X[selected_feature].std() + 1e-10)
        st.metric(
            "Drift Score (Z)",
            f"{drift:.3f}",
            delta="Stable" if drift < 0.5 else "UNSTABLE",
            delta_color="normal" if drift < 0.5 else "off",
        )


elif mode == "Distribution":
    st.header(f"📊 Distribution Analysis: {selected_feature}")

    col1, col2 = st.columns(2)

    with col1:
        # Train/Val/Test comparison
        st.subheader("Split Distribution Comparison")
        X_train, y_train = dataset.get_split("train")
        X_val, y_val = dataset.get_split("val")
        X_test, y_test = dataset.get_split("test")

        fig_dist = plot_distribution_comparison(
            X_train, X_val, X_test, selected_feature
        )
        st.plotly_chart(fig_dist, width="stretch")

        # Statistical tests
        from scipy import stats

        train_data = X_train[selected_feature].dropna()
        test_data = X_test[selected_feature].dropna()
        result = stats.ks_2samp(train_data.to_numpy(), test_data.to_numpy())

        ks_stat = result.statistic
        p_value = result.pvalue

        st.metric(
            "K-S Test p-value",
            f"{p_value:.4f}",
            delta="Same Dist" if p_value > 0.05 else "DIFFERENT",
            delta_color="normal" if p_value > 0.05 else "off",
        )

    with col2:
        # Feature vs Target scatter
        st.subheader(f"{selected_feature} vs Future Return")
        fig_scatter = plot_scatter_hexbin(X, y, selected_feature)
        st.plotly_chart(fig_scatter, width="stretch")

        # Quantile returns
        st.subheader("Quantile Return Profile")
        fig_quant = plot_quantile_returns(X, y, selected_feature, n_bins=10)
        st.plotly_chart(fig_quant, width="stretch")


elif mode == "Correlation":
    st.header("🔗 Correlation Structure")

    col1, col2 = st.columns([1, 2])

    with col1:
        # Feature-target correlation ranking
        st.subheader("Target Correlation Ranking")
        fig_rank = plot_feature_target_correlation(X, y, top_n=20)
        st.plotly_chart(fig_rank, width="stretch")

        # Rolling correlation
        st.subheader("Rolling Correlation")
        corr_feature = st.selectbox(
            "Correlate with",
            ["next_log_return"] + available_features[:10],
            key="rolling_corr",
        )
        y_target = y if corr_feature == "next_log_return" else X[corr_feature]
        fig_roll_corr = plot_rolling_correlation(
            X, selected_feature, corr_feature, window
        )
        st.plotly_chart(fig_roll_corr, width="stretch")

    with col2:
        # Correlation heatmap
        st.subheader("Feature Correlation Matrix")
        n_features = st.slider("Number of Features", 5, 50, 20)
        top_features = X.corrwith(y).abs().nlargest(n_features).index.tolist()
        fig_heat = plot_correlation_heatmap(X, top_features)
        st.plotly_chart(fig_heat, width="stretch")


elif mode == "Signal":
    st.header("📈 Predictive Signal Analysis")

    # Regime map
    st.subheader("Market Regime Visualization")
    method = st.selectbox("Reduction Method", ["umap", "pca"])
    color_by = st.selectbox("Color By", ["time", "return"])

    fig_regime = plot_regime_map(X, y, color_by, method)
    st.plotly_chart(fig_regime, width="stretch")

    # Lag structure
    st.subheader("Lag Structure Validation")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Target Autocorrelation")
        fig_acf = plot_acf_pacf(y, lags=40)
        st.plotly_chart(fig_acf, width="stretch")

    with col2:
        st.write(f"{selected_feature} Autocorrelation")
        fig_acf_feat = plot_acf_pacf(X[selected_feature], lags=40)
        st.plotly_chart(fig_acf_feat, width="stretch")


elif mode == "Diagnostics":
    st.header("🩺 Data Integrity & Pipeline Health")

    # NaN/Inf monitoring
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Train")
        train_stats = dataset.get_feature_stats("train")
        nan_train = train_stats["nan_pct"].sum()
        inf_train = train_stats["inf_pct"].sum()
        st.metric("Total NaN %", f"{nan_train:.4f}")
        st.metric("Total Inf %", f"{inf_train:.4f}")

    with col2:
        st.subheader("Validation")
        val_stats = dataset.get_feature_stats("val")
        nan_val = val_stats["nan_pct"].sum()
        inf_val = val_stats["inf_pct"].sum()
        st.metric("Total NaN %", f"{nan_val:.4f}")
        st.metric("Total Inf %", f"{inf_val:.4f}")

    with col3:
        st.subheader("Test")
        test_stats = dataset.get_feature_stats("test")
        nan_test = test_stats["nan_pct"].sum()
        inf_test = test_stats["inf_pct"].sum()
        st.metric("Total NaN %", f"{nan_test:.4f}")
        st.metric("Total Inf %", f"{inf_test:.4f}")

    # Split boundaries
    st.subheader("Split Boundary Check")
    fig_boundaries = go.Figure()

    for split_name, color in [("train", "blue"), ("val", "orange"), ("test", "green")]:
        X_split, _ = dataset.get_split(split_name)
        if selected_feature in X_split.columns:
            daily_mean = X_split[selected_feature].resample("D").mean()
            fig_boundaries.add_trace(
                go.Scatter(
                    x=daily_mean.index,
                    y=daily_mean,
                    name=f"{split_name} (daily mean)",
                    line=dict(width=1, color=color),
                )
            )

    # Add split boundary lines
    meta = dataset.metadata
    if "train_end" in meta and "val_start" in meta:
        fig_boundaries.add_vline(
            x=pd.to_datetime(meta["train_end"]),
            line_dash="dash",
            line_color="white",
            annotation_text="Train/Val",
        )
    if "val_end" in meta and "test_start" in meta:
        fig_boundaries.add_vline(
            x=pd.to_datetime(meta["val_end"]),
            line_dash="dash",
            line_color="white",
            annotation_text="Val/Test",
        )

    fig_boundaries.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig_boundaries, width="stretch")

    # Metadata inspection
    with st.expander("Raw Metadata"):
        st.json(meta)


elif mode == "Selection":
    st.header("🎯 Feature Selection Diagnostics")

    meta = dataset.metadata

    # Selection funnel
    if "selection_stages" in meta:
        st.subheader("Feature Selection Pipeline")
        funnel_data = meta["selection_stages"]
        fig_funnel = plot_feature_selection_funnel(funnel_data)
        st.plotly_chart(fig_funnel, width="stretch")

    # MI Ranking
    if "mi_scores" in meta and meta["mi_scores"]:
        st.subheader("Mutual Information Scores")
        threshold = meta.get("mi_threshold", None)
        fig_mi = plot_mi_ranking(meta["mi_scores"], threshold)
        st.plotly_chart(fig_mi, width="stretch")

    # VIF Analysis (if available in metadata)
    if "vif_values" in meta and meta["vif_values"]:
        st.subheader("VIF Distribution")
        fig_vif = plot_vif_analysis(meta["feature_names"], meta["vif_values"])
        st.plotly_chart(fig_vif, width="stretch")

    # Feature category breakdown
    st.subheader("Selected Feature Composition")
    categories = {
        "Technical": sum(
            1
            for f in dataset.feature_names
            if any(x in f.lower() for x in ["rsi", "macd", "ema", "bb_", "atr"])
        ),
        "Statistical": sum(
            1
            for f in dataset.feature_names
            if any(x in f.lower() for x in ["ret_", "skew", "kurt", "hurst"])
        ),
        "Volume": sum(
            1
            for f in dataset.feature_names
            if any(x in f.lower() for x in ["volume", "vwap", "rvol"])
        ),
        "Cross-Ticker": sum(
            1
            for f in dataset.feature_names
            if any(x in f.lower() for x in ["spy", "peer", "beta"])
        ),
        "Lags": sum(1 for f in dataset.feature_names if "lag" in f.lower()),
        "Other": 0,
    }
    categories["Other"] = len(dataset.feature_names) - sum(categories.values())

    fig_pie = go.Figure(
        data=[
            go.Pie(
                labels=list(categories.keys()),
                values=list(categories.values()),
                hole=0.4,
            )
        ]
    )
    fig_pie.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig_pie, width="stretch")


# Footer
st.markdown("---")
st.markdown(
    f"**Dataset**: {ticker} | **Features**: {dataset.n_features} | **Samples ({split})**: {len(X)}"
)
