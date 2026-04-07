"""
plots/xgboost_plots.py

Visualization functions for XGBoost model training results.
Creates publication-quality plots for model analysis and diagnostics.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set publication-quality style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14


def load_xgboost_results(
    results_dir: Path | str,
    ticker: str,
    mode: str = "train",
    seed: int = 42,
) -> Dict[str, Any]:
    """Load all XGBoost result files for a given ticker.
    
    Args:
        results_dir: Path to results directory
        ticker: Stock ticker symbol
        mode: Training mode ('train', 'val', 'test')
        seed: Random seed used in training
        
    Returns:
        Dictionary containing:
            - history: Training/validation loss curves
            - params: Hyperparameters and metadata
            - importances: Feature importance scores
            - feature_names: List of feature names (if available)
    """
    results_dir = Path(results_dir)
    tag = f"{ticker}_{mode}_seed{seed}"
    
    # Load history
    history_path = results_dir / f"xgb_history_{tag}.json"
    with open(history_path) as f:
        history = json.load(f)
    
    # Load params
    params_path = results_dir / f"xgb_params_{tag}.json"
    with open(params_path) as f:
        params = json.load(f)
    
    # Load importances
    importances_path = results_dir / f"xgb_importances_{tag}.npy"
    importances = np.load(importances_path)
    
    # Try to load feature names from metadata
    feature_names = None
    features_dir = Path("data/features")
    metadata_path = features_dir / ticker / "metadata.pkl"
    
    if metadata_path.exists():
        try:
            with open(metadata_path, "rb") as f:
                meta = pickle.load(f)
            feature_names = meta.get("feature_names", None)
        except Exception as e:
            print(f"Warning: Could not load feature names: {e}")
    
    return {
        "history": history,
        "params": params,
        "importances": importances,
        "feature_names": feature_names,
        "ticker": ticker,
        "mode": mode,
        "seed": seed,
    }


def plot_training_curves(
    history: Dict[str, List[float]],
    params: Dict[str, Any],
    save_path: Optional[Path | str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot training and validation RMSE curves.
    
    Args:
        history: Dictionary with 'train_rmse' and 'val_rmse' lists
        params: Parameters dictionary with 'best_iteration'
        save_path: Optional path to save figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    train_rmse = history.get("train_rmse", [])
    val_rmse = history.get("val_rmse", [])
    
    epochs = range(1, len(val_rmse) + 1) if val_rmse else []
    
    # Plot curves
    if train_rmse:
        ax.plot(epochs[:len(train_rmse)], train_rmse, 
                label="Train RMSE", linewidth=2, alpha=0.8)
    
    if val_rmse:
        ax.plot(epochs, val_rmse, 
                label="Validation RMSE", linewidth=2, alpha=0.8)
    
    # Mark best iteration
    best_iter = params.get("best_iteration", 0)
    if best_iter > 0 and val_rmse and best_iter <= len(val_rmse):
        best_rmse = val_rmse[best_iter - 1]
        ax.axvline(best_iter, color='red', linestyle='--', 
                   linewidth=1.5, alpha=0.7, label=f'Best Iteration ({best_iter})')
        ax.plot(best_iter, best_rmse, 'r*', markersize=15, 
                label=f'Best Val RMSE: {best_rmse:.6f}')
    
    # Formatting
    ax.set_xlabel('Boosting Round', fontweight='bold')
    ax.set_ylabel('RMSE', fontweight='bold')
    ax.set_title(f'{params.get("ticker", "Unknown")} - XGBoost Training Curves', 
                 fontweight='bold', pad=20)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Add text box with key info
    textstr = '\n'.join([
        f'Objective: {params.get("hyperparameters", {}).get("objective", "N/A")}',
        f'Max Depth: {params.get("hyperparameters", {}).get("max_depth", "N/A")}',
        f'Learning Rate: {params.get("hyperparameters", {}).get("learning_rate", "N/A")}',
        f'Runtime: {params.get("runtime_seconds", 0):.1f}s',
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_feature_importances(
    importances: np.ndarray,
    feature_names: Optional[List[str]] = None,
    lookback: int = 30,
    top_k: int = 20,
    save_path: Optional[Path | str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot feature importance scores.
    
    For flattened features (lookback × F), aggregates across time steps
    to show per-original-feature importances.
    
    Args:
        importances: Feature importance array (length = lookback × F)
        feature_names: List of original feature names (length F)
        lookback: Lookback window size used in training
        top_k: Number of top features to display
        save_path: Optional path to save figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure object
    """
    # Aggregate importances across lookback window
    n_flat = len(importances)
    
    if feature_names:
        n_features = len(feature_names)
        expected_flat = lookback * n_features
        
        if n_flat == expected_flat:
            # Reshape and average across time steps
            imp_reshaped = importances.reshape(lookback, n_features)
            imp_aggregated = imp_reshaped.mean(axis=0)
        else:
            # Mismatch - use raw importances
            print(f"Warning: Expected {expected_flat} importances but got {n_flat}")
            imp_aggregated = importances[:n_features] if n_flat >= n_features else importances
            feature_names = feature_names[:len(imp_aggregated)]
    else:
        # No feature names - use raw importances
        imp_aggregated = importances
        feature_names = [f"Feature {i}" for i in range(len(imp_aggregated))]
    
    # Sort by importance
    sorted_idx = np.argsort(imp_aggregated)[::-1]
    top_idx = sorted_idx[:min(top_k, len(sorted_idx))]
    
    top_importances = imp_aggregated[top_idx]
    top_names = [feature_names[i] for i in top_idx]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(top_names) * 0.3)))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_names)))
    bars = ax.barh(range(len(top_names)), top_importances, color=colors)
    
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names)
    ax.set_xlabel('Importance Score', fontweight='bold')
    ax.set_title(f'Top {len(top_names)} Feature Importances', 
                 fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, top_importances)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{val:.1f}',
                ha='left', va='center', fontsize=8, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved feature importances to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_metrics_summary(
    params: Dict[str, Any],
    save_path: Optional[Path | str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot validation metrics summary as a bar chart.
    
    Args:
        params: Parameters dictionary containing 'val_metrics'
        save_path: Optional path to save figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure object
    """
    val_metrics = params.get("val_metrics", {})
    
    if not val_metrics:
        print("Warning: No validation metrics found")
        return None
    
    # Select key metrics for visualization
    metric_keys = ['rmse', 'mae', 'r2', 'directional_accuracy', 'f1_ternary', 'auc_ternary']
    metric_labels = ['RMSE', 'MAE', 'R²', 'Directional Acc.', 'F1 (Ternary)', 'AUC (Ternary)']
    
    available_metrics = []
    available_labels = []
    available_values = []
    
    for key, label in zip(metric_keys, metric_labels):
        if key in val_metrics:
            available_metrics.append(key)
            available_labels.append(label)
            available_values.append(val_metrics[key])
    
    if not available_values:
        print("Warning: No displayable metrics found")
        return None
    
    # Create figure with two subplots: one for small values, one for large
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Separate metrics by scale
    small_idx = [i for i, v in enumerate(available_values) if abs(v) < 10]
    large_idx = [i for i, v in enumerate(available_values) if abs(v) >= 10]
    
    # Plot small-scale metrics
    if small_idx:
        small_labels = [available_labels[i] for i in small_idx]
        small_values = [available_values[i] for i in small_idx]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(small_values)))
        bars1 = ax1.bar(range(len(small_values)), small_values, color=colors)
        
        ax1.set_xticks(range(len(small_values)))
        ax1.set_xticklabels(small_labels, rotation=45, ha='right')
        ax1.set_ylabel('Metric Value', fontweight='bold')
        ax1.set_title('Validation Metrics (Normal Scale)', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.axhline(0, color='black', linewidth=0.8, linestyle='-')
        
        # Add value labels
        for bar, val in zip(bars1, small_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height,
                    f'{val:.4f}',
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=9, fontweight='bold')
    else:
        ax1.text(0.5, 0.5, 'No small-scale metrics', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_xticks([])
        ax1.set_yticks([])
    
    # Plot large-scale metrics (if any)
    if large_idx:
        large_labels = [available_labels[i] for i in large_idx]
        large_values = [available_values[i] for i in large_idx]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(large_values)))
        bars2 = ax2.bar(range(len(large_values)), large_values, color=colors)
        
        ax2.set_xticks(range(len(large_values)))
        ax2.set_xticklabels(large_labels, rotation=45, ha='right')
        ax2.set_ylabel('Metric Value', fontweight='bold')
        ax2.set_title('Validation Metrics (Large Scale)', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars2, large_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height,
                    f'{val:.1f}',
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=9, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No large-scale metrics', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_xticks([])
        ax2.set_yticks([])
    
    fig.suptitle(f'{params.get("ticker", "Unknown")} - Validation Metrics Summary',
                 fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved metrics summary to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_hyperparameters(
    params: Dict[str, Any],
    save_path: Optional[Path | str] = None,
    show: bool = True,
) -> plt.Figure:
    """Visualize hyperparameters as a table.
    
    Args:
        params: Parameters dictionary containing 'hyperparameters'
        save_path: Optional path to save figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure object
    """
    hyperparams = params.get("hyperparameters", {})
    
    if not hyperparams:
        print("Warning: No hyperparameters found")
        return None
    
    # Format hyperparameters for display
    param_names = []
    param_values = []
    
    for key, value in hyperparams.items():
        param_names.append(key)
        # Format value based on type
        if isinstance(value, float):
            if value < 0.001:
                param_values.append(f"{value:.2e}")
            else:
                param_values.append(f"{value:.4f}")
        else:
            param_values.append(str(value))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(6, len(param_names) * 0.4)))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table_data = [[name, value] for name, value in zip(param_names, param_values)]
    table = ax.table(cellText=table_data,
                     colLabels=['Parameter', 'Value'],
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.5, 0.5])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(param_names) + 1):
        if i % 2 == 0:
            for j in range(2):
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax.set_title(f'{params.get("ticker", "Unknown")} - XGBoost Hyperparameters',
                 fontweight='bold', fontsize=14, pad=20)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved hyperparameters to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_all_results(
    results_dir: Path | str,
    ticker: str,
    mode: str = "train",
    seed: int = 42,
    output_dir: Optional[Path | str] = None,
    show: bool = False,
) -> Dict[str, plt.Figure]:
    """Generate all plots for a given ticker's XGBoost results.
    
    Args:
        results_dir: Path to results directory
        ticker: Stock ticker symbol
        mode: Training mode ('train', 'val', 'test')
        seed: Random seed used in training
        output_dir: Directory to save plots (default: plots/)
        show: Whether to display plots interactively
        
    Returns:
        Dictionary of figure objects keyed by plot type
    """
    # Load results
    print(f"Loading results for {ticker}...")
    results = load_xgboost_results(results_dir, ticker, mode, seed)
    
    # Setup output directory
    if output_dir is None:
        output_dir = Path("plots")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tag = f"{ticker}_{mode}_seed{seed}"
    figures = {}
    
    # 1. Training curves
    print("Generating training curves...")
    fig1 = plot_training_curves(
        results["history"],
        results["params"],
        save_path=output_dir / f"xgb_training_curves_{tag}.png",
        show=show,
    )
    figures["training_curves"] = fig1
    
    # 2. Feature importances
    print("Generating feature importances...")
    lookback = results["params"].get("hyperparameters", {}).get("lookback", 30)
    fig2 = plot_feature_importances(
        results["importances"],
        results["feature_names"],
        lookback=lookback,
        top_k=20,
        save_path=output_dir / f"xgb_feature_importances_{tag}.png",
        show=show,
    )
    figures["feature_importances"] = fig2
    
    # 3. Metrics summary
    print("Generating metrics summary...")
    fig3 = plot_metrics_summary(
        results["params"],
        save_path=output_dir / f"xgb_metrics_summary_{tag}.png",
        show=show,
    )
    figures["metrics_summary"] = fig3
    
    # 4. Hyperparameters table
    print("Generating hyperparameters table...")
    fig4 = plot_hyperparameters(
        results["params"],
        save_path=output_dir / f"xgb_hyperparameters_{tag}.png",
        show=show,
    )
    figures["hyperparameters"] = fig4
    
    print(f"\nAll plots saved to {output_dir}/")
    print(f"Generated {len(figures)} plots for {ticker}")
    
    return figures


def create_summary_dashboard(
    results_dir: Path | str,
    ticker: str,
    mode: str = "train",
    seed: int = 42,
    save_path: Optional[Path | str] = None,
    show: bool = True,
) -> plt.Figure:
    """Create a comprehensive dashboard with all key visualizations.
    
    Args:
        results_dir: Path to results directory
        ticker: Stock ticker symbol
        mode: Training mode ('train', 'val', 'test')
        seed: Random seed used in training
        save_path: Optional path to save figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure object
    """
    # Load results
    results = load_xgboost_results(results_dir, ticker, mode, seed)
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Training curves (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    train_rmse = results["history"].get("train_rmse", [])
    val_rmse = results["history"].get("val_rmse", [])
    epochs = range(1, len(val_rmse) + 1) if val_rmse else []
    
    if train_rmse:
        ax1.plot(epochs[:len(train_rmse)], train_rmse, label="Train RMSE", linewidth=2)
    if val_rmse:
        ax1.plot(epochs, val_rmse, label="Val RMSE", linewidth=2)
    
    best_iter = results["params"].get("best_iteration", 0)
    if best_iter > 0 and val_rmse and best_iter <= len(val_rmse):
        ax1.axvline(best_iter, color='red', linestyle='--', alpha=0.7)
        ax1.plot(best_iter, val_rmse[best_iter-1], 'r*', markersize=12)
    
    ax1.set_xlabel('Boosting Round')
    ax1.set_ylabel('RMSE')
    ax1.set_title('Training Curves', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Feature importances (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    lookback = results["params"].get("hyperparameters", {}).get("lookback", 30)
    importances = results["importances"]
    feature_names = results["feature_names"]
    
    if feature_names:
        n_features = len(feature_names)
        n_flat = len(importances)
        expected_flat = lookback * n_features
        
        if n_flat == expected_flat:
            imp_reshaped = importances.reshape(lookback, n_features)
            imp_aggregated = imp_reshaped.mean(axis=0)
        else:
            imp_aggregated = importances[:n_features]
    else:
        imp_aggregated = importances
        feature_names = [f"F{i}" for i in range(len(imp_aggregated))]
    
    sorted_idx = np.argsort(imp_aggregated)[::-1][:10]
    top_imp = imp_aggregated[sorted_idx]
    top_names = [feature_names[i] for i in sorted_idx]
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_names)))
    ax2.barh(range(len(top_names)), top_imp, color=colors)
    ax2.set_yticks(range(len(top_names)))
    ax2.set_yticklabels(top_names, fontsize=9)
    ax2.set_xlabel('Importance')
    ax2.set_title('Top 10 Features', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Metrics (middle row, spanning both columns)
    ax3 = fig.add_subplot(gs[1, :])
    val_metrics = results["params"].get("val_metrics", {})
    
    metric_keys = ['rmse', 'mae', 'r2', 'directional_accuracy', 'f1_ternary', 'auc_ternary']
    metric_labels = ['RMSE', 'MAE', 'R²', 'Dir. Acc.', 'F1', 'AUC']
    
    available_values = [val_metrics.get(k, 0) for k in metric_keys if k in val_metrics]
    available_labels = [l for k, l in zip(metric_keys, metric_labels) if k in val_metrics]
    
    if available_values:
        # Normalize for visualization (except MAPE which can be huge)
        display_values = []
        for k, v in zip(metric_keys, available_values):
            if k == 'mape' and abs(v) > 100:
                display_values.append(np.sign(v) * np.log10(abs(v) + 1))
            else:
                display_values.append(v)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(display_values)))
        bars = ax3.bar(range(len(display_values)), display_values, color=colors)
        
        ax3.set_xticks(range(len(display_values)))
        ax3.set_xticklabels(available_labels)
        ax3.set_ylabel('Metric Value')
        ax3.set_title('Validation Metrics', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.axhline(0, color='black', linewidth=0.8)
        
        for bar, val in zip(bars, available_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, height,
                    f'{val:.4f}' if abs(val) < 10 else f'{val:.1f}',
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=8)
    
    # 4. Hyperparameters summary (bottom left)
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.axis('off')
    
    hyperparams = results["params"].get("hyperparameters", {})
    key_params = ['objective', 'n_estimators', 'max_depth', 'learning_rate', 
                  'subsample', 'tree_method', 'lookback']
    
    text_lines = ['Key Hyperparameters:', '']
    for key in key_params:
        if key in hyperparams:
            value = hyperparams[key]
            if isinstance(value, float):
                text_lines.append(f'{key}: {value:.4f}')
            else:
                text_lines.append(f'{key}: {value}')
    
    ax4.text(0.1, 0.9, '\n'.join(text_lines), transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 5. Training info (bottom right)
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    info_lines = [
        'Training Information:',
        '',
        f'Ticker: {ticker}',
        f'Mode: {mode}',
        f'Seed: {seed}',
        f'Best Iteration: {results["params"].get("best_iteration", "N/A")}',
        f'Runtime: {results["params"].get("runtime_seconds", 0):.2f}s',
        f'Best Val RMSE: {min(val_rmse) if val_rmse else "N/A"}',
    ]
    
    ax5.text(0.1, 0.9, '\n'.join(info_lines), transform=ax5.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Overall title
    fig.suptitle(f'{ticker} - XGBoost Training Dashboard', 
                 fontweight='bold', fontsize=16)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved dashboard to {save_path}")
    
    if show:
        plt.show()
    
    return fig


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate XGBoost result plots",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ticker", required=True, help="Stock ticker symbol")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--mode", default="train", choices=["train", "val", "test"],
                       help="Training mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default="plots", help="Output directory for plots")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    parser.add_argument("--dashboard", action="store_true", 
                       help="Create single dashboard instead of individual plots")
    
    args = parser.parse_args()
    
    if args.dashboard:
        # Create single dashboard
        create_summary_dashboard(
            args.results_dir,
            args.ticker,
            args.mode,
            args.seed,
            save_path=Path(args.output_dir) / f"xgb_dashboard_{args.ticker}_{args.mode}_seed{args.seed}.png",
            show=args.show,
        )
    else:
        # Create all individual plots
        plot_all_results(
            args.results_dir,
            args.ticker,
            args.mode,
            args.seed,
            args.output_dir,
            args.show,
        )
