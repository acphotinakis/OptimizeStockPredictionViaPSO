from __future__ import annotations

import os
from typing import Dict, List, Tuple, Sequence, Optional
import argparse
import numpy as np
import pandas as pd
import torch

from ..evaluation.ou_reporting import format_metrics, safe_filename
from ..evaluation.ou_img import plot_learning_curves, plot_search_convergence
from ..evaluation.yfinance_img import plot_forecast_comparison
from ..data.yfinance_data import PAPER_INDEX_SYMBOLS, PAPER_START, PAPER_END, latest_yfinance_end_date, download_close_prices
from ..models.yfinance_core import LSTMHyperparams, ForecastArtifacts, run_baseline_lstm, run_search_model

def compare_lookbacks(
    series: pd.Series,
    method: str,
    lookbacks: Sequence[int],
    seed: int,
    search_iters: int,
    swarm_size: int,
    batch_size: int,
    verbose: int = 0,
    device: Optional[torch.device] = None,
    amp: bool = False,
) -> pd.DataFrame:
    rows = []
    for lb in lookbacks:
        print(f"\n=== {method.upper()} | lookback={lb} ===")
        artifacts = run_search_model(series, lb, method, search_iters, swarm_size, batch_size, seed, verbose, device=device, amp=amp)
        row = {"lookback": lb, **artifacts.metrics}
        if artifacts.best_hparams is not None:
            row.update({
                "epochs": artifacts.best_hparams.epochs,
                "node1": artifacts.best_hparams.node1,
                "node2": artifacts.best_hparams.node2,
                "lr": artifacts.best_hparams.learning_rate,
            })
        rows.append(row)
        print(format_metrics(artifacts.metrics))
        print(f"best_hp={artifacts.best_hparams}")
    return pd.DataFrame(rows).sort_values(by="RMSE", ascending=True).reset_index(drop=True)


def history_to_rows(symbol_name: str, method: str, lookback: int, history: Dict[str, List[float]]) -> List[Dict[str, float]]:
    return [
        {"index": symbol_name, "method": method, "lookback": lookback, "epoch": epoch_idx, "loss": float(loss)}
        for epoch_idx, loss in enumerate(history.get("loss", []), start=1)
    ]



def metrics_row(index_name: str, symbol: str, method: str, lookback: int, art: ForecastArtifacts) -> Dict[str, object]:
    hp = art.best_hparams
    row: Dict[str, object] = {"index": index_name, "symbol": symbol, "method": method, "lookback": lookback, **art.metrics}
    if hp is not None:
        row.update({"epochs": hp.epochs, "node1": hp.node1, "node2": hp.node2, "learning_rate": hp.learning_rate, "batch_size": hp.batch_size})
    return row


def run_three_method_experiment(
    index_name: str,
    symbol: str,
    series: pd.Series,
    lookback: int,
    base_hp: LSTMHyperparams,
    search_iters: int,
    swarm_size: int,
    batch_size: int,
    seed: int,
    img_dir: str,
    verbose: int = 0,
    device: Optional[torch.device] = None,
    amp: bool = False,
) -> Tuple[List[Dict[str, object]], List[Dict[str, float]], List[Dict[str, object]]]:
    os.makedirs(img_dir, exist_ok=True)
    artifacts_by_method: Dict[str, ForecastArtifacts] = {}

    print(f"\n=== {index_name} ({symbol}) | LSTM ===")
    artifacts_by_method["lstm"] = run_baseline_lstm(series, lookback, base_hp, seed=seed, verbose=verbose, device=device, amp=amp)
    print(format_metrics(artifacts_by_method["lstm"].metrics))

    for method in ["pso", "ipso"]:
        print(f"\n=== {index_name} ({symbol}) | {method.upper()} ===")
        artifacts_by_method[method] = run_search_model(series, lookback, method, search_iters, swarm_size, batch_size, seed, verbose, device=device, amp=amp)
        print(format_metrics(artifacts_by_method[method].metrics))
        print(f"best_hp={artifacts_by_method[method].best_hparams}")

    safe_name = index_name.replace(" ", "_").replace("/", "_")
    plot_learning_curves({m: a.training_history for m, a in artifacts_by_method.items()}, f"Training learning curves: {index_name}", os.path.join(img_dir, f"{safe_name}_learning_curves.png"))
    plot_search_convergence({"pso": artifacts_by_method["pso"].search_history, "ipso": artifacts_by_method["ipso"].search_history}, f"PSO/IPSO search convergence: {index_name}", os.path.join(img_dir, f"{safe_name}_search_convergence.png"))
    plot_forecast_comparison(artifacts_by_method, f"Forecast comparison: {index_name}", os.path.join(img_dir, f"{safe_name}_forecast_comparison.png"))

    metric_rows = [metrics_row(index_name, symbol, method, lookback, art) for method, art in artifacts_by_method.items()]
    history_rows: List[Dict[str, float]] = []
    search_rows: List[Dict[str, object]] = []
    for method, art in artifacts_by_method.items():
        history_rows.extend(history_to_rows(index_name, method, lookback, art.training_history))
        for row in art.search_history:
            search_rows.append({"index": index_name, "symbol": symbol, "method": method, **row})
    return metric_rows, history_rows, search_rows


def run_index_suite(args: argparse.Namespace, device: torch.device) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.img_dir, exist_ok=True)
    end = PAPER_END if args.paper_window else (args.end or latest_yfinance_end_date())
    start = PAPER_START if args.paper_window else args.start

    base_hp = LSTMHyperparams(args.epochs, args.node1, args.node2, args.learning_rate, args.batch_size).clipped()

    symbols = PAPER_INDEX_SYMBOLS.copy()
    if args.indices:
        wanted = [x.strip().upper() for x in args.indices.split(",") if x.strip()]
        symbols = {k: v for k, v in PAPER_INDEX_SYMBOLS.items() if k.upper() in wanted or v.upper() in wanted}
        if not symbols:
            raise ValueError(f"No matching indices found for --indices={args.indices!r}. Valid keys: {list(PAPER_INDEX_SYMBOLS)}")

    all_metric_rows: List[Dict[str, object]] = []
    all_history_rows: List[Dict[str, float]] = []
    all_search_rows: List[Dict[str, object]] = []

    print(f"Using device: {device}")
    print(f"Data window: {start} to {end} (yfinance end is exclusive).")
    print(f"Running indices: {symbols}")
    for index_name, symbol in symbols.items():
        try:
            series = download_close_prices(symbol, start=start, end=end)
            print(f"Retrieved {len(series)} close-price samples for {index_name} ({symbol}).")
            metric_rows, history_rows, search_rows = run_three_method_experiment(
                index_name=index_name,
                symbol=symbol,
                series=series,
                lookback=args.lookback,
                base_hp=base_hp,
                search_iters=args.search_iters,
                swarm_size=args.swarm_size,
                batch_size=args.batch_size,
                seed=args.seed,
                img_dir=args.img_dir,
                verbose=args.verbose,
                device=device,
                amp=args.amp,
            )
            all_metric_rows.extend(metric_rows)
            all_history_rows.extend(history_rows)
            all_search_rows.extend(search_rows)
        except Exception as exc:
            print(f"WARNING: skipped {index_name} ({symbol}) due to error: {exc}")

    metrics_df = pd.DataFrame(all_metric_rows)
    history_df = pd.DataFrame(all_history_rows)
    search_df = pd.DataFrame(all_search_rows)

    metrics_path = args.save_csv or os.path.join(args.output_dir, "metrics_summary.csv")
    history_path = os.path.join(args.output_dir, "training_history.csv")
    search_path = os.path.join(args.output_dir, "search_history.csv")

    metrics_df.to_csv(metrics_path, index=False)
    history_df.to_csv(history_path, index=False)
    search_df.to_csv(search_path, index=False)

    print("\nSaved reports:")
    print(f"- Metrics summary: {metrics_path}")
    print(f"- Training history: {history_path}")
    print(f"- Search history: {search_path}")
    if not metrics_df.empty:
        print("\nMetrics summary:")
        print(metrics_df.sort_values(["index", "RMSE"]).to_string(index=False))
