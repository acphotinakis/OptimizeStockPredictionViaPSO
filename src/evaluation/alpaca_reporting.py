from __future__ import annotations

import os
from typing import Dict, List, Tuple, Sequence, Optional
import argparse
import dataclasses
import json
import numpy as np
import pandas as pd
import torch

from ..evaluation.ou_reporting import format_metrics, safe_filename
from ..evaluation.ou_img import plot_forecast, plot_learning_curves, plot_search_convergence
from ..evaluation.alpaca_img import plot_forecast_comparison
from ..data.alpaca_config import PAPER_START, PAPER_END, TARGET_COL
from ..data.alpaca_data import latest_end_date, get_symbol_map, parse_symbol_map, download_alpaca_symbol_frame
from ..models.alpaca_core import LSTMHyperparams, ForecastArtifacts, run_baseline_lstm, run_search_model
def history_to_rows(index_name: str, symbol: str, method: str, lookback: int, history: Dict[str, List[float]]) -> List[Dict[str, object]]:
    return [{"index": index_name, "symbol": symbol, "method": method, "lookback": lookback, "epoch": i, "loss": float(loss)} for i, loss in enumerate(history.get("loss", []), start=1)]


def metrics_row(index_name: str, symbol: str, method: str, lookback: int, feature_cols: Sequence[str], art: ForecastArtifacts) -> Dict[str, object]:
    hp = art.best_hparams
    row: Dict[str, object] = {"index": index_name, "symbol": symbol, "method": method, "lookback": lookback, "features": ",".join(feature_cols), **art.metrics}
    if hp is not None:
        row.update({"epochs": hp.epochs, "node1": hp.node1, "node2": hp.node2, "learning_rate": hp.learning_rate, "batch_size": hp.batch_size})
    return row


def run_three_method_experiment(index_name: str, symbol: str, df: pd.DataFrame, feature_cols: Sequence[str], lookback: int, base_hp: LSTMHyperparams, search_iters: int, swarm_size: int, batch_size: int, seed: int, img_dir: str, verbose: int = 0, device: Optional[torch.device] = None, amp: bool = False) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    os.makedirs(img_dir, exist_ok=True)
    artifacts_by_method: Dict[str, ForecastArtifacts] = {}
    print(f"\n=== {index_name} ({symbol}) | LSTM ===")
    artifacts_by_method["lstm"] = run_baseline_lstm(df, feature_cols, lookback, base_hp, seed=seed, verbose=verbose, device=device, amp=amp)
    print(format_metrics(artifacts_by_method["lstm"].metrics))
    for method in ["pso", "ipso"]:
        print(f"\n=== {index_name} ({symbol}) | {method.upper()} ===")
        artifacts_by_method[method] = run_search_model(df, feature_cols, lookback, method, search_iters, swarm_size, batch_size, seed, verbose, device=device, amp=amp)
        print(format_metrics(artifacts_by_method[method].metrics))
        print(f"best_hp={artifacts_by_method[method].best_hparams}")

    name = safe_filename(index_name)
    plot_learning_curves({m: a.training_history for m, a in artifacts_by_method.items()}, f"Training learning curves: {index_name}", os.path.join(img_dir, f"{name}_learning_curves.png"))
    plot_search_convergence({"pso": artifacts_by_method["pso"].search_history, "ipso": artifacts_by_method["ipso"].search_history}, f"PSO/IPSO convergence: {index_name}", os.path.join(img_dir, f"{name}_search_convergence.png"))
    plot_forecast_comparison(artifacts_by_method, f"Forecast comparison: {index_name}", os.path.join(img_dir, f"{name}_forecast_comparison.png"))

    metric_rows = [metrics_row(index_name, symbol, method, lookback, feature_cols, art) for method, art in artifacts_by_method.items()]
    history_rows: List[Dict[str, object]] = []
    search_rows: List[Dict[str, object]] = []
    for method, art in artifacts_by_method.items():
        history_rows.extend(history_to_rows(index_name, symbol, method, lookback, art.training_history))
        for row in art.search_history:
            search_rows.append({"index": index_name, "symbol": symbol, "method": method, **row})
    return metric_rows, history_rows, search_rows


def save_raw_bars(df: pd.DataFrame, output_dir: str, symbol: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{safe_filename(symbol)}_preprocessed_bars.csv")
    df.to_csv(path)
    print(f"Saved preprocessed bars: {path}")


def run_index_suite(args: argparse.Namespace, cfg: AlpacaConfig, device: torch.device, feature_cols: Sequence[str]) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.img_dir, exist_ok=True)
    end = PAPER_END if args.paper_window else (args.end or latest_end_date())
    start = PAPER_START if args.paper_window else args.start
    base_hp = LSTMHyperparams(args.epochs, args.node1, args.node2, args.learning_rate, args.batch_size).clipped()
    symbols = get_symbol_map(args.indices, custom_symbols=args.custom_symbols)

    all_metric_rows: List[Dict[str, object]] = []
    all_history_rows: List[Dict[str, object]] = []
    all_search_rows: List[Dict[str, object]] = []

    print(f"Using device: {device}")
    print(f"Paper trading base URL: {cfg.base_url}")
    print(f"Market data URL: {cfg.data_url}")
    print(f"Data window: {start} to {end}")
    print(f"Timeframe/feed/adjustment: {args.timeframe}/{cfg.feed}/{cfg.adjustment}")
    print(f"Features: {list(feature_cols)} -> target: {TARGET_COL}")
    print(f"Running index proxies: {symbols}")

    for index_name, symbol in symbols.items():
        try:
            df = download_alpaca_symbol_frame(cfg, symbol, start=start, end=end, timeframe=args.timeframe, feature_cols=feature_cols)
            print(f"Retrieved/preprocessed {len(df)} bars for {index_name} ({symbol}).")
            if args.save_preprocessed:
                save_raw_bars(df, args.output_dir, symbol)
            metric_rows, history_rows, search_rows = run_three_method_experiment(
                index_name=index_name,
                symbol=symbol,
                df=df,
                feature_cols=feature_cols,
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


def run_symbol_suite(args: argparse.Namespace, cfg: AlpacaConfig, device: torch.device, feature_cols: Sequence[str]) -> None:
    """Run the original three-method reporting pipeline on user-provided traded companies."""
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.img_dir, exist_ok=True)
    end = args.end or latest_end_date()
    start = args.start
    base_hp = LSTMHyperparams(args.epochs, args.node1, args.node2, args.learning_rate, args.batch_size).clipped()
    symbols = parse_symbol_map(args.symbols)
    if not symbols:
        raise ValueError("--symbols was provided but no valid symbols were parsed.")

    all_metric_rows: List[Dict[str, object]] = []
    all_history_rows: List[Dict[str, object]] = []
    all_search_rows: List[Dict[str, object]] = []

    print(f"Using device: {device}")
    print(f"Paper trading base URL: {cfg.base_url}")
    print(f"Market data URL: {cfg.data_url}")
    print(f"Data window: {start} to {end}")
    print(f"Timeframe/feed/adjustment: {args.timeframe}/{cfg.feed}/{cfg.adjustment}")
    print(f"Features: {list(feature_cols)} -> target: {TARGET_COL}")
    print(f"Running traded company symbols: {symbols}")

    for company_name, symbol in symbols.items():
        try:
            df = download_alpaca_symbol_frame(cfg, symbol, start=start, end=end, timeframe=args.timeframe, feature_cols=feature_cols)
            print(f"Retrieved/preprocessed {len(df)} bars for {company_name} ({symbol}).")
            if args.save_preprocessed:
                save_raw_bars(df, args.output_dir, symbol)
            metric_rows, history_rows, search_rows = run_three_method_experiment(
                index_name=company_name,
                symbol=symbol,
                df=df,
                feature_cols=feature_cols,
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
            print(f"WARNING: skipped {company_name} ({symbol}) due to error: {exc}")

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


def run_single_symbol(args: argparse.Namespace, cfg: AlpacaConfig, device: torch.device, feature_cols: Sequence[str]) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.img_dir, exist_ok=True)
    end = args.end or latest_end_date()
    print(f"Downloading {args.symbol} from Alpaca Market Data: {args.start} to {end}.")
    df = download_alpaca_symbol_frame(cfg, args.symbol.upper(), start=args.start, end=end, timeframe=args.timeframe, feature_cols=feature_cols)
    print(f"Retrieved/preprocessed {len(df)} bars. Features={list(feature_cols)} target={TARGET_COL}")
    if args.save_preprocessed:
        save_raw_bars(df, args.output_dir, args.symbol.upper())

    if args.method == "lstm":
        hp = LSTMHyperparams(args.epochs, args.node1, args.node2, args.learning_rate, args.batch_size).clipped()
        artifacts = run_baseline_lstm(df, feature_cols, args.lookback, hp, seed=args.seed, verbose=args.verbose, device=device, amp=args.amp)
    else:
        artifacts = run_search_model(df, feature_cols, args.lookback, args.method, args.search_iters, args.swarm_size, args.batch_size, args.seed, args.verbose, device=device, amp=args.amp)

    print("\nFinal metrics:")
    print(format_metrics(artifacts.metrics))
    print(f"Best hyperparameters: {artifacts.best_hparams}")

    payload = {
        "symbol": args.symbol.upper(),
        "lookback": args.lookback,
        "method": args.method,
        "features": list(feature_cols),
        "target": TARGET_COL,
        "metrics": artifacts.metrics,
        "best_hparams": dataclasses.asdict(artifacts.best_hparams) if artifacts.best_hparams else None,
        "search_history": artifacts.search_history,
        "device": str(device),
    }
    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved results to {args.save_json}")

    metrics_path = args.save_csv or os.path.join(args.output_dir, f"{safe_filename(args.symbol)}_{args.method}_metrics.csv")
    pd.DataFrame([metrics_row(args.symbol.upper(), args.symbol.upper(), args.method, args.lookback, feature_cols, artifacts)]).to_csv(metrics_path, index=False)
    print(f"Saved metrics CSV: {metrics_path}")

    if args.plot:
        outpath = os.path.join(args.img_dir, f"{safe_filename(args.symbol)}_{args.method}_forecast.png")
        plot_forecast(artifacts.y_true, artifacts.y_pred, title=f"{args.method.upper()} forecast for {args.symbol.upper()} (lookback={args.lookback})", outpath=outpath, show=True)
        print(f"Saved plot: {outpath}")
