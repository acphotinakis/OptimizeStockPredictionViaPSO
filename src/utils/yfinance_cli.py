from __future__ import annotations

import argparse
import dataclasses
import json
import os
from typing import Optional, Sequence
import pandas as pd
import torch

from ..utils.ou_runtime import configure_torch_runtime, resolve_device, set_global_seed
from ..evaluation.ou_reporting import format_metrics
from ..evaluation.ou_img import plot_forecast
from ..data.yfinance_data import DEFAULT_START, latest_yfinance_end_date, download_close_prices
from ..models.yfinance_core import LSTMHyperparams, run_baseline_lstm, run_search_model
from ..evaluation.yfinance_reporting import compare_lookbacks, run_index_suite

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Approximate IPSO-LSTM reproduction in PyTorch")
    parser.add_argument("--symbol", type=str, default="^AXJO")
    parser.add_argument("--start", type=str, default=DEFAULT_START, help="Start date for current/2026-backward data. Default is the paper start date.")
    parser.add_argument("--end", type=str, default="", help="Exclusive yfinance end date. Blank means latest available data.")
    parser.add_argument("--method", type=str, default="ipso", choices=["lstm", "pso", "ipso"])
    parser.add_argument("--run-index-suite", action="store_true", help="Run LSTM, LSTM+PSO, and LSTM+IPSO on the six paper indices.")
    parser.add_argument("--paper-window", action="store_true", help="Use the paper's 2008-07-02 to 2016-09-30 window instead of current 2026-backward data.")
    parser.add_argument("--indices", type=str, default="", help="Optional comma-separated subset, e.g. DJIA,SP500,NIFTY50.")
    parser.add_argument("--output-dir", type=str, default="results_index_suite", help="Directory for CSV/JSON report artifacts.")
    parser.add_argument("--img-dir", type=str, default="img", help="Directory for generated PNG image artifacts.")
    parser.add_argument("--save-csv", type=str, default="", help="CSV path for metrics summary. Defaults to output-dir/metrics_summary.csv.")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--run-all-lookbacks", action="store_true")
    parser.add_argument("--search-iters", type=int, default=10)
    parser.add_argument("--swarm-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--node1", type=int, default=100)
    parser.add_argument("--node2", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save-json", type=str, default="")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="PyTorch device. Use cuda for NVIDIA GPU.")
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision during training/inference. Faster on many NVIDIA GPUs, but slightly changes numeric results.")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic PyTorch settings where possible; may slow GPU training.")
    args, _ = parser.parse_known_args(argv)
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    device = resolve_device(args.device)
    set_global_seed(args.seed, deterministic=args.deterministic)
    configure_torch_runtime(device, deterministic=args.deterministic, amp=args.amp)

    if device.type == "cuda":
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Using device: {device}")

    if args.run_index_suite:
        run_index_suite(args, device=device)
        return

    if not args.end:
        args.end = latest_yfinance_end_date()

    print(f"Downloading {args.symbol} from Yahoo Finance...")
    series = download_close_prices(args.symbol, args.start, args.end)
    print(f"Retrieved {len(series)} close-price samples.")

    lookbacks = [5, 10, 15, 20, 30, 60]
    if args.run_all_lookbacks:
        if args.method == "lstm":
            rows = []
            base_hp = LSTMHyperparams(args.epochs, args.node1, args.node2, args.learning_rate, args.batch_size).clipped()
            for lb in lookbacks:
                print(f"\n=== LSTM | lookback={lb} ===")
                artifacts = run_baseline_lstm(series, lb, base_hp, seed=args.seed, verbose=args.verbose, device=device, amp=args.amp)
                row = {"lookback": lb, **artifacts.metrics, "epochs": base_hp.epochs, "node1": base_hp.node1, "node2": base_hp.node2, "lr": base_hp.learning_rate}
                rows.append(row)
                print(format_metrics(artifacts.metrics))
            df = pd.DataFrame(rows).sort_values(by="RMSE", ascending=True).reset_index(drop=True)
        else:
            df = compare_lookbacks(series, args.method, lookbacks, args.seed, args.search_iters, args.swarm_size, args.batch_size, args.verbose, device=device, amp=args.amp)
        print("\nSorted results by RMSE:")
        print(df.to_string(index=False))
        if args.save_json:
            df.to_json(args.save_json, orient="records", indent=2)
            print(f"Saved summary to {args.save_json}")
        return

    if args.method == "lstm":
        hp = LSTMHyperparams(args.epochs, args.node1, args.node2, args.learning_rate, args.batch_size).clipped()
        artifacts = run_baseline_lstm(series, args.lookback, hp, seed=args.seed, verbose=args.verbose, device=device, amp=args.amp)
    else:
        artifacts = run_search_model(series, args.lookback, args.method, args.search_iters, args.swarm_size, args.batch_size, args.seed, args.verbose, device=device, amp=args.amp)

    print("\nFinal metrics:")
    print(format_metrics(artifacts.metrics))
    print(f"Best hyperparameters: {artifacts.best_hparams}")

    if args.save_json:
        payload = {
            "symbol": args.symbol,
            "lookback": args.lookback,
            "method": args.method,
            "metrics": artifacts.metrics,
            "best_hparams": dataclasses.asdict(artifacts.best_hparams) if artifacts.best_hparams else None,
            "search_history": artifacts.search_history,
            "device": str(device),
        }
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved results to {args.save_json}")

    if args.plot:
        plot_forecast(artifacts.y_true, artifacts.y_pred, title=f"{args.method.upper()} forecast for {args.symbol} (lookback={args.lookback})")
