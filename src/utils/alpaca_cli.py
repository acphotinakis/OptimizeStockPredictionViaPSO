from __future__ import annotations

import argparse
from typing import Optional, Sequence
import torch

from ..utils.ou_runtime import configure_torch_runtime, resolve_device, set_global_seed
from ..data.alpaca_config import DEFAULT_START, DEFAULT_ALPACA_DATA_URL, DEFAULT_ALPACA_BASE_URL, DEFAULT_FEATURES, AlpacaConfig
from ..data.alpaca_data import parse_features
from ..evaluation.alpaca_reporting import run_index_suite, run_symbol_suite, run_single_symbol

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IPSO-LSTM forecasting using Alpaca Market Data bars")

    # Credentials / Alpaca endpoints
    parser.add_argument("--api-key", type=str, default="", help="Alpaca API key. Safer alternative: set ALPACA_API_KEY in your environment or .env file.")
    parser.add_argument("--api-secret", type=str, default="", help="Alpaca API secret. Safer alternative: set ALPACA_API_SECRET in your environment or .env file.")
    parser.add_argument("--alpaca-base-url", type=str, default=DEFAULT_ALPACA_BASE_URL, help="Paper/live trading API base URL. Default: paper trading URL.")
    parser.add_argument("--alpaca-data-url", type=str, default=DEFAULT_ALPACA_DATA_URL, help="Market data API base URL. Default: https://data.alpaca.markets/v2")
    parser.add_argument("--feed", type=str, default="iex", choices=["iex", "sip", "otc"], help="Alpaca stock data feed. IEX is commonly available on free plans; SIP may require a subscription.")
    parser.add_argument("--adjustment", type=str, default="raw", choices=["raw", "split", "dividend", "all"], help="Corporate-action adjustment mode for bars.")

    # Data selection
    parser.add_argument("--symbol", type=str, default="SPY", help="Single stock/ETF symbol for Alpaca Market Data.")
    parser.add_argument("--symbols", type=str, default="", help="Run the original three-method suite on traded companies, e.g. AAPL,GOOGL,NVDA,META. Also supports NAME:SYMBOL pairs.")
    parser.add_argument("--start", type=str, default=DEFAULT_START, help="Start date, e.g. 2008-07-02.")
    parser.add_argument("--end", type=str, default="", help="End date. Blank means today.")
    parser.add_argument("--timeframe", type=str, default="1Day", help="Alpaca timeframe, e.g. 1Day, 1Hour, 15Min.")
    parser.add_argument("--features", type=str, default=",".join(DEFAULT_FEATURES), help="Comma-separated features from o,h,l,c,v,n,vw plus return,log_return,range,body.")
    parser.add_argument("--run-index-suite", action="store_true", help="Run LSTM, PSO-LSTM, and IPSO-LSTM on the built-in index ETF proxies.")
    parser.add_argument("--paper-window", action="store_true", help="Use 2008-07-02 to 2016-09-30 window for paper-style reproducibility.")
    parser.add_argument("--indices", type=str, default="", help="Subset for built-in proxies, e.g. DJIA,SP500,NASDAQ100, or symbols like DIA,SPY.")
    parser.add_argument("--custom-symbols", type=str, default="", help="Override index suite, e.g. SP500:SPY,DJIA:DIA,MYINDEX:QQQ. For traded companies, prefer --symbols.")

    # Model/search
    parser.add_argument("--method", type=str, default="ipso", choices=["lstm", "pso", "ipso"])
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--search-iters", type=int, default=10)
    parser.add_argument("--swarm-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--node1", type=int, default=100)
    parser.add_argument("--node2", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)

    # Output / runtime
    parser.add_argument("--output-dir", type=str, default="results_alpaca_index_suite", help="Directory for CSV/JSON report artifacts.")
    parser.add_argument("--img-dir", type=str, default="img", help="Directory for generated PNG image artifacts.")
    parser.add_argument("--save-csv", type=str, default="", help="CSV path for metrics summary. Defaults to output-dir/metrics_summary.csv or symbol_method_metrics.csv.")
    parser.add_argument("--save-json", type=str, default="")
    parser.add_argument("--save-preprocessed", action="store_true", help="Save cleaned Alpaca bars used by the model.")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision.")
    parser.add_argument("--deterministic", action="store_true")
    args, _ = parser.parse_known_args(argv)
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    feature_cols = parse_features(args.features)
    cfg = AlpacaConfig.from_args(args)
    device = resolve_device(args.device)
    set_global_seed(args.seed, deterministic=args.deterministic)
    configure_torch_runtime(device, deterministic=args.deterministic, amp=args.amp)
    print(f"Using device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    if args.symbols:
        run_symbol_suite(args, cfg, device=device, feature_cols=feature_cols)
    elif args.run_index_suite:
        run_index_suite(args, cfg, device=device, feature_cols=feature_cols)
    else:
        run_single_symbol(args, cfg, device=device, feature_cols=feature_cols)
