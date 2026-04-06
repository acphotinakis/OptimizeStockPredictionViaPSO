import typer
from pathlib import Path
from hydra import initialize, compose
import argparse
import logging
import sys
from pathlib import Path
import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.alpaca_ingestor import AlpacaIngestor
from src.data.cleaner import DataCleaner
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config
import time


logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Command Handlers
# ══════════════════════════════════════════════════════════════════════════════


def cmd_ingest(args):
    class _Args:
        config = args.config
        tickers = args.tickers
        output = args.output
        skip_exisiting = args.skip_existing
        save_combined = args.save_combined


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hft_lstm",
        description="HFT LSTM System — S&P 500 Forecasting Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train                          # Train with SPY (default)
  python main.py train --symbol SPY --refresh  # Force re-download
  python main.py backtest --start 2022-01-01
  python main.py live --paper --dry-run
  python main.py report
  python main.py test
  python main.py status
""",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = sub.add_parser("train", help="Fetch data and train LSTM model")
    p_train.add_argument("--symbol", default="SPY")
    p_train.add_argument("--refresh", action="store_true")
    p_train.add_argument("--walk-forward", action="store_true")

    # backtest
    p_bt = sub.add_parser("backtest", help="Run historical backtest")
    p_bt.add_argument("--start", default=None)
    p_bt.add_argument("--end", default=None)

    # live
    p_live = sub.add_parser("live", help="Run one paper/live trading cycle")
    p_live.add_argument("--symbol", default="SPY")
    p_live.add_argument("--paper", action="store_true", default=True)
    p_live.add_argument("--live", action="store_true")
    p_live.add_argument("--dry-run", action="store_true")

    # report / test / health / status — no extra args
    sub.add_parser("report", help="Generate HTML performance report")
    sub.add_parser("test", help="Run unit-test suite")
    sub.add_parser("health", help="Print system health dashboard")
    sub.add_parser("status", help="Show model registry & experiment summary")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "train": cmd_train,
        "backtest": cmd_backtest,
        "live": cmd_live,
        "report": cmd_report,
        "test": cmd_test,
        "health": cmd_health,
        "status": cmd_status,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    t0 = time.perf_counter()
    handler(args)
    logger.info(f"Command '{args.command}' completed in {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
