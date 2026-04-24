"""
src/utils/logger.py
Centralised logging setup using standard logging library.
"""

import logging
import sys
from pathlib import Path
from enum import Enum


class LogFileMode(str, Enum):
    APPEND = "a"
    OVERWRITE = "w"


def setup_logger(
    log_file: str | Path | None = None,
    level: str = "INFO",
    mode: LogFileMode = LogFileMode.APPEND,
) -> None:
    """Configure logging for the project.

    Args:
        log_file: Optional path to write logs to disk.
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_format)

    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        # file_handler = logging.FileHandler(log_file)
        file_handler = logging.FileHandler(log_file, mode=mode.value)
        file_handler.setLevel(log_level)
        file_format = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        root_logger.addHandler(file_handler)


def _log_table_stats(alpaca_ingestor, raw_dir, tickers):
    from prettytable import PrettyTable

    table = PrettyTable()
    table.field_names = [
        "Ticker",
        "Rows",
        "Start",
        "End",
        "Mean Close",
        "Std Close",
        "Mean Vol",
        "Std Vol",
        "Missing %",
    ]

    stats_summary = []

    for ticker in tickers:
        path = raw_dir / f"{ticker}.parquet"
        if not path.exists():
            logger.warning("Missing raw data for %s", ticker)
            continue

        df = alpaca_ingestor.load_bars(path)

        # Ensure datetime index
        df.index = pd.to_datetime(df.index)

        rows = len(df)
        start = df.index.min()
        end = df.index.max()

        mean_close = df["close"].mean()
        std_close = df["close"].std()

        mean_vol = df["volume"].mean()
        std_vol = df["volume"].std()

        missing = df.isna().mean().mean()

        stats_summary.append(
            {"ticker": ticker, "mean_close": mean_close, "std_close": std_close}
        )

        table.add_row(
            [
                ticker,
                rows,
                str(start),
                str(end),
                f"{mean_close:.2f}",
                f"{std_close:.2f}",
                f"{mean_vol:.2f}",
                f"{std_vol:.2f}",
                f"{missing:.4%}",
            ]
        )

    logger.info("\n%s", table)

    # ---- Cross-ticker comparison ----
    df_stats = pd.DataFrame(stats_summary)

    logger.info("\n=== Cross-Ticker Dispersion ===")
    logger.info(
        "Mean Close (min/max): %.2f / %.2f",
        df_stats["mean_close"].min(),
        df_stats["mean_close"].max(),
    )

    logger.info(
        "Std Close (min/max): %.2f / %.2f",
        df_stats["std_close"].min(),
        df_stats["std_close"].max(),
    )
