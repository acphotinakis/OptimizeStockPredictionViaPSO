#!/usr/bin/env python3
"""
scripts/ingest_data.py

Ingest, clean, and align OHLCV data for the ticker universe.

Usage:
    python scripts/ingest_data.py --mode ingest --config config/default_config.yaml
    python scripts/ingest_data.py --mode clean
    python scripts/ingest_data.py --mode align
"""

import sys
import logging
import sys
from pathlib import Path
from typing import List


from pathlib import Path

# Resolve project root (adjust depth if needed)
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]  # adjust if structure changes

# Ensure only the project root (not file paths) is added
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


logger = logging.getLogger(__name__)


import pandas as pd
from pathlib import Path
from pathlib import Path
import pandas as pd

from src.utils.data_storage import _load_parquet_close_volume

import matplotlib

matplotlib.use("Agg")  # important for multiprocessing safety


def render_plot_job(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw_path, aligned_path, cleaned_path, plot_path, tf, title = args

    raw = _load_parquet_close_volume(raw_path, tf)
    aligned = _load_parquet_close_volume(aligned_path, tf)
    cleaned = _load_parquet_close_volume(cleaned_path, tf)

    fig, (ax_raw, ax_align, ax_clean) = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    # ------------------------------------------------------------
    # RAW
    # ------------------------------------------------------------
    ax_raw.plot(
        raw.index.to_numpy(),
        raw["close"].to_numpy(),
        label="raw",
        alpha=0.4,
        linewidth=1,
    )
    ax_raw.set_ylabel("Raw")
    ax_raw.grid(True, alpha=0.3)
    ax_raw.legend()

    # ------------------------------------------------------------
    # ALIGNED
    # ------------------------------------------------------------
    ax_align.plot(
        aligned.index.to_numpy(),
        aligned["close"].to_numpy(),
        label="aligned",
        alpha=0.6,
        linewidth=1,
    )
    ax_align.set_ylabel("Aligned")
    ax_align.grid(True, alpha=0.3)
    ax_align.legend()

    # ------------------------------------------------------------
    # CLEANED
    # ------------------------------------------------------------
    ax_clean.plot(
        cleaned.index.to_numpy(),
        cleaned["close"].to_numpy(),
        label="cleaned",
        alpha=1.0,
        linewidth=2,
    )
    ax_clean.set_ylabel("Cleaned")
    ax_clean.grid(True, alpha=0.3)
    ax_clean.legend()

    ax_raw.set_title(title)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=100)
    plt.close(fig)

    return str(plot_path)


from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def run_parallel_plotting(
    raw_dir: Path,
    aligned_dir: Path,
    cleaned_dir: Path,
    plot_root: Path,
    tickers: List[str],
    timeframes,
):
    jobs = []

    for ticker in tickers:
        ticker_plot_dir = plot_root / ticker
        ticker_plot_dir.mkdir(parents=True, exist_ok=True)

        for tf in timeframes:
            raw = raw_dir / f"{ticker}.parquet"
            aligned = aligned_dir / f"{ticker}.parquet"
            cleaned = cleaned_dir / f"{ticker}.parquet"

            if not (raw.exists() and aligned.exists() and cleaned.exists()):
                continue

            out_path = ticker_plot_dir / f"{tf}_overlay.png"

            jobs.append((raw, aligned, cleaned, out_path, tf, f"{ticker} {tf} overlay"))

    return jobs


from concurrent.futures import ProcessPoolExecutor, as_completed
import os


def run_batch(jobs):
    workers = max(1, os.cpu_count() - 1)

    print(f"Using {workers} processes for plotting {len(jobs)} files")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(render_plot_job, job) for job in jobs]

        for i, f in enumerate(as_completed(futures)):
            try:
                result = f.result()
                if i % 20 == 0:
                    print(f"Rendered: {result}")
            except Exception as e:
                print("Plot failed:", e)
