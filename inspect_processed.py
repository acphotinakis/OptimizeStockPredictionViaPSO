from pathlib import Path
import pandas as pd
import logging
from src.utils import logger
from src.utils.logger import setup_logger
import sys
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Resolve project root (adjust depth if needed)
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[0]  # adjust if structure changes

# Ensure only the project root (not file paths) is added
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Debug prints (optional)
print("Current file:", CURRENT_FILE)
print("Project root:", PROJECT_ROOT)
print("sys.path updated:")
print(sys.path)

logger = logging.getLogger(__name__)
setup_logger(log_file="logs/ingest_data.log", level="INFO")


def format_columns(cols, max_len=6):
    if len(cols) <= max_len:
        return ", ".join(cols)
    return ", ".join(cols[:max_len]) + f" ... (+{len(cols) - max_len})"


def extract_index_info(df: pd.DataFrame):
    idx = df.index

    return {
        "idx_name": idx.name,
        "idx_dtype": str(idx.dtype),
        "is_datetime": isinstance(idx, pd.DatetimeIndex),
        "is_monotonic": idx.is_monotonic_increasing,
        "has_duplicates": not idx.is_unique,
        "has_nan": idx.hasnans,
    }


def load_parquet_files(data_dir: str = "data/processed") -> dict:
    data_path = Path(data_dir)
    dfs = {}

    for file_path in sorted(data_path.glob("*.parquet")):
        ticker = file_path.stem
        df = pd.read_parquet(file_path)
        dfs[ticker] = df

    return dfs


# =========================================================
# SPY-ANCHORED INSPECTION (TRD-CORRECT)
# =========================================================


def inspect_and_plot_datetime_index(dfs: dict) -> None:
    """
    TRD-CORRECT:
    - SPY defines canonical time index
    - missingness is informational, not error
    - observability mask is first-class output
    """

    print("\n" + "=" * 80)
    print("SPY-ANCHORED DATASET DIAGNOSTICS (TRD MODE)")
    print("=" * 80)

    # -----------------------------
    # 1. CANONICAL INDEX (SPY)
    # -----------------------------
    if "SPY" not in dfs:
        raise ValueError("SPY must exist as canonical index anchor")

    spy_df = dfs["SPY"]

    if not isinstance(spy_df.index, pd.DatetimeIndex):
        raise ValueError("SPY index must be DatetimeIndex")

    canonical_index = spy_df.index.sort_values()

    logger.info(
        "SPY canonical index | rows=%d | start=%s | end=%s",
        len(canonical_index),
        canonical_index.min(),
        canonical_index.max(),
    )

    # -----------------------------
    # 2. OBSERVABILITY MATRIX
    # -----------------------------
    observability = {}

    alignment_report = {}

    for ticker, df in dfs.items():

        if not isinstance(df.index, pd.DatetimeIndex):
            print(f"[{ticker}] ERROR: invalid index type")
            continue

        idx = df.index

        # -------------------------------------------------
        # ALIGNMENT IS GIVEN (NOT MEASURED)
        # -------------------------------------------------
        # We DO NOT validate overlap as correctness metric.
        # We measure observability instead.

        aligned_mask = canonical_index.isin(idx)

        observed_count = aligned_mask.sum()
        missing_count = (~aligned_mask).sum()

        observed_ratio = observed_count / len(canonical_index)
        missing_ratio = missing_count / len(canonical_index)

        observability[ticker] = aligned_mask.astype(int)

        # -----------------------------
        # REPORT (TRD CORRECT VIEW)
        # -----------------------------
        print(
            f"[{ticker}] "
            f"observed={observed_ratio:.4f} | "
            f"missing={missing_ratio:.4f} | "
            f"raw_rows={len(idx)}"
        )

        # -----------------------------
        # ADDITIONAL STRUCTURE METRICS
        # -----------------------------
        alignment_report[ticker] = {
            "observed_ratio": observed_ratio,
            "missing_ratio": missing_ratio,
            "raw_rows": len(idx),
        }

    # -----------------------------
    # 3. GLOBAL OBSERVABILITY STATS
    # -----------------------------
    print("\n" + "-" * 80)
    print("OBSERVABILITY SUMMARY (MODEL RELEVANT)")
    print("-" * 80)

    ratios = [v["observed_ratio"] for v in alignment_report.values()]

    print(f"Tickers: {len(alignment_report)}")
    print(f"Mean observability: {np.mean(ratios):.4f}")
    print(f"Min observability: {np.min(ratios):.4f}")
    print(f"Max observability: {np.max(ratios):.4f}")

    # -----------------------------
    # 4. BUILD OBSERVABILITY MATRIX DF
    # -----------------------------
    obs_df = pd.DataFrame(observability, index=canonical_index)

    missing_by_time = 1 - obs_df.mean(axis=1)

    print("\n" + "-" * 80)
    print("SYSTEM-WIDE MISSINGNESS PROFILE")
    print("-" * 80)

    print(f"Global missing rate: {missing_by_time.mean():.4f}")
    print(f"Worst timestamp missingness: {missing_by_time.max():.4f}")

    # -----------------------------
    # 5. PLOT OBSERVABILITY (NOT INDEX SETS)
    # -----------------------------
    plt.figure(figsize=(12, 6))

    for ticker in obs_df.columns:
        plt.plot(
            obs_df.index.view("int64"),
            obs_df[ticker] + np.random.uniform(-0.01, 0.01, len(obs_df)),
            linewidth=0.5,
        )

    plt.title("Observability Across SPY-Aligned Time Grid")
    plt.xlabel("Time (ns since epoch)")
    plt.ylabel("Observability (1=present, 0=missing)")
    plt.tight_layout()
    plt.savefig("observability.png")
    plt.show()

    # -----------------------------
    # 6. STORE OUTPUT FOR MODEL PIPELINE
    # -----------------------------
    dfs["_observability_matrix"] = obs_df

    logger.info(
        "Observability matrix built | shape=(%d,%d)",
        obs_df.shape[0],
        obs_df.shape[1],
    )


def plot_ohlcv_overlap(
    data_dir: str = "data/cleaned",
    output_file: str = "out/ohlcv_overlap.png",
    normalize: bool = True,
    max_points: int = 5000,
) -> None:

    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    import pandas as pd

    data_path = Path(data_dir)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fields = ["open", "high", "low", "close", "volume"]
    dfs = {}

    # -----------------------------
    # LOAD + PREPROCESS ONCE
    # -----------------------------
    for file in sorted(data_path.glob("*.parquet")):
        ticker = file.stem
        df = pd.read_parquet(file)

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)

        df = df[fields]

        dfs[ticker] = df

    tickers = list(dfs.keys())

    fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True)

    # -----------------------------
    # FIELD LOOP
    # -----------------------------
    for i, field in enumerate(fields):
        ax = axes[i]

        for ticker in tickers:
            df = dfs[ticker]

            series = df[field].astype("float32")

            # -----------------------------
            # DOWN SAMPLE (KEY SPEEDUP)
            # -----------------------------
            if len(series) > max_points:
                series = series.iloc[:: len(series) // max_points]

            # -----------------------------
            # TRANSFORMS (FAST VECTOR OPS)
            # -----------------------------
            if normalize and field != "volume":
                mean = series.mean()
                std = series.std() + 1e-8
                series = (series - mean) / std

            elif normalize and field == "volume":
                series = np.log1p(series)

            ax.plot(series.index, series.values, linewidth=0.6, alpha=0.5)

        ax.set_title(field.upper())
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)  # lower DPI = faster write
    plt.close()

    logger.info(
        "Saved optimized OHLCV plot | tickers=%d | file=%s",
        len(tickers),
        str(output_path),
    )


if __name__ == "__main__":
    dfs = load_parquet_files("data/processed")
    # inspect_and_plot_datetime_index(dfs)
    plot_ohlcv_overlap(data_dir="data/processed", output_file="out/ohlcv_overlap.png")
