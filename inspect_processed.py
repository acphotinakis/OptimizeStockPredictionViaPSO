from pathlib import Path
import pandas as pd
from prettytable import PrettyTable


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


def print_dataset_summary(dfs: dict) -> None:
    table = PrettyTable()
    table.title = "DATASET SUMMARY"

    table.field_names = [
        "Ticker",
        "Rows",
        "Cols",
        "Index Name",
        "Index Type",
        "Datetime",
        "Sorted",
        "DupIdx",
        "NaNIdx",
        "Columns",
    ]

    for ticker in sorted(dfs.keys()):
        df = dfs[ticker]
        rows, cols = df.shape
        col_names = list(df.columns)

        idx_info = extract_index_info(df)

        table.add_row(
            [
                ticker,
                rows,
                cols,
                idx_info["idx_name"],
                idx_info["idx_dtype"],
                idx_info["is_datetime"],
                idx_info["is_monotonic"],
                idx_info["has_duplicates"],
                idx_info["has_nan"],
                format_columns(col_names),
            ]
        )

    print(table)


def inspect_parquet_directory(
    data_dir: str = "data/processed",
    n_head: int = 3,
    n_tail: int = 3,
) -> None:
    """
    Iterates through all parquet files in a directory and prints:
    - number of tickers (files)
    - per-ticker shape
    - columns
    - index metadata
    - head (first n rows)
    - tail (last n rows)
    - consolidated summary table
    """

    data_path = Path(data_dir)
    parquet_files = sorted(data_path.glob("*.parquet"))

    header = (
        f"\n{'=' * 80}\nPARQUET INSPECTION | files={len(parquet_files)}\n{'=' * 80}"
    )
    print(header)

    dfs = {}

    for file_path in parquet_files:
        ticker = file_path.stem
        df = pd.read_parquet(file_path)
        dfs[ticker] = df

        print(f"\n[{ticker}] shape={df.shape}")

        # columns
        cols = list(df.columns)
        print(f"Columns ({len(cols)}): {cols}")

        # index info
        idx_info = extract_index_info(df)
        print(
            f"Index -> name={idx_info['idx_name']} "
            f"dtype={idx_info['idx_dtype']} "
            f"datetime={idx_info['is_datetime']} "
            f"sorted={idx_info['is_monotonic']} "
            f"dup={idx_info['has_duplicates']} "
            f"nan={idx_info['has_nan']}"
        )

        # head / tail tables
        head = df.head(n_head)
        tail = df.tail(n_tail)

        head_table = PrettyTable()
        head_table.title = f"{ticker} | FIRST {n_head} ROWS"
        head_table.field_names = ["index"] + cols

        for idx, row in head.iterrows():
            head_table.add_row([idx] + list(row.values))

        tail_table = PrettyTable()
        tail_table.title = f"{ticker} | LAST {n_tail} ROWS"
        tail_table.field_names = ["index"] + cols

        for idx, row in tail.iterrows():
            tail_table.add_row([idx] + list(row.values))

        print(head_table)
        print()
        print(tail_table)
        print()

    # consolidated summary
    print_dataset_summary(dfs)


import pandas as pd
import matplotlib.pyplot as plt


def inspect_and_plot_datetime_index(dfs: dict) -> None:
    """
    Validates and visualizes datetime indices across multiple DataFrames.

    Checks:
    - timezone consistency
    - min/max timestamps
    - missing timestamps (gaps)
    - alignment across tickers

    Plot:
    - Overlapping time index curves (as ordinal positions)
    """

    print("\n" + "=" * 80)
    print("DATETIME INDEX INSPECTION")
    print("=" * 80)

    index_sets = {}
    global_min = None
    global_max = None

    # ---- Inspect each ticker ----
    for ticker, df in dfs.items():
        idx = df.index

        if not isinstance(idx, pd.DatetimeIndex):
            print(f"[{ticker}] ERROR: Not a DatetimeIndex")
            continue

        # timezone
        tz = idx.tz

        # bounds
        idx_min = idx.min()
        idx_max = idx.max()

        # monotonic
        is_sorted = idx.is_monotonic_increasing

        # duplicates
        has_dupes = not idx.is_unique

        # gaps (assumes regular freq)
        inferred_freq = pd.infer_freq(idx)
        if inferred_freq:
            expected = pd.date_range(
                start=idx_min, end=idx_max, freq=inferred_freq, tz=tz
            )
            missing = len(expected.difference(idx))
        else:
            missing = "unknown"

        print(
            f"[{ticker}] "
            f"tz={tz} "
            f"range=({idx_min} → {idx_max}) "
            f"sorted={is_sorted} "
            f"dupes={has_dupes} "
            f"freq={inferred_freq} "
            f"missing={missing}"
        )

        index_sets[ticker] = set(idx)

        # track global bounds
        global_min = idx_min if global_min is None else min(global_min, idx_min)
        global_max = idx_max if global_max is None else max(global_max, idx_max)

    # ---- Alignment check ----
    print("\n" + "-" * 80)
    print("INDEX ALIGNMENT CHECK")
    print("-" * 80)

    all_indices = list(index_sets.values())
    intersection = set.intersection(*all_indices)
    union = set.union(*all_indices)

    print(f"Common timestamps: {len(intersection)}")
    print(f"Total unique timestamps: {len(union)}")
    print(f"Alignment ratio: {len(intersection) / len(union):.4f}")

    # ---- Plot overlapping indices ----
    print("\n" + "-" * 80)
    print("PLOTTING INDEX ALIGNMENT")
    print("-" * 80)

    plt.figure(figsize=(12, 6))

    for ticker, df in dfs.items():
        idx = df.index

        # convert datetime to numeric for plotting
        x = idx.view("int64")  # nanoseconds since epoch
        y = [ticker] * len(x)

        plt.scatter(x, y, s=1, label=ticker)

    plt.title("Datetime Index Alignment Across Tickers")
    plt.xlabel("Time (ns since epoch)")
    plt.ylabel("Ticker")
    plt.tight_layout()
    plt.savefig("out.png")
    plt.show()


def load_parquet_files(data_dir: str = "data/processed") -> dict:
    data_path = Path(data_dir)
    dfs = {}

    for file_path in sorted(data_path.glob("*.parquet")):
        ticker = file_path.stem
        df = pd.read_parquet(file_path)
        dfs[ticker] = df

    return dfs


if __name__ == "__main__":
    inspect_parquet_directory("data/processed")
    dfs = load_parquet_files("data/processed")
    inspect_and_plot_datetime_index(dfs)
