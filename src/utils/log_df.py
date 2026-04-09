from prettytable import PrettyTable
import pandas as pd


def pretty_print_df(
    df: pd.DataFrame,
    name: str = "DataFrame",
    n_head: int = 3,
    n_tail: int = 3,
    logger=None,
):
    """
    Pretty print a DataFrame with:
      - shape
      - column names
      - dtypes
      - missing %
      - first N rows
      - last N rows
    """
    if df is None or df.empty:
        msg = f"{name}: EMPTY DataFrame"
        if logger:
            logger.info(msg)
        else:
            print(msg)
        return

    # --- Summary table ---
    summary = PrettyTable()
    summary.title = f"{name} Summary"
    summary.field_names = ["Metric", "Value"]

    summary.add_row(["Shape", df.shape])
    summary.add_row(["Rows", len(df)])
    summary.add_row(["Columns", len(df.columns)])
    summary.add_row(["Column Names", ", ".join(map(str, df.columns.tolist()))])

    # Dtypes
    dtype_str = ", ".join([f"{col}:{dtype}" for col, dtype in df.dtypes.items()])
    summary.add_row(["Dtypes", dtype_str])

    # Missing %
    missing_pct = df.isna().mean().mean() * 100
    summary.add_row(["Missing %", f"{missing_pct:.4f}%"])

    # --- Head table ---
    head_table = PrettyTable()
    head_table.title = f"{name} (First {n_head} rows)"
    head_table.field_names = ["Index"] + list(df.columns)

    for idx, row in df.head(n_head).iterrows():
        head_table.add_row([str(idx)] + list(row.values))

    # --- Tail table ---
    tail_table = PrettyTable()
    tail_table.title = f"{name} (Last {n_tail} rows)"
    tail_table.field_names = ["Index"] + list(df.columns)

    for idx, row in df.tail(n_tail).iterrows():
        tail_table.add_row([str(idx)] + list(row.values))

    # --- Output ---
    output = f"\n{summary}\n\n{head_table}\n\n{tail_table}\n"

    if logger:
        logger.info(output)
    else:
        print(output)
