import logging
from typing import Any, Dict

import pandas as pd


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Log Messages
# ---------------------------------------------------------------------------


def _log_table_stats(df: pd.DataFrame, ticker: str):
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

    stats_summary.append({"ticker": ticker, "mean_close": mean_close, "std_close": std_close})

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


def _log_ohlcv_validation_report(report: dict, ticker: str):
    from prettytable import PrettyTable

    table = PrettyTable()

    table.field_names = [
        "Ticker",
        "Initial Rows",
        "Final Rows",
        "Dropped",
        "Invalid %",
        "Missing Cols",
    ]

    initial = report.get("initial_rows", 0)
    final = report.get("final_rows", 0)
    dropped = report.get("dropped_count", report.get("invalid_count", 0))

    invalid_pct = (dropped / initial * 100) if initial > 0 else 0.0

    missing_cols = report.get("missing_columns", [])
    missing_str = ",".join(missing_cols) if missing_cols else "None"

    table.add_row(
        [
            ticker,
            initial,
            final,
            dropped,
            f"{invalid_pct:.2f}%",
            missing_str,
        ]
    )

    logger.info("\n%s", table)

    # ------------------------------------------------------------------
    # Breakdown (if available)
    # ------------------------------------------------------------------
    breakdown = report.get("breakdown")

    if breakdown:
        breakdown_table = PrettyTable()
        breakdown_table.field_names = ["Issue", "Count"]

        for k, v in breakdown.items():
            breakdown_table.add_row([k, v])

        logger.info("\n=== OHLCV Validation Breakdown (%s) ===\n%s", ticker, breakdown_table)

    # ------------------------------------------------------------------
    # Summary stats
    # ------------------------------------------------------------------
    logger.info(
        "OHLCV Summary | %s | initial=%d final=%d dropped=%d (%.2f%%)",
        ticker,
        initial,
        final,
        dropped,
        invalid_pct,
    )


def _log_observation_gap_report(report: dict, ticker: str):
    from prettytable import PrettyTable

    gap_start = report.get("gap_start")
    gap_sizes = report.get("gap_sizes")
    expected = report.get("expected_interval")

    logger.info("=== GAP REPORT (%s) ===", ticker)

    # ------------------------------------------------------------
    # Core summary
    # ------------------------------------------------------------
    total_gaps = int(gap_start.sum()) if gap_start is not None else 0
    max_gap = int(gap_sizes.max()) if gap_sizes is not None else 0
    median_gap = float(gap_sizes.median()) if gap_sizes is not None else 0.0

    summary_table = PrettyTable()
    summary_table.field_names = ["Metric", "Value"]

    summary_table.add_row(["Expected Interval", str(expected)])
    summary_table.add_row(["Total Gap Starts", total_gaps])
    summary_table.add_row(["Max Gap Size", max_gap])
    summary_table.add_row(["Median Gap Size", f"{median_gap:.2f}"])

    logger.info("\n%s", summary_table)

    # ------------------------------------------------------------
    # Gap size distribution
    # ------------------------------------------------------------
    if gap_sizes is not None and len(gap_sizes) > 0:
        dist_table = PrettyTable()
        dist_table.field_names = ["Statistic", "Value"]

        dist_table.add_row(["Min", int(gap_sizes.min())])
        dist_table.add_row(["Median", f"{gap_sizes.median():.2f}"])
        dist_table.add_row(["Max", int(gap_sizes.max())])
        dist_table.add_row(["Mean", f"{gap_sizes.mean():.2f}"])

        logger.info("\n=== Gap Size Distribution (%s) ===\n%s", ticker, dist_table)

    # ------------------------------------------------------------
    # Sample gap locations
    # ------------------------------------------------------------
    if gap_start is not None and total_gaps > 0:
        sample_idx = list(gap_start[gap_start].index[:10])

        logger.info(
            "Gap start samples (%s) | first_10_indices=%s",
            ticker,
            sample_idx,
        )

    # ------------------------------------------------------------
    # Interpretation summary
    # ------------------------------------------------------------
    if total_gaps == 0:
        logger.info("No gaps detected (%s) | data is continuous", ticker)
    else:
        logger.info(
            "Gaps detected (%s) | %d gap segments found | max gap=%d",
            ticker,
            total_gaps,
            max_gap,
        )


def log_cleaning_report(reports: Dict[str, Any]) -> None:
    """
    Logs full MockDataCleaner report in structured PrettyTable format.
    """

    from prettytable import PrettyTable

    logger.info("========== CLEANING PIPELINE REPORT ==========")

    # ------------------------------------------------------------
    # OHLCV REPORT
    # ------------------------------------------------------------
    if "ohlcv_report" in reports:
        r = reports["ohlcv_report"]

        t = PrettyTable()
        t.field_names = ["OHLCV Metric", "Value"]

        t.add_row(["Initial Rows", r.get("initial_rows", -1)])
        t.add_row(["Final Rows", r.get("final_rows")])
        t.add_row(["Dropped Rows", r.get("dropped_count")])
        t.add_row(["Valid Rows", r.get("valid_count")])
        t.add_row(["Invalid Rows", r.get("invalid_count")])
        t.add_row(["Drop Rate", f"{r.get('drop_rate', 0):.2%}"])

        logger.info("\n[STEP 3 - OHLCV VALIDATION]\n%s", t)

        if "breakdown" in r:
            b = r["breakdown"]
            bt = PrettyTable()
            bt.field_names = ["Violation Type", "Count"]

            for k, v in b.items():
                bt.add_row([k, v])

            logger.info("\n[OHLCV BREAKDOWN]\n%s", bt)

    # ------------------------------------------------------------
    # GAP REPORT
    # ------------------------------------------------------------
    if "gap_info" in reports:
        g = reports["gap_info"]

        # --------------------------------------------------------
        # Summary table
        # --------------------------------------------------------
        gt = PrettyTable()
        gt.field_names = ["Gap Metric", "Value"]

        gt.add_row(["Expected Interval", str(g.get("expected_interval"))])
        gt.add_row(["Total Gaps", g.get("total_gaps", 0)])
        gt.add_row(["Max Gap Size", g.get("max_gap", 0)])

        logger.info("\n[STEP 4 - GAP SUMMARY]\n%s", gt)

        # --------------------------------------------------------
        # Detailed gap segments table
        # --------------------------------------------------------
        segments = g.get("gap_segments", [])

        if segments:
            dt = PrettyTable()
            dt.field_names = ["Start", "End", "Missing Observations"]
            for seg in segments[:10]:
                dt.add_row(
                    [
                        str(seg["start"]),
                        str(seg["end"]),
                        seg["missing_observations"],
                    ]
                )
            # for seg in segments:
            #     dt.add_row(
            #         [
            #             str(seg["start"]),
            #             str(seg["end"]),
            #             seg["missing_observations"],
            #         ]
            #     )

            logger.info("\n[STEP 4 - GAP SEGMENTS]\n%s", dt)

        else:
            logger.info("[STEP 4 - GAP SEGMENTS] No gaps detected")
    # ------------------------------------------------------------
    # BOUNDED FORWARD FILL REPORT
    # ------------------------------------------------------------
    if "_bounded_forward_fill_report" in reports:
        r = reports["_bounded_forward_fill_report"]

        ft = PrettyTable()
        ft.field_names = ["Column", "Filled", "Skipped"]

        for col, stats in r.get("columns", {}).items():
            ft.add_row([col, stats["filled"], stats["skipped"]])

        summary = PrettyTable()
        summary.field_names = ["Metric", "Value"]
        summary.add_row(["Total Filled", r.get("total_filled")])
        summary.add_row(["Total Skipped", r.get("total_skipped")])
        summary.add_row(["Rows Before", r.get("rows_before")])
        summary.add_row(["Rows After", r.get("rows_after")])

        logger.info("\n[STEP 5 - FORWARD FILL]\n%s\n%s", summary, ft)

    # ------------------------------------------------------------
    # LONG GAP REMOVAL REPORT
    # ------------------------------------------------------------
    if "_remove_long_gaps_report" in reports:
        r = reports["_remove_long_gaps_report"]

        t = PrettyTable()
        t.field_names = ["Metric", "Value"]

        t.add_row(["Rows Before", r.get("rows_before")])
        t.add_row(["Rows After", r.get("rows_after")])
        t.add_row(["Rows Removed", r.get("rows_removed")])
        t.add_row(["Removed Fraction", f"{r.get('removed_fraction', 0):.2%}"])
        t.add_row(["Max Gap Threshold", r.get("max_gap_fill")])

        logger.info("\n[STEP 6 - LONG GAP REMOVAL]\n%s", t)

        if r.get("removed_indices_sample"):
            logger.info(
                "Sample removed indices: %s",
                r["removed_indices_sample"],
            )

    # ------------------------------------------------------------
    # FINAL VALIDATION REPORT
    # ------------------------------------------------------------
    if "_final_validation_report" in reports:
        r = reports["_final_validation_report"]

        t = PrettyTable()
        t.field_names = ["Metric", "Value"]

        t.add_row(["Rows", r.get("rows")])
        t.add_row(["NaN Count", r.get("nan_count")])
        t.add_row(["Close Invalid", r.get("close_invalid")])
        t.add_row(["Volume Invalid", r.get("volume_invalid")])
        t.add_row(["Is Valid", r.get("is_valid")])

        logger.info("\n[STEP 7 - FINAL VALIDATION]\n%s", t)

    logger.info("========== END CLEANING REPORT ==========")


def log_all_timeframe_rows(rows: list[dict]):
    from prettytable import PrettyTable

    df = pd.DataFrame(rows).copy()

    df["rows"] = df["df"].apply(len)
    df["start"] = df["df"].apply(lambda x: x.index.min())
    df["end"] = df["df"].apply(lambda x: x.index.max())
    df["mean_close"] = df["df"].apply(lambda x: x["close"].mean())
    df["volatility"] = df["df"].apply(lambda x: x["close"].pct_change().std())
    df["missing_rate"] = df["df"].apply(lambda x: x.isna().mean().mean())

    # df = df.drop(columns=["df"]).sort_values(["ticker", "feed", "timeframe"])

    table = PrettyTable()
    table.field_names = list(df.columns)

    for _, r in df.iterrows():
        table.add_row([r[c] for c in df.columns])

    logger.info("\n========== TIMEFRAME SUMMARY ==========\n%s", table)


def log_all_cleaning_reports(rows: list[dict]):
    import pandas as pd
    from prettytable import PrettyTable

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------
    # 1. Enforce correct timeframe ordering
    # ------------------------------------------------------------
    timeframe_order = ["1Min", "5Min", "15Min", "1Hour", "1Day"]

    df["timeframe"] = pd.Categorical(df["timeframe"], categories=timeframe_order, ordered=True)

    # Sort: ticker --> timeframe --> feed
    df = df.sort_values(["ticker", "timeframe", "feed"])

    # ------------------------------------------------------------
    # 2. Create column key (ticker | timeframe | feed)
    # ------------------------------------------------------------
    df["key"] = (
        df["ticker"].astype(str)
        + " | "
        + df["timeframe"].astype(str)
        + " | "
        + df["feed"].astype(str)
    )

    # ------------------------------------------------------------
    # 3. Select value columns
    # ------------------------------------------------------------
    value_cols = [c for c in df.columns if c not in ["ticker", "feed", "timeframe", "key"]]

    # ------------------------------------------------------------
    # 4. Pivot (metrics as rows)
    # ------------------------------------------------------------
    pivot_df = df.set_index("key")[value_cols].T

    # ------------------------------------------------------------
    # 5. Enforce correct column order AFTER pivot
    # ------------------------------------------------------------
    def sort_key(col: str):
        # "AAPL | 1Min | merged"
        ticker, tf, feed = [x.strip() for x in col.split("|")]
        return (
            ticker,
            timeframe_order.index(tf) if tf in timeframe_order else 999,
            feed,
        )

    sorted_cols = sorted(pivot_df.columns, key=sort_key)
    pivot_df = pivot_df[sorted_cols]

    # ------------------------------------------------------------
    # 6. Build PrettyTable
    # ------------------------------------------------------------
    table = PrettyTable()
    table.field_names = ["metric"] + list(pivot_df.columns)

    for metric, row in pivot_df.iterrows():
        table.add_row([metric] + [row[c] for c in pivot_df.columns])

    logger.info("\n========== CLEANING REPORT (PIVOTED) ==========\n%s", table)


def flatten_cleaning_report(reports: Dict[str, Any], ticker: str, feed: str, tf: str):
    row = {}

    def add(k, v):
        row[k] = v

    # metadata (IMPORTANT for cross-timeframe table)
    add("ticker", ticker)
    add("feed", feed)
    add("timeframe", tf)

    # ------------------------------------------------------------
    # OHLCV REPORT
    # ------------------------------------------------------------
    r = reports.get("ohlcv_report", {})
    add("initial_rows", r.get("initial_rows"))
    add("final_rows", r.get("final_rows"))
    add("dropped_rows", r.get("dropped_count"))
    add("valid_rows", r.get("valid_count"))
    add("invalid_rows", r.get("invalid_count"))
    add("drop_rate", f"{r.get('drop_rate', 0):.2%}")

    # ------------------------------------------------------------
    # GAP REPORT
    # ------------------------------------------------------------
    g = reports.get("gap_info", {})
    add("expected_interval", g.get("expected_interval"))
    add("total_gaps", g.get("total_gaps"))
    add("max_gap_size", g.get("max_gap"))
    add("min_gap_size", g.get("min_gap"))

    # ------------------------------------------------------------
    # BOUNDED FORWARD FILL REPORT
    # ------------------------------------------------------------
    bff = reports.get("_bounded_forward_fill_report", {})
    add("forward_fill_total_filled", bff.get("total_filled"))
    add("forward_fill_total_skipped", bff.get("total_skipped"))
    add("forward_fill_rows_before", bff.get("rows_before"))
    add("forward_fill_rows_after", bff.get("rows_after"))
    add("forward_fill_nan_before", bff.get("nan_before"))
    add("forward_fill_nan_after", bff.get("nan_after"))

    bff_cols = bff.get("columns", {})

    for col, metrics in bff_cols.items():
        add(f"ff_{col}_filled", metrics.get("filled"))
        add(f"ff_{col}_skipped", metrics.get("skipped"))
        add(f"ff_{col}_fill_ratio", metrics.get("fill_ratio"))

    add("ff_close_filled", bff_cols.get("close", {}).get("filled"))
    add("ff_close_skipped", bff_cols.get("close", {}).get("skipped"))
    add("ff_close_fill_ratio", bff_cols.get("close", {}).get("fill_ratio"))

    # ------------------------------------------------------------
    # LONG GAP REMOVAL
    # ------------------------------------------------------------
    lg = reports.get("_remove_long_gaps_report", {})
    add("long_gap_rows_before", lg.get("rows_before"))
    add("long_gap_rows_after", lg.get("rows_after"))
    add("long_gap_removed", lg.get("rows_removed"))
    add("long_gap_removed_fraction", lg.get("removed_fraction"))
    add("max_gap_threshold", lg.get("max_gap_fill"))

    # ------------------------------------------------------------
    # FINAL VALIDATION
    # ------------------------------------------------------------
    fv = reports.get("_final_validation_report", {})
    add("final_rows", fv.get("rows"))
    add("nan_count", fv.get("nan_count"))
    add("close_invalid", fv.get("close_invalid"))
    add("volume_invalid", fv.get("volume_invalid"))
    add("is_valid", fv.get("is_valid"))

    return row
