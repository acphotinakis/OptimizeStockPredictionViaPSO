"""
src/utils/cleaning_tracker.py

Dedicated tracker for data cleaning pipeline diagnostics.
Fully decoupled from experiment tracking.
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


class CleaningTracker:
    """Track data cleaning pipeline runs and diagnostics."""

    def __init__(self, db_path: Path = Path("results/database/cleaning.db")):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # -------------------------------------------------
    # DB Schema
    # -------------------------------------------------

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # ---------------------------------------------
        # Main cleaning runs
        # ---------------------------------------------
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS cleaning_runs (
                run_id TEXT PRIMARY KEY,
                ticker TEXT NOT NULL,
                timestamp TEXT NOT NULL,

                start_date TEXT,
                end_date TEXT,

                rows_before INTEGER,
                rows_after INTEGER,
                rows_removed INTEGER,
                pct_removed REAL,

                expected_bars INTEGER,
                coverage_ratio REAL
            )
        """
        )

        # ---------------------------------------------
        # Step-level audit
        # ---------------------------------------------
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS cleaning_steps (
                run_id TEXT,
                step_name TEXT,
                rows_affected INTEGER,
                FOREIGN KEY (run_id) REFERENCES cleaning_runs(run_id)
            )
        """
        )

        # ---------------------------------------------
        # Gap diagnostics
        # ---------------------------------------------
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS cleaning_gaps (
                run_id TEXT,

                total_missing INTEGER,
                num_gaps INTEGER,
                max_gap INTEGER,
                avg_gap REAL,

                short_filled INTEGER,
                long_dropped INTEGER,

                FOREIGN KEY (run_id) REFERENCES cleaning_runs(run_id)
            )
        """
        )

        # ---------------------------------------------
        # Outlier diagnostics
        # ---------------------------------------------
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS cleaning_outliers (
                run_id TEXT,

                count INTEGER,
                pct REAL,
                max_z REAL,
                mean_z REAL,

                FOREIGN KEY (run_id) REFERENCES cleaning_runs(run_id)
            )
        """
        )

        # ---------------------------------------------
        # Return distribution shifts
        # ---------------------------------------------
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS cleaning_returns (
                run_id TEXT,

                mean_before REAL,
                std_before REAL,
                skew_before REAL,
                kurt_before REAL,

                mean_after REAL,
                std_after REAL,
                skew_after REAL,
                kurt_after REAL,

                FOREIGN KEY (run_id) REFERENCES cleaning_runs(run_id)
            )
        """
        )

        # ---------------------------------------------
        # Structural metrics
        # ---------------------------------------------
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS cleaning_structure (
                run_id TEXT,

                post_long_gap_count INTEGER,
                num_sequences INTEGER,

                FOREIGN KEY (run_id) REFERENCES cleaning_runs(run_id)
            )
        """
        )

        conn.commit()
        conn.close()

    # -------------------------------------------------
    # Logging API
    # -------------------------------------------------

    def log_cleaning_run(
        self,
        ticker: str,
        report: Dict[str, Any],
    ) -> str:
        """Persist full cleaning report into database."""

        run_id = f"clean_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # ---------------------------------------------
        # Main run
        # ---------------------------------------------
        cursor.execute(
            """
            INSERT INTO cleaning_runs (
                run_id, ticker, timestamp,
                start_date, end_date,
                rows_before, rows_after, rows_removed, pct_removed,
                expected_bars, coverage_ratio
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                ticker,
                datetime.now().isoformat(),
                report["metadata"]["start_date"],
                report["metadata"]["end_date"],
                report["input_metrics"]["row_count"],
                report["output_metrics"]["row_count"],
                report["integrity"]["rows_removed_total"],
                report["integrity"]["pct_rows_removed"],
                report["density"]["expected_bars"],
                report["density"]["coverage_ratio"],
            ),
        )

        # ---------------------------------------------
        # Cleaning steps
        # ---------------------------------------------
        for step, value in report["cleaning_steps"].items():
            cursor.execute(
                """
                INSERT INTO cleaning_steps (run_id, step_name, rows_affected)
                VALUES (?, ?, ?)
                """,
                (run_id, step, int(value)),
            )

        # ---------------------------------------------
        # Gaps
        # ---------------------------------------------
        g = report["gaps"]
        cursor.execute(
            """
            INSERT INTO cleaning_gaps VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                g["total_missing"],
                g["num_gaps"],
                g["max_gap"],
                g["avg_gap"],
                g["short_filled"],
                g["long_dropped"],
            ),
        )

        # ---------------------------------------------
        # Outliers
        # ---------------------------------------------
        o = report["outliers"]
        cursor.execute(
            """
            INSERT INTO cleaning_outliers VALUES (?, ?, ?, ?, ?)
            """,
            (
                run_id,
                o["count"],
                o["count"] / report["input_metrics"]["row_count"],
                o["max_z"],
                o["mean_z"],
            ),
        )

        # ---------------------------------------------
        # Returns
        # ---------------------------------------------
        rb = report["returns"]["before"]
        ra = report["returns"]["after"]

        cursor.execute(
            """
            INSERT INTO cleaning_returns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                rb.get("mean"),
                rb.get("std"),
                rb.get("skew"),
                rb.get("kurtosis"),
                ra.get("mean"),
                ra.get("std"),
                ra.get("skew"),
                ra.get("kurtosis"),
            ),
        )

        # ---------------------------------------------
        # Structure
        # ---------------------------------------------
        structure = report.get("structure", {})

        cursor.execute(
            """
            INSERT INTO cleaning_structure VALUES (?, ?, ?)
            """,
            (
                run_id,
                int(structure.get("post_long_gap_count", 0)),
                int(structure.get("num_sequences", 0)),
            ),
        )

        conn.commit()
        conn.close()

        return run_id

    # -------------------------------------------------
    # Query Helpers (Optional but useful)
    # -------------------------------------------------

    def get_latest_run(self, ticker: str) -> Optional[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM cleaning_runs
            WHERE ticker = ?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (ticker,),
        )

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        columns = [col[0] for col in cursor.description]
        return dict(zip(columns, row))
