"""
src/utils/experiment_tracker.py

Unified experiment tracking across all model types.
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np


class ExperimentTracker:
    """Track experiments across all model types with SQLite backend."""

    def __init__(self, db_path: Path = Path("results/database/experiments.db")):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                run_id TEXT PRIMARY KEY,
                model_type TEXT NOT NULL,
                ticker TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                config_hash TEXT,
                status TEXT,
                params_json TEXT,
                metrics_json TEXT,
                artifacts_path TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics (
                run_id TEXT,
                metric_name TEXT,
                metric_value REAL,
                metric_type TEXT,
                FOREIGN KEY (run_id) REFERENCES experiments(run_id)
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS pso_iterations (
                run_id TEXT,
                iteration INTEGER,
                gbest_fitness REAL,
                swarm_diversity REAL,
                FOREIGN KEY (run_id) REFERENCES experiments(run_id)
            )
        """
        )

        conn.commit()
        conn.close()

    def start_experiment(
        self,
        model_type: str,
        ticker: str,
        params: Dict[str, Any],
        config: Optional[Dict] = None,
    ) -> str:
        """Start new experiment, return run_id."""
        run_id = f"{model_type}_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO experiments 
            (run_id, model_type, ticker, timestamp, status, params_json)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                run_id,
                model_type,
                ticker,
                datetime.now().isoformat(),
                "running",
                json.dumps(params),
            ),
        )

        conn.commit()
        conn.close()

        return run_id

    def log_metrics(
        self, run_id: str, metrics: Dict[str, float], metric_type: str = "eval"
    ):
        """Log metrics for an experiment."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for name, value in metrics.items():
            cursor.execute(
                """
                INSERT INTO metrics (run_id, metric_name, metric_value, metric_type)
                VALUES (?, ?, ?, ?)
            """,
                (run_id, name, float(value), metric_type),
            )

        # Update experiments table
        cursor.execute(
            """
            UPDATE experiments
            SET metrics_json = ?
            WHERE run_id = ?
        """,
            (json.dumps(metrics), run_id),
        )

        conn.commit()
        conn.close()

    def log_pso_iteration(
        self, run_id: str, iteration: int, gbest_fitness: float, diversity: float
    ):
        """Log PSO iteration metrics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO pso_iterations (run_id, iteration, gbest_fitness, swarm_diversity)
            VALUES (?, ?, ?, ?)
        """,
            (run_id, iteration, gbest_fitness, diversity),
        )

        conn.commit()
        conn.close()

    def finish_experiment(self, run_id: str, status: str = "completed"):
        """Mark experiment as finished."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE experiments
            SET status = ?
            WHERE run_id = ?
        """,
            (status, run_id),
        )

        conn.commit()
        conn.close()

    def get_best_run(
        self, ticker: str, model_type: str, metric: str = "sharpe"
    ) -> Optional[Dict]:
        """Get best run for a ticker and model type by metric."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT e.run_id, e.params_json, e.metrics_json, m.metric_value
            FROM experiments e
            JOIN metrics m ON e.run_id = m.run_id
            WHERE e.ticker = ? AND e.model_type = ? AND m.metric_name = ?
            ORDER BY m.metric_value DESC
            LIMIT 1
        """,
            (ticker, model_type, metric),
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "run_id": row[0],
                "params": json.loads(row[1]),
                "metrics": json.loads(row[2]),
                "best_metric_value": row[3],
            }
        return None

    def compare_models(self, ticker: str) -> Dict[str, Dict]:
        """Compare all models for a ticker."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT model_type, metrics_json
            FROM experiments
            WHERE ticker = ? AND status = 'completed'
            ORDER BY timestamp DESC
        """,
            (ticker,),
        )

        results = {}
        for row in cursor.fetchall():
            model_type = row[0]
            metrics = json.loads(row[1])
            if model_type not in results:
                results[model_type] = metrics

        conn.close()
        return results
