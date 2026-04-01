import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Matplotlib — lazy import ──────────────────────────────────────────────────
try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend (no display needed)
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False
    logger.warning("matplotlib not installed — visualisation disabled. pip install matplotlib")

PLOT_DIR = Path("data/reports/plots")
STYLE = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#c9d1d9",
    "ytick.color": "#c9d1d9",
    "grid.color": "#21262d",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.5,
    "font.family": "monospace",
}
C_ACTUAL = "#58a6ff"
C_PRED = "#f78166"
C_EQUITY = "#3fb950"
C_DD = "#f85149"
C_TRAIN = "#d2a8ff"


def _setup_ax(ax):
    ax.grid(True, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig, name: str, plot_dir: Path) -> Path:
    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / name
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=STYLE["figure.facecolor"])
    plt.close(fig)
    logger.info(f"Plot saved -> {path}")
    return path


def _require_mpl(func):
    def wrapper(*a, **kw):
        if not _MPL_AVAILABLE:
            logger.warning(f"Skipping {func.__name__}: matplotlib not available.")
            return None
        return func(*a, **kw)

    wrapper.__name__ = func.__name__
    return wrapper


# ══════════════════════════════════════════════════════════════════════════════
# Plot Functions
# ══════════════════════════════════════════════════════════════════════════════


def render_summary_statistics(stats: dict) -> str:
    """
    Generate a visually appealing transposed HTML table for summary statistics.
    Columns = features (open, high, low, close, volume, trade_count, vwap)
    Rows = metrics (count, mean, std, min, 25%, 50%, 75%, max)
    """
    if not stats:
        return "<p>No summary statistics available.</p>"

    # Define the order of columns and rows
    columns_order = ["open", "high", "low", "close", "volume", "trade_count", "vwap"]
    rows_order = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]

    # CSS styling
    style = """
    <style>
        .summary-table {
            border-collapse: collapse;
            width: 100%;
            max-width: 900px;
            margin: 12px 0;
            font-family: monospace;
        }
        .summary-table th, .summary-table td {
            border: 1px solid #30363d;
            padding: 8px 12px;
            text-align: center;
        }
        .summary-table th {
            background-color: #161b22;
            color: #58a6ff;
            font-weight: bold;
            text-transform: uppercase;
        }
        .summary-table tr:nth-child(even) {
            background-color: #0d1117;
        }
        .summary-table tr:nth-child(odd) {
            background-color: #161b22;
        }
        .summary-table tr:hover {
            background-color: #21262d;
        }
    </style>
    """

    # Build table header
    header_html = "<tr><th>Metric</th>"
    for col in columns_order:
        header_html += f"<th>{col}</th>"
    header_html += "</tr>"

    # Build table rows
    rows_html = ""
    for metric in rows_order:
        rows_html += f"<tr><td>{metric}</td>"
        for col in columns_order:
            value = stats.get(col, {}).get(metric, "—")
            # Format numeric values
            if isinstance(value, float):
                value = f"{value:,.4f}"
            rows_html += f"<td>{value}</td>"
        rows_html += "</tr>"

    table_html = (
        style
        + "<h3>Summary Statistics</h3>"
        + "<table class='summary-table'>"
        + header_html
        + rows_html
        + "</table>"
    )

    return table_html


def generate_validation_html_report(
    report_json_path: Path,
    output_path: Path = Path("data/reports/ingestion_report.html"),
) -> Path:
    """
    Generate an HTML ingestion report from ingestion_report.json.
    Includes per-ticker summary, missing values, and summary statistics.
    """

    # Load JSON data
    try:
        with open(report_json_path, "r") as f:
            report_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON report: {e}")
        return None

    # Helper to generate table row
    def _row(label, value):
        return f"<tr><td>{label}</td><td>{value}</td></tr>"

    # Start HTML (CSS braces escaped)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Alpaca Ingestion Report</title>
<style>
body {{ background:#0d1117; color:#c9d1d9; font-family:monospace; padding:24px; }}
h1,h2 {{ color:#58a6ff; }}
table {{ border-collapse:collapse; width:100%; max-width:900px; margin:16px 0; }}
th,td {{ border:1px solid #30363d; padding:8px 12px; text-align:left; }}
th {{ background:#161b22; color:#8b949e; font-size:0.85em; text-transform:uppercase; }}
tr:hover {{ background:#161b22; }}
.section {{ margin-top:32px; }}
.badge-success {{ color:#3fb950; font-weight:bold; }}
.badge-failed {{ color:#f85149; font-weight:bold; }}
.badge-skipped {{ color:#f0a500; font-weight:bold; }}
</style>
</head>
<body>
<h1>🚀 Alpaca Ingestion Report</h1>
<p>Total tickers: <strong>{len(report_data)}</strong></p>
"""

    # Iterate over each ticker entry
    for entry in report_data:
        status = entry.get("status", "unknown")
        if status == "success":
            badge = "<span class='badge-success'>✅ Success</span>"
        elif status == "skipped_existing":
            badge = "<span class='badge-skipped'>⏭ Skipped (exists)</span>"
        else:
            badge = "<span class='badge-failed'>❌ Failed</span>"

        html += f"""
<div class="section">
<h2>{entry.get('ticker','—')} {badge}</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
{_row("Rows fetched", entry.get("rows_fetched","—"))}
{_row("Chunks processed", entry.get("chunks_processed","—"))}
{_row("Start date", entry.get("start_date","—"))}
{_row("End date", entry.get("end_date","—"))}
{_row("Parquet file", entry.get("parquet_file","—"))}
{_row("Errors", entry.get("errors","—"))}
</table>
<h3>Missing Values (%)</h3>
<table>
<tr><th>Column</th><th>Count</th><th>Percent</th></tr>
"""
        missing = entry.get("missing_values", {})
        for col, mv in missing.items():
            html += f"<tr><td>{col}</td><td>{mv.get('count',0)}</td><td>{mv.get('percent',0.0):.2f}%</td></tr>"

        html += "</table>"

        stats = entry.get("summary_statistics", {})
        html += render_summary_statistics(stats)

        html += "</div>"

    html += "</body></html>"

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info(f"HTML ingestion report saved -> {output_path}")
    return output_path
