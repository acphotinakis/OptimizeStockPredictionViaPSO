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


import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import base64
import io
import logging

logger = logging.getLogger(__name__)

PLOT_DIR = Path("data/reports/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def _plot_dict_bar(data: dict, title: str, xlabel: str = "", ylabel: str = "Count") -> str:
    """
    Creates a horizontal bar plot from a dictionary and returns the base64-encoded PNG.
    """
    if not data:
        return "<p>No data to plot.</p>"

    fig, ax = plt.subplots(figsize=(6, 4))
    pd.Series(data).sort_values().plot(kind="barh", ax=ax, color="#58a6ff")
    ax.set_title(title, fontsize=10, color="#c9d1d9")
    ax.set_xlabel(xlabel, color="#c9d1d9")
    ax.set_ylabel(ylabel, color="#c9d1d9")
    ax.tick_params(colors="#c9d1d9")
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    return f'<img src="data:image/png;base64,{img_base64}" style="width:100%;max-width:600px;margin:12px 0;">'


def generate_portfolio_validation_report(report_json_path: Path, output_path: Path):
    """
    Generates a single HTML validation report for all tickers with a clickable ticker list.
    """
    with open(report_json_path, "r") as f:
        reports = json.load(f)

    if not isinstance(reports, list):
        raise ValueError("Expected a list of reports in JSON file.")

    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <title>Portfolio Data Validation Report</title>
    <style>
        body { background:#0d1117; color:#c9d1d9; font-family:monospace; padding:24px; }
        h1,h2,h3 { color:#58a6ff; }
        table { border-collapse:collapse; width:100%; max-width:700px; margin:16px 0; }
        th,td { border:1px solid #30363d; padding:8px 12px; text-align:left; }
        th { background:#161b22; color:#8b949e; }
        tr:hover { background:#161b22; }
        .ticker-list { margin-bottom:24px; }
        .ticker-link { margin-right:12px; color:#58a6ff; cursor:pointer; text-decoration:underline; }
        .ticker-section { margin-bottom:48px; padding-bottom:24px; border-bottom:1px solid #30363d; }
    </style>
    <script>
        function scrollToTicker(id) {
            const el = document.getElementById(id);
            if(el) el.scrollIntoView({behavior:'smooth', block:'start'});
        }
    </script>
    </head>
    <body>
    <h1>🚀 Portfolio Data Validation Report</h1>
    <div class="ticker-list"><strong>Tickers:</strong> 
    """

    # ticker navigation links
    for report in reports:
        ticker = report.get("ticker", "UNKNOWN")
        html += f'<span class="ticker-link" onclick="scrollToTicker(\'ticker_{ticker}\')">{ticker}</span>'
    html += "</div>"

    # ticker sections
    for report in reports:
        ticker = report.get("ticker", "UNKNOWN")
        html += f'<div class="ticker-section" id="ticker_{ticker}">'
        html += f'<h2>{ticker} — {"✅ Fit for Training" if report["is_fit_for_training"] else "❌ Not Fit for Training"}</h2>'

        # Failed checks
        failed = report.get("failed_reasons", [])
        if failed:
            html += "<h3>Failed Checks</h3><ul>"
            for reason in failed:
                html += f"<li style='color:#f85149'>{reason}</li>"
            html += "</ul>"

        # Timestamp integrity
        ts = report.get("timestamp_integrity", {})
        html += "<h3>Timestamp Integrity</h3><table><tr><th>Metric</th><th>Value</th></tr>"
        for k, v in ts.items():
            if k != "missing_indices":
                html += f"<tr><td>{k}</td><td>{v}</td></tr>"
        html += "</table>"

        # Null values plot
        nulls = report.get("null_values", {}).get("null_counts", {})
        html += "<h3>Null Values</h3>" + _plot_dict_bar(nulls, f"Null Values per Column ({ticker})")

        # Outliers plot
        outliers = report.get("outliers", {})
        html += "<h3>Outliers</h3>" + _plot_dict_bar(outliers, f"Outliers per Column ({ticker})")

        # Integrity issues
        integrity_issues = report.get("integrity", {}).get("issues", [])
        html += "<h3>Integrity Issues</h3>"
        if integrity_issues:
            html += "<ul>"
            for issue in integrity_issues:
                html += f"<li style='color:#f85149'>{issue}</li>"
            html += "</ul>"
        else:
            html += "<p style='color:#3fb950'>No integrity issues detected.</p>"

        html += "</div>"  # close ticker-section

    html += "</body></html>"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info(f"Portfolio HTML validation report generated -> {output_path}")
    return output_path


if __name__ == "__main__":
    generate_portfolio_validation_report(
        Path("data/reports/validation.json"), Path("data/reports/validation.html")
    )
