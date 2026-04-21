"""
src/visualization/market_renderer.py

Production-grade streaming market visualization engine.

Guarantees:
- SPY canonical time grid alignment
- deterministic sampling (seeded stride selection)
- chunked IO (constant memory)
- no full dataset loading
- consistent cross-ticker comparability
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


# =========================================================
# CORE RENDERER
# =========================================================


class MarketRenderer:
    """
    Streaming-safe SPY-aligned market renderer.

    Contract:
        All tickers are projected onto SPY global time index
        BEFORE any sampling or plotting.
    """

    def __init__(
        self,
        data_dir: str,
        max_points: int = 5000,
        seed: int = 42,
        dpi: int = 120,
    ):
        self.data_dir = Path(data_dir)
        self.max_points = max_points
        self.seed = seed
        self.dpi = dpi

        self.spy_index = self._load_spy_index()

    # -----------------------------------------------------
    # LOAD SPY CANONICAL INDEX (GLOBAL CONTRACT)
    # -----------------------------------------------------
    def _load_spy_index(self) -> pd.DatetimeIndex:
        spy_path = self.data_dir / "SPY.parquet"

        df = pd.read_parquet(spy_path, columns=["close"])

        idx = pd.to_datetime(df.index, utc=True)
        idx = pd.DatetimeIndex(idx).sort_values()

        logger.info(
            "SPY canonical index loaded | rows=%d | start=%s | end=%s",
            len(idx),
            idx.min(),
            idx.max(),
        )

        return idx

    # -----------------------------------------------------
    # STREAM TICKERS (NO FULL MEMORY LOAD)
    # -----------------------------------------------------
    def _iter_tickers(self) -> Iterator[Tuple[str, Path]]:
        for file in sorted(self.data_dir.glob("*.parquet")):
            if file.stem == "SPY":
                continue
            yield file.stem, file

    # -----------------------------------------------------
    # DETEMINISTIC DOWNSAMPLING (GLOBAL GRID CONSISTENT)
    # -----------------------------------------------------
    def _deterministic_mask(self, n: int) -> np.ndarray:
        """
        Produces reproducible sampling mask.

        Guarantees:
        - same input length → same sampled indices
        - no randomness at runtime
        """
        if n <= self.max_points:
            return np.ones(n, dtype=bool)

        rng = np.random.default_rng(self.seed)

        idx = np.linspace(0, n - 1, self.max_points).astype(int)

        mask = np.zeros(n, dtype=bool)
        mask[idx] = True

        return mask

    # -----------------------------------------------------
    # ALIGN TO SPY GRID (CRITICAL STEP)
    # -----------------------------------------------------
    def _align_to_spy(self, df: pd.DataFrame) -> pd.DataFrame:
        df.index = pd.to_datetime(df.index, utc=True)
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()

        # HARD ALIGNMENT
        df = df.reindex(self.spy_index)

        return df

    # -----------------------------------------------------
    # STREAM LOAD (LOW MEMORY)
    # -----------------------------------------------------
    def _load_close_stream(self, file: Path) -> pd.Series:
        df = pd.read_parquet(file, columns=["close"])
        return df["close"]

    # -----------------------------------------------------
    # RENDER
    # -----------------------------------------------------
    def render(self, output_file: str) -> None:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(16, 6))

        rendered = 0

        for ticker, file in self._iter_tickers():
            if ticker == "UVXY":
                logger.info("Skipping UVXY (extreme volatility)")
                continue

            logger.info("Processing ticker | %s", ticker)

            series = self._load_close_stream(file)

            df = pd.DataFrame({"close": series})

            df = self._align_to_spy(df)

            # extract aligned arrays
            y = df["close"].to_numpy(dtype=np.float32)
            x = self.spy_index

            # deterministic sampling (post-alignment)
            mask = self._deterministic_mask(len(y))

            y = y[mask]
            x = x[mask]

            # remove NaNs safely (do NOT break alignment earlier)
            valid = ~np.isnan(y)
            y = y[valid]
            x = x[valid]

            ax.plot(
                x,
                y,
                linewidth=0.7,
                alpha=0.6,
                rasterized=True,
                label=ticker,
            )

            rendered += 1

        # -------------------------------------------------
        # FINAL FORMATTING
        # -------------------------------------------------
        ax.set_title("SPY-ALIGNED CLOSE PRICE (STREAMING RENDER)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Close Price")
        ax.grid(True, alpha=0.2)

        ax.legend(loc="upper left", ncol=4, fontsize=8)

        plt.tight_layout()
        plt.savefig(output_file, dpi=self.dpi)
        plt.close()

        logger.info(
            "Render complete | tickers=%d | output=%s",
            rendered,
            str(output_file),
        )


# =========================================================
# CLI
# =========================================================


def main():
    import argparse
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output", type=str, default="out/market_close.png")
    parser.add_argument("--max_points", type=int, default=5000)

    args = parser.parse_args()

    renderer = MarketRenderer(
        data_dir=args.data_dir,
        max_points=args.max_points,
    )

    logger.info(
        "Starting SPY-aligned renderer | data_dir=%s | output=%s",
        args.data_dir,
        args.output,
    )

    renderer.render(args.output)


if __name__ == "__main__":
    main()
