from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch import nn

DEFAULT_ALPACA_DATA_URL = "https://data.alpaca.markets/v2"
DEFAULT_ALPACA_BASE_URL = "https://paper-api.alpaca.markets/v2"
PAPER_START = "2008-07-02"
PAPER_END = "2016-10-01"
DEFAULT_START = PAPER_START
INDEX_PROXY_SYMBOLS: Dict[str, str] = {
    "DJIA": "DIA",       # SPDR Dow Jones Industrial Average ETF Trust
    "SP500": "SPY",     # SPDR S&P 500 ETF Trust
    "NASDAQ100": "QQQ", # Invesco QQQ Trust
    "RUSSELL2000": "IWM",
    "NIKKEI225": "EWJ", # iShares MSCI Japan ETF; imperfect proxy
    "HANGSENG": "EWH",  # iShares MSCI Hong Kong ETF; imperfect proxy
    "NIFTY50": "INDA",  # iShares MSCI India ETF; imperfect proxy
    "CHINA": "MCHI",    # iShares MSCI China ETF; broad China proxy
}
RAW_BAR_COLUMNS = ["o", "h", "l", "c", "v", "n", "vw"]
DEFAULT_FEATURES = ["o", "h", "l", "c", "v", "vw"]
TARGET_COL = "c"

def load_dotenv_if_available() -> None:
    """Load a .env file without making python-dotenv a hard runtime requirement."""
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    for candidate in [Path.cwd() / ".env", Path(__file__).resolve().parent / ".env"]:
        if candidate.exists():
            load_dotenv(candidate)


@dataclass
class AlpacaConfig:
    api_key: str
    api_secret: str
    data_url: str = DEFAULT_ALPACA_DATA_URL
    base_url: str = DEFAULT_ALPACA_BASE_URL
    feed: str = "iex"
    adjustment: str = "raw"

    @staticmethod
    def from_args(args: argparse.Namespace) -> "AlpacaConfig":
        load_dotenv_if_available()
        api_key = args.api_key or os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID") or ""
        api_secret = args.api_secret or os.getenv("ALPACA_API_SECRET") or os.getenv("APCA_API_SECRET_KEY") or ""
        data_url = args.alpaca_data_url or os.getenv("ALPACA_DATA_URL") or DEFAULT_ALPACA_DATA_URL
        base_url = args.alpaca_base_url or os.getenv("ALPACA_BASE_URL") or DEFAULT_ALPACA_BASE_URL
        if not api_key or not api_secret:
            raise SystemExit(
                "Missing Alpaca credentials. Provide them with --api-key/--api-secret, "
                "environment variables ALPACA_API_KEY and ALPACA_API_SECRET, or a local .env file."
            )
        return AlpacaConfig(
            api_key=api_key.strip(),
            api_secret=api_secret.strip(),
            data_url=data_url.rstrip("/"),
            base_url=base_url.rstrip("/"),
            feed=args.feed,
            adjustment=args.adjustment,
        )

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Accept": "application/json",
        }
