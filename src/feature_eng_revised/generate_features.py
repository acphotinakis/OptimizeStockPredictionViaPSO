#!/usr/bin/env python3
"""Build feature matrices for each ticker.  Leakage-safe: universe selection
is done once per ticker on training data and frozen for val/test."""

import argparse
from datetime import datetime
import gc
import hashlib
import logging
import pickle
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import yaml
from numba import njit
from dataclasses import dataclass

# project_root = Path(__file__).parent.parent
# sys.path.insert(0, str(project_root))

# print(sys.path)
import sys
from pathlib import Path

# Resolve project root (adjust depth if needed)
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]  # adjust if structure changes

# Ensure only the project root (not file paths) is added
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Debug prints (optional)
print("Current file:", CURRENT_FILE)
print("Project root:", PROJECT_ROOT)
print("sys.path updated:")
print(sys.path)


logger = logging.getLogger(__name__)


_TRADING_MINUTES = 390
_TRADING_DAYS = 5


@dataclass
class FeatureState:
    close: pd.Series
    high: pd.Series
    low: pd.Series
    open: pd.Series
    volume: pd.Series

    tp: pd.Series
    tr: pd.Series
    prev_close: pd.Series


class FeatureGenerator:
    """Leakage-safe deterministic feature generator (TRD aligned)."""

    def __init__(self, df: pd.DataFrame):
        self._bind(df)
        self._fitted = False
        self._feature_names: List[str] = []

    def _bind(self, df: pd.DataFrame) -> None:
        self.state = FeatureState(
            close=df["close"],
            high=df["high"],
            low=df["low"],
            open=df["open"],
            volume=df["volume"],
            tp=(df["high"] + df["low"] + df["close"]) / 3,
            tr=pd.concat(
                [
                    (df["high"] - df["low"]),
                    (df["high"] - df["close"].shift(1)).abs(),
                    (df["low"] - df["close"].shift(1)).abs(),
                ],
                axis=1,
            ).max(axis=1),
            prev_close=df["close"].shift(1),
        )

    def generate_price_features_and_add_target_col(
        self, df: pd.DataFrame, target_col: str = "log_return", horizon: int = 1
    ) -> Tuple[pd.DataFrame, pd.Series]:

        logger.info("Stage 3.1: price features")

        required = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        out = pd.DataFrame(index=df.index)

        for c in required:
            out[c] = df[c].astype(np.float32)

        close = out["close"].astype(np.float32)

        # =========================
        # FEATURES
        # =========================
        out["daily_return"] = close.pct_change(fill_method=None)

        # =========================
        # TARGET: FORWARD RETURN (NO LEAKAGE)
        # =========================
        future_close = close.shift(-horizon)

        log_return = np.log(future_close / close)

        log_return = log_return.replace([np.inf, -np.inf], np.nan)

        out[target_col] = log_return.astype(np.float32)

        # =========================
        # REMOVE INVALID ROWS
        # =========================
        out = out.iloc[:-horizon].copy()  # remove last rows with NaN target

        self._feature_names = list(out.columns)
        self._fitted = True

        logger.info(f"Price features shape={out.shape}")

        return out, out[target_col]

    # ============================================================
    # 3.2 TREND INDICATORS
    # ============================================================
    def compute_trend_following_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        C = self.state.close

        ema12 = C.ewm(span=12, adjust=False).mean()
        ema20 = C.ewm(span=20, adjust=False).mean()
        ema25 = C.ewm(span=25, adjust=False).mean()
        ema26 = C.ewm(span=26, adjust=False).mean()
        ema60 = C.ewm(span=60, adjust=False).mean()

        sma5 = C.rolling(5, min_periods=1).mean()
        sma10 = C.rolling(10, min_periods=1).mean()
        sma20 = C.rolling(20, min_periods=1).mean()
        sma60 = C.rolling(60, min_periods=1).mean()

        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()

        out = pd.DataFrame(index=df.index)
        out["sma_5"] = sma5
        out["sma_10"] = sma10
        out["sma_20"] = sma20
        out["sma_60"] = sma60

        out["ema_12"] = ema12
        out["ema_20"] = ema20
        out["ema_25"] = ema25
        out["ema_60"] = ema60

        out["macd"] = macd
        out["macd_signal"] = macd_signal
        out["macd_hist"] = macd - macd_signal

        return out

    # ============================================================
    # 3.3 VOLATILITY
    # ============================================================
    def compute_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        C = self.state.close

        sma20 = C.rolling(20, min_periods=1).mean()
        std20 = C.rolling(20, min_periods=1).std()

        out = pd.DataFrame(index=df.index)

        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20

        out["bb_upper"] = bb_upper
        out["bb_mid"] = sma20
        out["bb_lower"] = bb_lower
        out["bb_width"] = 4 * std20 / (sma20 + 1e-10)
        out["bb_pct_b"] = (C - bb_lower) / (bb_upper - bb_lower + 1e-10)

        tr = self.state.tr

        for n in (14, 30):
            atr = tr.ewm(alpha=1 / n, adjust=False).mean()
            out[f"atr_{n}"] = atr

        return out

    # ============================================================
    # 3.4 MOMENTUM
    # ============================================================
    def compute_momentum_oscillator_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        C = self.state.close
        H = self.state.high
        L = self.state.low
        O = self.state.open
        V = self.state.volume
        tp = self.state.tp

        out = pd.DataFrame(index=df.index)

        delta = C.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        for p in (14, 30):
            avg_gain = gain.ewm(alpha=1 / p, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1 / p, adjust=False).mean()
            rs = avg_gain / (avg_loss + 1e-10)
            out[f"rsi_{p}"] = 100 - 100 / (1 + rs)

        n = 20
        tp_sma = tp.rolling(n, min_periods=n).mean()
        mean_dev = (tp - tp_sma).abs().rolling(n, min_periods=n).mean()
        out["cci_20"] = (tp - tp_sma) / (0.015 * mean_dev + 1e-10)

        out["mtm_6"] = C - C.shift(6)
        out["mtm_12"] = C - C.shift(12)

        out["roc_12"] = (C - C.shift(12)) / (C.shift(12) + 1e-10) * 100

        HH = H.rolling(10, min_periods=10).max()
        LL = L.rolling(10, min_periods=10).min()

        midpoint = (HH + LL) / 2
        range_ = HH - LL
        rel = C - midpoint

        D = rel.ewm(alpha=1 / 3).mean().ewm(alpha=1 / 3).mean()
        HLD = range_.ewm(alpha=1 / 3).mean().ewm(alpha=1 / 3).mean()

        out["smi_10_3"] = 200 * (D / (HLD + 1e-10))

        hl_range = (H - L).replace(0, np.nan)
        direction = np.where(C >= O, 1.0, -1.0)
        flow = ((C - O) / (hl_range + 1e-10)) * V * direction

        out["wvad"] = pd.Series(flow, index=df.index).cumsum()

        return out

    # ============================================================
    # 3.5 VOLUME
    # ============================================================
    def compute_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        C = self.state.close
        H = self.state.high
        L = self.state.low
        O = self.state.open
        V = self.state.volume
        tp = self.state.tp
        r = df["log_return"]

        out = pd.DataFrame(index=df.index)

        # ── Close Location Value ─────────────────────────────────────
        clv = ((C - L) - (H - C)) / (H - L + 1e-10)

        # ── Time Index (TRADING CALENDAR SAFE) ──────────────────────
        idx = df.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")

        et = idx.tz_convert("America/New_York")

        # New session detection (new trading day)
        et = idx.tz_convert("America/New_York")
        session_id = pd.Series(et.normalize(), index=df.index)

        # ── VWAP (session-reset, leakage-safe) ──────────────────────
        vwap = (
            (tp * V)
            .groupby(session_id)
            .cumsum()
            .div(V.groupby(session_id).cumsum() + 1e-10)
        )

        out["vwap"] = vwap
        out["price_to_vwap"] = C / (vwap + 1e-10)
        out["vwap_dev"] = (C - vwap) / (vwap + 1e-10)

        # ── Relative Volume ─────────────────────────────────────────
        out["rvol_20"] = V / (V.rolling(20, min_periods=1).mean() + 1e-10)
        out["rvol_60"] = V / (V.rolling(60, min_periods=1).mean() + 1e-10)

        # ── On-Balance Volume ───────────────────────────────────────
        direction = C.diff().fillna(0).apply(np.sign)
        obv = (direction * V.fillna(0)).cumsum()
        obv = pd.Series(obv, index=df.index)

        out["obv"] = obv
        out["obv_ema_20"] = obv.ewm(span=20, adjust=False).mean()
        out["obv_momentum_20"] = obv.diff(20) / (obv.abs().rolling(20).mean() + 1e-10)

        # ── Accumulation/Distribution Line ──────────────────────────
        adl = (clv.fillna(0) * V).cumsum()
        out["adl"] = adl
        out["adl_slope_10"] = adl.diff(10) / (adl.abs().rolling(10).mean() + 1e-10)

        # ── Chaikin Money Flow ───────────────────────────────────────
        out["cmf_20"] = (clv * V).rolling(20).sum() / (V.rolling(20).sum() + 1e-10)

        # ── Force Index ─────────────────────────────────────────────
        force = r * V
        out["force_1"] = force
        out["force_ema_13"] = force.ewm(span=13, adjust=False).mean()

        # ── Ease of Movement ────────────────────────────────────────
        hl_mid = (H + L) / 2
        out["eom_14"] = (
            ((hl_mid - hl_mid.shift(1)) / (V / (H - L + 1e-10) + 1e-10))
            .rolling(14, min_periods=1)
            .mean()
        )

        # ── Volume Price Trend ───────────────────────────────────────
        out["vpt"] = (r * V).cumsum()

        # ── Intraday Turnover Velocity ──────────────────────────────
        out["itv_20"] = V / (V.rolling(20, min_periods=1).sum() + 1e-10)

        # ── Bar Activity Score ───────────────────────────────────────
        out["bas"] = np.log1p(V) * (H - L) / (C.shift(1) + 1e-10)

        # ── Time-of-Day Encoding (REMOVED session_minute dependency) ─
        market_open = et.normalize() + pd.Timedelta(hours=9, minutes=30)

        sm = (et - market_open).total_seconds() / 60.0
        sm = pd.Series(sm, index=df.index)

        sm = sm.clip(lower=0, upper=_TRADING_MINUTES)

        tod = sm / _TRADING_MINUTES
        out["tod_sin"] = np.sin(2 * np.pi * tod)
        out["tod_cos"] = np.cos(2 * np.pi * tod)

        # ── Day-of-Week Encoding ─────────────────────────────────────
        dow = et.dayofweek.values.astype(float)
        out["dow_sin"] = np.sin(2 * np.pi * dow / _TRADING_DAYS)
        out["dow_cos"] = np.cos(2 * np.pi * dow / _TRADING_DAYS)

        return out

    # ============================================================
    # 3.6 STATISTICS
    # ============================================================
    def compute_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        r = df["log_return"]
        out = pd.DataFrame(index=df.index)

        for w, stats in {
            10: ["mean", "var"],
            20: ["mean", "var", "skew", "kurt"],
            60: ["mean", "var", "skew", "kurt"],
            120: ["mean"],
        }.items():
            roll = r.rolling(w, min_periods=max(1, w // 2))
            for s in stats:
                out[f"ret_{s}_{w}"] = getattr(roll, s)()

        for w in (20, 60):
            out[f"ret_autocorr_1_{w}"] = r.rolling(w, min_periods=max(1, w // 2)).apply(
                self._fast_autocorr, raw=True
            )

        for w in (20, 60):
            roll_C = self.state.close.rolling(w, min_periods=1)
            out[f"range_ratio_{w}"] = (roll_C.max() - roll_C.min()) / (
                roll_C.mean() + 1e-10
            )

        r_sq = r**2
        for w in (10, 30):
            out[f"rv_{w}"] = np.sqrt(r_sq.rolling(w, min_periods=1).sum())

        # out["hurst_exp_60"] = (
        #     r.rolling(60, min_periods=30)
        #     .apply(self._hurst_single, raw=True)
        #     .fillna(0.5)
        # )

        return out

    def compute_other_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        low14 = self.state.low.rolling(14, min_periods=1).min()
        high14 = self.state.high.rolling(14, min_periods=1).max()
        # ────────────────────────────────────────────  ─────────────────
        # STOCHASTIC
        # ─────────────────────────────────────────────────────────────
        range_ = (high14 - low14).replace(0, np.nan)
        stoch_k = 100 * (self.state.close - low14) / range_
        out["stoch_k"] = stoch_k
        out["stoch_d"] = stoch_k.rolling(3, min_periods=1).mean()

        # ─────────────────────────────────────────────────────────────
        # WILLIAMS %R
        # ─────────────────────────────────────────────────────────────
        out["williams_r"] = (
            -100 * (high14 - self.state.close) / (high14 - low14 + 1e-10)
        )

        # ─────────────────────────────────────────────────────────────
        # ROC
        # ─────────────────────────────────────────────────────────────
        out["roc_5"] = self.state.close.pct_change(5, fill_method=None) * 100
        out["roc_10"] = self.state.close.pct_change(10, fill_method=None) * 100

        # ─────────────────────────────────────────────────────────────
        # MFI (shared tp/material flow)
        # ─────────────────────────────────────────────────────────────
        tp = self.state.tp  # typical price
        mf = tp * self.state.volume  # raw money flow

        tp_prev = tp.shift(1)
        direction = np.where(tp > tp_prev, 1.0, -1.0)

        pos_mf = mf.where(direction > 0, 0.0)
        neg_mf = mf.where(direction < 0, 0.0)

        for p in (14, 30):
            pos_sum = pos_mf.rolling(p, min_periods=1).sum()
            neg_sum = neg_mf.rolling(p, min_periods=1).sum()

            money_ratio = pos_sum / (neg_sum + 1e-10)
            out[f"mfi_{p}"] = 100 - (100 / (1 + money_ratio))

        # ─────────────────────────────────────────────────────────────
        # ICHIMOKU
        # ─────────────────────────────────────────────────────────────
        tenkan = (
            self.state.high.rolling(9).max() + self.state.low.rolling(9).min()
        ) / 2
        kijun = (
            self.state.high.rolling(26).max() + self.state.low.rolling(26).min()
        ) / 2

        out["ichi_tenkan"] = tenkan
        out["ichi_kijun"] = kijun
        out["ichi_senkou_a"] = (tenkan + kijun) / 2

        # ─────────────────────────────────────────────────────────────
        # PARABOLIC SAR (unchanged, stateful)
        # ─────────────────────────────────────────────────────────────
        out["psar_value"], out["psar_signal"] = self._parabolic_sar(
            self.state.high, self.state.low
        )

        # ─────────────────────────────────────────────────────────────
        # DONCHIAN
        # ─────────────────────────────────────────────────────────────
        out["dc_upper"] = self.state.high.rolling(20, min_periods=1).max()
        out["dc_lower"] = self.state.low.rolling(20, min_periods=1).min()

        # ─────────────────────────────────────────────────────────────
        # REGRESSION SLOPE (unchanged expensive op)
        # ─────────────────────────────────────────────────────────────
        for w in (10, 30):
            norm = self.state.close.rolling(w, min_periods=w)
            out[f"lr_slope_{w}"] = (
                self.state.close.rolling(w, min_periods=w).apply(
                    lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True
                )
            ) / (norm.mean() + 1e-10)

        # ─────────────────────────────────────────────────────────────
        # Z-SCORE
        # ─────────────────────────────────────────────────────────────
        for w in (20, 60):
            mu = self.state.close.rolling(w, min_periods=1).mean()
            sd = self.state.close.rolling(w, min_periods=1).std() + 1e-10
            out[f"zscore_{w}"] = (self.state.close - mu) / sd

        # ─────────────────────────────────────────────────────────────
        # ADX / DMI (shared ATR reused)
        # ─────────────────────────────────────────────────────────────
        up_move = self.state.high.diff()
        down_move = -self.state.low.diff()

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        plus_dm = pd.Series(plus_dm, index=df.index)
        minus_dm = pd.Series(minus_dm, index=df.index)

        atr14 = self.state.tr.ewm(alpha=1 / 14, adjust=False).mean()

        plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / (atr14 + 1e-10)
        minus_di = (
            100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / (atr14 + 1e-10)
        )

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)

        out["adx_14"] = dx.ewm(alpha=1 / 14, adjust=False).mean()
        out["dmi_plus"] = plus_di
        out["dmi_minus"] = minus_di

        return out

    @staticmethod
    def _fast_autocorr(x: np.ndarray) -> float:
        return float(np.corrcoef(x[:-1], x[1:])[0, 1]) if len(x) > 2 else 0.0

    @staticmethod
    @njit
    def _hurst_single(x: np.ndarray) -> float:
        n = len(x)
        if n < 10:
            return 0.5
        mean_x = np.mean(x)
        dev = np.cumsum(x - mean_x)
        R = dev.max() - dev.min()
        S = np.std(x)
        if S < 1e-10:
            return 0.5
        val = np.log(R / (S + 1e-10)) / np.log(n)

        if val < 0.0:
            return 0.0
        elif val > 1.0:
            return 1.0
        else:
            return val

    def _parabolic_sar(
        self,
        high: pd.Series,
        low: pd.Series,
        af0: float = 0.02,
        af_max: float = 0.20,
    ) -> tuple[pd.Series, pd.Series]:
        n = len(high)
        sar = np.full(n, np.nan)
        sig = np.zeros(n)
        bull = True
        af = af0
        sar[0] = float(self.state.close.iloc[0])
        ep = float(self.state.high.iloc[0])

        for i in range(1, n):
            h, l = float(high.iloc[i]), float(low.iloc[i])
            prev = sar[i - 1]
            if bull:
                sar[i] = min(
                    prev + af * (ep - prev),
                    float(low.iloc[max(0, i - 1)]),
                    float(low.iloc[max(0, i - 2)]),
                )
                if h > ep:
                    ep, af = h, min(af + af0, af_max)
                if l < sar[i]:
                    bull, sar[i], ep, af = False, ep, l, af0
            else:
                sar[i] = max(
                    prev + af * (ep - prev),
                    float(high.iloc[max(0, i - 1)]),
                    float(high.iloc[max(0, i - 2)]),
                )
                if l < ep:
                    ep, af = l, min(af + af0, af_max)
                if h > sar[i]:
                    bull, sar[i], ep, af = True, ep, h, af0
            sig[i] = 1.0 if bull else -1.0

        return pd.Series(sar, index=high.index), pd.Series(sig, index=high.index)
