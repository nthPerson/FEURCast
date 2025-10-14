#!/usr/bin/env python3
"""
build_splg_features.py

Engineer a rich feature set for SPLG daily OHLCV data for use with tree-based
models (e.g., GradientBoostingRegressor). No external TA libraries required.

Input CSV must contain (names can vary by case/underscore/space; script normalizes):
    Date, Company Name, Ticker, Current Price|Current_Price, Open, Close, High, Low, Volume

Outputs a CSV with:
  - Original columns (normalized snake_case)
  - Dozens of engineered features across trend, momentum, volatility, volume, lags, calendar
  - Two target columns:
        target_close_t1  = next-day Close (regression)
        target_return_t1 = next-day % return (regression/classification cutoff)

Usage:
    python build_splg_features.py --input splg_raw.csv --output splg_features_full.csv
"""

import argparse
import sys
import numpy as np
import pandas as pd
from typing import List, Tuple
import os

# ----------------------------
# Utilities
# ----------------------------
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase column names and replace spaces/hyphens with underscores."""
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[\s\-]+", "_", regex=True)
    )
    return df

def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Coerce listed columns to numeric (errors become NaN)."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def require_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    # Classic Wilder RSI implementation (simple rolling averages)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def true_range(high: pd.Series, low: pd.Series, prev_close: pd.Series) -> pd.Series:
    a = high - low
    b = (high - prev_close).abs()
    c = (low - prev_close).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    # On-Balance Volume
    sign = np.sign(close.diff()).fillna(0.0)
    return (sign * volume).fillna(0.0).cumsum()

def volume_price_trend(close: pd.Series, volume: pd.Series) -> pd.Series:
    # VPT = cumulative sum of Volume * pct_change(Close)
    return (volume * close.pct_change(fill_method=None)).fillna(0.0).cumsum()

def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df = df.copy()
    # The date column should already be a datetime type (naive or aware)
    if not np.issubdtype(df[date_col].dtype, np.datetime64):
        # If not datetime, try to convert it
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    
    dt = df[date_col]
    df["dow"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["quarter"] = dt.dt.quarter
    df["year"] = dt.dt.year
    df["is_month_start"] = dt.dt.is_month_start.astype(int)
    df["is_month_end"] = dt.dt.is_month_end.astype(int)
    return df

def rolling_slope(x: pd.Series, window: int) -> pd.Series:
    """
    Simple rolling slope (% change) proxy for trend steepness:
    slope_t = (MA_t - MA_{t-1}) / MA_{t-1}
    """
    ma = x.rolling(window).mean()
    return (ma - ma.shift(1)) / ma.shift(1)

# ----------------------------
# Core feature engineering
# ----------------------------
def engineer_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = normalize_cols(df_raw)

    # Map possible aliases
    col_map = {
        "date": "date",
        "company_name": "company_name",
        "ticker": "ticker",
        "current_price": "current_price",  # if missing, we may alias to 'close'
        "open": "open",
        "close": "close",
        "high": "high",
        "low": "low",
        "volume": "volume",
    }

    # Accept both 'current_price' or 'current_price' derived from 'current price'
    # After normalization, 'current price' -> 'current_price' already.

    # Ensure required market columns exist
    required_any_current = ["date", "open", "close", "high", "low", "volume"]
    require_columns(df, required_any_current)

    # If current_price is missing or empty, create it as close (some vendors do this)
    if "current_price" not in df.columns:
        df["current_price"] = df["close"]

    # Parse date, sort, drop duplicates
    # The input dates have timezone info (e.g., '2005-11-15 00:00:00-05:00')
    # Parse them, convert to a consistent timezone, then make timezone-naive
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    # Convert from UTC to America/New_York timezone, then remove timezone info
    df["date"] = df["date"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)

    # Coerce numerics
    num_cols = ["open", "close", "high", "low", "volume", "current_price"]
    df = coerce_numeric(df, num_cols)

    # Basic sanity: drop rows without key prices
    df = df.dropna(subset=["open", "close", "high", "low", "volume"])

    # --- Price relationships ---
    df["high_low_spread"] = df["high"] - df["low"]
    df["high_low_pct"]    = (df["high"] - df["low"]) / df["close"]
    df["close_open_diff"] = df["close"] - df["open"]
    df["close_open_pct"]  = (df["close"] - df["open"]) / df["open"]
    df["high_close_diff"] = df["high"] - df["close"]
    df["low_close_diff"]  = df["low"] - df["close"]
    df["close_current_price_diff"] = df["close"] - df["current_price"]

    # --- Returns ---
    df["return_1d"]   = df["close"].pct_change()
    # Use natural log for stationarity; avoid divide-by-zero by shift safely
    df["log_return"]  = np.log(df["close"] / df["close"].shift(1))

    for w in [5, 10, 20]:
        df[f"return_{w}d"] = df["close"].pct_change(periods=w)

    for w in [5, 20]:
        df[f"rolling_mean_return_{w}d"] = df["return_1d"].rolling(w).mean()

    # --- Moving averages (SMA/EMA), ratios, slopes ---
    ma_windows = [5, 10, 20, 50, 100, 200]
    for w in ma_windows:
        df[f"ma_{w}"] = df["close"].rolling(w).mean()
        df[f"ema_{w}"] = ema(df["close"], w)
        df[f"price_ma_{w}_ratio"] = df["close"] / df[f"ma_{w}"]
        df[f"ma_{w}_slope"] = rolling_slope(df["close"], w)

    # Golden/death-cross style binary signals (interpretable to stakeholders)
    df["ma_cross_10_over_50"]  = (df["ma_10"] > df["ma_50"]).astype(int)
    df["ma_cross_20_over_50"]  = (df["ma_20"] > df["ma_50"]).astype(int)
    df["ma_cross_50_over_200"] = (df["ma_50"] > df["ma_200"]).astype(int)

    # --- Momentum indicators ---
    df["momentum_5"]  = df["close"] - df["close"].shift(5)
    df["momentum_10"] = df["close"] - df["close"].shift(10)
    df["roc_10"]      = df["close"].pct_change(periods=10)

    df["rsi_14"] = rsi(df["close"], 14)

    macd_fast, macd_slow, macd_signal = 12, 26, 9
    macd_line   = ema(df["close"], macd_fast) - ema(df["close"], macd_slow)
    macd_signal_line = ema(macd_line, macd_signal)
    df["macd"]        = macd_line
    df["macd_signal"] = macd_signal_line
    df["macd_hist"]   = macd_line - macd_signal_line

    # --- Volatility & risk ---
    for w in [5, 10, 20, 30]:
        df[f"rolling_std_{w}"] = df["close"].rolling(w).std()
        df[f"rolling_var_{w}"] = df["close"].rolling(w).var()

    # ATR (Average True Range, Wilder) over 14 days
    tr = true_range(df["high"], df["low"], df["close"].shift(1))
    df["atr_14"] = tr.rolling(14).mean()

    # Bollinger Bands (20-day)
    mid = df["ma_20"]
    std20 = df["close"].rolling(20).std()
    df["bb_upper_20"] = mid + 2 * std20
    df["bb_lower_20"] = mid - 2 * std20
    df["bb_pct_20"]   = (df["close"] - df["bb_lower_20"]) / (df["bb_upper_20"] - df["bb_lower_20"])

    # --- Volume features ---
    df["vol_change"]  = df["volume"].pct_change()
    df["vol_ma_5"]    = df["volume"].rolling(5).mean()
    df["vol_ma_20"]   = df["volume"].rolling(20).mean()
    df["vol_ratio_5_20"] = df["vol_ma_5"] / df["vol_ma_20"]

    df["obv"] = obv(df["close"], df["volume"])
    df["vpt"] = volume_price_trend(df["close"], df["volume"])

    # --- Lagged features (autoregressive signals) ---
    for lag in [1, 2, 3, 5, 10]:
        df[f"lag_{lag}_close"]  = df["close"].shift(lag)
        df[f"lag_{lag}_return"] = df["return_1d"].shift(lag)
        df[f"lag_{lag}_volume"] = df["volume"].shift(lag)

    # --- Rolling statistics (shape & bounds) ---
    for w in [10, 20, 30]:
        df[f"rolling_mean_close_{w}"] = df["close"].rolling(w).mean()
        df[f"rolling_max_{w}"] = df["close"].rolling(w).max()
        df[f"rolling_min_{w}"] = df["close"].rolling(w).min()
        # rolling skew/kurtosis can be noisy; still useful for trees
        df[f"rolling_skew_{w}"] = df["close"].rolling(w).skew()
        df[f"rolling_kurt_{w}"] = df["close"].rolling(w).kurt()

    # --- Calendar & seasonality ---
    df = add_calendar_features(df, "date")

    # --- Targets (shifted forward; **do not** use these as inputs!) ---
    df["target_close_t1"]  = df["close"].shift(-1)
    df["target_return_t1"] = (df["close"].shift(-1) / df["close"]) - 1

    # Drop columns that are entirely NaN (e.g., 'beta' if never populated)
    df = df.dropna(axis=1, how='all')
    
    # Drop rows with NaNs introduced by rolling windows and the forward target
    # This keeps rows where all columns have valid data
    df = df.dropna().reset_index(drop=True)

    return df

# ----------------------------
# Main / CLI
# ----------------------------
def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Engineer SPLG features for GBR.")
    p.add_argument("--input", "-i", required=True, help="Path to raw SPLG CSV.")
    p.add_argument("--output", "-o", required=True, help="Path to write features CSV.")
    p.add_argument("--show-info", action="store_true", help="Print dataframe info summary.")
    return p.parse_args(argv)

def main(argv: List[str]) -> int:
    # args = parse_args(argv)  # using hard-coded path instead

    #### Input and output paths ####
    SPLG_DATA_IN = "/home/robert/FEURCast/data/SPLG_history_full.csv"
    # SPLG_DATA_IN = "../../data/SPLG_history_full.csv"
    DATA_OUT_DIR = "data_out"
    os.makedirs(DATA_OUT_DIR, exist_ok=True)
    OUT_CSV_PATH = os.path.join(DATA_OUT_DIR, "rich_features_SPLG_history_full.csv")

    # Read
    df_raw = pd.read_csv(SPLG_DATA_IN)
    print(f'##### Raw sample: {df_raw.head()}')
    # df_raw = pd.read_csv(args.input)

    # Engineer
    df_feat = engineer_features(df_raw)

    # print info
    print("Feature rows:", len(df_feat))
    print("Feature cols:", len(df_feat.columns))
    print("Columns sample:", list(df_feat.columns)[:20], "...")
    print(df_feat.tail(3))
    # if args.show_info:
    #     print("Feature rows:", len(df_feat))
    #     print("Feature cols:", len(df_feat.columns))
    #     print("Columns sample:", list(df_feat.columns)[:20], "...")
    #     print(df_feat.tail(3))

    # Write
    df_feat.to_csv(OUT_CSV_PATH, index=False)
    # df_feat.to_csv(args.output, index=False)
    print(f"[OK] Wrote engineered features to: {OUT_CSV_PATH}")
    # print(f"[OK] Wrote engineered features to: {args.output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
