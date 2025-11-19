#!/usr/bin/env python3
"""
build_splg_features.py

Engineer a rich feature set for SPLG daily OHLCV data for use with tree-based
models (e.g., GradientBoostingRegressor). No external TA libraries required.
"""

import argparse
import sys
import numpy as np
import pandas as pd
from typing import List, Tuple
import os
from pathlib import Path
import logging
from datetime import datetime

# ----------------------------
# Setup Logging
# ----------------------------
def setup_logger(name: str) -> logging.Logger:
    """Configure logger for feature engineering"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    
    logger.addHandler(ch)
    
    return logger

logger = setup_logger(__name__)

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
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_values = 100 - (100 / (1 + rs))
    return rsi_values.clip(0, 100)

def true_range(high: pd.Series, low: pd.Series, prev_close: pd.Series) -> pd.Series:
    a = high - low
    b = (high - prev_close).abs()
    c = (low - prev_close).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    sign = np.sign(close.diff()).fillna(0.0)
    return (sign * volume).fillna(0.0).cumsum()

def volume_price_trend(close: pd.Series, volume: pd.Series) -> pd.Series:
    return (volume * close.pct_change(fill_method=None)).fillna(0.0).cumsum()

def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df = df.copy()
    if not np.issubdtype(df[date_col].dtype, np.datetime64):
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
    ma = x.rolling(window).mean()
    slope = (ma - ma.shift(1)) / ma.shift(1)
    return slope.replace([np.inf, -np.inf], 0).fillna(0)

# ----------------------------
# Core feature engineering
# ----------------------------
def engineer_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features from raw SPLG data.
    
    IMPORTANT: This function preserves ALL rows and does NOT drop NaN values.
    NaN handling is deferred to the training script (train_gbr_model.py).
    """
    logger.info("="*70)
    logger.info("Starting Feature Engineering")
    logger.info("="*70)
    
    df = normalize_cols(df_raw)
    
    logger.info(f"Input data: {len(df)} rows, {len(df.columns)} columns")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Ensure required market columns exist
    required_any_current = ["date", "open", "close", "high", "low", "volume"]
    require_columns(df, required_any_current)

    # If current_price is missing or empty, create it as close
    if "current_price" not in df.columns:
        df["current_price"] = df["close"]

    # Parse date, sort, drop duplicates
    logger.info("Parsing and sorting dates...")
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df["date"] = df["date"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    
    # Drop rows with invalid dates
    initial_count = len(df)
    df = df.dropna(subset=["date"])
    if len(df) < initial_count:
        logger.warning(f"Dropped {initial_count - len(df)} rows with invalid dates")
    
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    logger.info(f"After date cleaning: {len(df)} rows")

    # Coerce numerics
    num_cols = ["open", "close", "high", "low", "volume", "current_price"]
    df = coerce_numeric(df, num_cols)

    # Basic sanity: drop rows without key prices (these are truly unusable)
    initial_count = len(df)
    df = df.dropna(subset=["open", "close", "high", "low", "volume"])
    if len(df) < initial_count:
        logger.warning(f"Dropped {initial_count - len(df)} rows missing OHLCV data")
    logger.info(f"After OHLCV validation: {len(df)} rows")

    logger.info("Engineering features...")
    
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
        df[f"price_ma_{w}_ratio"] = (df["close"] / df[f"ma_{w}"]).replace([np.inf, -np.inf], 1.0).fillna(1.0)
        df[f"ma_{w}_slope"] = rolling_slope(df["close"], w)

    # Golden/death-cross style binary signals
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
    band_width = df["bb_upper_20"] - df["bb_lower_20"]
    df["bb_pct_20"] = ((df["close"] - df["bb_lower_20"]) / (band_width + 1e-10)).clip(0, 1)

    # --- Volume features ---
    df["vol_change"]  = df["volume"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    df["vol_ma_5"]    = df["volume"].rolling(5).mean()
    df["vol_ma_20"]   = df["volume"].rolling(20).mean()
    df["vol_ratio_5_20"] = (df["vol_ma_5"] / df["vol_ma_20"]).replace([np.inf, -np.inf], 1.0).fillna(1.0)

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
        df[f"rolling_skew_{w}"] = df["close"].rolling(w).skew()
        df[f"rolling_kurt_{w}"] = df["close"].rolling(w).kurt()

    # --- Calendar & seasonality ---
    df = add_calendar_features(df, "date")

    # --- Targets (shifted forward; **do not** use these as inputs!) ---
    logger.info("Creating target variables...")
    df["target_close_t1"]  = df["close"].shift(-1)
    df["target_return_t1"] = (df["close"].shift(-1) / df["close"]) - 1

    # Drop columns that are entirely NaN (e.g., 'beta' if never populated)
    df = df.dropna(axis=1, how='all')
    
    # CRITICAL: DO NOT drop rows with NaN here!
    # NaN handling is deferred to training script (train_gbr_model.py)
    # This preserves all historical data including:
    # - Early rows with NaN in rolling window features (first ~200 rows)
    # - Last row with NaN in targets (from forward shift)
    
    logger.info("="*70)
    logger.info("Feature Engineering Complete")
    logger.info("="*70)
    logger.info(f"Output data: {len(df)} rows, {len(df.columns)} columns")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Diagnostic: Count NaN values
    feature_cols = [col for col in df.columns if not col.startswith('target_') and col not in ['date', 'company_name', 'ticker']]
    feature_nans = df[feature_cols].isna().sum().sum()
    target_nans = df[['target_close_t1', 'target_return_t1']].isna().sum().sum()
    
    logger.info(f"NaN counts: Features={feature_nans}, Targets={target_nans}")
    logger.info("(NaN values will be handled during model training)")
    logger.info("="*70)
    
    # Reset index
    df = df.reset_index(drop=True)

    return df

# ----------------------------
# Main / CLI
# ----------------------------
def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Engineer SPLG features for GBR.")
    p.add_argument("--input", "-i", required=False, help="Path to raw SPLG CSV.")
    p.add_argument("--output", "-o", required=False, help="Path to write features CSV.")
    p.add_argument("--show-info", action="store_true", help="Print dataframe info summary.")
    return p.parse_args(argv)

def main(argv: List[str]) -> int:
    # Parse args
    args = parse_args(argv)
    
    # Resolve paths relative to this script's location
    SCRIPT_DIR = Path(__file__).parent
    DATA_DIR = SCRIPT_DIR.parent.parent.parent / "data"
    SPLG_DATA_IN = DATA_DIR / "SPLG_history_full.csv"
    
    # Output to pred_model/data_out directory
    DATA_OUT_DIR = SCRIPT_DIR / "data_out"
    os.makedirs(DATA_OUT_DIR, exist_ok=True)
    OUT_CSV_PATH = DATA_OUT_DIR / "rich_features_SPLG_history_full.csv"
    
    # Use command line args if provided, otherwise use defaults
    input_path = args.input if args.input else SPLG_DATA_IN
    output_path = args.output if args.output else OUT_CSV_PATH
    
    logger.info(f"Reading raw data from: {input_path}")
    
    # Read
    df_raw = pd.read_csv(input_path)
    logger.info(f"Raw data sample (first 5 rows):")
    logger.info(f"\n{df_raw.head()}")

    # Engineer
    df_feat = engineer_features(df_raw)

    # Print info
    if args.show_info:
        logger.info("\nFeature Engineering Summary:")
        logger.info(f"  Rows: {len(df_feat)}")
        logger.info(f"  Columns: {len(df_feat.columns)}")
        logger.info(f"  Sample columns: {list(df_feat.columns)[:20]}...")
        logger.info(f"\nLast 3 rows:")
        logger.info(f"\n{df_feat.tail(3)}")

    # Write
    df_feat.to_csv(output_path, index=False)
    logger.info(f"✓ Wrote engineered features to: {output_path}")
    logger.info(f"✓ Final output: {len(df_feat)} rows × {len(df_feat.columns)} columns")
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
