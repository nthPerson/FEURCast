"""
FUREcast - Latest Features Extractor

This module provides functions to extract the most recent feature values
from the SPLG dataset for making real-time predictions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
from typing import Optional

DATA_FILE_NAME = "rich_features_SPLG_history_full.csv"

def _repo_root(start: Path) -> Path:
    """
    Walk upwards from `start` until we find a directory that contains a 'data' folder.
    This makes pathing robust to different checkout locations and case differences (FEURCast vs feurcast).
    """
    start = start.resolve()
    for cand in [start] + list(start.parents):
        if (cand / "data").is_dir():
            return cand
    # Fallback: 2 parents up (streamlit/production/… -> repo root)
    return start.parents[3] if len(start.parents) >= 4 else start.parent

def _data_path() -> Path:
    # Optional: allow override via env var
    override = os.getenv("FEURCAST_DATA_DIR")
    if override:
        return (Path(override).expanduser().resolve() / DATA_FILE_NAME)
    return _repo_root(Path(__file__)) / "data" / DATA_FILE_NAME

DATA_PATH = _data_path()

def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    # Robust date parsing for "YYYY-MM-DD" and "YYYY-MM-DD HH:MM:SS"
    try:
        df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
    except TypeError:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        mask = df["date"].isna()
        if mask.any():
            df.loc[mask, "date"] = pd.to_datetime(
                df.loc[mask, "date"].astype(str).str.slice(0, 10),
                errors="coerce"
            )
    return df.dropna(subset=["date"])

def get_latest_features(n_days: int = 1) -> pd.DataFrame:
    """
    Get the most recent N days of features from the dataset.
    
    Args:
        n_days: Number of most recent days to retrieve (default 1)
    
    Returns:
        DataFrame with feature columns only (no metadata or targets)
    """
    # Load data
    df = pd.read_csv(DATA_PATH)
    df = _parse_dates(df)
    
    # Define columns
    metadata_cols = ['date', 'company_name', 'ticker']
    target_cols = ['target_close_t1', 'target_return_t1']
    
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in metadata_cols + target_cols]
    
    # Get latest N days
    latest_data = df.sort_values('date', ascending=False).head(n_days)
    
    return latest_data[feature_cols]


def get_features_for_date(date_str: str) -> Optional[pd.DataFrame]:
    """
    Get features for a specific date.
    
    Args:
        date_str: Date string in YYYY-MM-DD format
    
    Returns:
        DataFrame with features for that date, or None if date not found
    """
    # Load data
    df = pd.read_csv(DATA_PATH)
    df = _parse_dates(df)
    
    # Define columns
    metadata_cols = ['date', 'company_name', 'ticker']
    target_cols = ['target_close_t1', 'target_return_t1']
    feature_cols = [col for col in df.columns if col not in metadata_cols + target_cols]
    
    # Find matching date
    target_date = pd.to_datetime(date_str, errors='coerce')
    matching_row = df[df['date'] == target_date]
    
    if matching_row.empty:
        return None
    
    return matching_row[feature_cols]


def get_date_range() -> tuple:
    """
    Get the date range of available data.
    
    Returns:
        Tuple of (min_date, max_date) as strings
    """
    df = pd.read_csv(DATA_PATH)
    df = _parse_dates(df)
    
    return str(df['date'].min().date()), str(df['date'].max().date())


def get_latest_date() -> str:
    """
    Get the most recent date in the dataset.
    
    Returns:
        Date string in YYYY-MM-DD format
    """
    df = pd.read_csv(DATA_PATH)
    df = _parse_dates(df)
    
    return str(df['date'].max().date())


if __name__ == "__main__":
    # Test the functions
    print("Testing latest features extractor...")
    
    try:
        # Get date range
        min_date, max_date = get_date_range()
        print(f"\nData available from {min_date} to {max_date}")
        
        # Get latest features
        latest = get_latest_features(1)
        print(f"\nLatest features shape: {latest.shape}")
        print(f"Latest date: {get_latest_date()}")
        
        print("\n✓ Feature extractor is working correctly")
        
    except Exception as e:
        print(f"✗ Error: {e}")
