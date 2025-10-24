"""
FUREcast - Latest Features Extractor

This module provides functions to extract the most recent feature values
from the SPLG dataset for making real-time predictions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

# Setup paths
MODULE_DIR = Path(__file__).parent
PRED_MODEL_DIR = MODULE_DIR.parent
DATA_PATH = PRED_MODEL_DIR.parent.parent / "data" / "rich_features_SPLG_history_full.csv"


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
    df['date'] = pd.to_datetime(df['date'])
    
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
    df['date'] = pd.to_datetime(df['date'])
    
    # Define columns
    metadata_cols = ['date', 'company_name', 'ticker']
    target_cols = ['target_close_t1', 'target_return_t1']
    feature_cols = [col for col in df.columns if col not in metadata_cols + target_cols]
    
    # Find matching date
    target_date = pd.to_datetime(date_str)
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
    df['date'] = pd.to_datetime(df['date'])
    
    return str(df['date'].min().date()), str(df['date'].max().date())


def get_latest_date() -> str:
    """
    Get the most recent date in the dataset.
    
    Returns:
        Date string in YYYY-MM-DD format
    """
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    
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
