"""
FUREcast - SPLG Data Updater

This module handles automated fetching and updating of SPLG price data
using yfinance API. It maintains the historical dataset and triggers
feature engineering when new data is available.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, Optional
import logging

# Setup paths
MODULE_DIR = Path(__file__).parent
DATA_PATH = MODULE_DIR.parent.parent.parent / "data" / "rich_features_SPLG_history_full.csv"
RAW_DATA_PATH = MODULE_DIR.parent.parent.parent / "data" / "SPLG_history_full.csv"
LOG_DIR = MODULE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Setup logging
def setup_logger(name: str) -> logging.Logger:
    """Configure logger for data updates"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler
    log_file = LOG_DIR / f"data_update_{datetime.now().strftime('%Y%m%d')}.log"
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

logger = setup_logger(__name__)


def get_latest_date_in_dataset() -> Optional[datetime]:
    """
    Get the most recent date in the existing dataset.
    
    Returns:
        datetime object of latest date, or None if dataset doesn't exist
    """
    if not DATA_PATH.exists():
        logger.warning(f"Dataset not found at {DATA_PATH}")
        return None
    
    try:
        df = pd.read_csv(DATA_PATH)
        df['date'] = pd.to_datetime(df['date'])
        latest_date = df['date'].max()
        logger.info(f"Latest date in dataset: {latest_date.date()}")
        return latest_date
    except Exception as e:
        logger.error(f"Error reading dataset: {e}")
        return None


def fetch_new_splg_data(start_date: Optional[datetime] = None, 
                        end_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
    """
    Fetch SPLG price data from yfinance.
    
    Args:
        start_date: Start date for fetching data (default: 1 day after latest in dataset)
        end_date: End date for fetching data (default: today)
    
    Returns:
        DataFrame with SPLG price data, or None if no new data
    """
    # Determine date range
    if end_date is None:
        end_date = datetime.now()
    
    if start_date is None:
        latest_date = get_latest_date_in_dataset()
        if latest_date is None:
            # No existing dataset - fetch last 20 years
            start_date = datetime.now() - timedelta(days=365*20)
            logger.info("No existing dataset - fetching full history (20 years)")
        else:
            # Fetch from day after latest date
            start_date = latest_date + timedelta(days=1)
            logger.info(f"Fetching data from {start_date.date()} to {end_date.date()}")
    
    # Check if we need to fetch anything
    if start_date >= end_date:
        logger.info("Dataset is already up to date")
        return None
    
    try:
        # Fetch data from yfinance
        logger.info(f"Downloading SPLG data from yfinance...")
        splg = yf.Ticker("SPLG")
        
        # Get historical data
        hist = splg.history(start=start_date, end=end_date, interval="1d")
        
        if hist.empty:
            logger.info("No new data available from yfinance")
            return None
        
        # Reset index to make Date a column
        hist = hist.reset_index()
        
        # Rename columns to match our schema
        hist = hist.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # Add required columns
        hist['company_name'] = 'SPDR Portfolio S&P 500 ETF'
        hist['ticker'] = 'SPLG'
        hist['current_price'] = hist['close']  # Use close as current price
        
        # Get additional info
        info = splg.info
        hist['pe_ratio'] = info.get('trailingPE', 0.0)
        hist['yield'] = info.get('dividendYield', 0.0) * 100 if info.get('dividendYield') else 0.0
        
        # Select and order columns
        columns = [
            'date', 'company_name', 'ticker', 'current_price',
            'open', 'close', 'high', 'low', 'volume',
            'pe_ratio', 'yield'
        ]
        
        hist = hist[columns]
        
        logger.info(f"✓ Fetched {len(hist)} new records")
        logger.info(f"  Date range: {hist['date'].min().date()} to {hist['date'].max().date()}")
        
        return hist
        
    except Exception as e:
        logger.error(f"Error fetching data from yfinance: {e}")
        return None


def update_raw_dataset(new_data: pd.DataFrame) -> bool:
    """
    Append new data to the raw SPLG dataset.
    
    Args:
        new_data: DataFrame with new SPLG data
    
    Returns:
        True if update successful, False otherwise
    """
    try:
        # Ensure new_data has datetime date column
        if 'date' not in new_data.columns:
            logger.error("New data missing 'date' column")
            return False
        
        new_data = new_data.copy()
        new_data['date'] = pd.to_datetime(new_data['date'])
        
        # Load existing raw data if it exists
        if RAW_DATA_PATH.exists():
            existing_data = pd.read_csv(RAW_DATA_PATH)
            
            # Ensure existing data also has datetime
            if 'date' in existing_data.columns:
                existing_data['date'] = pd.to_datetime(existing_data['date'])
            else:
                logger.warning("Existing data missing 'date' column, using new data only")
                combined = new_data
                logger.info(f"Creating new raw dataset with {len(combined)} records")
                RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
                combined.to_csv(RAW_DATA_PATH, index=False)
                logger.info(f"✓ Saved raw dataset to {RAW_DATA_PATH}")
                return True
            
            # Combine with new data
            combined = pd.concat([existing_data, new_data], ignore_index=True)
            
            # Remove duplicates (keep latest)
            combined = combined.sort_values('date').drop_duplicates(subset='date', keep='last')
            
            logger.info(f"Combined dataset: {len(combined)} total records")
        else:
            combined = new_data
            logger.info(f"Creating new raw dataset with {len(combined)} records")
        
        # Ensure directory exists
        RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        # Save updated dataset
        combined.to_csv(RAW_DATA_PATH, index=False)
        logger.info(f"✓ Saved raw dataset to {RAW_DATA_PATH}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error updating raw dataset: {e}")
        return False


def check_for_updates() -> Tuple[bool, Optional[pd.DataFrame]]:
    """
    Check if new SPLG data is available and fetch it.
    
    Returns:
        Tuple of (has_updates: bool, new_data: Optional[DataFrame])
    """
    logger.info("=" * 70)
    logger.info("SPLG Data Update Check")
    logger.info("=" * 70)
    
    # Fetch new data
    new_data = fetch_new_splg_data()
    
    if new_data is None or new_data.empty:
        logger.info("No updates available")
        return False, None
    
    # Update raw dataset
    success = update_raw_dataset(new_data)
    
    if not success:
        logger.error("Failed to update raw dataset")
        return False, None
    
    logger.info("=" * 70)
    logger.info("✓ Data update complete")
    logger.info("=" * 70)
    
    return True, new_data


if __name__ == "__main__":
    # Test the data updater
    has_updates, new_data = check_for_updates()
    
    if has_updates:
        print(f"\n✓ Successfully updated dataset with {len(new_data)} new records")
        print(f"  Latest date: {new_data['date'].max().date()}")
    else:
        print("\n✓ Dataset is already up to date")
