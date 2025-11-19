"""
FUREcast - Feature Engineering Updater

This module handles automated feature engineering for new SPLG data.
It applies the same feature engineering logic as build_splg_features.py
but is optimized for incremental updates.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import sys

# Setup paths
MODULE_DIR = Path(__file__).parent
DATA_PATH = MODULE_DIR.parent.parent.parent / "data" / "rich_features_SPLG_history_full.csv"
RAW_DATA_PATH = MODULE_DIR.parent.parent.parent / "data" / "SPLG_history_full.csv"
FEATURE_ENGINEERING_PATH = MODULE_DIR / "data_out" / "rich_features_SPLG_history_full.csv"
# FEATURE_ENGINEERING_PATH = MODULE_DIR.parent.parent.parent / "wrangling" / "pred_model_feature_engineering"

# Setup logging
def setup_logger(name: str) -> logging.Logger:
    """Configure logger for feature engineering"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler
    log_dir = MODULE_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"feature_update_{datetime.now().strftime('%Y%m%d')}.log"
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


def rebuild_features_from_scratch() -> bool:
    """
    Rebuild entire feature set from raw data using the original build script.
    This is safer than incremental updates as it ensures consistency.
    
    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 70)
    logger.info("Feature Engineering - Full Rebuild")
    logger.info("=" * 70)
    
    try:
        # Import the feature building module
        # sys.path.insert(0, str(FEATURE_ENGINEERING_PATH))  # Created a copy of the wrangling build_splg_features in pred_model directory for easier maintenance
        from build_splg_features import main as build_features
        
        # Run the feature engineering
        logger.info("Running feature engineering pipeline...")
        build_features([])  # Pass empty list as argv since paths are hardcoded in the script
        
        # Copy the output to the main data directory
        source = FEATURE_ENGINEERING_PATH
        # source = FEATURE_ENGINEERING_PATH / "data_out" / "rich_features_SPLG_history_full.csv"
        
        if not source.exists():
            logger.error(f"Feature file not found at {source}")
            return False
        
        # Read and validate
        df = pd.read_csv(source)
        logger.info(f"✓ Generated features: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Copy to main data directory
        df.to_csv(DATA_PATH, index=False)
        logger.info(f"✓ Saved features to {DATA_PATH}")
        
        # Validate - check for infinite values
        inf_cols = []
        for col in df.columns:
            if df[col].dtype in ['float64', 'float32']:
                if np.isinf(df[col]).any():
                    inf_cols.append(col)
        
        if inf_cols:
            logger.warning(f"Warning: Found infinite values in {len(inf_cols)} columns: {inf_cols[:5]}")
        else:
            logger.info("✓ No infinite values detected")
        
        logger.info("=" * 70)
        logger.info("✓ Feature engineering complete")
        logger.info("=" * 70)
        
        return True
        
    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    # Test feature engineering
    success = rebuild_features_from_scratch()
    
    if success:
        print("\n✓ Feature engineering completed successfully")
    else:
        print("\n✗ Feature engineering failed - check logs")
