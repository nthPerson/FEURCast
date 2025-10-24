"""
FUREcast - GradientBoostingRegressor Model Training Script

This script trains a GBR model on the SPLG feature-engineered dataset for next-day return prediction.
Implements comprehensive training pipeline with evaluation, persistence, and logging.

Usage:
    python train_gbr_model.py [--quick] [--tune]
    
Options:
    --quick     Skip hyperparameter tuning, use recommended defaults
    --tune      Perform full GridSearch hyperparameter tuning (slow)
"""

import pandas as pd
import numpy as np
import joblib
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
import sys

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PRED_MODEL_DIR = SCRIPT_DIR.parent
DATA_PATH = PRED_MODEL_DIR.parent.parent.parent / "data" / "rich_features_SPLG_history_full.csv"
MODELS_DIR = PRED_MODEL_DIR / "models"
LOGS_DIR = PRED_MODEL_DIR / "logs"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = LOGS_DIR / f"training_log_{timestamp}.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def load_and_prepare_data():
    """
    Load SPLG feature-engineered dataset and prepare for training.
    
    Returns:
        tuple: (X, y, dates, feature_names, metadata)
    """
    logger.info(f"Loading data from {DATA_PATH}")
    
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
    
    # Parse dates
    df['date'] = pd.to_datetime(df['date'])
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Define columns
    metadata_cols = ['date', 'company_name', 'ticker']
    target_cols = ['target_close_t1', 'target_return_t1']
    
    # Extract feature columns (everything except metadata and targets)
    feature_cols = [col for col in df.columns if col not in metadata_cols + target_cols]
    
    logger.info(f"Feature columns: {len(feature_cols)}")
    logger.info(f"Target: target_return_t1 (next-day percentage return)")
    
    # Separate features and target
    X = df[feature_cols].copy()
    y = df['target_return_t1'].copy()
    dates = df['date'].copy()
    metadata = df[metadata_cols].copy()
    
    # ====== DATA CLEANING ======
    logger.info("Performing data quality checks...")
    
    # 1. Check for infinite values
    inf_counts = np.isinf(X).sum()
    if inf_counts.sum() > 0:
        logger.warning(f"Found {inf_counts.sum()} infinite values across {(inf_counts > 0).sum()} features")
        logger.info("Features with infinite values:")
        for col in inf_counts[inf_counts > 0].index:
            logger.info(f"  {col}: {inf_counts[col]} inf values")
        
        # Replace inf with NaN (will be handled next)
        logger.info("Replacing infinite values with NaN...")
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 2. Check for NaN values
    nan_counts = X.isna().sum()
    nan_target = y.isna().sum()
    
    if nan_counts.sum() > 0 or nan_target > 0:
        logger.warning(f"Found {nan_counts.sum()} NaN values in features, {nan_target} NaN in target")
        
        # Log features with high NaN percentage
        nan_pct = (nan_counts / len(X)) * 100
        high_nan_features = nan_pct[nan_pct > 5]
        if len(high_nan_features) > 0:
            logger.warning(f"Features with >5% NaN values:")
            for col, pct in high_nan_features.items():
                logger.warning(f"  {col}: {pct:.2f}%")
        
        # Drop rows with any NaN
        logger.info("Dropping rows with NaN values...")
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]
        dates = dates[valid_idx]
        metadata = metadata[valid_idx]
        logger.info(f"Remaining records after NaN removal: {len(X)}")
    
    # 3. Check for extreme outliers (beyond reasonable bounds)
    logger.info("Checking for extreme outliers...")
    extreme_outliers = {}
    for col in X.columns:
        q1, q99 = X[col].quantile([0.01, 0.99])
        extreme_low = (X[col] < q1 - 10 * (q99 - q1)).sum()
        extreme_high = (X[col] > q99 + 10 * (q99 - q1)).sum()
        if extreme_low > 0 or extreme_high > 0:
            extreme_outliers[col] = {'low': extreme_low, 'high': extreme_high}
    
    if extreme_outliers:
        logger.warning(f"Found extreme outliers in {len(extreme_outliers)} features")
        for col, counts in list(extreme_outliers.items())[:5]:  # Show first 5
            logger.warning(f"  {col}: {counts['low']} low, {counts['high']} high")
        if len(extreme_outliers) > 5:
            logger.warning(f"  ... and {len(extreme_outliers) - 5} more features")
    
    # 4. Verify target is reasonable
    target_outliers = (np.abs(y) > 0.2).sum()  # Returns > 20% are extreme for daily
    if target_outliers > 0:
        logger.warning(f"Found {target_outliers} target values with |return| > 20%")
        logger.info(f"Max absolute return: {np.abs(y).max():.4f}")
    
    # Final statistics
    logger.info(f"\nFinal cleaned dataset: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Target statistics:")
    logger.info(f"  Mean:   {y.mean():.6f}")
    logger.info(f"  Std:    {y.std():.6f}")
    logger.info(f"  Min:    {y.min():.6f}")
    logger.info(f"  Max:    {y.max():.6f}")
    logger.info(f"  Median: {y.median():.6f}")
    
    # Verify no issues remain
    assert not np.isinf(X.values).any(), "Infinite values still present after cleaning"
    assert not X.isna().any().any(), "NaN values still present after cleaning"
    assert not y.isna().any(), "NaN values in target after cleaning"
    
    logger.info("✓ Data quality checks passed")
    
    return X, y, dates, feature_cols, metadata


def create_time_based_splits(X, y, dates):
    """
    Create time-based train/validation/test splits.
    
    Uses 70% train, 15% validation, 15% test (chronological order).
    
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test, dates_train, dates_val, dates_test)
    """
    logger.info("Creating time-based train/val/test splits (70%/15%/15%)")
    
    n = len(X)
    train_size = int(0.70 * n)
    val_size = int(0.15 * n)
    
    # Time-ordered splits
    X_train = X.iloc[:train_size]
    X_val = X.iloc[train_size:train_size + val_size]
    X_test = X.iloc[train_size + val_size:]
    
    y_train = y.iloc[:train_size]
    y_val = y.iloc[train_size:train_size + val_size]
    y_test = y.iloc[train_size + val_size:]
    
    dates_train = dates.iloc[:train_size]
    dates_val = dates.iloc[train_size:train_size + val_size]
    dates_test = dates.iloc[train_size + val_size:]
    
    logger.info(f"Train: {dates_train.min()} to {dates_train.max()} ({len(X_train)} samples)")
    logger.info(f"Val:   {dates_val.min()} to {dates_val.max()} ({len(X_val)} samples)")
    logger.info(f"Test:  {dates_test.min()} to {dates_test.max()} ({len(X_test)} samples)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, dates_train, dates_val, dates_test


def scale_features(X_train, X_val, X_test):
    """
    Scale features using StandardScaler fitted on training data.
    
    Returns:
        tuple: (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
    """
    logger.info("Scaling features with StandardScaler")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info("Feature scaling complete")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def train_baseline_model(X_train, y_train):
    """
    Train baseline GBR model with default parameters.
    
    Returns:
        GradientBoostingRegressor: Trained baseline model
    """
    logger.info("Training baseline model with default parameters...")
    
    gbr_baseline = GradientBoostingRegressor(
        random_state=RANDOM_SEED,
        verbose=1
    )
    
    gbr_baseline.fit(X_train, y_train)
    
    logger.info("Baseline model training complete")
    return gbr_baseline


def train_tuned_model(X_train, y_train, quick=True):
    """
    Train GBR model with recommended or tuned hyperparameters.
    
    Args:
        X_train: Training features
        y_train: Training target
        quick: If True, use recommended params; if False, perform GridSearch
    
    Returns:
        GradientBoostingRegressor: Trained model
    """
    if quick:
        logger.info("Training with recommended hyperparameters (quick mode)...")
        
        gbr = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            min_samples_split=10,
            min_samples_leaf=4,
            subsample=0.8,
            max_features='sqrt',
            random_state=RANDOM_SEED,
            verbose=1,
            validation_fraction=0.1,
            n_iter_no_change=20
        )
        
        gbr.fit(X_train, y_train)
        logger.info("Quick training complete")
        
        return gbr
    
    else:
        logger.info("Performing GridSearch hyperparameter tuning (this may take a while)...")
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4],
            'subsample': [0.8, 0.9],
            'max_features': ['sqrt', 'log2']
        }
        
        # Use TimeSeriesSplit for proper temporal cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        gbr = GradientBoostingRegressor(random_state=RANDOM_SEED)
        
        grid_search = GridSearchCV(
            estimator=gbr,
            param_grid=param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=2
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV MSE: {-grid_search.best_score_:.6f}")
        
        return grid_search.best_estimator_


def evaluate_model(model, X, y, set_name=""):
    """
    Evaluate model performance with multiple metrics.
    
    Args:
        model: Trained model
        X: Features
        y: Target
        set_name: Name of dataset (Train/Val/Test)
    
    Returns:
        dict: Performance metrics
    """
    y_pred = model.predict(X)
    
    # Regression metrics
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # Mean Absolute Percentage Error (with handling for zero values)
    epsilon = 1e-10
    mape = np.mean(np.abs((y - y_pred) / (y + epsilon))) * 100
    
    # Financial metric: Directional accuracy
    directional_accuracy = np.mean((np.sign(y) == np.sign(y_pred)))
    
    # Log metrics
    logger.info(f"\n{set_name} Set Performance:")
    logger.info(f"  MSE:  {mse:.8f}")
    logger.info(f"  RMSE: {rmse:.6f}")
    logger.info(f"  MAE:  {mae:.6f}")
    logger.info(f"  R²:   {r2:.6f}")
    logger.info(f"  MAPE: {mape:.2f}%")
    logger.info(f"  Directional Accuracy: {directional_accuracy:.2%}")
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'mape': float(mape),
        'directional_accuracy': float(directional_accuracy),
        'predictions': y_pred.tolist() if len(y_pred) < 10000 else None  # Don't store huge arrays
    }


def save_model_artifacts(model, scaler, feature_names, metrics, model_name="gbr_model"):
    """
    Save trained model and all artifacts.
    
    Args:
        model: Trained GBR model
        scaler: Fitted StandardScaler
        feature_names: List of feature column names
        metrics: Dictionary of performance metrics
        model_name: Base name for saved files
    """
    logger.info("Saving model artifacts...")
    
    # Save model
    model_path = MODELS_DIR / f"{model_name}.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Saved model to {model_path}")
    
    # Save scaler
    scaler_path = MODELS_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    logger.info(f"Saved scaler to {scaler_path}")
    
    # Save feature names
    feature_names_path = MODELS_DIR / "feature_names.pkl"
    joblib.dump(feature_names, feature_names_path)
    logger.info(f"Saved feature names to {feature_names_path}")
    
    # Save metrics
    metrics_path = MODELS_DIR / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")
    
    # Save feature importances as CSV
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    importance_path = MODELS_DIR / "feature_importance.csv"
    feature_importance.to_csv(importance_path, index=False)
    logger.info(f"Saved feature importance to {importance_path}")
    
    # Create model metadata
    metadata = {
        'model_type': 'GradientBoostingRegressor',
        'version': timestamp,
        'training_date': datetime.now().isoformat(),
        'n_features': len(feature_names),
        'hyperparameters': model.get_params(),
        'random_seed': RANDOM_SEED,
        'data_path': str(DATA_PATH),
        'performance_metrics': metrics
    }
    
    metadata_path = MODELS_DIR / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved model metadata to {metadata_path}")
    
    logger.info("All artifacts saved successfully")


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description="Train GBR model for SPLG prediction")
    parser.add_argument('--quick', action='store_true', help='Skip hyperparameter tuning')
    parser.add_argument('--tune', action='store_true', help='Perform full GridSearch tuning')
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("FUREcast GBR Model Training Pipeline")
    logger.info("="*80)
    logger.info(f"Log file: {log_filename}")
    logger.info(f"Random seed: {RANDOM_SEED}")
    
    try:
        # Step 1: Load and prepare data
        X, y, dates, feature_names, metadata = load_and_prepare_data()
        
        # Step 2: Create splits
        X_train, X_val, X_test, y_train, y_val, y_test, dates_train, dates_val, dates_test = \
            create_time_based_splits(X, y, dates)
        
        # Step 3: Scale features
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = \
            scale_features(X_train, X_val, X_test)
        
        # Step 4: Train model
        if args.tune:
            model = train_tuned_model(X_train_scaled, y_train, quick=False)
        else:
            model = train_tuned_model(X_train_scaled, y_train, quick=True)
        
        # Step 5: Evaluate on all sets
        logger.info("\n" + "="*80)
        logger.info("MODEL EVALUATION")
        logger.info("="*80)
        
        train_metrics = evaluate_model(model, X_train_scaled, y_train, "Training")
        val_metrics = evaluate_model(model, X_val_scaled, y_val, "Validation")
        test_metrics = evaluate_model(model, X_test_scaled, y_test, "Test")
        
        # Compile all metrics
        all_metrics = {
            'train': train_metrics,
            'validation': val_metrics,
            'test': test_metrics,
            'training_info': {
                'n_train_samples': len(X_train),
                'n_val_samples': len(X_val),
                'n_test_samples': len(X_test),
                'n_features': len(feature_names),
                'train_date_range': [str(dates_train.min()), str(dates_train.max())],
                'val_date_range': [str(dates_val.min()), str(dates_val.max())],
                'test_date_range': [str(dates_test.min()), str(dates_test.max())]
            }
        }
        
        # Step 6: Save everything
        save_model_artifacts(model, scaler, feature_names, all_metrics)
        
        # Step 7: Summary
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETE")
        logger.info("="*80)
        logger.info(f"Final Test R² Score: {test_metrics['r2']:.6f}")
        logger.info(f"Final Test RMSE: {test_metrics['rmse']:.6f}")
        logger.info(f"Final Test Directional Accuracy: {test_metrics['directional_accuracy']:.2%}")
        logger.info(f"Top 5 Features:")
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        for idx, row in feature_importance.head(5).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        logger.info("\nAll artifacts saved to:")
        logger.info(f"  Models: {MODELS_DIR}")
        logger.info(f"  Logs: {log_filename}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
