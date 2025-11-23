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
from typing import Dict

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
    
    # Parse dates with flexible format handling
    # Use 'mixed' format to handle inconsistent date formats in the dataset
    try:
        df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
    except:
        # Fallback: let pandas infer the format
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Drop any rows where date parsing failed
    initial_count = len(df)
    df = df.dropna(subset=['date'])
    if len(df) < initial_count:
        logger.warning(f"Dropped {initial_count - len(df)} rows with invalid dates")
    
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
    
    # ====== DATA CLEANING FOR TRAINING ======
    logger.info("\n" + "="*60)
    logger.info("PERFORMING DATA QUALITY CHECKS AND CLEANING")
    logger.info("="*60)
    
    initial_rows = len(X)
    logger.info(f"Initial dataset size: {initial_rows} rows, {len(feature_cols)} features")
    
    # 1. Check for and handle infinite values
    inf_counts = np.isinf(X).sum()
    if inf_counts.sum() > 0:
        logger.warning(f"Found {inf_counts.sum()} infinite values across {(inf_counts > 0).sum()} features")
        logger.info("Features with infinite values (top 10):")
        for col in inf_counts[inf_counts > 0].nlargest(10).index:
            logger.info(f"  {col}: {inf_counts[col]} inf values")
        
        logger.info("Replacing infinite values with NaN...")
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
    else:
        logger.info("✓ No infinite values found")
    
    # 2. Check NaN statistics BEFORE cleaning
    nan_counts = X.isna().sum()
    nan_target = y.isna().sum()
    
    logger.info(f"\nNaN Statistics (before cleaning):")
    logger.info(f"  Total feature NaNs: {nan_counts.sum()}")
    logger.info(f"  Features with NaNs: {(nan_counts > 0).sum()} of {len(feature_cols)}")
    logger.info(f"  Target NaNs: {nan_target}")
    
    # Show features with high NaN counts
    high_nan_features = nan_counts[nan_counts > initial_rows * 0.05].sort_values(ascending=False)
    if len(high_nan_features) > 0:
        logger.info(f"\nFeatures with >5% NaN values:")
        for col, count in high_nan_features.head(10).items():
            pct = (count / initial_rows) * 100
            logger.info(f"  {col}: {count} ({pct:.1f}%)")
    
    # 3. Intelligent NaN handling strategy
    logger.info("\nApplying intelligent NaN handling strategy...")
    
    # Strategy A: Drop columns with >95% NaN (essentially useless)
    nan_threshold_drop = 0.95
    cols_to_drop = nan_counts[nan_counts > initial_rows * nan_threshold_drop].index.tolist()
    
    if cols_to_drop:
        logger.info(f"Dropping {len(cols_to_drop)} columns with >{nan_threshold_drop*100}% NaN:")
        for col in cols_to_drop:
            pct = (nan_counts[col] / initial_rows) * 100
            logger.info(f"  {col}: {pct:.1f}% NaN")
        X = X.drop(columns=cols_to_drop)
        feature_cols = [col for col in feature_cols if col not in cols_to_drop]
        logger.info(f"Remaining features: {len(X.columns)}")
    
    # Strategy B: Forward-fill NaN in rolling window features (ma_, ema_, rolling_, lag_, etc.)
    # These have NaN at the start due to insufficient lookback
    rolling_pattern_cols = [col for col in X.columns if any(
        pattern in col for pattern in ['ma_', 'ema_', 'rolling_', 'lag_', 'momentum_', 'rsi_', 'macd', 'atr_', 'bb_', 'obv', 'vpt']
    )]
    
    if rolling_pattern_cols:
        logger.info(f"\nForward-filling NaN in {len(rolling_pattern_cols)} rolling/technical indicator features...")
        nan_before = X[rolling_pattern_cols].isna().sum().sum()
        X[rolling_pattern_cols] = X[rolling_pattern_cols].fillna(method='ffill')
        # Backward fill any remaining NaN at the very start
        X[rolling_pattern_cols] = X[rolling_pattern_cols].fillna(method='bfill')
        nan_after = X[rolling_pattern_cols].isna().sum().sum()
        logger.info(f"  NaN values filled: {nan_before} → {nan_after}")
    
    # Strategy C: Drop rows with NaN in remaining features (basic features like price, volume)
    logger.info("\nDropping rows with NaN in basic features or target...")
    
    remaining_nan = X.isna().sum()
    if remaining_nan.sum() > 0:
        logger.info(f"Remaining NaN in {(remaining_nan > 0).sum()} features:")
        for col, count in remaining_nan[remaining_nan > 0].head(10).items():
            pct = (count / len(X)) * 100
            logger.info(f"  {col}: {count} ({pct:.1f}%)")
    
    valid_idx = ~(X.isna().any(axis=1) | y.isna())
    
    X = X[valid_idx].reset_index(drop=True)
    y = y[valid_idx].reset_index(drop=True)
    dates = dates[valid_idx].reset_index(drop=True)
    metadata = metadata[valid_idx].reset_index(drop=True)
    
    rows_dropped = initial_rows - len(X)
    logger.info(f"Dropped {rows_dropped} rows ({rows_dropped/initial_rows*100:.1f}%)")
    logger.info(f"✓ Final training set: {len(X)} samples")
    
    # 4. Verify data quality after cleaning
    assert not np.isinf(X.values).any(), "ERROR: Infinite values still present after cleaning!"
    assert not X.isna().any().any(), "ERROR: NaN values in features after cleaning!"
    assert not y.isna().any(), "ERROR: NaN values in target after cleaning!"
    
    logger.info("\n✓ All data quality checks passed")
    
    # 5. Check for extreme outliers in target
    target_stats = {
        'mean': y.mean(),
        'std': y.std(),
        'min': y.min(),
        'max': y.max(),
        'median': y.median(),
        'q01': y.quantile(0.01),
        'q99': y.quantile(0.99)
    }
    
    extreme_returns = (np.abs(y) > 0.2).sum()
    if extreme_returns > 0:
        logger.warning(f"Found {extreme_returns} target values with |return| > 20%")
    
    # 6. Final statistics
    logger.info(f"\nFinal cleaned dataset:")
    logger.info(f"  Shape: {X.shape[0]} samples × {X.shape[1]} features")
    logger.info(f"  Date range: {dates.min()} to {dates.max()}")
    logger.info(f"\nTarget (next-day return) statistics:")
    logger.info(f"  Mean:     {target_stats['mean']:>10.6f}")
    logger.info(f"  Std:      {target_stats['std']:>10.6f}")
    logger.info(f"  Min:      {target_stats['min']:>10.6f}")
    logger.info(f"  1st %ile: {target_stats['q01']:>10.6f}")
    logger.info(f"  Median:   {target_stats['median']:>10.6f}")
    logger.info(f"  99th %ile:{target_stats['q99']:>10.6f}")
    logger.info(f"  Max:      {target_stats['max']:>10.6f}")
    
    logger.info("="*60 + "\n")
    
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


def load_optimal_hyperparameters():
    """
    Load previously discovered optimal hyperparameters from model metadata.
    
    Returns:
        dict or None: Optimal hyperparameters if available, else None
    """
    metadata_path = MODELS_DIR / "model_metadata.json"
    
    if not metadata_path.exists():
        logger.info("No previous model metadata found - using default hyperparameters")
        return None
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Extract hyperparameters
        hyperparams = metadata.get('hyperparameters', {})
        
        # Filter to only GBR-specific parameters (exclude sklearn internals)
        gbr_params = {
            'n_estimators': hyperparams.get('n_estimators'),
            'learning_rate': hyperparams.get('learning_rate'),
            'max_depth': hyperparams.get('max_depth'),
            'min_samples_split': hyperparams.get('min_samples_split'),
            'min_samples_leaf': hyperparams.get('min_samples_leaf'),
            'subsample': hyperparams.get('subsample'),
            'max_features': hyperparams.get('max_features')
        }
        
        # Check if we have valid parameters
        if all(v is not None for v in gbr_params.values()):
            logger.info("Loaded optimal hyperparameters from previous tuning:")
            for key, value in gbr_params.items():
                logger.info(f"  {key}: {value}")
            return gbr_params
        else:
            logger.info("Previous hyperparameters incomplete - using defaults")
            return None
            
    except Exception as e:
        logger.warning(f"Error loading previous hyperparameters: {e}")
        return None


def smape(y_true, y_pred, epsilon=1e-10):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + epsilon))

def compute_baselines(y_train, y_val, y_test):
    """Naive baselines: previous day (lag-1), zero, mean."""
    def metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        s_mape = smape(y_true, y_pred)
        dir_acc = np.mean(np.sign(y_true) == np.sign(y_pred))
        return dict(mse=mse, rmse=rmse, mae=mae, r2=r2, smape=s_mape, directional_accuracy=dir_acc)
    # Lag baseline (cannot predict first item ⇒ drop first sample for fair comparison)
    lag_train_pred = y_train.shift(1).fillna(0)
    lag_val_pred = y_val.shift(1).fillna(y_train.iloc[-1])
    lag_test_pred = y_test.shift(1).fillna(y_val.iloc[-1])
    zero_train = np.zeros_like(y_train)
    zero_val = np.zeros_like(y_val)
    zero_test = np.zeros_like(y_test)
    mean_val = np.full_like(y_val, y_train.mean())
    mean_test = np.full_like(y_test, y_train.mean())
    baselines = {
        'lag': {
            'train': metrics(y_train[1:], lag_train_pred[1:]),
            'val': metrics(y_val, lag_val_pred),
            'test': metrics(y_test, lag_test_pred)
        },
        'zero': {
            'val': metrics(y_val, zero_val),
            'test': metrics(y_test, zero_test)
        },
        'mean': {
            'val': metrics(y_val, mean_val),
            'test': metrics(y_test, mean_test)
        }
    }
    return baselines

def evaluate_model(model, X, y, set_name=""):
    """Extended evaluation (adds sMAPE; suppress unusable MAPE)."""
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    s_mape = smape(y, y_pred)
    directional_accuracy = np.mean(np.sign(y) == np.sign(y_pred))
    logger.info(f"\n{set_name} Set Performance:")
    logger.info(f"  MSE:   {mse:.8f}")
    logger.info(f"  RMSE:  {rmse:.6f}")
    logger.info(f"  MAE:   {mae:.6f}")
    logger.info(f"  R²:    {r2:.6f}")
    logger.info(f"  sMAPE: {s_mape:.2f}%")
    logger.info(f"  Directional Accuracy: {directional_accuracy:.2%}")
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'smape': float(s_mape),
        'directional_accuracy': float(directional_accuracy),
        'predictions': y_pred.tolist() if len(y_pred) < 10000 else None
    }


def train_tuned_model(X_train, y_train, quick=True):
    """
    Train GBR model with recommended or tuned hyperparameters.
    
    Args:
        X_train: Training features
        y_train: Training target
        quick: If True, use recommended/previous params; if False, perform GridSearch
    
    Returns:
        GradientBoostingRegressor: Trained model
    """
    if quick:
        logger.info("Training in quick mode with enhanced defaults...")
        optimal_params = load_optimal_hyperparameters()
        if optimal_params:
            base_params = optimal_params
        else:
            base_params = {
                'n_estimators': 1200,
                'learning_rate': 0.01,
                'max_depth': 5,
                'min_samples_split': 10,
                'min_samples_leaf': 7,
                'subsample': 0.7,
                'max_features': 'sqrt'
            }
        gbr = GradientBoostingRegressor(
            **base_params,
            random_state=RANDOM_SEED,
            verbose=1,
            validation_fraction=0.1,
            n_iter_no_change=50,
            tol=1e-4
        )
        gbr.fit(X_train, y_train)
        logger.info("Enhanced quick training complete")
        return gbr
    else:
        logger.info("Expanded GridSearch with TimeSeriesSplit...")
        param_grid = {
            'n_estimators': [600, 900, 1200],
            'learning_rate': [0.01, 0.02, 0.03],
            'max_depth': [4, 5, 6],
            'min_samples_split': [10, 20],
            'min_samples_leaf': [3, 5, 8],
            'subsample': [0.6, 0.7, 0.8],
            'max_features': ['sqrt', 'log2']
        }
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
        
        logger.info(f"\n{'='*60}")
        logger.info("HYPERPARAMETER TUNING RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Best parameters found:")
        for param, value in grid_search.best_params_.items():
            logger.info(f"  {param}: {value}")
        logger.info(f"Best CV MSE: {-grid_search.best_score_:.6f}")
        logger.info(f"\nThese parameters will be used for future --quick training")
        logger.info(f"{'='*60}\n")
        
        return grid_search.best_estimator_


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
        'hyperparameters': {
            k: v for k, v in model.get_params().items()
            if k in ['n_estimators','learning_rate','max_depth','min_samples_split','min_samples_leaf',
                     'subsample','max_features','validation_fraction','n_iter_no_change','tol','random_state']
        },
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
        baselines = compute_baselines(y_train, y_val, y_test)
        all_metrics = {
            'train': train_metrics,
            'validation': val_metrics,
            'test': test_metrics,
            'baselines': baselines,
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
