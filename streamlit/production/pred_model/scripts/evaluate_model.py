"""
FUREcast - Model Evaluation and Visualization Script

This script loads a trained GBR model and generates comprehensive evaluation metrics
and visualizations for academic documentation.

Usage:
    python evaluate_model.py
"""

import pandas as pd
import numpy as np
import joblib
import json
import logging
from datetime import datetime
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PRED_MODEL_DIR = SCRIPT_DIR.parent
DATA_PATH = PRED_MODEL_DIR.parent.parent.parent / "data" / "rich_features_SPLG_history_full.csv"
MODELS_DIR = PRED_MODEL_DIR / "models"
PLOTS_DIR = PRED_MODEL_DIR / "plots"
LOGS_DIR = PRED_MODEL_DIR / "logs"

# Ensure directories exist
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = LOGS_DIR / f"evaluation_log_{timestamp}.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


def load_model_artifacts():
    """Load trained model and associated artifacts."""
    logger.info("Loading model artifacts...")
    
    model = joblib.load(MODELS_DIR / "gbr_model.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")
    
    with open(MODELS_DIR / "metrics.json", 'r') as f:
        metrics = json.load(f)
    
    logger.info("Model artifacts loaded successfully")
    return model, scaler, feature_names, metrics


def load_and_prepare_data(feature_names):
    """Load data and recreate train/val/test splits with cleaning."""
    logger.info("Loading data...")
    
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    
    metadata_cols = ['date', 'company_name', 'ticker']
    target_cols = ['target_close_t1', 'target_return_t1']
    
    X = df[feature_names].copy()
    y = df['target_return_t1'].copy()
    dates = df['date'].copy()
    
    # ====== DATA CLEANING (SAME AS TRAINING) ======
    logger.info("Performing data quality checks...")
    
    # 1. Replace infinite values with NaN
    inf_counts = np.isinf(X).sum()
    if inf_counts.sum() > 0:
        logger.warning(f"Found {inf_counts.sum()} infinite values across {(inf_counts > 0).sum()} features")
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 2. Drop rows with NaN
    nan_counts = X.isna().sum()
    nan_target = y.isna().sum()
    
    if nan_counts.sum() > 0 or nan_target > 0:
        logger.warning(f"Found {nan_counts.sum()} NaN values in features, {nan_target} in target")
        logger.info("Dropping rows with NaN values...")
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]
        dates = dates[valid_idx]
        logger.info(f"Remaining records after cleaning: {len(X)}")
    
    # Verify no issues remain
    assert not np.isinf(X.values).any(), "Infinite values still present after cleaning"
    assert not X.isna().any().any(), "NaN values still present after cleaning"
    assert not y.isna().any(), "NaN values in target after cleaning"
    
    logger.info("✓ Data quality checks passed")
    
    # Recreate splits (same as training)
    n = len(X)
    train_size = int(0.70 * n)
    val_size = int(0.15 * n)
    
    X_train = X.iloc[:train_size]
    X_val = X.iloc[train_size:train_size + val_size]
    X_test = X.iloc[train_size + val_size:]
    
    y_train = y.iloc[:train_size]
    y_val = y.iloc[train_size:train_size + val_size]
    y_test = y.iloc[train_size + val_size:]
    
    dates_train = dates.iloc[:train_size]
    dates_val = dates.iloc[train_size:train_size + val_size]
    dates_test = dates.iloc[train_size + val_size:]
    
    logger.info(f"Data loaded: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    return (X_train, X_val, X_test, y_train, y_val, y_test, 
            dates_train, dates_val, dates_test)


def plot_training_progress(model):
    """Plot training loss curve."""
    logger.info("Generating training progress plot...")
    
    train_score = model.train_score_
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_score, label='Training Loss', linewidth=2, color='#2E86AB')
    plt.xlabel('Boosting Iteration', fontsize=12)
    plt.ylabel('Loss (Negative MSE)', fontsize=12)
    plt.title('GBR Training Progress', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plot_path = PLOTS_DIR / 'training_progress.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved training progress plot to {plot_path}")


def plot_predictions_vs_actuals(y_train, y_val, y_test, 
                                y_train_pred, y_val_pred, y_test_pred):
    """Plot predictions vs actuals for all sets."""
    logger.info("Generating predictions vs actuals plot...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    datasets = [
        (y_train, y_train_pred, 'Training', '#2E86AB'),
        (y_val, y_val_pred, 'Validation', '#A23B72'),
        (y_test, y_test_pred, 'Test', '#F18F01')
    ]
    
    for ax, (y_true, y_pred, title, color) in zip(axes, datasets):
        ax.scatter(y_true, y_pred, alpha=0.5, s=10, color=color)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 
                'r--', linewidth=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Return', fontsize=11)
        ax.set_ylabel('Predicted Return', fontsize=11)
        ax.set_title(f'{title} Set', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Add R² score
        r2 = r2_score(y_true, y_pred)
        ax.text(0.05, 0.95, f'R² = {r2:.4f}', 
                transform=ax.transAxes, 
                fontsize=10, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plot_path = PLOTS_DIR / 'predictions_vs_actuals.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved predictions vs actuals plot to {plot_path}")


def plot_residuals(y_train, y_val, y_test, 
                   y_train_pred, y_val_pred, y_test_pred):
    """Plot residuals for all sets."""
    logger.info("Generating residuals plot...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    datasets = [
        (y_train, y_train_pred, 'Training', '#2E86AB'),
        (y_val, y_val_pred, 'Validation', '#A23B72'),
        (y_test, y_test_pred, 'Test', '#F18F01')
    ]
    
    for ax, (y_true, y_pred, title, color) in zip(axes, datasets):
        residuals = y_true - y_pred
        ax.scatter(y_pred, residuals, alpha=0.5, s=10, color=color)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        
        ax.set_xlabel('Predicted Return', fontsize=11)
        ax.set_ylabel('Residuals', fontsize=11)
        ax.set_title(f'{title} Set Residuals', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add residual statistics
        mean_res = residuals.mean()
        std_res = residuals.std()
        ax.text(0.05, 0.95, f'Mean: {mean_res:.6f}\nStd: {std_res:.6f}', 
                transform=ax.transAxes, 
                fontsize=9, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plot_path = PLOTS_DIR / 'residuals.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved residuals plot to {plot_path}")


def plot_feature_importance(model, feature_names):
    """Plot top 30 feature importances."""
    logger.info("Generating feature importance plot...")
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True).tail(30)
    
    plt.figure(figsize=(10, 12))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_importance)))
    plt.barh(range(len(feature_importance)), 
             feature_importance['importance'],
             color=colors)
    plt.yticks(range(len(feature_importance)), feature_importance['feature'])
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title('Top 30 Feature Importances', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_path = PLOTS_DIR / 'feature_importance.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved feature importance plot to {plot_path}")


def plot_time_series_predictions(dates_test, y_test, y_test_pred):
    """Plot predictions over time on test set."""
    logger.info("Generating time series predictions plot...")
    
    plt.figure(figsize=(16, 8))
    plt.plot(dates_test, y_test.values, label='Actual Returns', 
             alpha=0.7, linewidth=1.5, color='#2E86AB')
    plt.plot(dates_test, y_test_pred, label='Predicted Returns', 
             alpha=0.7, linewidth=1.5, color='#F18F01')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Return', fontsize=12)
    plt.title('Model Predictions vs Actual Returns (Test Set)', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plot_path = PLOTS_DIR / 'time_series_predictions.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved time series predictions plot to {plot_path}")


def plot_cumulative_returns(dates_test, y_test, y_test_pred):
    """Plot cumulative returns comparison."""
    logger.info("Generating cumulative returns plot...")
    
    cumulative_actual = (1 + y_test).cumprod()
    cumulative_predicted = (1 + pd.Series(y_test_pred, index=y_test.index)).cumprod()
    
    plt.figure(figsize=(14, 7))
    plt.plot(dates_test, cumulative_actual, label='Actual Cumulative Return', 
             linewidth=2, color='#2E86AB')
    plt.plot(dates_test, cumulative_predicted, label='Strategy Cumulative Return', 
             linewidth=2, color='#F18F01')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return (Starting from 1.0)', fontsize=12)
    plt.title('Cumulative Returns: Actual vs Model-Based Strategy', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plot_path = PLOTS_DIR / 'cumulative_returns.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved cumulative returns plot to {plot_path}")


def plot_error_distribution(y_train, y_val, y_test, 
                            y_train_pred, y_val_pred, y_test_pred):
    """Plot error distributions."""
    logger.info("Generating error distribution plot...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    datasets = [
        (y_train, y_train_pred, 'Training', '#2E86AB'),
        (y_val, y_val_pred, 'Validation', '#A23B72'),
        (y_test, y_test_pred, 'Test', '#F18F01')
    ]
    
    for ax, (y_true, y_pred, title, color) in zip(axes, datasets):
        errors = y_true - y_pred
        ax.hist(errors, bins=50, alpha=0.7, edgecolor='black', color=color)
        ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Prediction Error', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{title} Set Error Distribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        mean_err = errors.mean()
        std_err = errors.std()
        ax.text(0.70, 0.95, f'Mean: {mean_err:.6f}\nStd: {std_err:.6f}', 
                transform=ax.transAxes, 
                fontsize=9, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plot_path = PLOTS_DIR / 'error_distribution.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved error distribution plot to {plot_path}")


def calculate_financial_metrics(y_test, y_test_pred):
    """Calculate additional financial performance metrics."""
    logger.info("Calculating financial metrics...")
    
    # Sharpe Ratio (assuming 252 trading days, 2% risk-free rate)
    returns = pd.Series(y_test_pred)
    excess_return = returns.mean() * 252 - 0.02
    volatility = returns.std() * np.sqrt(252)
    sharpe = excess_return / volatility if volatility > 0 else 0
    
    # Max Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = ((cumulative - running_max) / running_max)
    max_drawdown = drawdown.min()
    
    # Profit Factor (simple simulation)
    wins = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    profit_factor = wins / losses if losses > 0 else np.inf
    
    # Win Rate
    win_rate = (returns > 0).mean()
    
    financial_metrics = {
        'sharpe_ratio': float(sharpe),
        'max_drawdown': float(max_drawdown),
        'profit_factor': float(profit_factor),
        'win_rate': float(win_rate),
        'avg_win': float(returns[returns > 0].mean() if (returns > 0).any() else 0),
        'avg_loss': float(returns[returns < 0].mean() if (returns < 0).any() else 0)
    }
    
    logger.info(f"  Sharpe Ratio: {sharpe:.4f}")
    logger.info(f"  Max Drawdown: {max_drawdown:.2%}")
    logger.info(f"  Profit Factor: {profit_factor:.2f}")
    logger.info(f"  Win Rate: {win_rate:.2%}")
    
    return financial_metrics


def main():
    """Main evaluation pipeline."""
    logger.info("="*80)
    logger.info("FUREcast GBR Model Evaluation")
    logger.info("="*80)
    
    try:
        # Load model
        model, scaler, feature_names, metrics = load_model_artifacts()
        
        # Load data
        (X_train, X_val, X_test, y_train, y_val, y_test, 
         dates_train, dates_val, dates_test) = load_and_prepare_data(feature_names)
        
        # Scale features
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Generate predictions
        y_train_pred = model.predict(X_train_scaled)
        y_val_pred = model.predict(X_val_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Generate all plots
        logger.info("\n" + "="*80)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("="*80)
        
        plot_training_progress(model)
        plot_predictions_vs_actuals(y_train, y_val, y_test, 
                                   y_train_pred, y_val_pred, y_test_pred)
        plot_residuals(y_train, y_val, y_test, 
                      y_train_pred, y_val_pred, y_test_pred)
        plot_feature_importance(model, feature_names)
        plot_time_series_predictions(dates_test, y_test, y_test_pred)
        plot_cumulative_returns(dates_test, y_test, y_test_pred)
        plot_error_distribution(y_train, y_val, y_test, 
                               y_train_pred, y_val_pred, y_test_pred)
        
        # Calculate financial metrics
        logger.info("\n" + "="*80)
        logger.info("FINANCIAL METRICS")
        logger.info("="*80)
        financial_metrics = calculate_financial_metrics(y_test, y_test_pred)
        
        # Update metrics file with financial metrics
        metrics['financial_metrics'] = financial_metrics
        metrics_path = MODELS_DIR / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info("\n" + "="*80)
        logger.info("EVALUATION COMPLETE")
        logger.info("="*80)
        logger.info(f"All plots saved to: {PLOTS_DIR}")
        logger.info(f"Updated metrics saved to: {metrics_path}")
        logger.info(f"Log file: {log_filename}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed with error: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
