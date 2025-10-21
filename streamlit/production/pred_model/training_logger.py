"""
FUREcast - Training Results Logger

This module maintains a log of all model training runs with their metrics.
It provides functions to log training results and query historical performance.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Setup paths
MODULE_DIR = Path(__file__).parent
LOGS_DIR = MODULE_DIR / "logs"
TRAINING_LOG_FILE = LOGS_DIR / "training_history.jsonl"
LOGS_DIR.mkdir(exist_ok=True)


def log_training_results(metrics: Dict[str, Any], 
                         metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Append training results to the training history log.
    
    Args:
        metrics: Dictionary of model metrics from training
        metadata: Optional additional metadata (hyperparameters, data info, etc.)
    """
    # Create log entry
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'date': datetime.now().strftime('%Y-%m-%d'),
        'time': datetime.now().strftime('%H:%M:%S'),
        'metrics': metrics,
        'metadata': metadata or {}
    }
    
    # Append to log file (JSON Lines format)
    with open(TRAINING_LOG_FILE, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
    
    print(f"✓ Logged training results to {TRAINING_LOG_FILE}")


def get_training_history(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load training history from log file.
    
    Args:
        limit: Maximum number of recent entries to return (None = all)
    
    Returns:
        List of training log entries, most recent first
    """
    if not TRAINING_LOG_FILE.exists():
        return []
    
    entries = []
    with open(TRAINING_LOG_FILE, 'r') as f:
        for line in f:
            entries.append(json.loads(line.strip()))
    
    # Reverse to get most recent first
    entries.reverse()
    
    if limit:
        entries = entries[:limit]
    
    return entries


def get_latest_training_results() -> Optional[Dict[str, Any]]:
    """
    Get the most recent training results.
    
    Returns:
        Dictionary with latest training results, or None if no history
    """
    history = get_training_history(limit=1)
    return history[0] if history else None


def get_training_summary() -> pd.DataFrame:
    """
    Get a summary DataFrame of all training runs.
    
    Returns:
        DataFrame with key metrics from each training run
    """
    history = get_training_history()
    
    if not history:
        return pd.DataFrame()
    
    # Extract key metrics into flat structure
    rows = []
    for entry in history:
        metrics = entry.get('metrics', {})
        test_metrics = metrics.get('test', {})
        
        row = {
            'timestamp': entry['timestamp'],
            'date': entry['date'],
            'time': entry['time'],
            'test_r2': test_metrics.get('r2', None),
            'test_rmse': test_metrics.get('rmse', None),
            'test_mae': test_metrics.get('mae', None),
            'test_directional_accuracy': test_metrics.get('directional_accuracy', None),
            'n_samples': entry.get('metadata', {}).get('n_samples', None),
            'n_features': entry.get('metadata', {}).get('n_features', None)
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def print_training_summary(limit: int = 10) -> None:
    """
    Print a formatted summary of recent training runs.
    
    Args:
        limit: Number of recent runs to display
    """
    history = get_training_history(limit=limit)
    
    if not history:
        print("No training history available")
        return
    
    print("=" * 90)
    print(f"Training History (Last {min(len(history), limit)} runs)")
    print("=" * 90)
    
    for i, entry in enumerate(history, 1):
        metrics = entry.get('metrics', {})
        test = metrics.get('test', {})
        
        print(f"\n{i}. {entry['date']} {entry['time']}")
        print(f"   Test R²: {test.get('r2', 'N/A'):.4f}")
        print(f"   Test RMSE: {test.get('rmse', 'N/A'):.6f}")
        print(f"   Directional Accuracy: {test.get('directional_accuracy', 'N/A'):.2f}%")
        
        # Financial metrics if available
        if 'sharpe_ratio' in test:
            print(f"   Sharpe Ratio: {test.get('sharpe_ratio', 'N/A'):.2f}")
            print(f"   Win Rate: {test.get('win_rate', 'N/A'):.2f}%")
    
    print("\n" + "=" * 90)


def compare_with_previous() -> Optional[Dict[str, float]]:
    """
    Compare latest training results with previous run.
    
    Returns:
        Dictionary with performance changes, or None if < 2 runs
    """
    history = get_training_history(limit=2)
    
    if len(history) < 2:
        return None
    
    latest = history[0]['metrics']['test']
    previous = history[1]['metrics']['test']
    
    changes = {
        'r2_change': latest.get('r2', 0) - previous.get('r2', 0),
        'rmse_change': latest.get('rmse', 0) - previous.get('rmse', 0),
        'directional_accuracy_change': latest.get('directional_accuracy', 0) - previous.get('directional_accuracy', 0)
    }
    
    return changes


if __name__ == "__main__":
    # Display training history
    print_training_summary(limit=10)
    
    # Show comparison with previous if available
    changes = compare_with_previous()
    if changes:
        print("\nChange from Previous Training:")
        print(f"  R² change: {changes['r2_change']:+.4f}")
        print(f"  RMSE change: {changes['rmse_change']:+.6f}")
        print(f"  Directional Accuracy change: {changes['directional_accuracy_change']:+.2f}%")
