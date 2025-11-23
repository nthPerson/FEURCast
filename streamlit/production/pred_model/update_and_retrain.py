#!/usr/bin/env python3
"""
FUREcast - Automated Update and Retrain

This script orchestrates the complete workflow:
1. Check for new SPLG data from yfinance
2. Update the raw dataset
3. Rebuild features from updated data
4. Retrain the prediction model
5. Log training results

Usage:
    python update_and_retrain.py [--force] [--quick] [--tune]
    
Options:
    --force  Force update and retrain even if no new data
    --quick  Use quick training mode (fast, no GridSearch)
    --tune   Perform hyperparameter tuning (GridSearch)
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import json

# Setup paths and imports
MODULE_DIR = Path(__file__).parent
sys.path.insert(0, str(MODULE_DIR))

from data_updater import check_for_updates, logger as data_logger
from feature_updater import rebuild_features_from_scratch, logger as feature_logger
from training_logger import log_training_results, print_training_summary, compare_with_previous
import subprocess


def print_header(title: str) -> None:
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def train_model(quick: bool = False, tune: bool = False) -> bool:
    """
    Run the model training script.
    
    Args:
        quick: If True, use quick training mode
        tune: If True, perform hyperparameter tuning (overrides quick)
    
    Returns:
        True if training successful, False otherwise
    """
    print_header("STEP 3: Model Training")
    
    try:
        # Build command
        script_path = MODULE_DIR / "scripts" / "train_gbr_model.py"
        cmd = [sys.executable, str(script_path)]

        if tune:
            cmd.append("--tune")
            print("Using tuning mode (GridSearch)")
        elif quick:
            cmd.append("--quick")
            print("Using quick training mode (no GridSearch)")
        else:
            print("Using default training mode (quick-enhanced defaults)")

        # Run training
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode != 0:
            print(f"✗ Training failed with exit code {result.returncode}")
            return False
        
        print("✓ Model training complete")
        return True
        
    except Exception as e:
        print(f"✗ Error during training: {e}")
        return False


def evaluate_model() -> bool:
    """
    Run the model evaluation script to generate plots and update metrics.
    
    Returns:
        True if evaluation successful, False otherwise
    """
    print_header("STEP 4: Model Evaluation")
    try:
        script_path = MODULE_DIR / "scripts" / "evaluate_model.py"
        cmd = [sys.executable, str(script_path)]
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            print(f"✗ Evaluation failed with exit code {result.returncode}")
            return False
        print("✓ Model evaluation complete")
        return True
    except Exception as e:
        print(f"✗ Error during evaluation: {e}")
        return False


def load_training_metrics() -> dict:
    """
    Load the latest training metrics from the metrics.json file.
    
    Returns:
        Dictionary with training metrics
    """
    metrics_path = MODULE_DIR / "models" / "metrics.json"
    
    if not metrics_path.exists():
        print(f"Warning: Metrics file not found at {metrics_path}")
        return {}
    
    with open(metrics_path, 'r') as f:
        return json.load(f)


def load_model_metadata() -> dict:
    """
    Load model metadata including hyperparameters and data info.
    
    Returns:
        Dictionary with model metadata
    """
    metadata_path = MODULE_DIR / "models" / "model_metadata.json"
    
    if not metadata_path.exists():
        return {}
    
    with open(metadata_path, 'r') as f:
        return json.load(f)


def main():
    """Main orchestration function"""
    parser = argparse.ArgumentParser(description='Update SPLG data and retrain model')
    parser.add_argument('--force', action='store_true', help='Force update and retrain even if no new data')
    parser.add_argument('--quick', action='store_true', help='Use quick training mode')
    parser.add_argument('--tune', action='store_true', help='Perform hyperparameter tuning (GridSearch)')
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("  FUREcast - Automated Data Update and Model Retraining")
    print("  " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("=" * 80)
    
    # STEP 1: Check for new data
    print_header("STEP 1: Data Update Check")
    
    has_updates, new_data = check_for_updates()
    
    if not has_updates and not args.force:
        print("\n✓ No updates needed - dataset is current")
        print("\nUse --force to retrain anyway")
        return 0
    
    if args.force and not has_updates:
        print("\n⚠️  Forcing update with no new data (--force flag)")
    
    # STEP 2: Rebuild features
    print_header("STEP 2: Feature Engineering")
    
    feature_success = rebuild_features_from_scratch()
    
    if not feature_success:
        print("\n✗ Feature engineering failed - aborting")
        return 1
    
    # STEP 3: Train model
    train_success = train_model(quick=args.quick, tune=args.tune)
    
    if not train_success:
        print("\n✗ Model training failed - aborting")
        return 1

    # STEP 4: Evaluate model
    eval_success = evaluate_model()
    if not eval_success:
        print("\n✗ Evaluation failed - aborting")
        return 1

    # STEP 5: Log results
    print_header("STEP 5: Logging Results")
    try:
        metrics = load_training_metrics()  # Reload after evaluation (financial metrics added)
        metadata = load_model_metadata()
        metadata['update_info'] = {
            'had_new_data': has_updates,
            'forced_update': args.force,
            'quick_training': args.quick,
            'new_records': len(new_data) if has_updates and new_data is not None else 0,
            'evaluation_completed': eval_success
        }
        log_training_results(metrics, metadata)
        print("✓ Results logged successfully")
    except Exception as e:
        print(f"⚠️  Warning: Could not log results: {e}")

    # STEP 6: Training Summary
    print_header("Training Summary")
    if metrics:
        test = metrics.get('test', {})
        fin = metrics.get('financial_metrics', {})
        print(f"Test R²: {test.get('r2', 'N/A'):.4f}")
        print(f"Test RMSE: {test.get('rmse', 'N/A'):.6f}")
        print(f"Test MAE: {test.get('mae', 'N/A'):.6f}")
        print(f"Directional Accuracy: {test.get('directional_accuracy', 'N/A'):.2f}%")
        if fin:
            print("\nFinancial Metrics:")
            print(f"  Sharpe Ratio: {fin.get('sharpe_ratio', 0):.2f}")
            print(f"  Max Drawdown: {fin.get('max_drawdown', 0):.2%}")
            print(f"  Win Rate: {fin.get('win_rate', 0):.2%}")
            print(f"  Profit Factor: {fin.get('profit_factor', 0):.2f}")

    changes = compare_with_previous()
    if changes:
        print(f"\nChange from Previous Training:")
        print(f"  R² change: {changes['r2_change']:+.4f}")
        print(f"  RMSE change: {changes['rmse_change']:+.6f}")
        print(f"  Directional Accuracy change: {changes['directional_accuracy_change']:+.2f}%")

    print("\n" + "=" * 80)
    print("  ✓ Update, Retrain, and Evaluate Complete")
    print("=" * 80 + "\n")
    print("Next steps:")
    print("  • Restart Streamlit app to use updated model")
    print("  • View training history: python training_logger.py")
    print("  • Check plots: ls -lh plots/")
    print("  • View logs: ls -lh logs/\n")
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
