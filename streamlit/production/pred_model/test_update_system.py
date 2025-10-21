#!/usr/bin/env python3
"""
Quick test script for the automated update system.
Tests each component individually to verify proper functionality.
"""

import sys
from pathlib import Path

# Setup paths
MODULE_DIR = Path(__file__).parent
sys.path.insert(0, str(MODULE_DIR))

def test_data_updater():
    """Test data fetching from yfinance"""
    print("\n" + "=" * 80)
    print("TEST 1: Data Updater")
    print("=" * 80)
    
    try:
        from data_updater import get_latest_date_in_dataset, fetch_new_splg_data
        from datetime import datetime, timedelta
        
        # Check current dataset date
        latest_date = get_latest_date_in_dataset()
        if latest_date:
            print(f"✓ Current dataset ends at: {latest_date.date()}")
        else:
            print("⚠️  No existing dataset found")
        
        # Test fetching recent data (last 5 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        
        print(f"\nTesting data fetch from {start_date.date()} to {end_date.date()}...")
        new_data = fetch_new_splg_data(start_date=start_date, end_date=end_date)
        
        if new_data is not None and not new_data.empty:
            print(f"✓ Fetched {len(new_data)} records")
            print(f"  Columns: {list(new_data.columns)}")
            print(f"  Sample:\n{new_data.head(2)}")
            return True
        else:
            print("⚠️  No data returned (may be weekend/holiday)")
            return True  # Not a failure, just no market data
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_engineering():
    """Test feature engineering can run"""
    print("\n" + "=" * 80)
    print("TEST 2: Feature Engineering")
    print("=" * 80)
    
    try:
        # Check if build script exists
        fe_path = MODULE_DIR.parent.parent.parent / "wrangling" / "pred_model_feature_engineering" / "build_splg_features.py"
        
        if not fe_path.exists():
            print(f"✗ Feature engineering script not found at {fe_path}")
            return False
        
        print(f"✓ Feature engineering script found")
        print(f"  Location: {fe_path}")
        
        # Check if data file exists
        data_path = MODULE_DIR.parent.parent.parent / "data" / "rich_features_SPLG_history_full.csv"
        if data_path.exists():
            import pandas as pd
            df = pd.read_csv(data_path)
            print(f"✓ Current feature dataset: {df.shape[0]} rows, {df.shape[1]} columns")
            print(f"  Latest date: {df['date'].max()}")
        else:
            print("⚠️  No feature dataset found yet")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_training_logger():
    """Test training logger"""
    print("\n" + "=" * 80)
    print("TEST 3: Training Logger")
    print("=" * 80)
    
    try:
        from training_logger import get_training_history, print_training_summary
        
        history = get_training_history(limit=5)
        
        if history:
            print(f"✓ Found {len(history)} training runs in history")
            print("\nMost recent runs:")
            print_training_summary(limit=3)
        else:
            print("⚠️  No training history yet (will be created after first training)")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_model_artifacts():
    """Test model files exist"""
    print("\n" + "=" * 80)
    print("TEST 4: Model Artifacts")
    print("=" * 80)
    
    try:
        models_dir = MODULE_DIR / "models"
        
        if not models_dir.exists():
            print("⚠️  Models directory not found (run training first)")
            return True
        
        required_files = ['gbr_model.pkl', 'scaler.pkl', 'feature_names.pkl', 'metrics.json']
        
        for file in required_files:
            file_path = models_dir / file
            if file_path.exists():
                size = file_path.stat().st_size
                print(f"✓ {file}: {size:,} bytes")
            else:
                print(f"✗ Missing: {file}")
        
        # Try loading model
        try:
            from predict import load_model
            model_bundle = load_model()
            print(f"\n✓ Model loaded successfully")
            print(f"  Features: {len(model_bundle['feature_names'])}")
            print(f"  Test R²: {model_bundle['metrics']['test']['r2']:.4f}")
        except Exception as e:
            print(f"⚠️  Could not load model: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_dependencies():
    """Test required dependencies"""
    print("\n" + "=" * 80)
    print("TEST 5: Dependencies")
    print("=" * 80)
    
    dependencies = {
        'yfinance': 'Data fetching',
        'pandas': 'Data manipulation',
        'numpy': 'Numerical computing',
        'sklearn': 'Machine learning',
        'joblib': 'Model persistence'
    }
    
    all_ok = True
    for module, purpose in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {module:15s} - {purpose}")
        except ImportError:
            print(f"✗ {module:15s} - MISSING")
            all_ok = False
    
    return all_ok


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("  FUREcast - Automated Update System Test Suite")
    print("=" * 80)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Data Updater", test_data_updater),
        ("Feature Engineering", test_feature_engineering),
        ("Training Logger", test_training_logger),
        ("Model Artifacts", test_model_artifacts)
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("  Test Summary")
    print("=" * 80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:10s} {name}")
    
    print("\n" + "=" * 80)
    print(f"  Results: {passed}/{total} tests passed")
    print("=" * 80)
    
    if passed == total:
        print("\n✓ All tests passed! System is ready.")
        print("\nNext steps:")
        print("  1. Run: python update_and_retrain.py --quick")
        print("  2. Set up scheduling: ./setup_scheduler.sh")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
