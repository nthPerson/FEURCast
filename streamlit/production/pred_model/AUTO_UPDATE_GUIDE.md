# FUREcast - Automated Data Update & Model Retraining System

## 📋 Overview

This system automatically:
1. ✅ Fetches latest SPLG price data from yfinance (free API)
2. ✅ Updates the historical dataset
3. ✅ Rebuilds all 110 features using the feature engineering pipeline
4. ✅ Retrains the GBR prediction model
5. ✅ Logs training results for performance tracking
6. ✅ Maintains training history for analysis

## 🚀 Quick Start

### Manual Update & Retrain

```bash
cd /home/robert/FEURCast/streamlit/production/pred_model

# Full training (300 estimators, ~2-3 minutes)
python update_and_retrain.py

# Quick training (20 estimators, ~10 seconds)
python update_and_retrain.py --quick

# Force retrain even if no new data
python update_and_retrain.py --force
```

### Automated Scheduling

Set up automated daily updates:

```bash
cd /home/robert/FEURCast/streamlit/production/pred_model
./setup_scheduler.sh
```

This will guide you through setting up a cron job with options:
- **Daily at 6:00 PM ET** (recommended)
- Every 3 hours during market hours
- Twice daily (8am and 6pm)
- Custom schedule

## 📂 System Components

### 1. `data_updater.py`
**Purpose**: Fetches new SPLG data from yfinance

**Key Functions**:
- `check_for_updates()` - Main entry point, checks for and downloads new data
- `fetch_new_splg_data()` - Queries yfinance API for SPLG data
- `update_raw_dataset()` - Appends new data to raw CSV

**Data Source**: Yahoo Finance via yfinance library (free, no API key needed)

**Update Frequency**: Can run as often as desired - automatically detects if new data is available

**Output**: Updates `/home/robert/FEURCast/data/SPLG_history_full.csv`

### 2. `feature_updater.py`
**Purpose**: Rebuilds feature-engineered dataset

**Key Functions**:
- `rebuild_features_from_scratch()` - Runs the full feature engineering pipeline

**Process**:
1. Calls the original `build_splg_features.py` script
2. Generates all 115 features (110 used in model)
3. Applies safe division fixes to prevent inf values
4. Validates output for data quality

**Output**: Updates `/home/robert/FEURCast/data/rich_features_SPLG_history_full.csv`

### 3. `training_logger.py`
**Purpose**: Maintains training history and performance tracking

**Key Functions**:
- `log_training_results()` - Append new training results
- `get_training_history()` - Load historical training data
- `print_training_summary()` - Display formatted history
- `compare_with_previous()` - Calculate performance changes

**Storage**: JSON Lines format in `logs/training_history.jsonl`

**Usage**:
```bash
# View training history
python training_logger.py

# Compare latest with previous
python training_logger.py  # shows comparison automatically
```

### 4. `update_and_retrain.py`
**Purpose**: Main orchestration script

**Workflow**:
```
Check for new data
    ↓
Update raw dataset
    ↓
Rebuild features
    ↓
Train model
    ↓
Log results
    ↓
Display summary
```

**Options**:
- `--quick` - Use 20 estimators (fast, for testing)
- `--force` - Retrain even without new data

### 5. `setup_scheduler.sh`
**Purpose**: Interactive cron job setup

**Features**:
- Pre-configured schedule options
- Custom schedule support
- View/edit/remove existing jobs
- Manual test execution

## 📊 Feature Engineering

### Features Kept (110 total)
All features with importance > 0 are maintained:
- **Moving averages**: MA_5, MA_10, MA_20, MA_50, MA_100, MA_200
- **Price ratios**: price_ma_X_ratio (6 variations)
- **Technical indicators**: RSI, MACD, Bollinger Bands, ATR
- **Returns**: lag returns (1-10 days), rolling returns (5d, 10d, 20d)
- **Volume**: Volume changes, ratios, OBV, VPT
- **Statistical**: Rolling std, variance, skewness, kurtosis
- **Momentum**: Momentum_5, Momentum_10, ROC

### Features Removed (15 total)
Features with 0.0 importance excluded from model:
- `yield`, `month`, `year`, `volume`, `pe_ratio`
- `is_month_start`, `is_month_end`
- `ma_50`, `ma_cross_10_over_50`, `ma_cross_50_over_200`
- `rolling_max_30`, `rolling_var_20`
- `close_current_price_diff`, `log_return`
- `ema_20`, `ema_100`

**Note**: While removed from model, they're still calculated to maintain data consistency.

## 🔄 Update Frequency Recommendations

### Data Updates
**Recommended**: 1-2 times per day
- **Why**: yfinance provides daily close data; intraday updates won't have new information
- **Best times**: After market close (4pm ET) and evening (6-8pm ET)

### Model Retraining
**Recommended**: Once daily after data update
- **Why**: Daily retraining keeps model current without overfitting
- **When**: 6-8pm ET after markets close and data is final

### Optimal Schedule
```bash
# Daily at 6:00 PM ET (Monday-Friday)
0 18 * * 1-5 cd /path/to/pred_model && python update_and_retrain.py

# Or twice daily (morning check + evening update)
0 8,18 * * 1-5 cd /path/to/pred_model && python update_and_retrain.py
```

## 📈 Monitoring & Logging

### Training History
View all training runs:
```bash
python training_logger.py
```

Example output:
```
==================================================================================
Training History (Last 10 runs)
==================================================================================

1. 2025-10-20 18:00:15
   Test R²: -0.0067
   Test RMSE: 0.009932
   Directional Accuracy: 54.31%
   Sharpe Ratio: 4.07
   Win Rate: 90.28%

2. 2025-10-19 18:00:22
   Test R²: -0.0070
   Test RMSE: 0.009945
   Directional Accuracy: 54.00%
   ...
```

### Log Files
All logs stored in `pred_model/logs/`:

- **`data_update_YYYYMMDD.log`** - Data fetching logs
- **`feature_update_YYYYMMDD.log`** - Feature engineering logs
- **`training_log_YYYYMMDD_HHMMSS.txt`** - Model training logs
- **`training_history.jsonl`** - Complete training history (JSON Lines)
- **`cron_update.log`** - Automated cron job output

View logs:
```bash
# Today's data update log
tail -f logs/data_update_$(date +%Y%m%d).log

# Latest training log
ls -t logs/training_log_*.txt | head -1 | xargs cat

# Training history summary
tail -20 logs/training_history.jsonl | jq .
```

## 🧪 Testing the System

### Test Complete Workflow
```bash
cd /home/robert/FEURCast/streamlit/production/pred_model

# Run quick test (takes ~30 seconds)
python update_and_retrain.py --quick --force

# Expected output:
# ✓ Data update check complete
# ✓ Feature engineering complete
# ✓ Model training complete
# ✓ Results logged
```

### Test Individual Components

**Test data fetcher**:
```bash
python data_updater.py
```

**Test feature engineering**:
```bash
python feature_updater.py
```

**Test training logger**:
```bash
python training_logger.py
```

### Verify Output

Check updated files:
```bash
# Raw data
ls -lh ../../../data/SPLG_history_full.csv

# Feature-engineered data
ls -lh ../../../data/rich_features_SPLG_history_full.csv

# Model artifacts
ls -lh models/

# Latest plots
ls -lh plots/
```

## 🔧 Troubleshooting

### Issue: "No new data available"
**Cause**: yfinance has no updates for today yet  
**Solution**: Markets may not be closed yet, or data not published. Run with `--force` to retrain anyway.

### Issue: "Feature engineering failed"
**Cause**: Corrupted raw data or missing dependencies  
**Solution**: 
```bash
# Check raw data
python -c "import pandas as pd; print(pd.read_csv('../../../data/SPLG_history_full.csv').tail())"

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: "Model training failed"
**Cause**: Infinite values or NaN in features  
**Solution**: Feature engineering should handle this automatically. Check logs for details.

### Issue: Cron job not running
**Cause**: Cron misconfiguration or environment issues  
**Solution**:
```bash
# Check cron is running
sudo systemctl status cron

# View cron logs
grep CRON /var/log/syslog

# Test environment
env - /bin/bash -c "cd /path/to/pred_model && python update_and_retrain.py --quick"
```

## 📱 Integration with Streamlit App

The system is fully integrated - the Streamlit app automatically uses the latest model:

1. **Automatic detection**: App checks for model at startup
2. **Hot reload**: Restart app to load newly trained model
3. **Fallback**: Uses simulation if model not available

**After updating model**:
```bash
# Restart Streamlit app
# Press Ctrl+C in terminal, then:
streamlit run app.py
```

## 🎯 Performance Tracking

### Key Metrics to Monitor

**Model Performance**:
- **R² Score**: Should stay close to 0 (stock returns are hard to predict)
- **Directional Accuracy**: Target >50% (better than random)
- **RMSE**: Lower is better (typical: 0.008-0.012)

**Financial Metrics**:
- **Sharpe Ratio**: >2 is good, >3 is excellent
- **Win Rate**: Percentage of profitable predictions
- **Max Drawdown**: Maximum loss from peak

### Performance Degradation
If performance drops significantly:

1. Check training history for trend
2. Review feature importance for changes
3. Consider retuning hyperparameters
4. Verify data quality (check for outliers)

### Model Improvement
To improve performance:

1. Add more features (sector momentum, market sentiment)
2. Try different algorithms (XGBoost, LightGBM)
3. Implement ensemble methods
4. Use longer training history

## 💡 Best Practices

### DO:
- ✅ Run daily updates consistently
- ✅ Monitor training logs regularly
- ✅ Keep backup of well-performing models
- ✅ Test with `--quick` before full training
- ✅ Review performance trends weekly

### DON'T:
- ❌ Update more than 2-3 times daily (no benefit)
- ❌ Skip data quality checks
- ❌ Ignore consistent performance drops
- ❌ Run without logging enabled
- ❌ Delete training history logs

## 📚 Additional Resources

### File Structure
```
pred_model/
├── data_updater.py          # Fetch SPLG data
├── feature_updater.py       # Rebuild features
├── training_logger.py       # Log training results
├── update_and_retrain.py    # Main orchestration script
├── setup_scheduler.sh       # Cron setup helper
├── logs/                    # All log files
│   ├── data_update_*.log
│   ├── feature_update_*.log
│   ├── training_log_*.txt
│   ├── training_history.jsonl
│   └── cron_update.log
├── models/                  # Trained model artifacts
├── plots/                   # Evaluation visualizations
└── scripts/                 # Training scripts
    ├── train_gbr_model.py
    └── evaluate_model.py
```

### Related Documentation
- **Model Training**: `GBR_MODEL_TRAINING_GUIDE.md`
- **Feature Engineering**: `DATA_DICTIONARY.md`
- **Quick Start**: `QUICK_START_MODEL_TRAINING.md`
- **Training Results**: `TRAINING_RESULTS.md`

## 🆘 Support

For issues or questions:
1. Check logs in `pred_model/logs/`
2. Review this documentation
3. Test individual components
4. Check training history for patterns

---

**Last Updated**: 2025-10-20  
**System Version**: 1.0  
**Python**: 3.11+  
**Dependencies**: yfinance, pandas, numpy, scikit-learn, joblib
