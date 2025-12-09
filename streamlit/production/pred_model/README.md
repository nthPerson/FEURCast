# FUREcast SPYM Prediction Model (Legacy SPLG Historical Data)

This directory contains the GradientBoostingRegressor (GBR) model for predicting next-day returns on the SPYM ETF (formerly SPLG). Historical data was collected under the SPLG ticker and filenames (e.g., `SPLG_history_full.csv`, `rich_features_SPLG_history_full.csv`) are retained for backward compatibility, but all future updates use SPYM exclusively.
**Dataset**: `rich_features_SPLG_history_full.csv` (legacy SPLG naming; includes SPYM appended rows after ticker migration)

## Directory Structure
# FUREcast SPLG Prediction Model

This directory contains the GradientBoostingRegressor (GBR) model for predicting next-day returns on the SPLG ETF, along with training scripts, evaluation tools, and inference interfaces.

## Directory Structure

```
pred_model/
├── models/              # Trained model artifacts (generated after training)
│   ├── gbr_model.pkl           # Trained GBR model
│   ├── scaler.pkl              # Feature scaler (StandardScaler)
│   ├── feature_names.pkl       # List of feature names
│   ├── feature_importance.csv  # Feature importance rankings
│   ├── metrics.json            # Performance metrics
│   └── model_metadata.json     # Model version and training info
├── plots/               # Evaluation visualizations (generated after evaluation)
│   ├── training_progress.png
│   ├── predictions_vs_actuals.png
│   ├── residuals.png
│   ├── feature_importance.png
│   ├── time_series_predictions.png
│   ├── cumulative_returns.png
│   └── error_distribution.png
├── logs/                # Training and evaluation logs
│   ├── training_log_YYYYMMDD_HHMMSS.txt
│   └── evaluation_log_YYYYMMDD_HHMMSS.txt
├── scripts/             # Training and evaluation scripts
│   ├── train_gbr_model.py      # Model training script
│   └── evaluate_model.py       # Model evaluation script
├── predict.py           # Prediction interface module
├── get_latest_features.py  # Helper to extract latest features
└── README.md            # This file
```

## Quick Start

### 1. Train the Model

**Quick training (recommended for first run):**
```bash
cd /home/robert/FEURCast/streamlit/production/pred_model
python scripts/train_gbr_model.py --quick
```

**With hyperparameter tuning (slower but potentially better performance):**
```bash
python scripts/train_gbr_model.py --tune
```

**Training Output:**
- Trained model saved to `models/gbr_model.pkl`
- Scaler saved to `models/scaler.pkl`
- Feature names saved to `models/feature_names.pkl`
- Performance metrics saved to `models/metrics.json`
- Feature importance saved to `models/feature_importance.csv`
- Training log saved to `logs/training_log_YYYYMMDD_HHMMSS.txt`

### 2. Evaluate the Model

**Generate comprehensive evaluation metrics and visualizations:**
```bash
python scripts/evaluate_model.py
```

**Evaluation Output:**
- 7 visualization plots saved to `plots/`
- Updated metrics with financial metrics in `models/metrics.json`
- Evaluation log saved to `logs/evaluation_log_YYYYMMDD_HHMMSS.txt`

### 3. Make Predictions

**Using the prediction module in Python:**
```python
from predict import load_model, predict_with_explanation
from get_latest_features import get_latest_features

# Load model
model_bundle = load_model()

# Get latest features
features = get_latest_features(1)

# Make prediction
result = predict_with_explanation(model_bundle, features)

print(f"Predicted Return: {result['predicted_return']:.4%}")
print(f"Direction: {result['direction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"\nTop Features:")
for feat in result['top_features']:
    print(f"  {feat['name']}: {feat['importance']:.4f}")
```

**Testing the prediction interface:**
```bash
python predict.py
```

## Model Details

### Training Data
- **Dataset**: `rich_features_SPLG_history_full.csv`
- **Records**: 4,748 daily observations
- **Date Range**: 2006-09-01 to 2025-09-24 (~19 years)
- **Features**: 112 engineered features across 10 categories
- **Target**: `target_return_t1` (next-day percentage return)

### Feature Categories
1. **Original Price/Volume** (11 features): OHLCV, PE ratio, yield
2. **Price Relationships** (7 features): Spreads, ratios, differences
3. **Returns** (7 features): 1d/5d/10d/20d returns, log returns
4. **Trend** (30 features): Moving averages, ratios, slopes, crosses
5. **Momentum** (7 features): RSI, MACD, ROC, momentum
6. **Volatility** (13 features): Rolling std/var, ATR, Bollinger Bands
7. **Volume** (6 features): Volume MAs, ratios, OBV, VPT
8. **Lags** (15 features): Autoregressive signals (1-10 days)
9. **Rolling Stats** (15 features): Max, min, skew, kurtosis
10. **Calendar** (8 features): Day of week, month, quarter, year

### Model Architecture
- **Algorithm**: GradientBoostingRegressor (scikit-learn)
- **Hyperparameters** (quick mode):
  - n_estimators: 300
  - learning_rate: 0.05
  - max_depth: 4
  - min_samples_split: 10
  - min_samples_leaf: 4
  - subsample: 0.8
  - max_features: 'sqrt'
  - validation_fraction: 0.1
  - n_iter_no_change: 20 (early stopping)
- **Random Seed**: 42 (for reproducibility)

### Train/Validation/Test Split
- **Train**: 70% (earliest data)
- **Validation**: 15% (middle data)
- **Test**: 15% (most recent data)
- **Split Method**: Time-based (chronological order preserved)

### Performance Metrics

**Regression Metrics:**
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score
- Mean Absolute Percentage Error (MAPE)

**Financial Metrics:**
- Directional Accuracy (% of correct up/down predictions)
- Sharpe Ratio (risk-adjusted returns)
- Maximum Drawdown (largest peak-to-trough decline)
- Profit Factor (gross profit / gross loss)
- Win Rate (% of positive returns)

*Note: Actual performance metrics will be populated after training.*

## Usage in Streamlit Application

The trained model is automatically integrated into the Streamlit app via `tools.py`:

```python
# In tools.py
prediction = predict_splg(use_real_model=True)

# Returns:
# {
#   'predicted_return': 0.0035,  # 0.35% return
#   'direction': 'up',
#   'confidence': 0.72,           # 72% confidence
#   'top_features': [
#     {'name': 'ma_20', 'importance': 0.23, 'value': 45.67},
#     {'name': 'rsi_14', 'importance': 0.18, 'value': 58.3},
#     ...
#   ]
# }
```

The app automatically:
- Detects if the trained model exists
- Falls back to simulated predictions if model is not found
- Displays actual feature importances from the trained model
- Uses real model performance metrics in the UI

## Requirements

### Python Packages
```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
```

Install with:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib
```

## Model Training Pipeline

### Step 1: Data Loading & Preprocessing
- Load CSV with 4,748 records
- Parse dates and sort chronologically
- Separate features, target, and metadata
- Verify no NaN values

### Step 2: Train/Val/Test Split
- Create time-based splits (70/15/15)
- Preserve chronological order
- Log date ranges for each set

### Step 3: Feature Scaling
- Fit StandardScaler on training data
- Transform train/val/test sets
- Save scaler for production use

### Step 4: Model Training
- Train GBR with recommended hyperparameters
- Optional: GridSearch with TimeSeriesSplit for tuning
- Use early stopping to prevent overfitting
- Log training progress

### Step 5: Evaluation
- Compute metrics on train/val/test sets
- Calculate directional accuracy
- Generate feature importance rankings
- Log all results

### Step 6: Persistence
- Save model, scaler, and feature names
- Export metrics as JSON
- Save feature importance as CSV
- Create model metadata with version info

## Model Evaluation Visualizations

### 1. Training Progress
- Shows loss curve over boosting iterations
- Helps identify convergence and overfitting

### 2. Predictions vs Actuals
- Scatter plots for train/val/test sets
- Perfect prediction line for reference
- R² scores displayed

### 3. Residuals
- Residual scatter plots for each set
- Check for systematic bias
- Verify homoscedasticity

### 4. Feature Importance
- Horizontal bar chart of top 30 features
- Identifies key prediction drivers
- Helps with model interpretation

### 5. Time Series Predictions
- Line plot of predictions vs actuals over time (test set)
- Shows temporal prediction quality
- Helps identify periods of good/poor performance

### 6. Cumulative Returns
- Compares actual vs strategy returns
- Shows potential trading performance
- Assumes simple long/short strategy

### 7. Error Distribution
- Histograms of prediction errors
- Check for normality of residuals
- Identify outliers

## Academic Documentation

For comprehensive model training guidance, see:
- `/wrangling/pred_model_feature_engineering/GBR_MODEL_TRAINING_GUIDE.md`
- `/wrangling/pred_model_feature_engineering/DATA_DICTIONARY.md`
- `/wrangling/pred_model_feature_engineering/QUICK_START_MODEL_TRAINING.md`

## Troubleshooting

### Model Not Found Error
**Problem**: `FileNotFoundError: Model artifacts not found`

**Solution**: Train the model first:
```bash
python scripts/train_gbr_model.py --quick
```

### Import Errors in Streamlit
**Problem**: `Import "predict" could not be resolved`

**Solution**: This is expected before training. The app will fall back to simulated predictions until the model is trained.

### Low Model Performance
**Problem**: R² score < 0.1 or directional accuracy < 52%

**Solutions**:
1. Try hyperparameter tuning: `python scripts/train_gbr_model.py --tune`
2. Check for data quality issues in the CSV
3. Review feature engineering pipeline
4. Consider ensemble methods or alternative algorithms

### Memory Issues
**Problem**: Out of memory during training

**Solutions**:
1. Reduce `n_estimators` in training script
2. Use fewer features (feature selection)
3. Reduce training data size
4. Run on machine with more RAM

## Model Versioning

Each trained model includes metadata with:
- Training timestamp
- Data hash (for reproducibility)
- Hyperparameters used
- Performance metrics
- Feature names and count

Models are saved with version info in `models/model_metadata.json`.

## Next Steps

After training and evaluation:

1. **Review Performance**: Check `models/metrics.json` and plots in `plots/`
2. **Iterate if Needed**: Tune hyperparameters or features based on results
3. **Deploy to App**: Restart Streamlit app to use real model
4. **Monitor**: Track prediction accuracy over time
5. **Retrain**: Periodically retrain with updated data

## Contact & Support

For questions or issues:
- Review training guide: `GBR_MODEL_TRAINING_GUIDE.md`
- Check logs in `logs/` directory
- Examine metrics in `models/metrics.json`

---

**Last Updated**: October 19, 2025  
**Model Version**: Will be populated after first training  
**Status**: Ready for training
