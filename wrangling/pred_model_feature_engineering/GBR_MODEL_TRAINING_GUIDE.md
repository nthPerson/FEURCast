# Gradient Boosting Regressor (GBR) Model Training Guide

## Document Purpose

This document provides comprehensive documentation for training a Gradient Boosting Regressor (GBR) model on the SPLG ETF feature-engineered dataset. It is designed to guide AI agents and human researchers through the complete model training pipeline for the FEURCast academic project.

---

## Table of Contents

1. [Dataset Overview](#dataset-overview)
2. [Feature Description](#feature-description)
3. [Data Generation Process](#data-generation-process)
4. [Model Training Pipeline](#model-training-pipeline)
5. [Performance Metrics](#performance-metrics)
6. [Integration with Streamlit Application](#integration-with-streamlit-application)
7. [Reproducibility Guidelines](#reproducibility-guidelines)

---

## Dataset Overview

### File Information
- **Filename**: `rich_features_SPLG_history_full.csv`
- **Location**: `/home/robert/FEURCast/wrangling/pred_model_feature_engineering/data_out/`
- **Size**: 4,748 records × 115 columns
- **Date Range**: September 1, 2006 to September 24, 2025 (~19 years)
- **Temporal Resolution**: Daily

### Source Data
The dataset is derived from SPLG (SPDR Portfolio S&P 500 ETF) historical price data, originally containing:
- Date, Company Name, Ticker
- Current Price, Open, Close, High, Low, Volume
- PE Ratio, Yield, Beta (note: Beta column was entirely NaN and removed during processing)

### Target Variables

The dataset includes two target variables for next-day prediction:

1. **`target_close_t1`** (float64)
   - Next-day closing price (absolute value)
   - Range: $7.94 to $78.68
   - Mean: $30.49
   - Use case: Direct price prediction

2. **`target_return_t1`** (float64)
   - Next-day percentage return: (Close_{t+1} / Close_t) - 1
   - Range: -11.68% to +9.53%
   - Mean: +0.039% (average daily return)
   - Use case: Return prediction, classification (up/down), risk modeling

---

## Feature Description

The dataset contains **115 columns** organized into the following categories:

### 1. Metadata & Original Columns (11 columns)

| Column | Type | Description |
|--------|------|-------------|
| `date` | object | Trading date (YYYY-MM-DD format, timezone-naive) |
| `company_name` | object | ETF name: "SPDR Portfolio S&P 500 ETF" |
| `ticker` | object | Ticker symbol: "SPLG" |
| `current_price` | float64 | Current price (same as close in most records) |
| `open` | float64 | Opening price for the trading day |
| `close` | float64 | Closing price for the trading day |
| `high` | float64 | Highest price during the trading day |
| `low` | float64 | Lowest price during the trading day |
| `volume` | int64 | Trading volume (number of shares traded) |
| `pe_ratio` | float64 | Price-to-earnings ratio |
| `yield` | float64 | Dividend yield percentage |

### 2. Price Relationship Features (7 columns)

| Column | Type | Description |
|--------|------|-------------|
| `high_low_spread` | float64 | Difference between high and low prices |
| `high_low_pct` | float64 | High-low spread as percentage of close |
| `close_open_diff` | float64 | Close minus open (intraday change) |
| `close_open_pct` | float64 | Intraday return: (close - open) / open |
| `high_close_diff` | float64 | High minus close |
| `low_close_diff` | float64 | Low minus close |
| `close_current_price_diff` | float64 | Close minus current_price |

### 3. Return Features (7 columns)

| Column | Type | Description |
|--------|------|-------------|
| `return_1d` | float64 | 1-day percentage return |
| `log_return` | float64 | Log return: ln(close_t / close_{t-1}) |
| `return_5d` | float64 | 5-day percentage return |
| `return_10d` | float64 | 10-day percentage return |
| `return_20d` | float64 | 20-day percentage return |
| `rolling_mean_return_5d` | float64 | 5-day moving average of daily returns |
| `rolling_mean_return_20d` | float64 | 20-day moving average of daily returns |

### 4. Moving Average Features (30 columns)

**Simple Moving Averages (SMA)**: 5, 10, 20, 50, 100, 200 periods
- `ma_5`, `ma_10`, `ma_20`, `ma_50`, `ma_100`, `ma_200` (float64)

**Exponential Moving Averages (EMA)**: 5, 10, 20, 50, 100, 200 periods
- `ema_5`, `ema_10`, `ema_20`, `ema_50`, `ema_100`, `ema_200` (float64)

**Price-to-MA Ratios**: Current price divided by MA (trend strength indicators)
- `price_ma_5_ratio`, `price_ma_10_ratio`, `price_ma_20_ratio`
- `price_ma_50_ratio`, `price_ma_100_ratio`, `price_ma_200_ratio` (float64)

**MA Slopes**: Rate of change of moving averages (trend direction)
- `ma_5_slope`, `ma_10_slope`, `ma_20_slope`
- `ma_50_slope`, `ma_100_slope`, `ma_200_slope` (float64)

**MA Cross Signals**: Binary indicators (1=bullish, 0=bearish)
- `ma_cross_10_over_50` (int64): Golden cross indicator (10-day > 50-day)
- `ma_cross_20_over_50` (int64): 20-day > 50-day
- `ma_cross_50_over_200` (int64): Golden cross (50-day > 200-day)

### 5. Momentum Indicators (7 columns)

| Column | Type | Description |
|--------|------|-------------|
| `momentum_5` | float64 | Price change over 5 days: close_t - close_{t-5} |
| `momentum_10` | float64 | Price change over 10 days |
| `roc_10` | float64 | 10-day rate of change (same as return_10d) |
| `rsi_14` | float64 | 14-day Relative Strength Index (0-100) |
| `macd` | float64 | MACD line: EMA(12) - EMA(26) |
| `macd_signal` | float64 | MACD signal line: EMA(9) of MACD |
| `macd_hist` | float64 | MACD histogram: MACD - MACD_signal |

### 6. Volatility & Risk Features (13 columns)

**Rolling Standard Deviations**: 5, 10, 20, 30 periods
- `rolling_std_5`, `rolling_std_10`, `rolling_std_20`, `rolling_std_30` (float64)

**Rolling Variances**: 5, 10, 20, 30 periods
- `rolling_var_5`, `rolling_var_10`, `rolling_var_20`, `rolling_var_30` (float64)

**Other Volatility Metrics**:
- `atr_14` (float64): 14-day Average True Range (Wilder's ATR)
- `bb_upper_20` (float64): Upper Bollinger Band (MA_20 + 2×STD_20)
- `bb_lower_20` (float64): Lower Bollinger Band (MA_20 - 2×STD_20)
- `bb_pct_20` (float64): Bollinger Band percentile (0=at lower, 1=at upper)

### 7. Volume Features (6 columns)

| Column | Type | Description |
|--------|------|-------------|
| `vol_change` | float64 | Percentage change in volume from previous day |
| `vol_ma_5` | float64 | 5-day moving average of volume |
| `vol_ma_20` | float64 | 20-day moving average of volume |
| `vol_ratio_5_20` | float64 | Ratio of 5-day to 20-day volume MA |
| `obv` | float64 | On-Balance Volume (cumulative volume with sign of price change) |
| `vpt` | float64 | Volume Price Trend (cumulative volume × % price change) |

### 8. Lagged Features (15 columns)

Autoregressive features capturing past values at lags: 1, 2, 3, 5, 10 days

- **Lagged Close Prices**: `lag_1_close`, `lag_2_close`, `lag_3_close`, `lag_5_close`, `lag_10_close` (float64)
- **Lagged Returns**: `lag_1_return`, `lag_2_return`, `lag_3_return`, `lag_5_return`, `lag_10_return` (float64)
- **Lagged Volume**: `lag_1_volume`, `lag_2_volume`, `lag_3_volume`, `lag_5_volume`, `lag_10_volume` (float64)

### 9. Rolling Statistics (15 columns)

**10-day Rolling Window**:
- `rolling_mean_close_10`, `rolling_max_10`, `rolling_min_10` (float64)
- `rolling_skew_10`, `rolling_kurt_10` (float64): Distribution shape metrics

**20-day Rolling Window**:
- `rolling_mean_close_20`, `rolling_max_20`, `rolling_min_20` (float64)
- `rolling_skew_20`, `rolling_kurt_20` (float64)

**30-day Rolling Window**:
- `rolling_mean_close_30`, `rolling_max_30`, `rolling_min_30` (float64)
- `rolling_skew_30`, `rolling_kurt_30` (float64)

### 10. Calendar Features (8 columns)

| Column | Type | Description |
|--------|------|-------------|
| `dow` | int64 | Day of week (0=Monday, 6=Sunday) |
| `month` | int64 | Month (1-12) |
| `quarter` | int64 | Quarter (1-4) |
| `year` | int64 | Year (2006-2025) |
| `is_month_start` | int64 | Binary: 1 if first trading day of month |
| `is_month_end` | int64 | Binary: 1 if last trading day of month |

### 11. Target Variables (2 columns)

See [Target Variables](#target-variables) section above.

---

## Data Generation Process

### Source Script
**File**: `build_splg_features.py`  
**Location**: `/home/robert/FEURCast/wrangling/pred_model_feature_engineering/`

### Generation Pipeline

#### Step 1: Data Ingestion
- Reads raw SPLG history CSV from `/home/robert/FEURCast/data/SPLG_history_full.csv`
- Original dataset: 4,996 records spanning 2005-11-15 to 2025-09-24

#### Step 2: Data Normalization
- Column names converted to lowercase snake_case
- Dates parsed with timezone awareness and converted to America/New_York timezone
- Numeric columns coerced to appropriate types
- Rows sorted by date, duplicates removed

#### Step 3: Feature Engineering
Features are computed in the following order:
1. Basic price relationships (spreads, percentages)
2. Return calculations (1d, 5d, 10d, 20d)
3. Moving averages (SMA, EMA) with ratios and slopes
4. Momentum indicators (RSI, MACD, ROC)
5. Volatility metrics (standard deviations, ATR, Bollinger Bands)
6. Volume features (OBV, VPT, volume MAs)
7. Lagged features (autoregressive signals)
8. Rolling statistics (mean, max, min, skewness, kurtosis)
9. Calendar features (day of week, month, quarter, etc.)
10. Target variables (forward-shifted close and return)

#### Step 4: Data Cleaning
- Columns with 100% NaN values are dropped (e.g., original `beta` column)
- Rows with any NaN values are dropped
- **Records lost**: ~248 rows (due to 200-day MA warm-up period and target shift)
- **Final output**: 4,748 complete records

#### Step 5: Export
- Saved to CSV: `data_out/rich_features_SPLG_history_full.csv`
- All features are float64 except metadata (object) and categorical/binary features (int64)

### Reproducing the Dataset

To regenerate the dataset:

```bash
cd /home/robert/FEURCast/wrangling/pred_model_feature_engineering
python3 build_splg_features.py
```

**Requirements**:
- Python 3.7+
- pandas
- numpy

---

## Model Training Pipeline

### Recommended Framework: Scikit-learn

Use `sklearn.ensemble.GradientBoostingRegressor` for its:
- Mature implementation
- Extensive documentation
- Easy integration with scikit-learn ecosystem
- Built-in feature importance analysis

### Training Pipeline Steps

#### 1. Data Loading & Preprocessing

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

# Load data
df = pd.read_csv('data_out/rich_features_SPLG_history_full.csv')

# Parse date column (for time-based splits)
df['date'] = pd.to_datetime(df['date'])

# Separate features and targets
metadata_cols = ['date', 'company_name', 'ticker']
target_cols = ['target_close_t1', 'target_return_t1']
feature_cols = [col for col in df.columns if col not in metadata_cols + target_cols]

X = df[feature_cols]
y_price = df['target_close_t1']  # For price prediction
y_return = df['target_return_t1']  # For return prediction
dates = df['date']

# Handle any remaining NaNs (defensive programming)
assert X.isna().sum().sum() == 0, "Features contain NaN values"
assert y_price.isna().sum() == 0, "Target contains NaN values"
```

#### 2. Train/Validation/Test Split

**Option A: Simple Time-Based Split** (Recommended for initial experiments)

```python
# 70% train, 15% validation, 15% test (time-ordered)
train_size = int(0.70 * len(df))
val_size = int(0.15 * len(df))

X_train = X.iloc[:train_size]
X_val = X.iloc[train_size:train_size+val_size]
X_test = X.iloc[train_size+val_size:]

y_train = y_return.iloc[:train_size]
y_val = y_return.iloc[train_size:train_size+val_size]
y_test = y_return.iloc[train_size+val_size:]

dates_train = dates.iloc[:train_size]
dates_val = dates.iloc[train_size:train_size+val_size]
dates_test = dates.iloc[train_size+val_size:]

print(f"Train: {dates_train.min()} to {dates_train.max()} ({len(X_train)} samples)")
print(f"Val:   {dates_val.min()} to {dates_val.max()} ({len(X_val)} samples)")
print(f"Test:  {dates_test.min()} to {dates_test.max()} ({len(X_test)} samples)")
```

**Option B: Time Series Cross-Validation** (For robust evaluation)

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
# Use this for hyperparameter tuning with GridSearchCV or cross_val_score
```

#### 3. Feature Scaling (Optional but Recommended)

While tree-based models don't require scaling, it can help with:
- Regularization effectiveness
- Numerical stability
- Comparison with other model types

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Save scaler for production
import joblib
joblib.dump(scaler, 'models/scaler.pkl')
```

#### 4. Model Training

**Initial Baseline Model**:

```python
# Simple baseline with default parameters
gbr_baseline = GradientBoostingRegressor(
    random_state=42,
    verbose=1
)

gbr_baseline.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_train_pred = gbr_baseline.predict(X_train)
y_val_pred = gbr_baseline.predict(X_val)

print("Baseline Training Performance:")
print(f"  MSE: {mean_squared_error(y_train, y_train_pred):.6f}")
print(f"  MAE: {mean_absolute_error(y_train, y_train_pred):.6f}")
print(f"  R²:  {r2_score(y_train, y_train_pred):.6f}")

print("\nBaseline Validation Performance:")
print(f"  MSE: {mean_squared_error(y_val, y_val_pred):.6f}")
print(f"  MAE: {mean_absolute_error(y_val, y_val_pred):.6f}")
print(f"  R²:  {r2_score(y_val, y_val_pred):.6f}")
```

**Hyperparameter Tuning**:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0],
    'max_features': ['sqrt', 'log2', None]
}

# Use TimeSeriesSplit for time-aware CV
tscv = TimeSeriesSplit(n_splits=5)

gbr = GradientBoostingRegressor(random_state=42)

grid_search = GridSearchCV(
    estimator=gbr,
    param_grid=param_grid,
    cv=tscv,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best CV score:", -grid_search.best_score_)

# Train final model with best parameters
gbr_best = grid_search.best_estimator_
```

**Recommended Starting Parameters** (based on domain knowledge):

```python
gbr_tuned = GradientBoostingRegressor(
    n_estimators=300,        # More trees for better learning
    learning_rate=0.05,       # Lower LR with more trees
    max_depth=4,              # Prevent overfitting
    min_samples_split=10,     # Require minimum samples for splits
    min_samples_leaf=4,       # Smooth predictions
    subsample=0.8,            # Stochastic gradient boosting
    max_features='sqrt',      # Random feature selection
    random_state=42,
    verbose=1,
    validation_fraction=0.1,  # Use 10% for early stopping
    n_iter_no_change=20       # Stop if no improvement for 20 rounds
)

gbr_tuned.fit(X_train, y_train)
```

#### 5. Model Evaluation

```python
# Predictions on all sets
y_train_pred = gbr_tuned.predict(X_train)
y_val_pred = gbr_tuned.predict(X_val)
y_test_pred = gbr_tuned.predict(X_test)

# Calculate metrics
def evaluate_model(y_true, y_pred, set_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Financial metrics
    directional_accuracy = np.mean((np.sign(y_true) == np.sign(y_pred)))
    
    print(f"\n{set_name} Set Performance:")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  R²:   {r2:.6f}")
    print(f"  Directional Accuracy: {directional_accuracy:.2%}")
    
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'dir_acc': directional_accuracy}

train_metrics = evaluate_model(y_train, y_train_pred, "Training")
val_metrics = evaluate_model(y_val, y_val_pred, "Validation")
test_metrics = evaluate_model(y_test, y_test_pred, "Test")
```

#### 6. Model Persistence

```python
import joblib

# Save the trained model
joblib.dump(gbr_tuned, 'models/gbr_model.pkl')

# Save feature names for production
feature_names = X_train.columns.tolist()
joblib.dump(feature_names, 'models/feature_names.pkl')

# Save metrics
import json
metrics = {
    'train': train_metrics,
    'validation': val_metrics,
    'test': test_metrics,
    'training_date': pd.Timestamp.now().isoformat(),
    'n_features': len(feature_names),
    'n_train_samples': len(X_train),
    'n_val_samples': len(X_val),
    'n_test_samples': len(X_test)
}

with open('models/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
```

---

## Performance Metrics

### Required Metrics for Academic Documentation

#### 1. Regression Metrics

- **Mean Squared Error (MSE)**: Average squared difference between predictions and actuals
- **Root Mean Squared Error (RMSE)**: Square root of MSE (same units as target)
- **Mean Absolute Error (MAE)**: Average absolute difference
- **R² Score**: Proportion of variance explained (0-1, higher is better)
- **Mean Absolute Percentage Error (MAPE)**: Average percentage error

#### 2. Financial Metrics

- **Directional Accuracy**: Percentage of correct up/down predictions
- **Sharpe Ratio**: Risk-adjusted return of trading strategy based on predictions
- **Maximum Drawdown**: Largest peak-to-trough decline in cumulative returns
- **Profit Factor**: Gross profit / gross loss from simulated trading

#### 3. Feature Importance Analysis

```python
# Get feature importances
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': gbr_tuned.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 Most Important Features:")
print(feature_importance.head(20))

# Save feature importance
feature_importance.to_csv('models/feature_importance.csv', index=False)
```

### Visualization Requirements

Create the following plots for documentation:

#### 1. Training Progress Plot

```python
import matplotlib.pyplot as plt

# Plot training loss curve
train_score = gbr_tuned.train_score_
val_score = np.zeros(len(train_score))  # If using validation_fraction

plt.figure(figsize=(10, 6))
plt.plot(train_score, label='Training Loss', linewidth=2)
plt.xlabel('Boosting Iteration')
plt.ylabel('Loss (Negative MSE)')
plt.title('GBR Training Progress')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('plots/training_progress.png', dpi=300, bbox_inches='tight')
plt.close()
```

#### 2. Predictions vs Actuals Plot

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, y_true, y_pred, title in zip(
    axes, 
    [y_train, y_val, y_test],
    [y_train_pred, y_val_pred, y_test_pred],
    ['Training', 'Validation', 'Test']
):
    ax.scatter(y_true, y_pred, alpha=0.5, s=10)
    ax.plot([y_true.min(), y_true.max()], 
            [y_true.min(), y_true.max()], 
            'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Return')
    ax.set_ylabel('Predicted Return')
    ax.set_title(f'{title} Set')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/predictions_vs_actuals.png', dpi=300, bbox_inches='tight')
plt.close()
```

#### 3. Residuals Plot

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, y_true, y_pred, title in zip(
    axes,
    [y_train, y_val, y_test],
    [y_train_pred, y_val_pred, y_test_pred],
    ['Training', 'Validation', 'Test']
):
    residuals = y_true - y_pred
    ax.scatter(y_pred, residuals, alpha=0.5, s=10)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Predicted Return')
    ax.set_ylabel('Residuals')
    ax.set_title(f'{title} Set Residuals')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/residuals.png', dpi=300, bbox_inches='tight')
plt.close()
```

#### 4. Feature Importance Plot

```python
plt.figure(figsize=(10, 12))
top_features = feature_importance.head(30)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance')
plt.title('Top 30 Feature Importances')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('plots/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
```

#### 5. Time Series of Predictions

```python
plt.figure(figsize=(16, 8))

# Plot test set predictions over time
plt.plot(dates_test, y_test.values, label='Actual Returns', alpha=0.7, linewidth=1.5)
plt.plot(dates_test, y_test_pred, label='Predicted Returns', alpha=0.7, linewidth=1.5)
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
plt.xlabel('Date')
plt.ylabel('Return')
plt.title('Model Predictions vs Actual Returns (Test Set)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/time_series_predictions.png', dpi=300, bbox_inches='tight')
plt.close()
```

#### 6. Cumulative Returns Comparison

```python
# Calculate cumulative returns
cumulative_actual = (1 + y_test).cumprod()
cumulative_predicted = (1 + y_test_pred).cumprod()

plt.figure(figsize=(14, 7))
plt.plot(dates_test, cumulative_actual, label='Actual Cumulative Return', linewidth=2)
plt.plot(dates_test, cumulative_predicted, label='Strategy Cumulative Return', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Cumulative Return (Starting from 1.0)')
plt.title('Cumulative Returns: Actual vs Model-Based Strategy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/cumulative_returns.png', dpi=300, bbox_inches='tight')
plt.close()
```

#### 7. Error Distribution

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, y_true, y_pred, title in zip(
    axes,
    [y_train, y_val, y_test],
    [y_train_pred, y_val_pred, y_test_pred],
    ['Training', 'Validation', 'Test']
):
    errors = y_true - y_pred
    ax.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{title} Set Error Distribution')
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('plots/error_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
```

---

## Integration with Streamlit Application

### Model Loading in Streamlit

```python
import joblib
import pandas as pd
import numpy as np

@st.cache_resource
def load_model():
    """Load the trained GBR model and preprocessing artifacts."""
    model = joblib.load('models/gbr_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    
    with open('models/metrics.json', 'r') as f:
        metrics = json.load(f)
    
    return model, scaler, feature_names, metrics

# In your Streamlit app
model, scaler, feature_names, metrics = load_model()

# Display model performance
st.subheader("Model Performance")
st.metric("Test R² Score", f"{metrics['test']['r2']:.4f}")
st.metric("Test RMSE", f"{metrics['test']['rmse']:.6f}")
st.metric("Directional Accuracy", f"{metrics['test']['dir_acc']:.2%}")
```

### Making Predictions

```python
def make_prediction(input_features: pd.DataFrame) -> dict:
    """
    Make a prediction using the trained model.
    
    Parameters:
    -----------
    input_features : pd.DataFrame
        DataFrame with columns matching training features
    
    Returns:
    --------
    dict with predicted return and confidence metrics
    """
    # Ensure correct feature order
    X = input_features[feature_names]
    
    # Scale features (if scaler was used)
    X_scaled = scaler.transform(X)
    
    # Predict
    prediction = model.predict(X_scaled)[0]
    
    return {
        'predicted_return': prediction,
        'predicted_direction': 'UP' if prediction > 0 else 'DOWN',
        'confidence': abs(prediction)  # Simple confidence metric
    }

# Example usage in Streamlit
if st.button("Predict Tomorrow's Return"):
    # Get latest feature values (from your data pipeline)
    latest_features = get_latest_features()  # You implement this
    
    result = make_prediction(latest_features)
    
    st.write(f"Predicted Return: {result['predicted_return']:.4%}")
    st.write(f"Direction: {result['predicted_direction']}")
```

### Model Retraining Interface (Optional)

```python
if st.sidebar.button("Retrain Model"):
    with st.spinner("Retraining model..."):
        # Load latest data
        df = pd.read_csv('data_out/rich_features_SPLG_history_full.csv')
        
        # Re-run training pipeline
        # (Use the code from Model Training Pipeline section)
        
        st.success("Model retrained successfully!")
        st.experimental_rerun()
```

---

## Reproducibility Guidelines

### Environment Setup

**Create a requirements file** (`requirements_model_training.txt`):

```
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
joblib==1.3.1
```

**Installation**:
```bash
pip install -r requirements_model_training.txt
```

### Random Seed Management

Set random seeds for reproducibility:

```python
import random
import numpy as np

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Use random_state parameter in all sklearn functions
GradientBoostingRegressor(random_state=RANDOM_SEED)
train_test_split(X, y, random_state=RANDOM_SEED)
```

### Directory Structure

Organize your model training project:

```
wrangling/pred_model_feature_engineering/
├── build_splg_features.py          # Feature engineering script
├── train_gbr_model.py              # Model training script (to be created)
├── evaluate_model.py               # Evaluation script (to be created)
├── data_out/
│   └── rich_features_SPLG_history_full.csv
├── models/
│   ├── gbr_model.pkl               # Trained model
│   ├── scaler.pkl                  # Feature scaler
│   ├── feature_names.pkl           # Feature list
│   └── metrics.json                # Performance metrics
├── plots/
│   ├── training_progress.png
│   ├── predictions_vs_actuals.png
│   ├── residuals.png
│   ├── feature_importance.png
│   ├── time_series_predictions.png
│   ├── cumulative_returns.png
│   └── error_distribution.png
└── logs/
    └── training_log_YYYYMMDD_HHMMSS.txt
```

### Logging

Implement comprehensive logging:

```python
import logging
from datetime import datetime

# Setup logging
log_filename = f"logs/training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Use throughout training
logger.info("Starting model training")
logger.info(f"Training samples: {len(X_train)}")
logger.info(f"Best parameters: {grid_search.best_params_}")
logger.info(f"Test R² score: {test_metrics['r2']:.6f}")
```

### Model Versioning

Track model versions:

```python
import hashlib
import json

def create_model_metadata(model, X_train, y_train, metrics):
    """Create comprehensive model metadata for versioning."""
    
    # Create hash of training data
    data_hash = hashlib.md5(
        pd.concat([X_train, y_train], axis=1).values.tobytes()
    ).hexdigest()
    
    metadata = {
        'model_type': 'GradientBoostingRegressor',
        'version': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'data_hash': data_hash,
        'n_features': X_train.shape[1],
        'n_train_samples': len(X_train),
        'hyperparameters': model.get_params(),
        'performance_metrics': metrics,
        'training_date': datetime.now().isoformat(),
        'feature_names': X_train.columns.tolist()
    }
    
    return metadata

# Save metadata
metadata = create_model_metadata(gbr_tuned, X_train, y_train, test_metrics)
with open('models/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

---

## Additional Considerations

### Hyperparameter Tuning Tips

1. **Start with default parameters** to establish a baseline
2. **Tune learning_rate and n_estimators together**: Lower LR needs more estimators
3. **Adjust max_depth and min_samples_split** to control overfitting
4. **Use early stopping** with validation_fraction and n_iter_no_change
5. **Consider computational cost**: Large param grids can take hours/days

### Common Pitfalls to Avoid

1. **Data leakage**: Never use future information in features
2. **Look-ahead bias**: Ensure all features use only past data
3. **Overfitting**: Monitor train vs validation performance gap
4. **Time series violations**: Use time-based splits, not random splits
5. **Feature scaling inconsistency**: Apply same scaler to train/val/test

### Model Interpretation

For academic reporting, include:

1. **Feature importance analysis**: Which features drive predictions?
2. **Partial dependence plots**: How do key features affect predictions?
3. **SHAP values** (optional): Local feature importance for individual predictions
4. **Error analysis**: When and why does the model fail?

### Performance Benchmarks

Compare your GBR model against:

1. **Naive baseline**: Previous day's return
2. **Linear regression**: Simple OLS model
3. **Random Forest**: Alternative ensemble method
4. **XGBoost/LightGBM**: Advanced gradient boosting libraries

---

## Conclusion

This guide provides a comprehensive framework for training and evaluating a Gradient Boosting Regressor model on the SPLG feature-engineered dataset. Follow these steps systematically to ensure reproducible, well-documented results suitable for academic presentation.

**Next Steps**:
1. Create `train_gbr_model.py` implementing the training pipeline
2. Create `evaluate_model.py` for comprehensive evaluation
3. Generate all required plots and metrics
4. Document results in your academic paper/report
5. Integrate the trained model into the Streamlit application

**Key Success Metrics**:
- R² > 0.1 (financial time series are noisy)
- Directional accuracy > 52% (better than random)
- Stable performance across train/val/test sets
- Interpretable feature importances aligned with financial theory

---

**Document Version**: 1.0  
**Last Updated**: October 14, 2025  
**Author**: FEURCast Team  
**Related Files**: 
- `build_splg_features.py`
- `rich_features_SPLG_history_full.csv`
