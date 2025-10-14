# Quick Start: GBR Model Training

## TL;DR for AI Agents

**Dataset**: `data_out/rich_features_SPLG_history_full.csv`
- 4,748 records × 115 columns
- Date range: 2006-09-01 to 2025-09-24
- Targets: `target_close_t1` (price), `target_return_t1` (return)
- 102 feature columns (technical indicators, momentum, volatility, volume, calendar)

**Key Information**:
- Use **time-based splits** (not random) to avoid look-ahead bias
- Target: `target_return_t1` (next-day return percentage)
- Recommended model: `sklearn.ensemble.GradientBoostingRegressor`
- All features are pre-computed, no additional engineering needed

---

## Minimal Working Example

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = pd.read_csv('data_out/rich_features_SPLG_history_full.csv')

# Define feature and target columns
metadata_cols = ['date', 'company_name', 'ticker']
target_cols = ['target_close_t1', 'target_return_t1']
feature_cols = [col for col in df.columns if col not in metadata_cols + target_cols]

# Prepare data
X = df[feature_cols]
y = df['target_return_t1']  # Predicting returns

# Time-based split (70/15/15)
train_end = int(0.70 * len(df))
val_end = int(0.85 * len(df))

X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

# Train model
model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
    verbose=1
)

model.fit(X_train, y_train)

# Evaluate
y_test_pred = model.predict(X_test)
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.6f}")
print(f"Test R²: {r2_score(y_test, y_test_pred):.6f}")
print(f"Directional Accuracy: {np.mean(np.sign(y_test) == np.sign(y_test_pred)):.2%}")

# Save model
import joblib
joblib.dump(model, 'models/gbr_model.pkl')
```

---

## Feature Categories (102 features)

1. **Price Features** (7): spreads, diffs, ratios
2. **Returns** (7): 1d, 5d, 10d, 20d returns + rolling means
3. **Moving Averages** (30): SMA/EMA (5,10,20,50,100,200) + ratios + slopes
4. **Momentum** (7): RSI, MACD, momentum, ROC
5. **Volatility** (13): std, var, ATR, Bollinger Bands
6. **Volume** (6): volume MAs, OBV, VPT
7. **Lags** (15): 1,2,3,5,10-day lags of close/return/volume
8. **Rolling Stats** (15): 10/20/30-day mean/max/min/skew/kurt
9. **Calendar** (8): day of week, month, quarter, year, month start/end

**Original Columns** (11): date, company_name, ticker, current_price, open, close, high, low, volume, pe_ratio, yield

**Targets** (2): target_close_t1, target_return_t1

---

## Critical Requirements

### Data Handling
- ✅ **Use time-based splits** (chronological order matters)
- ✅ **Never shuffle** the data randomly
- ✅ **No feature engineering needed** (all features pre-computed)
- ✅ **Check for NaNs** (should be none, but defensive programming is good)

### Model Training
- ✅ Set `random_state=42` for reproducibility
- ✅ Use `n_estimators=300` and `learning_rate=0.05` as starting point
- ✅ Monitor for overfitting (train vs val performance)
- ✅ Save model with `joblib`

### Evaluation
- ✅ Report RMSE, MAE, R² on test set
- ✅ Calculate directional accuracy (sign agreement)
- ✅ Plot predictions vs actuals
- ✅ Analyze feature importance

---

## Expected Performance

**Realistic Benchmarks** (financial time series):
- R² score: 0.05 - 0.15 (anything above 0.10 is good!)
- Directional accuracy: 52% - 58% (barely above random is significant)
- RMSE: ~0.01 (1% error on daily returns)

Financial markets are extremely noisy. Don't expect R² > 0.5.

---

## Files to Create

1. **`train_gbr_model.py`**: Main training script
2. **`evaluate_model.py`**: Generate metrics and plots
3. **`models/gbr_model.pkl`**: Trained model
4. **`models/feature_names.pkl`**: Feature list
5. **`models/metrics.json`**: Performance metrics
6. **`plots/*.png`**: Visualization outputs

---

## Integration with Streamlit

```python
import joblib
import streamlit as st

@st.cache_resource
def load_model():
    return joblib.load('models/gbr_model.pkl')

model = load_model()

# Make predictions
prediction = model.predict(latest_features)[0]
st.metric("Tomorrow's Predicted Return", f"{prediction:.2%}")
```

---

## Common Issues & Solutions

**Issue**: Model shows R² < 0
- **Solution**: Check target variable, ensure train/test split is correct

**Issue**: Perfect training accuracy but poor test
- **Solution**: Overfitting. Reduce max_depth, increase min_samples_split

**Issue**: "Feature names mismatch" error
- **Solution**: Ensure exact feature order matches training

**Issue**: Predictions all near zero
- **Solution**: Target standardization might help, or model needs more capacity

---

## Complete Training Pipeline

For the full training pipeline with hyperparameter tuning, cross-validation, comprehensive evaluation, and all visualization code, see:

📄 **[GBR_MODEL_TRAINING_GUIDE.md](./GBR_MODEL_TRAINING_GUIDE.md)**

---

**Last Updated**: October 14, 2025
