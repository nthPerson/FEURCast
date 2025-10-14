# SPLG Feature Engineering & Model Training Documentation

## Overview

This directory contains the complete pipeline for engineering features from SPLG (SPDR Portfolio S&P 500 ETF) historical data and training a Gradient Boosting Regressor (GBR) model for next-day return prediction.

**Project**: FEURCast (Financial Equity Uncertainty Regression & Classification Anticipation System Tool)  
**Academic Context**: Comprehensive documentation for reproducible research

---

## 📁 Directory Contents

```
pred_model_feature_engineering/
├── build_splg_features.py              # Feature engineering script
├── data_out/
│   └── rich_features_SPLG_history_full.csv  # Output dataset (4,748 × 115)
├── DATA_DICTIONARY.md                  # Complete column reference (115 features)
├── GBR_MODEL_TRAINING_GUIDE.md         # Comprehensive training guide
├── QUICK_START_MODEL_TRAINING.md       # TL;DR for quick implementation
└── README.md                           # This file
```

---

## 📚 Documentation Files

### 1. **DATA_DICTIONARY.md** 
*Comprehensive feature catalog*

- Full table of all 115 columns with descriptions
- Feature categories (Price, Trend, Momentum, Volatility, Volume, etc.)
- Data types and value ranges
- Feature engineering notes
- Data quality validation checklist

**Use this when**: You need to understand what each feature represents

### 2. **GBR_MODEL_TRAINING_GUIDE.md**
*Complete model training documentation*

- Full dataset overview with statistics
- Detailed feature descriptions organized by category
- Step-by-step model training pipeline
- Hyperparameter tuning recommendations
- Performance metrics and evaluation code
- Visualization examples (7 types of plots)
- Streamlit integration guide
- Reproducibility guidelines

**Use this when**: Setting up the full training pipeline from scratch

### 3. **QUICK_START_MODEL_TRAINING.md**
*Minimal working example*

- TL;DR summary for AI agents
- 30-line working code example
- Critical requirements checklist
- Expected performance benchmarks
- Common issues and solutions

**Use this when**: You want to train a model quickly without reading everything

---

## 🚀 Quick Start

### Step 1: Generate Features (Already Done)

```bash
python3 build_splg_features.py
```

**Output**: `data_out/rich_features_SPLG_history_full.csv` (4,748 records, 115 columns)

### Step 2: Train Model (Next Step)

```python
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

# Load data
df = pd.read_csv('data_out/rich_features_SPLG_history_full.csv')

# Prepare features and target
feature_cols = [col for col in df.columns 
                if col not in ['date', 'company_name', 'ticker', 
                              'target_close_t1', 'target_return_t1']]
X = df[feature_cols]
y = df['target_return_t1']

# Time-based split (70/15/15)
train_end = int(0.70 * len(df))
val_end = int(0.85 * len(df))

X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

# Train
model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
import joblib
joblib.dump(model, 'models/gbr_model.pkl')
```

See **QUICK_START_MODEL_TRAINING.md** for complete example.

---

## 📊 Dataset Overview

| Attribute | Value |
|-----------|-------|
| **Filename** | `rich_features_SPLG_history_full.csv` |
| **Records** | 4,748 |
| **Columns** | 115 |
| **Date Range** | 2006-09-01 to 2025-09-24 (~19 years) |
| **Features** | 102 (price, trend, momentum, volatility, volume, lags, calendar) |
| **Targets** | 2 (next-day price & return) |
| **Missing Values** | 0 (complete dataset) |

### Feature Breakdown

| Category | Count | Examples |
|----------|-------|----------|
| Metadata | 3 | date, company_name, ticker |
| Original | 8 | open, close, high, low, volume, pe_ratio, yield |
| Price Relationships | 7 | high_low_spread, close_open_pct |
| Returns | 7 | return_1d, return_5d, log_return |
| Moving Averages | 30 | ma_5, ema_20, price_ma_200_ratio, ma_50_slope |
| Momentum | 7 | rsi_14, macd, macd_hist, momentum_10 |
| Volatility | 13 | rolling_std_20, atr_14, bb_pct_20 |
| Volume | 6 | obv, vpt, vol_ratio_5_20 |
| Lagged Features | 15 | lag_1_close, lag_5_return, lag_10_volume |
| Rolling Statistics | 15 | rolling_max_20, rolling_skew_30, rolling_kurt_10 |
| Calendar | 8 | dow, month, quarter, year, is_month_start |
| **Targets** | 2 | **target_close_t1, target_return_t1** |

---

## 🎯 Target Variables

### Primary Target: `target_return_t1`
- **Description**: Next-day percentage return
- **Formula**: (Close_{t+1} / Close_t) - 1
- **Range**: -11.68% to +9.53%
- **Mean**: +0.039% daily
- **Recommended for**: Return prediction, directional trading strategies

### Secondary Target: `target_close_t1`
- **Description**: Next-day closing price
- **Range**: $7.94 to $78.68
- **Mean**: $30.49
- **Use case**: Absolute price forecasting

---

## 🔬 Model Training Requirements

### Critical Considerations

✅ **Time-Based Splits**: Always use chronological splits (no random shuffling)  
✅ **No Look-Ahead Bias**: All features use only past data  
✅ **Reproducibility**: Set `random_state=42`  
✅ **Feature Order**: Save and maintain exact column order  
✅ **Evaluation Metrics**: RMSE, R², directional accuracy  

### Recommended Model

**scikit-learn GradientBoostingRegressor**
- Mature, well-documented
- Built-in feature importance
- Easy integration with Streamlit
- Starting parameters:
  - `n_estimators=300`
  - `learning_rate=0.05`
  - `max_depth=4`
  - `subsample=0.8`

### Expected Performance

Financial time series are inherently noisy. Realistic benchmarks:

| Metric | Good | Excellent |
|--------|------|-----------|
| R² Score | 0.05 - 0.10 | > 0.10 |
| Directional Accuracy | 52% - 55% | > 55% |
| RMSE | ~0.01 (1%) | < 0.01 |

*Note: R² > 0.10 on daily returns is significant in finance*

---

## 📈 Deliverables for Academic Project

### Required Outputs

1. **Trained Model**: `models/gbr_model.pkl`
2. **Performance Metrics**: `models/metrics.json`
3. **Feature Importance**: `models/feature_importance.csv`
4. **Visualizations**: 
   - Training progress plot
   - Predictions vs actuals (train/val/test)
   - Residuals analysis
   - Feature importance bar chart
   - Time series predictions
   - Cumulative returns comparison
   - Error distribution histograms

5. **Model Metadata**: 
   - Hyperparameters used
   - Training/validation/test split details
   - Feature names and count
   - Training timestamp
   - Data hash for version tracking

### Documentation Requirements

- ✅ Dataset description (DATA_DICTIONARY.md)
- ✅ Feature engineering process (build_splg_features.py)
- ✅ Model training methodology (GBR_MODEL_TRAINING_GUIDE.md)
- ✅ Performance evaluation metrics and visualizations
- ✅ Reproducibility instructions (this README)

---

## 🔗 Integration with Streamlit App

### Model Loading

```python
import joblib
import streamlit as st

@st.cache_resource
def load_model():
    model = joblib.load('models/gbr_model.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    return model, feature_names

model, features = load_model()
```

### Making Predictions

```python
# Assuming you have latest_features DataFrame with 102 feature columns
prediction = model.predict(latest_features[features])[0]

st.metric("Tomorrow's Predicted Return", f"{prediction:.2%}")
if prediction > 0:
    st.success("📈 Model predicts UP")
else:
    st.error("📉 Model predicts DOWN")
```

---

## 🧪 Testing & Validation

### Data Quality Checks

```python
import pandas as pd

df = pd.read_csv('data_out/rich_features_SPLG_history_full.csv')

# Validate shape
assert df.shape == (4748, 115)

# Check for missing values
assert df.isna().sum().sum() == 0

# Verify chronological order
dates = pd.to_datetime(df['date'])
assert dates.is_monotonic_increasing

# Check target range (no extreme outliers)
assert df['target_return_t1'].abs().max() < 0.5

print("✅ All data quality checks passed!")
```

### Model Sanity Checks

```python
# Load model
model = joblib.load('models/gbr_model.pkl')

# Check feature count
assert model.n_features_in_ == 102

# Verify predictions are reasonable
sample_pred = model.predict(X_test.iloc[:10])
assert sample_pred.min() > -0.5  # Not too negative
assert sample_pred.max() < 0.5   # Not too positive

print("✅ Model sanity checks passed!")
```

---

## 📝 Citation

If using this work in academic publications:

```bibtex
@software{feurcast_gbr_2025,
  title = {FEURCast: Gradient Boosting Regressor for SPLG Return Prediction},
  author = {FEURCast Team},
  year = {2025},
  url = {https://github.com/nthPerson/FEURCast},
  note = {Feature-engineered dataset with 102 technical indicators for ETF prediction}
}
```

---

## 🆘 Getting Help

### Common Issues

**Issue**: "Feature mismatch" error during prediction
- **Solution**: Ensure feature order matches training (`feature_names.pkl`)

**Issue**: R² score is negative
- **Solution**: Check train/test split, verify target variable is correct

**Issue**: Model overfits (perfect train, poor test)
- **Solution**: Reduce `max_depth`, increase `min_samples_split`

**Issue**: All predictions near zero
- **Solution**: Normal for regression on returns; check directional accuracy instead

### Additional Resources

- scikit-learn GBR docs: https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting
- Time series CV: https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split
- Feature importance: https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html

---

## 📅 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-10-14 | Initial documentation release |

---

## 👥 Contributing

For questions or contributions to this academic project:
- Review existing documentation first
- Follow reproducibility guidelines
- Document all changes thoroughly
- Maintain chronological data integrity

---

## ⚖️ License

This is an academic project. Please consult with the project maintainers regarding usage and citation requirements.

---

**Last Updated**: October 14, 2025  
**Maintained by**: FEURCast Team  
**Repository**: https://github.com/nthPerson/FEURCast
