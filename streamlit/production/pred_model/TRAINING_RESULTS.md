# SPLG GBR Model Training Results

**Training Date**: [Will be populated after training]  
**Model Version**: [Will be populated after training]  
**Status**: ⏳ Not yet trained

---

## Quick Summary

| Metric | Training | Validation | Test |
|--------|----------|------------|------|
| **R² Score** | - | - | - |
| **RMSE** | - | - | - |
| **MAE** | - | - | - |
| **Directional Accuracy** | - | - | - |

---

## Model Configuration

### Hyperparameters
```json
{
  "n_estimators": 300,
  "learning_rate": 0.05,
  "max_depth": 4,
  "min_samples_split": 10,
  "min_samples_leaf": 4,
  "subsample": 0.8,
  "max_features": "sqrt",
  "random_state": 42
}
```

### Data Split
- **Training Set**: 70% of data (earliest observations)
- **Validation Set**: 15% of data (middle period)
- **Test Set**: 15% of data (most recent observations)
- **Total Samples**: 4,748 records
- **Features**: 112 engineered features

---

## Performance Analysis

### Regression Metrics

**Mean Squared Error (MSE)**
- Training: -
- Validation: -
- Test: -

**Root Mean Squared Error (RMSE)**
- Training: -
- Validation: -
- Test: -

**Mean Absolute Error (MAE)**
- Training: -
- Validation: -
- Test: -

**R² Score**
- Training: -
- Validation: -
- Test: -

### Financial Metrics

**Directional Accuracy** (% of correct up/down predictions)
- Training: -
- Validation: -
- Test: -

**Sharpe Ratio** (risk-adjusted returns)
- Test: -

**Maximum Drawdown**
- Test: -

**Profit Factor** (gross profit / gross loss)
- Test: -

**Win Rate** (% of positive returns)
- Test: -

---

## Feature Importance Analysis

### Top 10 Most Important Features

| Rank | Feature Name | Importance | Category |
|------|--------------|------------|----------|
| 1 | - | - | - |
| 2 | - | - | - |
| 3 | - | - | - |
| 4 | - | - | - |
| 5 | - | - | - |
| 6 | - | - | - |
| 7 | - | - | - |
| 8 | - | - | - |
| 9 | - | - | - |
| 10 | - | - | - |

*Full feature importance rankings available in `models/feature_importance.csv`*

---

## Visualizations

All plots are saved in the `plots/` directory:

1. **training_progress.png** - Loss curve over boosting iterations
2. **predictions_vs_actuals.png** - Scatter plots for train/val/test sets
3. **residuals.png** - Residual analysis for each set
4. **feature_importance.png** - Top 30 features bar chart
5. **time_series_predictions.png** - Predictions over time on test set
6. **cumulative_returns.png** - Strategy performance vs actual returns
7. **error_distribution.png** - Histogram of prediction errors

---

## Model Interpretation

### What the Model Learned
[Will be populated after training with interpretation of top features]

### Prediction Confidence
[Will be populated with analysis of when model is most/least confident]

### Limitations & Considerations
[Will be populated with identified limitations and edge cases]

---

## Deployment Status

### Integration with Streamlit App
- ✅ Prediction interface implemented (`predict.py`)
- ✅ Feature extraction module implemented (`get_latest_features.py`)
- ✅ Simulator integration completed (`tools.py`)
- ⏳ Model training pending

### Production Readiness
- [x] Training pipeline complete
- [x] Evaluation framework complete
- [x] Inference interface complete
- [ ] Model trained and validated
- [ ] Performance benchmarks met
- [ ] Documentation complete

---

## Next Steps

1. **Train the model**:
   ```bash
   cd /home/robert/FEURCast/streamlit/production/pred_model
   ./train_and_evaluate.sh
   ```

2. **Review results**: Check `models/metrics.json` and plots in `plots/`

3. **Test predictions**: Run `python predict.py` to verify inference works

4. **Deploy to app**: Restart Streamlit app to use real model

5. **Monitor performance**: Track accuracy on new data over time

---

## Training Log Location

Training logs are saved to `logs/training_log_YYYYMMDD_HHMMSS.txt`  
Evaluation logs are saved to `logs/evaluation_log_YYYYMMDD_HHMMSS.txt`

---

## Reproducibility

- **Random Seed**: 42
- **Data Version**: `rich_features_SPLG_history_full.csv` (2006-09-01 to 2025-09-24)
- **Training Script**: `scripts/train_gbr_model.py`
- **Evaluation Script**: `scripts/evaluate_model.py`

---

**This document will be automatically updated after model training completes.**
