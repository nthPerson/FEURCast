# 🔄 Automated Data Update & Model Retraining - Quick Reference

## ⚡ Quick Commands

```bash
# Navigate to pred_model directory
cd /home/robert/FEURCast/streamlit/production/pred_model

# Test the system
python test_update_system.py

# Run update & retrain (quick mode)
python update_and_retrain.py --quick

# Run update & retrain (full mode)
python update_and_retrain.py

# Force retrain even if no new data
python update_and_retrain.py --force

# View training history
python training_logger.py

# Set up automated scheduling
./setup_scheduler.sh
```

## 📋 What It Does

1. **Fetches new SPYM data** from yfinance (legacy SPLG rows retained; file names unchanged)
2. **Updates raw dataset** at `/data/SPLG_history_full.csv`
3. **Rebuilds features** (all 115 features, 110 used in model)
4. **Retrains GBR model** with latest data
5. **Logs results** to training history
6. **Displays performance** summary

## 🎯 Recommended Schedule

**Daily at 6:00 PM ET** (after market close):
```bash
0 18 * * 1-5 cd /path/to/pred_model && python update_and_retrain.py
```

Set up with:
```bash
./setup_scheduler.sh
# Select option 1 (Daily at 6:00 PM ET)
```

## 📊 Monitoring

### View logs
```bash
# Today's data update
tail -f logs/data_update_$(date +%Y%m%d).log

# Latest training run
ls -t logs/training_log_*.txt | head -1 | xargs less

# Training history
python training_logger.py
```

### Check status
```bash
# See if new data available
python -c "from data_updater import check_for_updates; check_for_updates()"

# View latest model metrics
cat models/metrics.json | jq '.test'

# Check cron jobs
crontab -l | grep update_and_retrain
```

## 🧪 Testing

```bash
# Test complete system
python test_update_system.py

# Test just data fetcher
python data_updater.py

# Test just feature engineering
python feature_updater.py

# Quick end-to-end test (30 seconds)
python update_and_retrain.py --quick --force
```

## 📁 File Locations

```
pred_model/
├── update_and_retrain.py    ← Main script
├── data_updater.py           ← Fetches SPYM data (legacy file names use SPLG)
├── feature_updater.py        ← Rebuilds features
├── training_logger.py        ← Logs results
├── test_update_system.py     ← Test suite
├── setup_scheduler.sh        ← Cron setup helper
├── logs/                     ← All logs
│   ├── training_history.jsonl
│   ├── data_update_*.log
│   └── training_log_*.txt
├── models/                   ← Model artifacts
└── plots/                    ← Evaluation plots
```

## 🔧 Troubleshooting

### No new data available
→ Markets may not be open/closed yet. Use `--force` to retrain anyway.

### Feature engineering failed
→ Check `logs/feature_update_*.log` for details

### Model training failed
→ Check `logs/training_log_*.txt` for details

### Cron job not running
→ Check: `grep CRON /var/log/syslog`

## 💡 Tips

- Run `--quick` mode during testing (20 estimators, ~10 seconds)
- Full training takes 2-3 minutes (300 estimators)
- Check training history weekly to monitor performance
- Keep logs for at least 30 days
- Restart Streamlit app after retraining to use new model

## 📚 Full Documentation

See `AUTO_UPDATE_GUIDE.md` for complete documentation including:
- Detailed component descriptions
- Feature engineering details
- Performance monitoring
- Best practices
- Advanced troubleshooting

---

**Status**: ✅ All systems operational  
**Last Test**: 2025-10-20  
**Current Model**: Test R² = -0.0063, Directional Accuracy = 54.31%
