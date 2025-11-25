# 🎯 QUICK START - Automated Updates

## Test Everything Works
```bash
cd /home/robert/FEURCast/streamlit/production/pred_model
python test_update_system.py
```
Expected: `✓ Results: 5/5 tests passed`

## Run Manual Update (SPYM data)
```bash
# Quick test (20 sec)
python update_and_retrain.py --quick

# Full training (3 min)
python update_and_retrain.py
```

## Set Up Daily Automation
```bash
./setup_scheduler.sh
```
Select: **Option 1** (Daily at 6:00 PM ET)

## Monitor Performance
```bash
# View training history
python training_logger.py

# Check today's logs
tail -f logs/data_update_$(date +%Y%m%d).log

# See cron job output
tail -f logs/cron_update.log
```

## After Each Retrain
```bash
# Restart Streamlit to use new model
cd ../
streamlit run app.py
```

---

## 📚 Documentation

- **`AUTO_UPDATE_GUIDE.md`** - Complete documentation
- **`UPDATE_QUICK_START.md`** - Quick reference
- **`AUTO_UPDATE_IMPLEMENTATION.md`** - What was built

---

**Ticker Change Note**: As of late October 2025 the SPLG ticker migrated to **SPYM**. The updater now fetches SPYM exclusively as of 11/24/25. Historical files keep the `SPLG` naming (`SPLG_history_full.csv`, `rich_features_SPLG_history_full.csv`) for compatibility.

**Ongoing Effects:** The GBR model will now update automatically every day at 6pm ET using SPYM data.
