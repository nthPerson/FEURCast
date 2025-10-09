# Troubleshooting Guide

## Common Issues and Solutions

### 1. "No module named 'dotenv'" or similar import errors

**Solution:**
```bash
pip install -r requirements.txt
```

Make sure you're in the `gbr_ui_test` directory when running this command.

---

### 2. "OpenAI API key not found" or authentication errors

**Problem:** The `.env` file is not properly configured.

**Solution:**
1. Check that `.env` exists in the workspace root (parent of `gbr_ui_test`):
   ```bash
   ls ../.env
   ```

2. Verify it contains your API key:
   ```bash
   cat ../.env
   ```

3. The file should look like:
   ```
   OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxx
   ```

4. If the file doesn't exist, copy the template:
   ```bash
   cp .env.example ../.env
   ```
   Then edit `../.env` with your actual API key.

---

### 3. Streamlit won't start or shows errors

**Solution 1:** Check Python version (needs 3.8+)
```bash
python --version
```

**Solution 2:** Try installing Streamlit explicitly
```bash
pip install streamlit --upgrade
```

**Solution 3:** Clear Streamlit cache
```bash
streamlit cache clear
```

---

### 4. Charts not displaying or Plotly errors

**Solution:**
```bash
pip install plotly --upgrade
```

If issues persist, try:
```bash
pip uninstall plotly
pip install plotly==5.17.0
```

---

### 5. "Module not found" when running app.py

**Problem:** Python can't find the simulator or llm_interface modules.

**Solution:** Make sure you're running the app from the `gbr_ui_test` directory:
```bash
cd gbr_ui_test
streamlit run app.py
```

---

### 6. LLM responses are slow or timing out

**Cause:** OpenAI API calls can take a few seconds.

**Expected behavior:** You should see spinner messages while waiting:
- "Generating prediction..."
- "Planning analysis..."
- "Composing answer..."

If it hangs for more than 30 seconds, check your internet connection and API key.

---

### 7. Simulated data looks unrealistic

**Not an error!** This is a demo with simulated data. The purpose is to show the UI and architecture, not provide real market predictions.

In the full implementation:
- Real historical data will replace simulated prices
- Trained GBR model will replace LLM-generated predictions
- Live API calls will replace mock data

---

### 8. Permission denied when running setup.sh

**Solution:** Make the script executable
```bash
chmod +x setup.sh
./setup.sh
```

---

## Getting Help

If you encounter issues not covered here:

1. **Check the terminal output** - error messages usually indicate the problem
2. **Verify all files are present** - see File Structure in README.md
3. **Check Python packages** - run `pip list` to see installed versions
4. **Test OpenAI connection** - try this in Python:
   ```python
   from dotenv import load_dotenv
   import os
   load_dotenv('../.env')
   print(os.getenv('OPENAI_API_KEY'))
   ```
   It should print your API key (not "None")

---

## Known Limitations (By Design)

These are **intentional** for the demo:

- ✅ All market data is simulated
- ✅ Predictions come from OpenAI, not a trained GBR model
- ✅ Tool execution is mocked/simplified
- ✅ No database or persistent storage
- ✅ Limited error handling
- ✅ No data caching (each run re-generates)

The purpose is to demonstrate the **architecture and user experience**, not full functionality.
