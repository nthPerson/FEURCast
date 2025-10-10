# Technical Architecture Documentation

## System Overview

FUREcast implements a **hybrid AI architecture** combining:
- Classical ML (GradientBoostingRegressor) for predictions
- Large Language Models (GPT-4) for orchestration and explanation
- Deterministic tools for data processing

This document explains how to extend this demo into a production system.

---

## Directory Structure

```
gbr_ui_test/
├── app.py                  # Streamlit UI - main entry point
├── simulator.py            # Data & tool simulations (REPLACE in production)
├── llm_interface.py        # LLM orchestration (KEEP in production)
├── requirements.txt        # Python dependencies
├── setup.sh               # Quick setup script
├── README.md              # User documentation
├── DEMO_GUIDE.md          # Presentation guide
├── TROUBLESHOOTING.md     # Common issues
├── ARCHITECTURE.md        # This file
└── .env.example           # Environment template
```

---

## Component Deep Dive

### 1. `app.py` - User Interface Layer

**Current Implementation:**
- Streamlit-based web interface
- Two modes: Lite (basic) and Pro (LLM-enabled)
- Session state management
- Chart rendering via Plotly

**Production Considerations:**
- Add authentication/user sessions
- Implement proper error boundaries
- Add logging for user queries
- Consider rate limiting for LLM queries
- Add caching with `@st.cache_data`

**Key Functions:**
```python
render_lite_mode()      # Simple prediction interface
render_pro_mode()       # NL query interface
render_prediction_card() # Model output display
render_feature_importance() # Explainability UI
```

---

### 2. `simulator.py` - Tool & Data Layer

**Current Implementation:**
All functions return simulated data. In production, replace with:

#### `fetch_prices(tickers, start, end)`
**Current:** Generates synthetic OHLCV data
**Production:**
```python
import yfinance as yf

def fetch_prices(tickers, start, end):
    """Fetch real market data"""
    data = yf.download(tickers, start=start, end=end, group_by='ticker')
    # Transform to standard format
    # Cache in SQLite
    return df
```

#### `compute_risk(df, metric, window)`
**Current:** Simulates volatility/Sharpe/drawdown
**Production:**
```python
def compute_risk(df, metric='volatility', window=60):
    """Calculate real risk metrics"""
    results = {}
    for ticker in df['ticker'].unique():
        returns = df[df['ticker'] == ticker]['close'].pct_change()
        
        if metric == 'volatility':
            vol = returns.tail(window).std() * np.sqrt(252)
            results[ticker] = vol
        # ... other metrics
    
    return results
```

#### `predict_splg()`
**Current:** LLM generates simulated prediction
**Production:**
```python
import joblib

# Load trained model
model = joblib.load('models/gbr_model.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_names = joblib.load('models/feature_names.pkl')

def predict_splg():
    """Get GBR prediction for next-day SPLG return"""
    # Fetch latest data
    df = fetch_prices(['SPLG'], start='2024-01-01', end='today')
    
    # Engineer features (same as training)
    features = engineer_features(df)
    latest = features.iloc[-1][feature_names]
    
    # Scale and predict
    X = scaler.transform([latest])
    pred = model.predict(X)[0]
    
    # Get feature importances
    importances = model.feature_importances_
    top_features = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    
    return {
        'predicted_return': pred,
        'direction': 'up' if pred > 0.002 else 'down' if pred < -0.002 else 'neutral',
        'confidence': calculate_confidence(model, X),
        'top_features': [
            {'name': name, 'importance': imp}
            for name, imp in top_features
        ]
    }
```

#### Visualization Functions
**Keep these mostly as-is**, but feed them real data:
- `create_price_chart()` - Works with real dataframes
- `create_sector_risk_treemap()` - Update sector data source
- `create_feature_importance_chart()` - Same interface
- `viz_from_spec()` - Keep the spec-based approach

---

### 3. `llm_interface.py` - LLM Orchestration Layer

**Current Implementation:**
This is mostly **production-ready**. Minor changes needed:

#### `route_query(query)`
**Current:** Works well
**Production additions:**
- Add query validation/sanitization
- Implement caching for common queries
- Add fallback for API failures
- Log all queries with timestamps

```python
def route_query(query: str, use_cache=True) -> Dict[str, Any]:
    """Route with caching"""
    if use_cache:
        cache_key = hashlib.md5(query.encode()).hexdigest()
        cached = get_from_cache(cache_key)
        if cached:
            return cached
    
    plan = _route_query_llm(query)
    
    if use_cache:
        save_to_cache(cache_key, plan, ttl=3600)
    
    return plan
```

#### `compose_answer(query, tool_results, plan)`
**Current:** Works well
**Production additions:**
- Add citation links to data sources
- Implement answer validation
- Add fact-checking against results
- Store composed answers for auditing

#### `explain_prediction(prediction)`
**Current:** Works well
**Keep as-is** - this demonstrates educational value

---

## Data Flow Diagram

```
┌─────────────────┐
│   User Input    │
│  (Streamlit)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM Router     │  ← llm_interface.route_query()
│ (Intent Class)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Tool Planner   │  ← Generates JSON execution plan
│   (LLM JSON)    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│        Tool Executor                │
│  ┌────────────┐  ┌──────────────┐  │
│  │ fetch_     │  │ predict_     │  │
│  │ prices()   │  │ splg()       │  │
│  └────────────┘  └──────────────┘  │
│  ┌────────────┐  ┌──────────────┐  │
│  │ compute_   │  │ viz_from_    │  │
│  │ risk()     │  │ spec()       │  │
│  └────────────┘  └──────────────┘  │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ Answer Composer │  ← llm_interface.compose_answer()
│  (LLM Synth)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  UI Renderer    │
│   (Streamlit)   │
└─────────────────┘
```

---

## Database Schema (Production)

### Table: `prices`
```sql
CREATE TABLE prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    date DATE NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER,
    UNIQUE(ticker, date)
);
CREATE INDEX idx_ticker_date ON prices(ticker, date);
```

### Table: `predictions`
```sql
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    ticker TEXT NOT NULL,
    predicted_return REAL,
    direction TEXT,
    confidence REAL,
    model_version TEXT
);
```

### Table: `queries`
```sql
CREATE TABLE queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    user_id TEXT,
    query TEXT,
    intent TEXT,
    tools_used TEXT, -- JSON
    response TEXT
);
```

### Table: `sectors`
```sql
CREATE TABLE sectors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sector TEXT NOT NULL UNIQUE,
    etf_ticker TEXT,
    weight_in_splg REAL
);
```

---

## Feature Engineering Pipeline

For GBR model training, implement these features:

### Price-Based Features
```python
def engineer_price_features(df):
    """Technical indicators from price data"""
    df['returns'] = df['close'].pct_change()
    df['returns_1d_lag'] = df['returns'].shift(1)
    df['returns_3d_lag'] = df['returns'].shift(3)
    df['returns_5d_lag'] = df['returns'].shift(5)
    
    df['ma_5'] = df['close'].rolling(5).mean()
    df['ma_20'] = df['close'].rolling(20).mean()
    df['ma_50'] = df['close'].rolling(50).mean()
    df['ma_200'] = df['close'].rolling(200).mean()
    
    df['volatility_5d'] = df['returns'].rolling(5).std()
    df['volatility_20d'] = df['returns'].rolling(20).std()
    
    return df
```

### Technical Indicators
```python
import ta

def engineer_technical_features(df):
    """RSI, MACD, Bollinger Bands"""
    df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    bollinger = ta.volatility.BollingerBands(df['close'])
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_low'] = bollinger.bollinger_lband()
    df['bb_width'] = df['bb_high'] - df['bb_low']
    
    return df
```

### Temporal Features
```python
def engineer_temporal_features(df):
    """Time-based features"""
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    
    return df
```

### Crisis Flags
```python
def engineer_crisis_features(df, crisis_dates):
    """Market crisis indicators"""
    df['in_crisis'] = 0
    df['days_since_crisis'] = 999
    
    for crisis_start, crisis_end in crisis_dates:
        mask = (df['date'] >= crisis_start) & (df['date'] <= crisis_end)
        df.loc[mask, 'in_crisis'] = 1
    
    # Calculate days since last crisis
    # ... implementation
    
    return df
```

---

## Model Training Pipeline

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

def train_gbr_model(df):
    """Train GBR model with time-series cross-validation"""
    
    # Engineer all features
    df = engineer_price_features(df)
    df = engineer_technical_features(df)
    df = engineer_temporal_features(df)
    df = engineer_crisis_features(df, CRISIS_DATES)
    
    # Define target (next-day return)
    df['target'] = df['returns'].shift(-1)
    
    # Drop NaN rows
    df = df.dropna()
    
    # Feature selection
    feature_cols = [c for c in df.columns if c not in 
                   ['date', 'ticker', 'target', 'open', 'high', 'low', 'close', 'volume']]
    
    X = df[feature_cols]
    y = df['target']
    
    # Time series split
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Hyperparameter tuning
    best_params = {
        'n_estimators': 200,
        'max_depth': 5,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'min_samples_split': 50
    }
    
    # Train final model
    model = GradientBoostingRegressor(**best_params)
    model.fit(X, y)
    
    # Evaluate
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    
    # Direction accuracy
    direction_acc = np.mean((y > 0) == (y_pred > 0))
    
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Direction Accuracy: {direction_acc:.2%}")
    
    # Save model
    joblib.dump(model, 'models/gbr_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(feature_cols, 'models/feature_names.pkl')
    
    return model
```

---

## API Integration

### yfinance (Primary)
```python
import yfinance as yf

def fetch_from_yfinance(ticker, start, end):
    """Free, reliable, good for MVP"""
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        return data
    except Exception as e:
        logging.error(f"yfinance error: {e}")
        return None
```

### Alpha Vantage (Backup)
```python
from alpha_vantage.timeseries import TimeSeries

def fetch_from_alphavantage(ticker):
    """Better for real-time, needs API key"""
    ts = TimeSeries(key=ALPHAVANTAGE_KEY, output_format='pandas')
    data, meta = ts.get_daily(symbol=ticker, outputsize='full')
    return data
```

---

## Deployment Checklist

### Streamlit Community Cloud
1. ✅ Push to GitHub repository
2. ✅ Add `requirements.txt`
3. ✅ Add `secrets.toml` for API keys:
   ```toml
   OPENAI_API_KEY = "sk-..."
   ```
4. ✅ Test locally first
5. ✅ Deploy via Streamlit Cloud dashboard
6. ✅ Set up custom domain (optional)

### Environment Variables Needed
- `OPENAI_API_KEY` - For LLM orchestration
- `ALPHAVANTAGE_KEY` - (Optional) For market data
- `DATABASE_URL` - (Optional) For external DB

### Performance Optimization
```python
# Cache expensive operations
@st.cache_data(ttl=3600)
def load_model():
    return joblib.load('models/gbr_model.pkl')

@st.cache_data(ttl=600)
def fetch_latest_prices(ticker):
    return yf.download(ticker, period='1y')
```

---

## Testing Strategy

### Unit Tests
```python
# tests/test_simulator.py
def test_fetch_prices():
    df = fetch_prices(['SPLG'], '2024-01-01', '2024-01-31')
    assert not df.empty
    assert 'close' in df.columns
    assert df['close'].min() > 0

def test_predict_splg():
    pred = predict_splg()
    assert 'predicted_return' in pred
    assert 'direction' in pred
    assert pred['confidence'] >= 0 and pred['confidence'] <= 1
```

### Integration Tests
```python
# tests/test_integration.py
def test_end_to_end_query():
    query = "Is now a good time to invest?"
    plan = route_query(query)
    assert 'intent' in plan
    assert 'tools' in plan
```

---

## Cost Estimation

### OpenAI API Costs (GPT-4o-mini)
- Input: $0.150 per 1M tokens
- Output: $0.600 per 1M tokens

**Typical Query:**
- Route: ~500 tokens ($0.0003)
- Compose: ~800 tokens ($0.0005)
- **Total per query: ~$0.001**

**Monthly estimate (100 users, 10 queries each):**
- 1,000 queries × $0.001 = **$1/month**

Very affordable for educational project!

### Alternatives if Needed
- Use GPT-4o-mini (even cheaper)
- Cache common queries
- Implement query rate limits
- Use open-source models (Llama, Mistral) via Hugging Face

---

## Security Considerations

1. **API Key Protection**
   - Never commit `.env` to Git
   - Use Streamlit secrets in production
   - Rotate keys regularly

2. **Input Validation**
   - Sanitize user queries
   - Whitelist allowed tools
   - Validate LLM outputs before execution

3. **Rate Limiting**
   ```python
   from functools import lru_cache
   from time import time
   
   @lru_cache(maxsize=100)
   def rate_limited_query(user_id, query, timestamp):
       # Allow max 10 queries per minute
       pass
   ```

4. **Logging**
   ```python
   import logging
   
   logging.basicConfig(
       filename='furecast.log',
       level=logging.INFO,
       format='%(asctime)s - %(levelname)s - %(message)s'
   )
   ```

---

## Performance Benchmarks (Target)

- Page load: < 2 seconds
- Model prediction: < 500ms
- LLM routing: < 2 seconds
- Chart rendering: < 1 second
- Total query response: < 5 seconds

---

## Future Enhancements

1. **Multi-Model Ensemble**
   - Combine GBR + LSTM + Prophet
   - Weighted voting system

2. **Real-Time Updates**
   - WebSocket for live prices
   - Auto-refresh predictions

3. **User Accounts**
   - Save favorite queries
   - Track prediction accuracy over time
   - Personalized dashboards

4. **Mobile Optimization**
   - Responsive design
   - Touch-friendly charts

5. **Export Features**
   - PDF reports
   - CSV data downloads
   - Email summaries

---

## Contact & Contribution

For questions about this architecture:
1. Review this document
2. Check TROUBLESHOOTING.md
3. Consult with team lead

**Key Principle:** Keep tools deterministic, LLM for orchestration only.
