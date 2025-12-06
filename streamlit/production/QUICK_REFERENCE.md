# Quick Reference: Treemap Integration

## 🎯 What Changed

### Before
```
Pro Mode → Quick Analytics → 3 Tabs
├── Sector Risk (simulated aggregated data)
├── Price Trends
└── Feature Analysis
```

### After
```
Pro Mode → Quick Analytics → 4 Tabs
├── Sector Risk (simulated aggregated data)
├── Holdings Detail (NEW! Real SPLG data with drill-down)
├── Price Trends
└── Feature Analysis
```

## 🔍 How to Use

### Method 1: Tab Navigation
1. Open app in Pro Mode
2. Scroll to "Quick Analytics"
3. Click "Holdings Detail" tab
4. Select color metric from dropdown
5. Click sectors to drill down
6. Hover over holdings for details

### Method 2: Natural Language
1. Type in query box: "Show me the top holdings in SPLG"
2. Click "Analyze"
3. View treemap with color selector
4. Explore expandable sector summary

## 📊 Color Metric Guide

| Metric | What It Shows | Best For |
|--------|---------------|----------|
| **DailyChangePct** | Daily price change % | Performance tracking |
| **PE** | Price-to-Earnings | Value investing |
| **Beta** | Volatility vs market | Risk assessment |
| **DividendYield** | Annual dividend % | Income investing |

## 🎨 Visual Legend

**Treemap Colors (DailyChangePct selected):**
- 🟢 Green = Positive returns (stock up today)
- 🟡 Yellow = Neutral (near 0% change)
- 🔴 Red = Negative returns (stock down today)

**Box Sizes:**
- Larger boxes = Higher SPLG weight
- Smaller boxes = Lower SPLG weight

## 📈 Example Use Cases

### 1. Finding Top Holdings
**Query:** "Show me the top holdings in SPLG"
**Result:** Treemap shows NVDA (8.0%), MSFT (6.8%), AAPL (6.5%) as largest boxes

### 2. Analyzing Sector Risk
**Action:** Select "Beta" as color metric
**Insight:** See which holdings/sectors have highest volatility

### 3. Income Investing
**Action:** Select "DividendYield" as color metric
**Insight:** Identify high dividend payers (darker colors)

### 4. Value Hunting
**Action:** Select "PE" as color metric
**Insight:** Find potentially undervalued stocks (lower PE ratios)

### 5. Daily Performance
**Action:** Default "DailyChangePct" metric
**Insight:** See which sectors/stocks moved most today

## 🔧 Technical Details

### Data Path
```
/home/robert/FEURCast/data/treemap_nodes.csv
```

### New Functions (in simulator.py)
```python
create_sector_holdings_treemap(color_metric='DailyChangePct')
get_sector_summary()
```

### Files Modified
- ✅ `/streamlit/production/app.py`
- ✅ `/streamlit/production/simulator.py`

### Files Created
- 📄 `TREEMAP_INTEGRATION.md` (detailed docs)
- 📄 `TREEMAP_COMPLETE.md` (this summary)
- 📄 `verify_treemap.py` (test script)

## 🚀 Launch Command

```bash
cd /home/robert/FEURCast/streamlit/production
streamlit run app.py
```

## 💡 Tips

1. **Click to Drill Down**: Click any sector to see its holdings
2. **Hover for Details**: Hover over any box to see all metrics
3. **Switch Metrics**: Try different color metrics to gain different insights
4. **Check Summary**: Scroll to see the sector summary table
5. **Use Queries**: Try natural language queries about holdings

## 🤖 Ask FUREcast Tooling Plan

### Objectives
- Keep the learning agent grounded in approved CSV data: `Financial Crisis Timeline by Day (since 2005).csv`, `investment_glossary.csv`, `holdings-with-sectors.csv`, `rich_features_SPLG_history_full.csv`, `treemap_nodes.csv`.
- Execute the LLM “System Plan” end-to-end: run each tool, store outputs, and render the requested visualization or surface a graceful fallback.
- Support the six Example Queries in the Ask FUREcast UI without synthetic data.

### Implementation Steps
1. **Data Access Tool** – add `query_rich_features` (no caching needed) that filters `rich_features_SPLG_history_full.csv` by dates, sectors, and columns, returning JSON-safe payloads plus warnings when filters exceed data bounds.
2. **Planner Schema** – update `route_query` to emit `plan_version`, stricter visualization specs, and tool definitions for `query_rich_features`, `predict_splg`, `compute_risk`, `fetch_prices`, `viz_from_spec`. Reject specs that need unavailable data.
3. **Execution Layer** – create a coordinator that iterates through the tool list, calls the simulator helpers, stores `results[data_key]`, and logs recoverable errors instead of crashing.
4. **Visualization Factory (Complete)** – `viz_from_spec` now resolves `spec['data_key']` against executed `tool_results`, supports `price`, `line`, `bar`, `treemap`, `feature_importance`, and `table`, and validates axes/metrics/date ranges. Errors bubble up as descriptive messages so the UI can warn instead of crashing.
5. **Sector Metrics** – derive comparisons and risk ranks from `treemap_nodes.csv` or `holdings-with-sectors.csv` (weights, beta, dividend yield, etc.) for bar charts and narratives.
6. **UI Integration (Complete)** – `app.py` drives Ask FUREcast via `generate_tool_plan` ➜ `execute_tool_plan`, shows raw `tool_results`, passes the `visualization` spec into `viz_from_spec`, renders tables straight from planner data, and removes keyword-based fallbacks.

### Visualization Spec Schema (v1)
```json
{
	"plan_version": 1,
	"visualization": {
		"type": "price|line|bar|treemap|feature_importance|table",
		"data_key": "name-of-tool-result",
		"title": "Optional chart title",
		"specs": {
			"x": "column name",
			"y": "column name",
			"group": "optional grouping column",
			"metric": "close|open|high|low|...",
			"start_date": "YYYY-MM-DD",
			"end_date": "YYYY-MM-DD",
			"sectors": ["Technology", "Utilities"],
			"top_n": 10
		}
	}
}
```

### Safety Requirements
- Every visualization call must verify that its source CSV exists. If not, display a user-facing warning and skip the chart.
- Specs that reference unsupported columns or missing tool results should be caught early with clear feedback (e.g., “Visualization skipped: requested metric not available in holdings data”).
- Keep runtime exceptions from propagating to Streamlit; log the issue and fall back to text responses.

### Future Enhancements
- Extend `query_rich_features` with lightweight aggregations needed by future Example Queries (rolling averages, percentile ranks, etc.).
- Document new spec fields inline as they are introduced and bump `plan_version` so the UI can branch safely.
- Add automated checks (or tests) ensuring the six Example Queries run successfully after tooling changes.

## 📊 Data Overview

**Total Holdings:** 305 companies
**Sectors:** 11 (Technology, Healthcare, Financial Services, etc.)
**Top 3 by Weight:**
1. NVDA (Technology) - 8.0%
2. MSFT (Technology) - 6.8%
3. AAPL (Technology) - 6.5%

**Total Technology Weight:** ~28-30%
**Total Healthcare Weight:** ~13%
**Total Financial Services Weight:** ~13%

## ⚡ Quick Troubleshooting

**Issue:** Treemap not showing
**Fix:** Check that `/data/treemap_nodes.csv` exists

**Issue:** Shows simulated data instead
**Fix:** Verify CSV file path is correct

**Issue:** Can't drill down
**Fix:** Make sure you're clicking on sector boxes, not company boxes

**Issue:** Hover not working
**Fix:** Try moving mouse slowly over boxes

---

**Ready to explore your SPLG holdings! 🎉**
