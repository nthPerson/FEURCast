"""
FUREcast - GBR Demo Application
"""

import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import sys
import os
import pandas as pd
import io
from PIL import Image

# Safe rerun helper to support Streamlit versions where experimental_rerun / rerun may be missing
def safe_rerun():
    """
    Attempt to trigger a Streamlit rerun in a way that's compatible across versions.
    Falls back to st.stop() if no programmatic rerun API is available.
    """
    try:
        # Preferred: modern Streamlit
        st.experimental_rerun()
    except Exception:
        try:
            # Older Streamlit versions
            st.rerun()
        except Exception:
            # Final fallback: stop the script so the UI can be refreshed by the user
            st.stop()
# Import from pred_model package
from pred_model import get_latest_date_in_dataset

# Import from production package modules
from simulator import (
    predict_splg,
    create_price_chart,
    create_sector_risk_treemap,
    create_sector_holdings_treemap,
    get_sector_summary,
    create_feature_importance_chart,
    create_sector_comparison_chart,
)
from llm_interface import (
    route_query,
    compose_answer,
    explain_prediction
)

# Load environment variables
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '../..', '.env'))
MARKETSENTIMENT_API_KEY = os.getenv('MARKETSENTIMENT_API_KEY')
FRED_KEY= os.getenv('FRED_KEY')
# ---------- STREAMLIT CONFIGURATION ----------

# Page configuration
st.set_page_config(
    page_title="FUREcast - SPLG Analytics Demo",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #2E86AB; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.2rem; color: #666; margin-bottom: 2rem; }
    .feature-item { display: flex; justify-content: space-between; padding: 0.5rem; border-bottom: 1px solid #eee; }
    .disclaimer { background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 1rem; border-radius: 4px; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)


# ---------- SESSION STATE ----------
def initialize_session_state():
    if 'mode' not in st.session_state: st.session_state.mode = 'Lite'
    if 'prediction_cache' not in st.session_state: st.session_state.prediction_cache = None
    if 'last_query' not in st.session_state: st.session_state.last_query = ""
    if 'execute_query' not in st.session_state: st.session_state.execute_query = False
    # Metric display names to DataFrame column mapping
    if 'metric_mapping' not in st.session_state:
        st.session_state.metric_mapping = {
            "Closing": "close",
            "Opening": "open",
            "Daily High": "high",
            "Daily Low": "low",
            "Daily Current": "current_price"
        }
    if "metric" not in st.session_state: st.session_state.metric = "Closing"
    if "start_date" not in st.session_state: 
        st.session_state.start_date = pd.to_datetime("2023-01-01")
    if "end_date" not in st.session_state: 
        st.session_state.end_date = pd.to_datetime("2025-09-24")  # Maximum date in our dataset
    if "max_dataset_date" not in st.session_state:
        st.session_state.max_dataset_date = get_latest_date_in_dataset()
        # st.session_state.max_dataset_date = pd.to_datetime("2025-09-24")
    # New: control whether event highlights are shown
    if "show_events" not in st.session_state:
        st.session_state.show_events = True
    # New: current page for simple navigation ('home' or 'performance')
    if "page" not in st.session_state:
        st.session_state.page = "home"


# ---------- SIDEBAR ----------
def render_sidebar():
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/c/c8/FURECast_SPLG.png", width='stretch')
        st.markdown("### 🎓 Demo Mode")
        st.session_state.mode = st.radio(
            "Select Mode:",
            options=['Lite', 'Pro'],
            key="sidebar_mode_radio",  # ✅ unique key
            help="Lite: Basic prediction + charts\nPro: Full LLM interface + all tools"
        )
        # Navigation buttons: Home + Model Performance (styled, centered)
        # Inject sidebar-specific button styles (colors, sizing, boxed background)
        st.markdown(
            """
        
            """,
            unsafe_allow_html=True,
        )

        # Place the two navigation buttons inside a small boxed area and center them
        st.markdown('<div class="sidebar-button-box">', unsafe_allow_html=True)
        col_l, col_c, col_r = st.columns([1, 6, 1])
        with col_c:
            if st.button("Home", key="sidebar_home", use_container_width=True):
                st.session_state.page = "home"
                safe_rerun()
            if st.button("Model Performance Metrics", key="open_model_perf", use_container_width=True):
                st.session_state.page = "performance"
                safe_rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.sidebar.header("Chart Filters")
        st.session_state.metric = st.selectbox(
            "Select Price Metric",
            options=["Closing", "Opening", "Daily High", "Daily Low", "Daily Current"],
            index=["Closing", "Opening", "Daily High", "Daily Low", "Daily Current"].index(st.session_state.metric),
            key="metric_selector"
        )

        # (Removed) Show Event Highlights control was here

        # Define the maximum available date from our dataset (last available date in historical data)
        MAX_DATASET_DATE = pd.to_datetime(get_latest_date_in_dataset())
    
        # Convert Timestamp to date for comparison and display
        current_end_date = st.session_state.end_date.date() if isinstance(st.session_state.end_date, pd.Timestamp) else st.session_state.end_date
        current_start_date = st.session_state.start_date.date() if isinstance(st.session_state.start_date, pd.Timestamp) else st.session_state.start_date
        
        st.session_state.start_date = st.date_input(
            "Start Date",
            value=current_start_date,
            key="start_date_selector",
            max_value=MAX_DATASET_DATE.date()
        )
        st.session_state.end_date = st.date_input(
            "End Date",
            value=min(current_end_date, MAX_DATASET_DATE.date()),
            key="end_date_selector",
            max_value=MAX_DATASET_DATE.date()
        )
        
        # Convert back to Timestamp for consistency
        st.session_state.start_date = pd.to_datetime(st.session_state.start_date)
        st.session_state.end_date = pd.to_datetime(st.session_state.end_date)

        st.markdown("---")
        st.markdown("### ℹ️ About FUREcast")
        st.markdown("""
        Demo showcasing SPLG ETF analysis with GradientBoostingRegressor and LLM orchestration.
        All data and predictions are simulated.
        """)
        with st.expander("📊 Data Sources (Simulated)", expanded=False):
            st.markdown("- SPLG historical data (2005-2025)\n- Sector ETF data\n- Technical indicators\n- Risk metrics")
        with st.expander("🛠️ Architecture", expanded=False):
            st.markdown("1. User Query\n2. LLM Router\n3. Tool Planner\n4. Tool Executor\n5. Answer Composer\n6. UI Renderer")
        st.markdown('<div class="disclaimer">⚠️ <strong>Educational Only</strong><br>Not financial advice.</div>', unsafe_allow_html=True)


# ---------- LITE MODE ----------
def render_prediction_card(prediction):
    direction = prediction['direction']
    pred_return = prediction['predicted_return'] * 100
    confidence = prediction['confidence']
    if direction == 'up': 
        color, emoji = '#28a745', '📈'
    elif direction == 'down': 
        color, emoji = '#dc3545', '📉'
    else: 
        color, emoji = '#ffc107', '➡️'

    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        st.markdown(f"""
        <div style='background-color: {color}; padding: 1.5rem; border-radius: 10px; color: white;'>
            <h2 style='margin: 0; color: white;'>{emoji} Model Prediction</h2>
            <h1 style='margin: 0.5rem 0; color: white;'>{direction.upper()}</h1>
            <p style='margin: 0; font-size: 1.2rem;'>Expected Return: {pred_return:+.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric(
            "Confidence Score",
            confidence, # confidence is currently a string (used to be a float), so just pass as parameter
            # f"{float(confidence):.1%}",  # original AI-generated confidence was a float
            help="Model's confidence in this prediction"
        )
    
    with col3:
        st.metric(
            "Timeframe",
            "Next Day",
            help="Prediction horizon"
        )


def render_feature_importance(features):
    st.markdown("##### 🔍 Top Model Features")
    col1, col2 = st.columns([3,2])
    with col1:
        fig = create_feature_importance_chart(features)
        st.plotly_chart(fig, config={'width': 'stretch'}, key='feature_importance_chart')
    with col2:
        for i, feat in enumerate(features, 1):
            st.markdown(f"<div class='feature-item'><span><strong>{i}.</strong> {feat['name']}</span>"
                        f"<span><strong>{feat['importance']:.2%}</strong></span></div>", unsafe_allow_html=True)
        try:
            st.markdown(f"**Interpretation:** {explain_prediction(st.session_state.prediction_cache)}")
        except:
            st.markdown("*Feature importance shows key indicators influencing prediction.*")


def render_lite_mode():
    
    import requests
    # 🗞️ News Channel (TOP SECTION)
    url = f"https://finnhub.io/api/v1/news?category=general&token={MARKETSENTIMENT_API_KEY}"

    try:
        response = requests.get(url, timeout=5)
        articles = response.json()
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        articles = []

    st.markdown("##### 🗞️ Latest Market Headlines", unsafe_allow_html=True)

    if not articles or not isinstance(articles, list):
        st.warning("No recent S&P 500 news found.")
    else:
        articles = sorted(articles, key=lambda x: x.get("datetime", 0), reverse=True)
        display_articles = articles[:20]

        headlines_html = ""
        for a in display_articles:
            headline = a.get("headline", "Untitled")
            link = a.get("url", "#")
            sentiment = "neutral"
            if any(w in headline.lower() for w in ["up", "gain", "growth", "rally", "record"]):
                sentiment = "positive"
            elif any(w in headline.lower() for w in ["down", "loss", "drop", "fall", "decline"]):
                sentiment = "negative"

            color = (
                "#00ff99" if sentiment == "positive"
                else "#ff4d4d" if sentiment == "negative"
                else "white"
            )
            headlines_html += f'📈 <a href="{link}" target="_blank" style="color:{color}; text-decoration:none; margin-right:50px;">{headline}</a>'

        st.markdown(
            f"""
            <div style="background-color:#001f3f; padding:5px; border-radius:8px; margin-bottom:5px;">
                <marquee behavior="scroll" direction="left" scrollamount="5"
                         style="font-size:20px; font-weight:500;">
                    {headlines_html}
                </marquee>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
    # =============================
    # 🌍 Macro & Sentiment Dashboard (FRED + Sentiment Combined)
    # =============================
    import requests
    FRED_KEY = "167c610d0808df0df6fc03d8a7c9f611"  # 🔑 Replace with your own
    base_url = "https://api.stlouisfed.org/fred/series/observations"

    # --- Fetch Unemployment Rate (UNRATE) ---
    params_un = {
        "series_id": "UNRATE",
        "api_key": FRED_KEY,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 1
    }
    try:
        resp_un = requests.get(base_url, params=params_un, timeout=5)
        data_un = resp_un.json().get("observations", [])
        unemployment_rate = float(data_un[0].get("value", 0)) if data_un else None
    except Exception:
        unemployment_rate = None

    # --- Fetch Public Debt to GDP Ratio (GFDEGDQ188S) ---
    params_debt = {
        "series_id": "GFDEGDQ188S",
        "api_key": FRED_KEY,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 1
    }
    try:
        resp_debt = requests.get(base_url, params=params_debt, timeout=5)
        data_debt = resp_debt.json().get("observations", [])
        public_debt_pct = float(data_debt[0].get("value", 0)) if data_debt else None
    except Exception:
        public_debt_pct = None

    # --- Threshold Logic ---
    def indicator_status(value, good_max):
        if value is None:
            return "N/A", "grey"
        if value <= good_max:
            return "Acceptable", "green"
        else:
            return "Bad", "red"

    un_status, un_color = indicator_status(unemployment_rate, 6.0) # Unemployment: 0 to 6% acceptable (the closer to 0 the better) 6.1% and above BAD
    debt_status, debt_color = indicator_status(public_debt_pct, 70.0) #Public Debt: 0 to 70% acceptable (the closer to 0 the better) 71% and above BAD


    # --- Display Combined Indicators ---
    st.markdown("##### 🌍 Macro & Sentiment Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "💼 Unemployment Rate",
            f"{unemployment_rate:.1f}%" if unemployment_rate is not None else "N/A",
            un_status
        )

    with col2:
        st.metric(
            "🏛️ Public Debt (% of GDP)",
            f"{public_debt_pct:.1f}%" if public_debt_pct is not None else "N/A",
            debt_status
        )

    with col3:
        if "bullish" in locals() and bullish is not None:
            sentiment_label = (
                "Bullish" if bullish > 0.55 else
                "Bearish" if bullish < 0.45 else
                "Neutral"
            )
            st.metric(
                "📊 Market Sentiment",
                sentiment_label,
                f"{bullish*100:.1f}%" if bullish else "N/A"
            )
        else:
            pass ## if no data is shown on sentiments then we are passing this section KPI or use if you want to display st.metric("📊 Market Sentiment", " ", " ")

    # --- Optional Styling ---
    st.markdown("""
    <style>[data-testid="stMetricValue"] {font-size: 1.4rem !important;}
    [data-testid="stMetricLabel"] {color: #1c1c1c;}
    </style>
    """, unsafe_allow_html=True)
    # =============================
    # 📈 Core UI: Header + Prediction
    # =============================
    st.markdown('<p class="main-header">📈 FUREcast SPLG Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Educational GBR-Based Market Analytics</p>', unsafe_allow_html=True)

    # --- Run model prediction ---
    if st.session_state.prediction_cache is None or st.button("🔄 Refresh Prediction", key="refresh_lite"):
        st.session_state.prediction_cache = predict_splg()
    prediction = st.session_state.prediction_cache

    render_prediction_card(prediction)
    st.markdown("---")
    render_feature_importance(prediction['top_features'])
    st.markdown("---")

    # =============================
    # 📊 Price Chart Section
    # =============================
    metric = st.session_state.metric
    start_date = st.session_state.start_date
    end_date = st.session_state.end_date
    max_date = st.session_state.max_dataset_date

    if start_date > end_date:
        st.error("🚫 Start date must be before end date.")
        st.stop()
        
    # Ensure end date doesn't exceed dataset limit for price charts
    if end_date > max_date:
        end_date = max_date
        st.warning(f"⚠️ End date adjusted to maximum available date: {max_date.date()}")

    st.markdown(f"##### 📊 {metric} Price Chart")
    with st.spinner("Loading chart..."):
        try:
            # Map display name to DataFrame column name
            df_column = st.session_state.metric_mapping[metric]
            # Pass show_events flag from session state
            fig = create_price_chart(df_column, pd.to_datetime(start_date), pd.to_datetime(end_date), show_events=st.session_state.show_events)
            st.plotly_chart(fig, config={"responsive": True, "width": 'stretch'}, key='price_chart')
        except Exception as e:
            st.error(f"❌ Chart failed to render: {e}")


    # =============================
    # 💡 How to Use Section
    # =============================
    st.info("""
    **💡 How to Use This Tool:**
    1. Review model prediction  
    2. Check confidence score  
    3. Examine feature influence  
    4. Compare with recent price trends  
    *Educational use only.*
    """)


# ---------- PRO MODE ----------
def render_pro_mode():
    """Render Pro mode interface with LLM query capabilities"""
    st.markdown('<p class="main-header">🚀 FUREcast Pro Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Investment Insights & Natural Language Interface</p>', unsafe_allow_html=True)
    
    # Get max dataset date for use throughout Pro mode
    max_dataset_date = st.session_state.max_dataset_date
    
    # Quick prediction card (collapsible)
    with st.expander("📈 Current SPLG Prediction", expanded=True):
        if st.session_state.prediction_cache is None:
            with st.spinner("Generating prediction..."):
                st.session_state.prediction_cache = predict_splg()
        
        prediction = st.session_state.prediction_cache
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Direction", prediction['direction'].upper())
        with col2:
            st.metric("Expected Return", f"{prediction['predicted_return']*100:+.2f}%")
        with col3:
            st.metric("Confidence", f"{prediction['confidence']}")
            # st.metric("Confidence", f"{prediction['confidence']:.1%}")
        with col4:
            if st.button("🔄 Refresh", key="pro_refresh_button"):
                st.session_state.prediction_cache = predict_splg()
                safe_rerun()
    
    st.markdown("---")
    
    # Natural language query interface
    st.markdown("### 💬 Ask FUREcast")
    st.markdown("*Enter your question about SPLG, sectors, risk, or market trends...*")
    
    # Example queries
    with st.expander("💡 Example Queries"):
        examples = [
            "Is now a good time to invest in SPLG?",
            "Which sectors look stable this quarter?",
            "Compare Technology vs Utilities performance",
            "What influenced today's prediction?",
            "Show me the top holdings in SPLG"
        ]
        cols = st.columns(2)
        for i, example in enumerate(examples):
            with cols[i % 2]:
                if st.button(example, key=f"example_query_button_{i}", use_container_width=True):
                    st.session_state.last_query = example
                    st.session_state.execute_query = True
                    st.rerun()
    
    # Query input
    query = st.text_input(
        "Your Question:",
        value=st.session_state.last_query,
        placeholder="e.g., Which sectors have the lowest volatility?",
        label_visibility="collapsed",
        key="query_input_box"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        submit = st.button("🔍 Analyze", type="primary", use_container_width=True, key="analyze_button")
    with col2:
        if st.button("Clear", use_container_width=True, key="clear_button"):
            st.session_state.last_query = ""
            st.session_state.execute_query = False
            st.rerun()
    
    # Execute query if submitted via button OR if example query was clicked
    should_execute = (submit and query) or (st.session_state.execute_query and query)
    
    if should_execute:
        # Reset the execute flag
        st.session_state.execute_query = False
        st.session_state.last_query = query
        
        with st.spinner("🤔 Planning analysis..."):
            # Route query
            plan = route_query(query)
        
        st.markdown("---")
        
        # Show what the system is doing
        with st.expander("🔧 System Plan", expanded=False):
            st.json(plan)
        
        # Execute tools based on plan
        with st.spinner("📊 Executing analysis..."):
            tool_results = {}
            intent = plan.get('intent', 'general_question')
            
            if intent == 'market_outlook' or 'predict' in intent:
                tool_results['prediction'] = st.session_state.prediction_cache
            
            if intent == 'sector_analysis' or 'sector' in query.lower():
                tool_results['sector_data'] = {
                    'risk_rankings': {
                        'Consumer Staples': 'Low Risk',
                        'Utilities': 'Low Risk',
                        'Healthcare': 'Medium Risk',
                        'Technology': 'High Risk',
                        'Energy': 'High Risk'
                    }
                }
            
            if intent == 'feature_explanation' or 'feature' in query.lower() or 'influence' in query.lower():
                tool_results['features'] = prediction['top_features']
        
        # Generate response
        with st.spinner("✍️ Composing answer..."):
            response = compose_answer(query, tool_results, plan)
        
        # Display results
        st.markdown("### 📝 Analysis Results")
        st.markdown(response)
        
        st.markdown("---")
        
        # Generate visualizations based on intent
        st.markdown("### 📊 Visualizations")
        
        viz_plan = plan.get('visualization') or {}
        viz_type = viz_plan.get('type', 'price')
        
        if 'holding' in query.lower() or 'stock' in query.lower() or 'company' in query.lower():
            # Show detailed holdings treemap
            st.markdown("**SPLG Holdings Drill-Down**")
            color_metric = st.selectbox(
                "Color by:",
                ["DailyChangePct", "PE", "Beta", "DividendYield"],
                index=0,
                key="query_treemap_color"
            )
            fig = create_sector_holdings_treemap(color_metric=color_metric)
            st.plotly_chart(fig, config={"width": 'stretch'}, key='sector_holdings_treemap')
            
            # Show summary table
            summary_df = get_sector_summary()
            if not summary_df.empty:
                with st.expander("📊 Sector Summary Table", expanded=False):
                    st.dataframe(summary_df, width='stretch', hide_index=True)
        
        elif 'sector' in query.lower() and 'risk' in query.lower():
            fig = create_sector_risk_treemap()
            st.plotly_chart(fig, config={"width": 'stretch'}, key='sector_risk_treemap')
        
        elif 'compare' in query.lower() or 'performance' in query.lower():
            # Extract sectors from query (simplified)
            sectors = ['Technology', 'Utilities', 'Healthcare']
            fig = create_sector_comparison_chart(sectors)
            st.plotly_chart(fig, config={"width": 'stretch'}, key='compare_chart')
        
        elif 'feature' in query.lower() or 'influence' in query.lower():
            fig = create_feature_importance_chart(prediction['top_features'])
            st.plotly_chart(fig, config={"width": 'stretch'}, key='feature_importance_chart')
        
        else:
            # Default to price chart - FIX: Add all required parameters
            default_start = max_dataset_date - pd.Timedelta(days=180)
            fig = create_price_chart('close', default_start, max_dataset_date, show_events=st.session_state.show_events)
            st.plotly_chart(fig, config={"width": 'stretch'}, key='price_chart')
    
    elif not query and submit:
        st.warning("Please enter a question to analyze.")
    
    # Additional tools section - MOVED OUTSIDE the conditional logic
    st.markdown("---")
    st.markdown("### 📊 Quick Analytics")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Sector Risk", "Holdings Detail", "Price Trends", "Feature Analysis"])

    with tab1:
        st.markdown("**Sector Risk Treemap** - Size by market cap, color by volatility")
        fig = create_sector_risk_treemap()
        st.plotly_chart(fig, config={"width": 'stretch'}, key='sector_risk_treemap_2')

    with tab2:
        st.markdown("**SPLG Holdings Drill-Down** - Explore sectors and individual holdings")
        st.caption("Size = SPLG Weight (%). Click on sectors to drill down into holdings. Hover for detailed KPIs.")
        
        color_option = st.selectbox(
            "Color by:",
            ["DailyChangePct", "PE", "Beta", "DividendYield"],
            index=0,
            key="holdings_treemap_color"
        )
        fig = create_sector_holdings_treemap(color_metric=color_option)
        st.plotly_chart(fig, config={"width": 'stretch'}, key='sector_holdings_treemap_2')
        
        st.markdown("**Sector Summary (Weighted by SPLG)**")
        summary_df = get_sector_summary()
        if not summary_df.empty:
            st.dataframe(summary_df, width='stretch', hide_index=True)

    with tab3:
        st.markdown("**SPLG Historical Performance**")
        metric = st.session_state.metric
        start_date = st.session_state.start_date
        end_date = st.session_state.end_date
        # Map display name to DataFrame column name
        df_column = st.session_state.metric_mapping[metric]
        # Pass show_events flag
        fig = create_price_chart(df_column, pd.to_datetime(start_date), pd.to_datetime(end_date), show_events=st.session_state.show_events)
        st.plotly_chart(fig, config={"width": 'stretch'}, key='price_chart_2')

    with tab4:
        st.markdown("**Current Model Feature Importance**")
        fig = create_feature_importance_chart(prediction['top_features'])
        st.plotly_chart(fig, config={"width": 'stretch'}, key='feature_importance_chart_2')
        
        st.markdown("**Explanation:**")
        with st.spinner("Generating explanation..."):
            try:
                explanation = explain_prediction(st.session_state.prediction_cache)
                st.info(explanation)
            except:
                st.info("Technical indicators drive the model's predictions by capturing market momentum, trends, and volatility patterns.")



# ---------- MAIN ----------
def main():
    initialize_session_state()
    render_sidebar()

    # Simple page routing
    if st.session_state.page == "performance":
        render_performance_page()
        return

    if st.session_state.mode == 'Lite':
        render_lite_mode()
    else:
        render_pro_mode()

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; color:#666; padding:2rem;'>
        <p><strong>FUREcast GBR Demo</strong> | Educational Analytics Platform</p>
        <p style='font-size:0.9rem;'>Built with Streamlit, OpenAI, and Plotly | Demo Version</p>
        <p style='font-size:0.8rem;'>⚠️ All data simulated for demonstration purposes</p>
    </div>
    """, unsafe_allow_html=True)


# --- remove or comment out any earlier immediate call to main() so functions defined later are available ---
# if __name__ == "__main__":
#     main()

def render_performance_page():
    """
    Show model performance plots in tabs (like Quick Analytics).
    Prefer dynamic plotting functions from pred_model.plots; fallback to PNGs
    in pred_model/plots. Each plot gets its own tab for selection.
    """
    import inspect

    st.markdown('<p class="main-header">📈 Model Performance Metrics</p>', unsafe_allow_html=True)
    st.markdown("Select a plot tab to view. Prefer dynamically generated figures from pred_model.plots, otherwise show PNGs.")

    items = []  # list of (label, fig_or_path)

    # 1) Try to load plotting functions from pred_model.plots
    try:
        import pred_model.plots as perf_plots
        preferred = [
            "make_pred_vs_actual_figure",
            "make_feature_importance_figure",
            "make_residuals_figure",
            "make_error_distribution_figure",
            "make_training_validation_test_plots",
        ]
        added = set()
        for name in preferred:
            if hasattr(perf_plots, name) and callable(getattr(perf_plots, name)):
                try:
                    fig = getattr(perf_plots, name)()
                    items.append((name, fig))
                    added.add(name)
                except Exception:
                    pass

        # auto-discover additional no-arg make_* functions
        for n, fn in inspect.getmembers(perf_plots, inspect.isfunction):
            if n.startswith("make_") and n not in added:
                sig = inspect.signature(fn)
                # allow functions with only optional args or *args/**kwargs
                if all(p.default != inspect._empty or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
                       for p in sig.parameters.values()):
                    try:
                        fig = fn()
                        items.append((n, fig))
                    except Exception:
                        pass
    except Exception:
        # dynamic generation not available; we'll fallback to PNGs below
        pass

    # 2) Fallback: PNG files in pred_model/plots
    if not items:
        plots_dir = os.path.join(os.path.dirname(__file__), "pred_model", "plots")
        if os.path.isdir(plots_dir):
            png_files = sorted([f for f in os.listdir(plots_dir) if f.lower().endswith(".png")])
            for fname in png_files:
                items.append((os.path.splitext(fname)[0], os.path.join(plots_dir, fname)))

    if not items:
        st.info("No performance plots available (no dynamic functions and no PNGs found).")
        if st.button("Back to Dashboard", key="back_from_perf"):
            st.session_state.page = "home"
            safe_rerun()
        return

    # Create tabs for each plot (like Quick Analytics)
    tabs = st.tabs([label for label, _ in items])
    for tab, (label, content) in zip(tabs, items):
        with tab:
            # Matplotlib-like figure (has savefig)
            try:
                if hasattr(content, "savefig"):
                    buf = io.BytesIO()
                    content.savefig(buf, format="png", bbox_inches="tight")
                    buf.seek(0)
                    st.image(buf, use_column_width=True)
                    buf.close()
                    continue
            except Exception:
                pass

            # Plotly figure
            try:
                if isinstance(content, go.Figure):
                    st.plotly_chart(content, use_container_width=True)
                    continue
            except Exception:
                pass

            # PNG file path
            if isinstance(content, str) and os.path.isfile(content):
                st.image(content, use_column_width=True)
                continue

            # bytes / raw image
            if isinstance(content, (bytes, bytearray)):
                st.image(content, use_column_width=True)
                continue

            st.warning(f"Could not render plot: {label}")

    st.markdown("---")
    if st.button("Back to Dashboard", key="back_from_perf"):
        st.session_state.page = "home"
        safe_rerun()
def render_performance_page_dynamic():
    st.markdown("Generated plots (no disk files).")

    # example: call function that returns a Matplotlib fig object from pred_model.plots
    try:
        from pred_model.plots import make_pred_vs_actual_figure, make_feature_importance_figure
        figs = [
            ("Pred vs Actual", make_pred_vs_actual_figure()),
            ("Feature Importance", make_feature_importance_figure()),
        ]
        for title, fig in figs:
            st.markdown(f"**{title}**")
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            st.image(buf, use_column_width=True)
            buf.close()
    except Exception as e:
        st.error(f"Could not generate plots dynamically: {e}")

    if st.button("Back to Dashboard", key="back_from_perf"):
        st.session_state.page = "home"
        safe_rerun()

if __name__ == "__main__":
    main()