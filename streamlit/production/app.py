"""
FUREcast - GBR Demo Application
"""

import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import sys
import os
import pandas as pd

# Add parent directory to path for .env access
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    if "metric" not in st.session_state: st.session_state.metric = "Closing"
    if "start_date" not in st.session_state: st.session_state.start_date = pd.to_datetime("2023-01-01")
    if "end_date" not in st.session_state: st.session_state.end_date = pd.to_datetime("2025-12-31")


# ---------- SIDEBAR ----------
def render_sidebar():
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/c/c8/FURECast_SPLG.png", use_container_width=True)
        st.markdown("### 🎓 Demo Mode")
        st.session_state.mode = st.radio(
            "Select Mode:",
            options=['Lite', 'Pro'],
            key="sidebar_mode_radio",  # ✅ unique key
            help="Lite: Basic prediction + charts\nPro: Full LLM interface + all tools"
        )

        st.sidebar.header("Chart Filters")
        st.session_state.metric = st.selectbox(
            "Select Price Metric",
            options=["Closing", "Opening", "Daily High", "Daily Low", "Daily Current"],
            index=["Closing", "Opening", "Daily High", "Daily Low", "Daily Current"].index(st.session_state.metric),
            key="metric_selector"
        )
        st.session_state.start_date = st.date_input("Start Date", value=st.session_state.start_date, key="start_date_selector")
        st.session_state.end_date = st.date_input("End Date", value=st.session_state.end_date, key="end_date_selector")

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
    if direction == 'up': color, emoji = '#28a745', '📈'
    elif direction == 'down': color, emoji = '#dc3545', '📉'
    else: color, emoji = '#ffc107', '➡️'

    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        st.markdown(f"<div style='background-color:{color}; padding:1.5rem; border-radius:10px; color:white;'>"
                    f"<h2>{emoji} Model Prediction</h2>"
                    f"<h1>{direction.upper()}</h1>"
                    f"<p>Expected Return: {pred_return:+.2f}%</p></div>", unsafe_allow_html=True)
    with col2: st.metric("Confidence Score", f"{confidence:.1%}")
    with col3: st.metric("Timeframe", "Next Day")


def render_feature_importance(features):
    st.markdown("#### 🔍 Top Model Features")
    col1, col2 = st.columns([3,2])
    with col1:
        fig = create_feature_importance_chart(features)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        for i, feat in enumerate(features, 1):
            st.markdown(f"<div class='feature-item'><span><strong>{i}.</strong> {feat['name']}</span>"
                        f"<span><strong>{feat['importance']:.2%}</strong></span></div>", unsafe_allow_html=True)
        try:
            st.markdown(f"**Interpretation:** {explain_prediction(st.session_state.prediction_cache)}")
        except:
            st.markdown("*Feature importance shows key indicators influencing prediction.*")


def render_lite_mode():
    st.markdown('<p class="main-header">📈 FUREcast SPLG Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Educational GBR-Based Market Analytics</p>', unsafe_allow_html=True)
    
    if st.session_state.prediction_cache is None or st.button("🔄 Refresh Prediction", key="refresh_lite"):
        st.session_state.prediction_cache = predict_splg()
    prediction = st.session_state.prediction_cache

    render_prediction_card(prediction)
    st.markdown("---")
    render_feature_importance(prediction['top_features'])
    st.markdown("---")

    metric = st.session_state.metric
    start_date = st.session_state.start_date
    end_date = st.session_state.end_date
    if start_date > end_date:
        st.error("🚫 Start date must be before end date.")
        return

    st.markdown(f"#### 📊 {metric.capitalize()} Price Chart")
    with st.spinner("Loading chart..."):
        try:
            fig = create_price_chart(metric, pd.to_datetime(start_date), pd.to_datetime(end_date))
            st.plotly_chart(fig, config={"responsive": True}, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Chart failed to render: {e}")

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
            st.metric("Confidence", f"{prediction['confidence']:.1%}")
        with col4:
            if st.button("🔄 Refresh", key="pro_refresh_button"):
                st.session_state.prediction_cache = predict_splg()
                st.rerun()
    
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
            st.rerun()
    
    if submit and query:
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
        
        viz_type = plan.get('visualization', {}).get('type', 'price')
        
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
            st.plotly_chart(fig, use_container_width=True)
            
            # Show summary table
            summary_df = get_sector_summary()
            if not summary_df.empty:
                with st.expander("📊 Sector Summary Table", expanded=False):
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        elif 'sector' in query.lower() and 'risk' in query.lower():
            fig = create_sector_risk_treemap()
            st.plotly_chart(fig, use_container_width=True)
        
        elif 'compare' in query.lower() or 'performance' in query.lower():
            # Extract sectors from query (simplified)
            sectors = ['Technology', 'Utilities', 'Healthcare']
            fig = create_sector_comparison_chart(sectors)
            st.plotly_chart(fig, use_container_width=True)
        
        elif 'feature' in query.lower() or 'influence' in query.lower():
            fig = create_feature_importance_chart(prediction['top_features'])
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            # Default to price chart
            fig = create_price_chart('SPLG', 180)
            st.plotly_chart(fig, use_container_width=True)
    
    elif not query and submit:
        st.warning("Please enter a question to analyze.")
    
    # Additional tools section
    if not query:
        st.markdown("---")
        st.markdown("### 📊 Quick Analytics")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Sector Risk", "Holdings Detail", "Price Trends", "Feature Analysis"])
        
        with tab1:
            st.markdown("**Sector Risk Treemap** - Size by market cap, color by volatility")
            fig = create_sector_risk_treemap()
            st.plotly_chart(fig, use_container_width=True)
        
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
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("**Sector Summary (Weighted by SPLG)**")
            summary_df = get_sector_summary()
            if not summary_df.empty:
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        with tab3:
            st.markdown("**SPLG Historical Performance**")
            metric = st.session_state.metric
            start_date = st.session_state.start_date
            end_date = st.session_state.end_date
            fig = create_price_chart(metric, pd.to_datetime(start_date), pd.to_datetime(end_date))
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.markdown("**Current Model Feature Importance**")
            fig = create_feature_importance_chart(prediction['top_features'])
            st.plotly_chart(fig, use_container_width=True)
            
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


if __name__ == "__main__":
    main()