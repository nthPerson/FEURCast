"""
FUREcast - GBR Demo Application

A skeleton Streamlit UI demonstrating the planned FUREcast architecture
with simulated GradientBoostingRegressor predictions and LLM orchestration.
"""

import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import sys
import os

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
    viz_from_spec
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

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-card {
        padding: 1.5rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .disclaimer {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .feature-item {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem;
        border-bottom: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'mode' not in st.session_state:
        st.session_state.mode = 'Lite'
    if 'prediction_cache' not in st.session_state:
        st.session_state.prediction_cache = None
    if 'last_query' not in st.session_state:
        st.session_state.last_query = ""


def render_sidebar():
    """Render sidebar with mode selection and info"""
    with st.sidebar:
        st.image("https://via.placeholder.com/200x80/2E86AB/FFFFFF?text=FUREcast", use_container_width=True)
        
        st.markdown("### 🎓 Demo Mode")
        mode = st.radio(
            "Select Mode:",
            options=['Lite', 'Pro'],
            help="Lite: Basic prediction + charts\nPro: Full LLM interface + all tools"
        )
        st.session_state.mode = mode
        
        st.markdown("---")
        
        st.markdown("### ℹ️ About FUREcast")
        st.markdown("""
        This is a **demo skeleton** showcasing the planned architecture for SPLG ETF analysis using:
        
        - **GradientBoostingRegressor** for predictions
        - **OpenAI LLM** for natural language interface
        - **Tool orchestration** for dynamic analytics
        
        All data and predictions are **simulated** for demonstration purposes.
        """)
        
        st.markdown("---")
        
        with st.expander("📊 Data Sources (Simulated)"):
            st.markdown("""
            - SPLG historical data (2005-2025)
            - Sector ETF data (XLK, XLV, XLF, etc.)
            - Technical indicators (RSI, MACD, MA)
            - Risk metrics (volatility, Sharpe, drawdown)
            """)
        
        with st.expander("🛠️ Architecture"):
            st.markdown("""
            **Pipeline:**
            1. User Query
            2. LLM Router (intent classification)
            3. Tool Planner (execution plan)
            4. Tool Executor (fetch data/predict)
            5. Answer Composer (LLM synthesis)
            6. UI Renderer (Streamlit display)
            """)
        
        st.markdown("---")
        
        st.markdown('<div class="disclaimer">⚠️ <strong>Educational Only</strong><br>Not financial advice. Demo purposes only.</div>', 
                   unsafe_allow_html=True)


def render_prediction_card(prediction):
    """Render the main GBR prediction card"""
    direction = prediction['direction']
    pred_return = prediction['predicted_return'] * 100
    confidence = prediction['confidence']
    
    # Determine color based on direction
    if direction == 'up':
        color = '#28a745'
        emoji = '📈'
    elif direction == 'down':
        color = '#dc3545'
        emoji = '📉'
    else:
        color = '#ffc107'
        emoji = '➡️'
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
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
    """Render feature importance section"""
    st.markdown("#### 🔍 Top Model Features")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Show chart
        fig = create_feature_importance_chart(features)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Feature Importance Rankings:**")
        for i, feat in enumerate(features, 1):
            st.markdown(f"""
            <div class='feature-item'>
                <span><strong>{i}.</strong> {feat['name']}</span>
                <span><strong>{feat['importance']:.2%}</strong></span>
            </div>
            """, unsafe_allow_html=True)
        
        # Get explanation
        with st.spinner("Generating explanation..."):
            try:
                explanation = explain_prediction(st.session_state.prediction_cache)
                st.markdown(f"**Interpretation:** {explanation}")
            except:
                st.markdown("*Feature importance shows which technical indicators most influenced this prediction.*")


def render_lite_mode():
    """Render Lite mode interface"""
    st.markdown('<p class="main-header">📈 FUREcast SPLG Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Educational GBR-Based Market Analytics</p>', unsafe_allow_html=True)
    
    # Get or generate prediction
    if st.session_state.prediction_cache is None or st.button("🔄 Refresh Prediction"):
        with st.spinner("Generating prediction..."):
            st.session_state.prediction_cache = predict_splg()
    
    prediction = st.session_state.prediction_cache
    
    # Main prediction card
    render_prediction_card(prediction)
    
    st.markdown("---")
    
    # Feature importance
    render_feature_importance(prediction['top_features'])
    
    st.markdown("---")
    
    # Price chart
    st.markdown("#### 📊 SPLG Historical Price")
    with st.spinner("Loading chart..."):
        fig = create_price_chart('SPLG', 180)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Educational note
    st.info("""
    **💡 How to Use This Tool:**
    
    1. Review the model's prediction (up/down/neutral)
    2. Check the confidence score - higher is more certain
    3. Examine which features drove the prediction
    4. Compare with recent price trends in the chart
    5. Remember: This is a learning tool, not investment advice!
    
    *Switch to **Pro Mode** in the sidebar for natural language queries and advanced analytics.*
    """)


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
            if st.button("🔄 Refresh"):
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
                if st.button(example, key=f"example_{i}", use_container_width=True):
                    st.session_state.last_query = example
    
    # Query input
    query = st.text_input(
        "Your Question:",
        value=st.session_state.last_query,
        placeholder="e.g., Which sectors have the lowest volatility?",
        label_visibility="collapsed"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        submit = st.button("🔍 Analyze", type="primary", use_container_width=True)
    with col2:
        if st.button("Clear", use_container_width=True):
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
            
            # Simulate tool execution based on intent
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
                with st.expander("📊 Sector Summary Table"):
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
            
            # Color metric selector
            color_option = st.selectbox(
                "Color by:",
                ["DailyChangePct", "PE", "Beta", "DividendYield"],
                index=0,
                key="treemap_color"
            )
            
            # Create and display the treemap
            fig = create_sector_holdings_treemap(color_metric=color_option)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show sector summary table
            st.markdown("**Sector Summary (Weighted by SPLG)**")
            summary_df = get_sector_summary()
            if not summary_df.empty:
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        with tab3:
            st.markdown("**SPLG Historical Performance**")
            fig = create_price_chart('SPLG', 365)
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


def main():
    """Main application entry point"""
    initialize_session_state()
    render_sidebar()
    
    # Render appropriate mode
    if st.session_state.mode == 'Lite':
        render_lite_mode()
    else:
        render_pro_mode()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p><strong>FUREcast GBR Demo</strong> | Educational Analytics Platform</p>
        <p style='font-size: 0.9rem;'>Built with Streamlit, OpenAI, and Plotly | Demo Version</p>
        <p style='font-size: 0.8rem;'>⚠️ All data simulated for demonstration purposes</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
