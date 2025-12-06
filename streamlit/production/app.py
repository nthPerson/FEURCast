"""
FUREcast - SPLG GBR Analytics Dashboard
"""

import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import sys
import os
import io
import pandas as pd
import html
from pathlib import Path

# Safe rerun helper to support Streamlit versions where experimental_rerun / rerun may be missing
def safe_rerun():
    try:
        st.experimental_rerun()
    except Exception:
        try:
            st.rerun()
        except Exception:
            st.stop()

# --- Glossary page renderer (ensure this is defined before main()) ---
def render_glossary_page():
    """
    Render the Investment Glossary page. Expects investment_glossary.csv at repo root
    or data/ folder. Shows table, search box, download and Back button.
    """
    st.markdown('<p class="main-header">Investment Glossary</p>', unsafe_allow_html=True)
    st.markdown("Definitions of common finance/investment terms.")

    # Try a couple of likely locations
    candidates = [
        os.path.join(os.path.dirname(__file__), '..', '..', 'investment_glossary.csv'),
        os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'investment_glossary.csv'),
        os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'glossary.csv'),
    ]
    csv_path = next((p for p in candidates if os.path.isfile(p)), None)

    if csv_path is None:
        st.error("Glossary CSV not found. Expected at one of:\n" + "\n".join(candidates))
        if st.button("Back to Dashboard", key="back_from_glossary_missing"):
            st.session_state.page = "home"
            safe_rerun()
        return

    try:
        df = pd.read_csv(csv_path, dtype=str, engine='python')
        # drop empty trailing columns
        df = df.loc[:, df.notna().any(axis=0)]
        # normalize column names
        cols = [c.strip() for c in df.columns]
        df.columns = cols

        # If first two columns look like Term / Definition, rename
        if len(cols) >= 2:
            if 'term' in cols[0].lower() or 'word' in cols[0].lower():
                df = df.rename(columns={cols[0]: 'Term', cols[1]: 'Definition'})

        # removed the loaded-count message per request

        query = st.text_input("Search terms or definitions (substring):", value="", key="glossary_search")
        if query:
            mask = df.apply(lambda row: row.astype(str).str.contains(query, case=False, na=False).any(), axis=1)
            display_df = df[mask].reset_index(drop=True)
            # render table without the index column
            html = display_df.to_html(index=False, classes="table table-striped", border=0)
            st.markdown(html, unsafe_allow_html=True)
        else:
            # render full table without the index column
            html = df.to_html(index=False, classes="table table-striped", border=0)
            st.markdown(html, unsafe_allow_html=True)

        col_dl, col_back = st.columns([1,1])
        with col_dl:
            csv_bytes = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv_bytes, file_name="investment_glossary.csv", mime="text/csv")
        with col_back:
            if st.button("Back to Dashboard", key="back_from_glossary"):
                st.session_state.page = "home"
                safe_rerun()

    except Exception as e:
        st.error(f"Failed to load glossary: {e}")
        if st.button("Back to Dashboard", key="back_from_glossary_err"):
            st.session_state.page = "home"
            safe_rerun()

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
    get_holdings_top_n,
)
from llm_interface import (
    route_query,
    compose_answer,
    explain_prediction
)

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '../..', '.env'))
except Exception:
    pass

def get_secret(name: str, default: str = "") -> str:
    val = os.getenv(name)
    if val:
        return val
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default

MARKETSENTIMENT_API_KEY = get_secret('MARKETSENTIMENT_API_KEY', '')
FRED_KEY = get_secret('FRED_KEY', '')
# Note: bullish/bearish sentiment variables removed; UI shows headline-based summary only
# ---------- STREAMLIT CONFIGURATION ----------

# Page configuration
st.set_page_config(
    page_title="FUREcast - SPLG Analytics Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Primary headers */
    .main-header, p.main-header { font-size: 2.5rem; font-weight: bold; color: #2E86AB; margin-bottom: 0.5rem; }
    .sub-header, p.sub-header { font-size: 1.2rem; color: #666; margin-bottom: 2rem; }
    /* Chart-style header - ensure block-level and stacking so it remains visible above nearby panels */
    .chart-header, p.chart-header, div.chart-header { font-size: 1.80rem; font-weight: 800; margin: 0.6rem 0; display:block; position:relative; z-index:50; }

    /* Provide theme-aware CSS variables for nav and header colors so other rules can reuse them */
    html[data-theme='dark'], body[data-theme='dark'] {
        --app-text-primary: #ffffff;
        --app-text-on-surface: #e6eef6;
        --nav-bg: #0f1720;
        --nav-text: #ffffff;
        --nav-border: rgba(255,255,255,0.06);
    }
    html[data-theme='light'], body[data-theme='light'] {
        --app-text-primary: #0b0d10;
        --app-text-on-surface: #1c1c1c;
        --nav-bg: #f6f8fb;
        --nav-text: #0b0d10;
        --nav-border: rgba(15,23,42,0.06);
    }

    /* Apply theme-aware colors to headers using the variables above */
    html[data-theme='dark'] .chart-header,
    body[data-theme='dark'] .chart-header,
    html[data-theme='dark'] p.chart-header,
    body[data-theme='dark'] p.chart-header,
    html[data-theme='dark'] .main-header,
    body[data-theme='dark'] .main-header,
    html[data-theme='dark'] p.main-header,
    body[data-theme='dark'] p.main-header,
    html[data-theme='dark'] .sub-header,
    body[data-theme='dark'] .sub-header,
    html[data-theme='dark'] p.sub-header,
    body[data-theme='dark'] p.sub-header {
        color: var(--app-text-primary) !important;
    }

    html[data-theme='light'] .chart-header,
    body[data-theme='light'] .chart-header,
    html[data-theme='light'] p.chart-header,
    body[data-theme='light'] p.chart-header,
    html[data-theme='light'] .main-header,
    body[data-theme='light'] .main-header,
    html[data-theme='light'] p.main-header,
    body[data-theme='light'] p.main-header,
    html[data-theme='light'] .sub-header,
    body[data-theme='light'] .sub-header,
    html[data-theme='light'] p.sub-header,
    body[data-theme='light'] p.sub-header {
        color: var(--app-text-primary) !important;
    }

    /* Keep basic nav layout rules here, but let render_top_nav define colors. */
    section[data-testid='stHorizontalBlock'] > div div button,
    .nav-tile {
        border-radius: 10px !important;
        padding: 0 18px !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        text-align: center !important;
        cursor: pointer !important;
        display: block !important;
        align-items: center !important;
        justify-content: center !important;
        width: 100% !important;
        height: 56px !important;
        line-height: 1.1 !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        box-sizing: border-box !important;
    }

    .feature-item { display: flex; justify-content: space-between; padding: 0.5rem; border-bottom: 1px solid #eee; }
    .disclaimer { background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 1rem; border-radius: 4px; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# Helper to render a chart/header with an info icon that shows a native browser tooltip on hover
def header_with_info(text: str, tooltip: str = None):
    if tooltip:
        tip = html.escape(tooltip, quote=True)
        # small circled info icon; title attribute provides the hover tooltip
        st.markdown(f'<div class="chart-header">{text} <span class="info-icon" title="{tip}">ⓘ</span></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chart-header">{text}</div>', unsafe_allow_html=True)

# Minimal style for the info icon (uses theme variables defined earlier)
st.markdown("""
<style>
    .info-icon {
        font-size:0.95rem; margin-left:8px; padding:2px 6px; border-radius:12px; cursor:help; vertical-align:middle;
        background: transparent; color: var(--app-text-primary, #0b0d10); border: 1px solid rgba(0,0,0,0.08);
    }
    html[data-theme='dark'] .info-icon, body[data-theme='dark'] .info-icon { color: var(--app-text-primary); border-color: rgba(255,255,255,0.08); }
    html[data-theme='light'] .info-icon, body[data-theme='light'] .info-icon { color: var(--app-text-primary); border-color: rgba(0,0,0,0.06); }
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
    # Ensure we capture the latest available dataset date first, then default the
    # date filters relative to that anchor.
    if "max_dataset_date" not in st.session_state:
        st.session_state.max_dataset_date = pd.to_datetime(get_latest_date_in_dataset())
    else:
        st.session_state.max_dataset_date = pd.to_datetime(st.session_state.max_dataset_date)

    if "start_date" not in st.session_state:
        six_months_prior = st.session_state.max_dataset_date - pd.DateOffset(months=6)
        st.session_state.start_date = six_months_prior
    if "end_date" not in st.session_state:
        # Default end date now dynamically set to latest available date
        st.session_state.end_date = st.session_state.max_dataset_date
    # New: control whether event highlights are shown
    if "show_events" not in st.session_state:
        st.session_state.show_events = True
    # New: current page for simple navigation ('home' or 'performance')
    if "page" not in st.session_state:
        st.session_state.page = "home"


# --------- Navigation helpers & callbacks ----------
def _sidebar_mode_changed():
    # keep session_state.mode in sync when the top-nav mode radio changes.
    # Use a unique widget key `topnav_mode` to avoid duplication.
    mode = st.session_state.get("topnav_mode", st.session_state.get("mode", "Lite"))
    st.session_state.mode = mode


def _set_page(page_key: str, nav_label: str = None):
    """Helper for nav buttons: set page and optional sidebar_nav label then rerun."""
    st.session_state.page = page_key
    if nav_label is not None:
        st.session_state.sidebar_nav = nav_label
    safe_rerun()


def render_top_nav():
    """Render a simple top horizontal navigation bar placed above main content."""
    # Shortened labels so they remain single-line and neat
    nav_items = [
        ("Home", "home"),
        ("GBR Model Details", "gbr_details"),
        ("Performance", "performance"),
        ("Glossary", "glossary"),
    ]
    current = st.session_state.get("page", "home")

    # Single full-width row: nav buttons span the page like the headline ribbon
    cols = st.columns([1] * len(nav_items))

    # Inline CSS for nav tiles (active + inactive)
    active_color = "#f4f6f8"  # light grey background
    active_text = "#0b0d10"   # dark text for high contrast on light bg
    active_font_px = 19

    css = f"""
    <style>
    /* Unified light-style nav tiles: light background, dark text, uniform height */
    :root {{
        --nav-bg: #f6f8fb;
        --nav-text: #0b0d10;
        --nav-border: rgba(15,23,42,0.06);
        --nav-active-bg: #eaf6ff;
        --nav-accent: #f5f7f9;
        --nav-home-bg: #0b5ed7;
        --nav-home-text: #ffffff;
    }}

    html[data-theme='dark'], body[data-theme='dark'] {{
        --nav-bg: #0f1720;
        --nav-text: #ffffff;
        --nav-border: rgba(255,255,255,0.12);
        --nav-active-bg: #111827;
        --nav-home-bg: #2563eb;   /* slightly brighter blue on dark */
        --nav-home-text: #f9fafb;
    }}

    section[data-testid='stHorizontalBlock'] > div div button,
    .nav-tile {{
        background: var(--nav-bg) !important;
        color: var(--nav-text) !important;
        border-radius: 10px !important;
        padding: 0 18px !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        border: 2px solid var(--nav-border) !important; /* keep same width as active */
        text-align: center !important;
        cursor: pointer !important;
        display: block !important; /* ensure consistent sizing */
        align-items: center !important;
        justify-content: center !important;
        width: 100% !important; /* fill the column so widths are uniform */
        height: 56px !important; /* fixed uniform height */
        line-height: 1.1 !important;
        /* keep label text on a single line, ellipsize if too long */
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        box-sizing: border-box !important;
    }}

    /* Make button inner text wrap nicely and center vertically */
    section[data-testid='stHorizontalBlock'] > div div button > div {{
        display:flex; align-items:center; justify-content:center; height:100%; width:100%;
    }}

    /* Hover effect: subtle, flatter shadow for a cleaner look */
    section[data-testid='stHorizontalBlock'] > div div button:hover,
    .nav-tile:hover {{
        box-shadow: 0 2px 6px rgba(0,0,0,0.04) !important;
    }}

    /* Active appearance: subtle blue accent border and slightly different background */
    .nav-active,
    section[data-testid='stHorizontalBlock'] > div div button.nav-active {{
        /* Only change background and text color for active state so size remains identical */
        background: var(--nav-active-bg) !important;
        color: var(--nav-text) !important;
        border: 2px solid var(--nav-border) !important; /* same border width as inactive */
        box-shadow: none !important; /* avoid adding shadows that change perceived size */
    }}

    /* Reduce any extra margin Streamlit may add so tiles align neatly */
    section[data-testid='stHorizontalBlock'] > div > div {{ padding: 6px !important; }}
    </style>
    """

    # Simple CSS only: no JS-based overrides for active nav styling.
    st.markdown(css, unsafe_allow_html=True)

    # Render each nav item as a Streamlit button so DOM is consistent and size doesn't change
    for (base_label, key), col in zip(nav_items, cols):
        # Home button simply routes to the last-used mode for the Home page
        if key == "home":
            clicked = col.button("Home", key=f"topnav_{key}", use_container_width=True)
            if clicked:
                st.session_state.page = "home"
                safe_rerun()
        else:
            clicked = col.button(base_label, key=f"topnav_{key}", use_container_width=True)
            if clicked:
                st.session_state.page = key
                safe_rerun()



# ---------- SIDEBAR ----------
def render_sidebar():
    with st.sidebar:
        current = st.session_state.get("page", "home")

        # Branding logo at top
        APP_DIR = Path(__file__).resolve().parent  # Get the parent directory of this file
        LOGO_PATH = APP_DIR / "FUREcast_logo.png"
        st.image(str(LOGO_PATH), width='stretch')

        # Mode toggle button only when on the Home page
        if current == "home":
            current_mode = st.session_state.get("mode", "Lite")
            if current_mode == "Lite":
                toggle_label = "Click Here for Pro Mode 😎"
                next_mode = "Pro"
            else:
                toggle_label = "Click Here for Lite Mode 🚀"
                next_mode = "Lite"

            if st.button(toggle_label, key="sidebar_mode_toggle", use_container_width=True):
                st.session_state.mode = next_mode
                st.session_state.page = "home"
                safe_rerun()

        # Show GBR architecture diagram only on the GBR Model Details page
        if current == "gbr_details":
            gbr_img_path = os.path.join(
                os.path.dirname(__file__),
                "pred_model",
                "models",
                "GBR_architecture_diagram.png",
            )
            if os.path.isfile(gbr_img_path):
                st.image(gbr_img_path, caption="GBR Model Architecture", use_container_width=True)

        if current in ("home", "performance"):
            st.markdown("### ℹ️ About FUREcast")
            st.markdown("""
            Interactive dashboard showcasing SPLG (now SPYM) ETF analysis with GradientBoostingRegressor and LLM orchestration. See dropdowns below for additional details.
            """)
            st.sidebar.header("Price Chart Filters")
            st.session_state.metric = st.selectbox(
                "Select Price Metric",
                options=["Closing", "Opening", "Daily High", "Daily Low", "Daily Current"],
                index=["Closing", "Opening", "Daily High", "Daily Low", "Daily Current"].index(st.session_state.metric),
                key="metric_selector"
            )

            # Define the maximum available date from our dataset (last available date in historical data)
            MAX_DATASET_DATE = pd.to_datetime(st.session_state.get("max_dataset_date", get_latest_date_in_dataset()))

            # Convert Timestamp to date for comparison and display
            current_end_date = st.session_state.end_date.date() if isinstance(st.session_state.end_date, pd.Timestamp) else st.session_state.end_date
            current_start_date = st.session_state.start_date.date() if isinstance(st.session_state.start_date, pd.Timestamp) else st.session_state.start_date

            st.session_state.start_date = st.date_input(
                "Start Date",
                value=current_start_date,
                key="start_date_selector",
                max_value=MAX_DATASET_DATE.date(),
                format="MM/DD/YYYY"
            )
            st.session_state.end_date = st.date_input(
                "End Date",
                value=min(current_end_date, MAX_DATASET_DATE.date()),
                key="end_date_selector",
                max_value=MAX_DATASET_DATE.date(),
                format="MM/DD/YYYY"
            )

            # Convert back to Timestamp for consistency
            st.session_state.start_date = pd.to_datetime(st.session_state.start_date)
            st.session_state.end_date = pd.to_datetime(st.session_state.end_date)

            # st.markdown("### ℹ️ About FUREcast")
            with st.expander("Data Sources", expanded=False):
                st.markdown("- SPLG/SPYM historical data (2005-2025)\n- Sector ETF data\n- Technical indicators\n- Risk metrics\n- Market events")
            with st.expander("Prediction Model", expanded=False):
                st.markdown("""
                **Gradient Boosting Regressor (GBR)**
                - Ensemble learning method combining multiple decision trees
                - Trained on 100+ engineered features including technical indicators, sector ETF data, and market metrics
                - Predicts next-day SPLG price movement
                - Performance metrics available in Performance page
                - Hyperparameters optimized via grid search with cross-validation
                
                **Key Features:**
                - Historical price data (open, high, low, close, volume)
                - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
                - Sector ETF performance metrics
                - Market volatility measures
                - Rolling statistics and momentum indicators
                """)
            with st.expander("Visualizations", expanded=False):
                st.markdown("""
                **Available Visualizations:**
                - Historical Price Charts (Closing, Opening, High, Low, Current)
                - Sector Risk Treemap (by volatility)
                - Sector Holdings Drill-Down Treemap
                - Feature Importance Chart
                - Sector Comparison Chart
                - Model Performance Metrics:
                  - Prediction vs Actual
                  - Residuals Analysis
                  - Error Distribution
                  - Training Progress
                  - Cumulative Returns
                  - Time Series Predictions
                """)
            with st.expander("Learning Agent Architecture", expanded=False):
            # with st.expander("Agent Architecture", expanded=False):
                st.markdown("1. User-Entered Query\n2. LLM Intent Router\n3. Tool Planner\n4. Tool Executor\n5. Answer Composer\n6. UI Rendering")
                # st.markdown("1. User Query\n2. LLM Router\n3. Tool Planner\n4. Tool Executor\n5. Answer Composer\n6. UI Renderer")
            # st.markdown('<div class="disclaimer">⚠️ <strong>Educational Use Only</strong><br>Not financial advice.</div>', unsafe_allow_html=True)

            # st.markdown("---")
            # st.sidebar.header("Price Chart Filters")
            # st.session_state.metric = st.selectbox(
            #     "Select Price Metric",
            #     options=["Closing", "Opening", "Daily High", "Daily Low", "Daily Current"],
            #     index=["Closing", "Opening", "Daily High", "Daily Low", "Daily Current"].index(st.session_state.metric),
            #     key="metric_selector"
            # )

            # # Define the maximum available date from our dataset (last available date in historical data)
            # MAX_DATASET_DATE = pd.to_datetime(get_latest_date_in_dataset())

            # # Convert Timestamp to date for comparison and display
            # current_end_date = st.session_state.end_date.date() if isinstance(st.session_state.end_date, pd.Timestamp) else st.session_state.end_date
            # current_start_date = st.session_state.start_date.date() if isinstance(st.session_state.start_date, pd.Timestamp) else st.session_state.start_date

            # st.session_state.start_date = st.date_input(
            #     "Start Date",
            #     value=current_start_date,
            #     key="start_date_selector",
            #     max_value=MAX_DATASET_DATE.date()
            # )
            # st.session_state.end_date = st.date_input(
            #     "End Date",
            #     value=min(current_end_date, MAX_DATASET_DATE.date()),
            #     key="end_date_selector",
            #     max_value=MAX_DATASET_DATE.date()
            # )

            # # Convert back to Timestamp for consistency
            # st.session_state.start_date = pd.to_datetime(st.session_state.start_date)
            # st.session_state.end_date = pd.to_datetime(st.session_state.end_date)
        else:
            # Compact sidebar for Glossary or other pages where filters are not needed
            st.markdown("### ℹ️ About FUREcast")
            st.markdown("Sidebar filters hidden for this page.")
            if st.button("Back to Dashboard", key="back_from_glossary_sidebar"):
                st.session_state.page = "home"
                safe_rerun()


# ---------- LITE MODE ----------
def render_prediction_card(prediction):
    """Render a full-width colored prediction card (Lite mode).
    Card now includes direction, expected return, confidence and timeframe
    inside a single uniformly colored panel, matching Pro mode styling.
    """
    direction = prediction['direction']
    pred_return = prediction['predicted_return'] * 100
    confidence = prediction['confidence']

    if direction == 'up':
        color, emoji = '#28a745', '📈'
    elif direction == 'down':
        color, emoji = '#dc3545', '📉'
    else:
        color, emoji = '#ffc107', '➡️'

    st.markdown(
        f"""
        <div style="background-color:{color}; padding:1.4rem 1.8rem; border-radius:12px; color:#fff; width:100%; box-sizing:border-box;">
            <div style="display:flex; flex-wrap:wrap; align-items:center; gap:2.5rem;">
                <div style="min-width:160px;">
                    <div style="font-size:1.1rem; font-weight:600;">{emoji} Model Prediction</div>
                    <div style="font-size:2.1rem; font-weight:800; line-height:1; margin-top:4px;">{direction.upper()}</div>
                </div>
                <div style="font-size:1.05rem; font-weight:500;">Expected Return:<br><span style="font-size:1.4rem; font-weight:700;">{pred_return:+.2f}%</span></div>
                <div style="font-size:1.05rem; font-weight:500;">Confidence:<br><span style="font-size:1.4rem; font-weight:700;">{confidence}</span></div>
                <div style="font-size:1.05rem; font-weight:500;">Timeframe:<br><span style="font-size:1.4rem; font-weight:700;">Next Day</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feature_importance(features):
    header_with_info('Top Model Features', 'Shows the model\'s top features and their relative importance in the current prediction — educational, not causal.')
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


def render_latest_headlines_strip():
    """Render the scrolling latest headlines strip with no section header.

    This is called at the top of each page, above all other content.
    """
    import requests
    url = f"https://finnhub.io/api/v1/news?category=general&token={MARKETSENTIMENT_API_KEY}"

    try:
        response = requests.get(url, timeout=5)
        articles = response.json()
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        articles = []

    if not articles or not isinstance(articles, list):
        st.warning("No recent S&P 500 news found.")
        return

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


def compute_macro_and_sentiment():
    """Fetch macro series and compute statuses/colors once per run.

    Returns unemployment_rate, public_debt_pct, un_status, un_color,
    debt_status, debt_color. Coloring thresholds mirror the original
    Lite/Pro implementations.
    """
    import requests
    base_url = "https://api.stlouisfed.org/fred/series/observations"

    if not FRED_KEY:
        unemployment_rate = None
        public_debt_pct = None
    else:
        params_un = {
            "series_id": "UNRATE",
            "api_key": FRED_KEY,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 1,
        }
        try:
            resp_un = requests.get(base_url, params=params_un, timeout=5)
            data_un = resp_un.json().get("observations", [])
            unemployment_rate = float(data_un[0].get("value", 0)) if data_un else None
        except Exception:
            unemployment_rate = None

        params_debt = {
            "series_id": "GFDEGDQ188S",
            "api_key": FRED_KEY,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 1,
        }
        try:
            resp_debt = requests.get(base_url, params=params_debt, timeout=5)
            data_debt = resp_debt.json().get("observations", [])
            public_debt_pct = float(data_debt[0].get("value", 0)) if data_debt else None
        except Exception:
            public_debt_pct = None

    def indicator_status(value, good_max):
        if value is None:
            return "N/A", "grey"
        if value <= good_max:
            return "Acceptable", "#0b8a3e"
        else:
            return "Bad", "#d62828"

    un_status, un_color = indicator_status(unemployment_rate, 6.0)
    debt_status, debt_color = indicator_status(public_debt_pct, 70.0)
    return unemployment_rate, public_debt_pct, un_status, un_color, debt_status, debt_color

def render_macro_and_sentiment_tiles(unemployment_rate, public_debt_pct, un_status, un_color, debt_status, debt_color):
    """Render compact Macro & Sentiment tiles used globally below headlines.

    Values and colors are passed in so we can reuse the same data
    for all pages without recomputing. Coloring comes from the
    shared indicator_status logic and must remain unchanged.
    """
    # Smaller typography / padding so tiles are visually de-emphasized
    col1, col2 = st.columns([1, 1], gap="small")

    panel_style = (
        "padding:8px; border-radius:10px; "
        "background:linear-gradient(180deg,#fbfdff,#f1f6fb); "
        "border:1px solid rgba(15,23,42,0.06); width:100%; box-sizing:border-box; color:#0b0d10;"
    )

    with col1:
        un_display = f"{unemployment_rate:.1f}%" if unemployment_rate is not None else "N/A"
        st.markdown(
            f"<div style='{panel_style}'>"
            f"<div style='display:flex; align-items:center; justify-content:center; gap:8px; font-size:0.95rem; text-align:center;'>"
            f"<span style='font-weight:600; color:#0b0d10; opacity:0.95;'>Unemployment Rate</span>"
            f"<span style='font-weight:700; color:#0b0d10;'>{un_display}</span>"
            f"<span style='font-weight:700; color:{un_color};'>{un_status}</span>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    with col2:
        debt_display = f"{public_debt_pct:.1f}%" if public_debt_pct is not None else "N/A"
        st.markdown(
            f"<div style='{panel_style}'>"
            f"<div style='display:flex; align-items:center; justify-content:center; gap:8px; font-size:0.95rem; text-align:center;'>"
            f"<span style='font-weight:600; color:#0b0d10; opacity:0.95;'>Public Debt (% of GDP)</span>"
            f"<span style='font-weight:700; color:#0b0d10;'>{debt_display}</span>"
            f"<span style='font-weight:700; color:{debt_color};'>{debt_status}</span>"
            f"</div></div>",
            unsafe_allow_html=True,
        )



def render_lite_mode():
    # Lite mode now assumes macro & sentiment tiles have already been
    # rendered globally near the top of the page.
    # =============================
    # 📈 Core UI: Header + Prediction
    # =============================
    header_with_info('FUREcast SPLG Predictor', 'Model prediction from a Gradient Boosting Regressor trained on historical features. Predictions are illustrative and not financial advice.')
    st.markdown('<div class="sub-header">Educational GBR-Based Market Analytics</div>', unsafe_allow_html=True)

    # --- Run model prediction ---
    if st.session_state.prediction_cache is None:
        st.session_state.prediction_cache = predict_splg()
    prediction = st.session_state.prediction_cache

    # Show refresh button only if using simulated model
    if prediction.get('model_source') == 'simulated':
        if st.button("🔄 Refresh Prediction", key="refresh_lite"):
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
        st.error("Start date must be before end date.")
        st.stop()
        
    # Ensure end date doesn't exceed dataset limit for price charts
    if end_date > max_date:
        end_date = max_date
        st.warning(f"⚠️ End date adjusted to maximum available date: {max_date.date()}")

    header_with_info(f"{metric} Price Chart", 'Interactive historical price chart for the selected metric. Use the date selectors to adjust the displayed range.')
    with st.spinner("Loading chart..."):
        try:
            # Map display name to DataFrame column name
            df_column = st.session_state.metric_mapping[metric]
            # Pass show_events flag from session state
            fig = create_price_chart(df_column, pd.to_datetime(start_date), pd.to_datetime(end_date), show_events=st.session_state.show_events)
            st.plotly_chart(fig, config={"responsive": True, "width": 'stretch'}, key='price_chart')
        except Exception as e:
            st.error(f"❌ Chart failed to render: {e}")

    # Credit for "Exogenous Market Events" (these are the red dots overlayed on the graph that represent events that impact the market)
    st.info("""
    📰 **Exogenous market event data source**:
    - Professionally aggregated by Fernando Cerdá
      - Financial Advisor, Investment Strategist, and FUREcast's subject matter expert
      - Author of *From Beginner to Investor: Quick Guide to Start Investing* (available on Amazon)
    - Provides 'Event', 'Category', and 'Econimic/Market Impact (S&P 500)' in Price Chart above (red colored points)
    """)

    # ============================
    # 💡 How to Use Section
    # =============================
    # st.info("""
    # **💡 How to Use This Tool:**
    # 1. Review model prediction  
    # 2. Examine feature influence  
    # 3. Compare with recent price trends  
    # """)


# ---------- PRO MODE ----------
def render_pro_mode():


    # =============================
    # 📈 Core UI: Header + Prediction
    # =============================
    header_with_info('FUREcast Pro Analytics', 'Pro mode includes advanced analysis tools and a natural language interface; outputs are explanatory and educational.')
    st.markdown('<div class="sub-header">AI-Powered Investment Insights & Natural Language Interface</div>', unsafe_allow_html=True)
    
    # Get max dataset date for use throughout Pro mode
    max_dataset_date = st.session_state.max_dataset_date
    
    # Quick prediction card (collapsible)
    with st.expander("Current SPLG Prediction", expanded=True):
        if st.session_state.prediction_cache is None:
            with st.spinner("Generating prediction..."):
                st.session_state.prediction_cache = predict_splg()

        prediction = st.session_state.prediction_cache
        direction = prediction['direction']
        if direction == 'up':
            color, emoji = '#28a745', '📈'
        elif direction == 'down':
            color, emoji = '#dc3545', '📉'
        else:
            color, emoji = '#ffc107', '➡️'

        # Colored panel mimicking Lite card but inside expander
        st.markdown(
            f"""
            <div style="background-color:{color}; padding:1rem 1.4rem; border-radius:10px; color:#fff; width:100%; box-sizing:border-box; margin-bottom:0.75rem;">
                <div style="display:flex; flex-wrap:wrap; gap:2rem; align-items:flex-start;">
                    <div style="min-width:150px;">
                        <div style="font-size:1.0rem; font-weight:600;">{emoji} Prediction</div>
                        <div style="font-size:1.9rem; font-weight:800; line-height:1; margin-top:4px;">{direction.upper()}</div>
                    </div>
                    <div style="font-size:0.95rem; font-weight:500;">Expected Return:<br><span style="font-size:1.3rem; font-weight:700;">{prediction['predicted_return']*100:+.2f}%</span></div>
                    <div style="font-size:0.95rem; font-weight:500;">Confidence:<br><span style="font-size:1.3rem; font-weight:700;">{prediction['confidence']}</span></div>
                    <div style="font-size:0.95rem; font-weight:500;">Timeframe:<br><span style="font-size:1.3rem; font-weight:700;">Next Day</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # Conditional refresh button only if simulated
        if prediction.get('model_source') == 'simulated':
            if st.button("🔄 Refresh Prediction", key="pro_refresh_button"):
                st.session_state.prediction_cache = predict_splg()
                safe_rerun()
    
    st.markdown("---")
    
    # Natural language query interface
    header_with_info('💬 Ask FUREcast - Learning Agent', 'Ask questions in natural language. Responses combine LLM-generated text with model outputs and visualizations — treat as educational commentary.')
    # header_with_info('💬 Ask FUREcast', 'Ask questions in natural language. Responses combine LLM-generated text with model outputs and visualizations — treat as educational commentary.')
    st.markdown('<div class="sub-header">Ask our Learning Agent questions about SPLG, sectors, risk, or market trends...</div>', unsafe_allow_html=True)
    # st.markdown('<div class="sub-header">Enter your question about SPLG, sectors, risk, or market trends...</div>', unsafe_allow_html=True)
    
    # Example queries
    with st.expander("Example Queries"):
        examples = [
            "Is now a good time to invest in SPLG?",
            "Which sectors look stable this quarter?",
            "Compare Technology vs Utilities performance",
            "Show me the top holdings in SPLG",
            "What influenced today's prediction?",
            "What sectors have the highest risk?"
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
        submit = st.button("Analyze", type="primary", use_container_width=True, key="analyze_button")
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
        
        with st.spinner("Planning analysis..."):
            # Route query and persist the raw input so users can inspect it later
            plan = route_query(query) or {}
            plan['user_query'] = query
        
        st.markdown("---")
        
        # Show what the system is doing
        with st.expander("🔧 System Plan", expanded=False):
            st.json(plan)
        
        # Execute tools based on plan
        with st.spinner("Executing analysis..."):
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
        header_with_info('Analysis Results', 'Composed answer summarizing model outputs and tool results related to your query. Educational use only — not financial advice.')
        st.markdown(response)
        
        st.markdown("---")
        
        # Generate visualizations based on intent
        header_with_info('Visualizations', 'Charts and treemaps generated to support the analysis. Interactive elements help explore model behavior and data.')
        
        viz_plan = plan.get('visualization') or {}
        viz_type = viz_plan.get('type', 'price')

        # If the LLM explicitly requested a table visualization (e.g. top holdings),
        # honor that first by rendering a DataFrame derived from holdings data.
        if viz_type == 'table':
            specs = viz_plan.get('specs', {}) or {}
            requested_columns = specs.get('columns') or []

            try:
                # Default to top 20 holdings; this can be tuned later or
                # extended to read a "top_n" field from specs.
                holdings_df = get_holdings_top_n(20)

                # Reorder / subset according to the plan's column list when possible.
                if requested_columns:
                    cols_available = [c for c in requested_columns if c in holdings_df.columns]
                    if cols_available:
                        display_df = holdings_df[cols_available]
                    else:
                        display_df = holdings_df
                else:
                    display_df = holdings_df

                st.markdown("**Top SPLG Holdings (from plan)**")
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            except Exception as e:
                st.warning(f"Unable to render requested table visualization: {e}")

        # Keyword-based fallbacks for additional visual context
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
                with st.expander("Sector Summary Table", expanded=False):
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
    header_with_info('Advanced Analytics', 'Deep-dive tools: Sector Risk, Holdings drill-down, Price Trends, and Feature Analysis. Use these to learn how sector composition and holdings influence ETF behavior.')
    
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
    # Top of page: headlines strip, then compact macro & sentiment tiles,
    # then main navigation and sidebar.
    render_latest_headlines_strip()

    (   unemployment_rate,
		public_debt_pct,
		un_status,
		un_color,
		debt_status,
		debt_color,
	) = compute_macro_and_sentiment()

    render_macro_and_sentiment_tiles(
		unemployment_rate,
		public_debt_pct,
		un_status,
		un_color,
		debt_status,
		debt_color,
	)

    # st.markdown("---")
    st.markdown("\n")
    st.markdown("\n")

    render_top_nav()
    render_sidebar()


    # Simple page routing
    if st.session_state.page == "performance":
        render_performance_page()
        return

    # NEW: route glossary page
    if st.session_state.page == "glossary":
        render_glossary_page()
        return

    # NEW: route GBR model details page
    if st.session_state.page == "gbr_details":
        render_gbr_model_details_page()
        return

    if st.session_state.mode == 'Lite':
        render_lite_mode()
    else:
        render_pro_mode()

    st.markdown("---")
    # Footer: show demo/warning text only when using simulated model
    is_simulated = False
    try:
        pred_cache = st.session_state.get('prediction_cache')
        if isinstance(pred_cache, dict) and pred_cache.get('model_source') == 'simulated':
            is_simulated = True
    except Exception:
        is_simulated = False

    demo_suffix = " | Demo Version" if is_simulated else ""
    warning_line = (
        "<p style='font-size:0.8rem;'>⚠️ Model predictions simulated for demonstration purposes</p>"
        if is_simulated else ""
    )

    st.markdown(f"""
    <div style='text-align:center; color:#666; padding:2rem;'>
        <p><strong>FUREcast SPLG Dashboard</strong> | Educational Analytics Platform</p>
        <p style='font-size:0.9rem;'>Built with Streamlit, scikit-learn, OpenAI API, and Plotly{demo_suffix}</p>
        {warning_line}
    </div>
    """, unsafe_allow_html=True)

def render_performance_page():
    """
    Show model performance plots in tabs (like Quick Analytics).
    Prefer dynamic plotting functions from pred_model.plots; fallback to PNGs
    in pred_model/plots. Each plot gets its own tab for selection.
    """
    import inspect

    st.markdown('<p class="main-header">Model Performance Metrics</p>', unsafe_allow_html=True)
    st.markdown("Select a plot tab to view model performance plots.")

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

    # map internal labels / filenames to friendly display names
    label_map = {
        "cumulative_returns": "Cumulative Returns",  # common spelling
        "error_distribution": "Error Distribution",
        "feature_importance": "Top Features",
        "predictions_vs_actuals": "Prediction vs Actual",
        "residuals": "Residuals",
        "time_series_predictions": "Time Series Predictions",
        "training_progress": "GBR Training Progress"
    }

    # build display labels (preserve original items list for content)
    display_labels = []
    for label, _ in items:
        key = label.lower().replace(" ", "_")
        display_labels.append(label_map.get(key, label.replace("_", " ").title()))

    # Create tabs for each plot (like Quick Analytics) using the friendly labels
    tabs = st.tabs(display_labels)
    for tab, (label, content) in zip(tabs, items):
        with tab:
            # Matplotlib-like figure (has savefig)
            try:
                if hasattr(content, "savefig"):
                    buf = io.BytesIO()
                    content.savefig(buf, format="png", bbox_inches="tight")
                    buf.seek(0)
                    st.image(buf, use_container_width=True)
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
                st.image(content, use_container_width=True)
                continue

            # bytes / raw image
            if isinstance(content, (bytes, bytearray)):
                st.image(content, use_container_width=True)
                continue

            st.warning(f"Could not render plot: {label}")

    # ========== Financial Performance Metrics ==========
    st.markdown("---")
    st.markdown('<p class="chart-header">Financial Performance Metrics</p>', unsafe_allow_html=True)
    st.markdown("These metrics evaluate the model's performance from a trading/investment perspective.")
    
    # Load metrics from metrics.json
    metrics_path = os.path.join(os.path.dirname(__file__), "pred_model", "models", "metrics.json")
    financial_metrics_data = []
    try:
        import json
        with open(metrics_path, 'r') as f:
            metrics_json = json.load(f)
        
        if 'financial_metrics' in metrics_json:
            fm = metrics_json['financial_metrics']
            
            # Define descriptions for each financial metric
            fin_metric_descriptions = {
                'sharpe_ratio': ('Sharpe Ratio', 'Risk-adjusted return measure; higher values indicate better risk-adjusted performance. Values > 1.0 are generally considered good.'),
                'max_drawdown': ('Max Drawdown', 'Largest peak-to-trough decline in portfolio value. Expressed as a negative percentage; closer to 0% is better.'),
                'profit_factor': ('Profit Factor', 'Ratio of gross profits to gross losses. Values > 1.0 indicate profitability; higher is better.'),
                'win_rate': ('Win Rate', 'Percentage of trades that were profitable. Expressed as a decimal (e.g., 0.54 = 54% win rate).'),
                'avg_win': ('Average Win', 'Mean return on profitable trades. Higher values indicate larger average gains.'),
                'avg_loss': ('Average Loss', 'Mean return on losing trades. Values closer to 0 indicate smaller average losses.'),
            }
            
            for key, value in fm.items():
                display_name, description = fin_metric_descriptions.get(key, (key.replace('_', ' ').title(), 'No description available.'))
                # Format the value appropriately
                if key == 'sharpe_ratio':
                    formatted_value = f"{value:.4f}"
                elif key == 'max_drawdown':
                    formatted_value = f"{value * 100:.2f}%"
                elif key == 'profit_factor':
                    formatted_value = f"{value:.4f}"
                elif key == 'win_rate':
                    formatted_value = f"{value * 100:.2f}%"
                elif key in ('avg_win', 'avg_loss'):
                    formatted_value = f"{value * 100:.4f}%"
                else:
                    formatted_value = f"{value:.6f}"
                
                financial_metrics_data.append({
                    'Metric': display_name,
                    'Value': formatted_value,
                    'Description': description
                })
            
            if financial_metrics_data:
                import pandas as pd
                fin_df = pd.DataFrame(financial_metrics_data)
                st.dataframe(fin_df, use_container_width=True, hide_index=True)
            else:
                st.info("No financial metrics available in metrics.json.")
        else:
            st.info("Financial metrics not found in metrics.json.")
    except Exception as e:
        st.error(f"Could not load financial metrics: {e}")

    # ========== Regression Performance Metrics ==========
    st.markdown("---")
    st.markdown('<p class="chart-header">Regression Performance Metrics</p>', unsafe_allow_html=True)
    st.markdown("These metrics evaluate the model's regression accuracy on the held-out test set.")
    
    regression_metrics_data = []
    try:
        # Reuse metrics_json if already loaded, otherwise load again
        if 'metrics_json' not in dir():
            import json
            with open(metrics_path, 'r') as f:
                metrics_json = json.load(f)
        
        if 'test' in metrics_json:
            test_metrics = metrics_json['test']
            
            # Define descriptions for each regression metric
            reg_metric_descriptions = {
                'mse': ('MSE (Mean Squared Error)', 'Average of squared prediction errors. Lower values indicate better fit; sensitive to outliers.'),
                'rmse': ('RMSE (Root Mean Squared Error)', 'Square root of MSE; in the same units as the target variable. Lower is better.'),
                'mae': ('MAE (Mean Absolute Error)', 'Average of absolute prediction errors. Less sensitive to outliers than MSE. Lower is better.'),
                'r2': ('R² (Coefficient of Determination)', 'Proportion of variance explained by the model. Values closer to 1.0 indicate better fit; can be negative for poor models.'),
                'smape': ('SMAPE (Symmetric Mean Absolute Percentage Error)', 'Percentage error metric bounded between 0-200%. Lower values indicate better accuracy.'),
                'directional_accuracy': ('Directional Accuracy', 'Percentage of predictions that correctly predicted the direction of price movement. Values > 0.5 indicate better than random guessing.'),
            }
            
            # Only include the regression metrics (exclude 'predictions' list)
            regression_keys = ['mse', 'rmse', 'mae', 'r2', 'smape', 'directional_accuracy']
            
            for key in regression_keys:
                if key in test_metrics:
                    value = test_metrics[key]
                    display_name, description = reg_metric_descriptions.get(key, (key.upper(), 'No description available.'))
                    
                    # Format the value appropriately
                    if key == 'directional_accuracy':
                        formatted_value = f"{value * 100:.2f}%"
                    elif key == 'smape':
                        formatted_value = f"{value:.2f}%"
                    elif key == 'r2':
                        formatted_value = f"{value:.6f}"
                    elif key in ('mse', 'rmse', 'mae'):
                        formatted_value = f"{value:.6e}" if value < 0.001 else f"{value:.6f}"
                    else:
                        formatted_value = f"{value:.6f}"
                    
                    regression_metrics_data.append({
                        'Metric': display_name,
                        'Value': formatted_value,
                        'Description': description
                    })
            
            if regression_metrics_data:
                import pandas as pd
                reg_df = pd.DataFrame(regression_metrics_data)
                st.dataframe(reg_df, use_container_width=True, hide_index=True)
            else:
                st.info("No regression metrics available in the test object.")
        else:
            st.info("Test metrics not found in metrics.json.")
    except Exception as e:
        st.error(f"Could not load regression metrics: {e}")

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
            st.image(buf, use_container_width=True)
            buf.close()
    except Exception as e:
        st.error(f"Could not generate plots dynamically: {e}")

    if st.button("Back to Dashboard", key="back_from_perf"):
        st.session_state.page = "home"
        safe_rerun()


def _load_feature_definitions_from_markdown():
    """Load feature definitions from DATA_DICTIONARY.md into a dict[name] -> description.

    Parses the Column Index markdown table and returns only the column name and description.
    """
    md_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "wrangling",
        "pred_model_feature_engineering",
        "DATA_DICTIONARY.md",
    )
    # Fallback when running from repo root layout (e.g., testing contexts)
    if not os.path.isfile(md_path):
        md_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "wrangling",
            "pred_model_feature_engineering",
            "DATA_DICTIONARY.md",
        )

    mapping = {}
    try:
        with open(md_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        in_table = False
        for line in lines:
            if line.strip().startswith("| 1 | date "):
                in_table = True
            if not in_table:
                continue
            if not line.strip() or line.lstrip().startswith("---"):
                continue
            parts = [p.strip() for p in line.strip().split("|")]
            if len(parts) < 6:
                continue
            # parts: [ '', '#', 'Column Name', 'Data Type', 'Category', 'Description', '' ]
            try:
                feature_name = parts[2]
                description = parts[5]
            except IndexError:
                continue
            if feature_name and feature_name != "Column Name" and feature_name not in mapping:
                mapping[feature_name] = description
    except Exception:
        # Fail silently; caller will handle missing descriptions gracefully.
        mapping = {}
    return mapping


def _load_feature_importance():
    """Load feature importance CSV used by the current GBR model.

    Returns a DataFrame with columns ['feature', 'importance'] sorted by importance desc.
    """
    csv_path = os.path.join(
        os.path.dirname(__file__),
        "pred_model",
        "models",
        "feature_importance.csv",
    )
    try:
        df = pd.read_csv(csv_path)
        if {"feature", "importance"}.issubset(df.columns):
            df = df.sort_values("importance", ascending=False).reset_index(drop=True)
            return df
    except Exception:
        pass
    return pd.DataFrame(columns=["feature", "importance"])


def _get_gbr_hyperparameter_definitions():
    """Definitions for the GradientBoostingRegressor hyperparameters used in model_metadata.json."""
    return {
        "learning_rate": "Step size shrinkage applied to each tree's contribution; smaller values require more trees.",
        "max_depth": "Maximum depth of individual regression trees that make up the ensemble.",
        "max_features": "Number of features considered when looking for the best split (e.g., 'sqrt' = square root of total features).",
        "min_samples_leaf": "Minimum number of samples required to be at a leaf (terminal) node.",
        "min_samples_split": "Minimum number of samples required to split an internal node.",
        "n_estimators": "Number of boosting stages (trees) in the ensemble.",
        "n_iter_no_change": "Number of iterations with no improvement on the validation loss before early stopping.",
        "random_state": "Seed used by the random number generator for reproducible model training.",
        "subsample": "Fraction of training samples used for fitting each base learner; values <1.0 introduce stochasticity.",
        "tol": "Minimum loss improvement required to continue training when early stopping is enabled.",
        "validation_fraction": "Proportion of training data set aside as validation set for early stopping.",
    }


def render_gbr_model_details_page():
    """Render detailed information about the GBR model features and hyperparameters."""
    st.markdown('<p class="main-header">GBR Model Details</p>', unsafe_allow_html=True)
    st.markdown("Explore the engineered features and GradientBoostingRegressor hyperparameters used by the FURECast SPLG prediction model.")

    st.markdown("---")

    # Load data
    feat_imp_df = _load_feature_importance()
    feat_defs = _load_feature_definitions_from_markdown()
    hyper_defs = _get_gbr_hyperparameter_definitions()

    # Unified search across both tables
    search_query = st.text_input(
        "Search features or hyperparameters:",
        value="",
        key="gbr_details_search",
        help="Filters both Model Features and Hyperparameters tables."
    )

    # ----- Model Features Table -----
    st.markdown("### Model Features")
    if feat_imp_df.empty:
        st.warning("Feature importance data not available.")
    else:
        features_table = feat_imp_df.copy()
        features_table["Definition"] = features_table["feature"].map(feat_defs).fillna("(No definition found in data dictionary)")
        features_table.rename(columns={
            "feature": "Feature",
            "importance": "Current Prediction Importance",
        }, inplace=True)
        # Convert importance to percentage string for display
        features_table["Current Prediction Importance"] = features_table["Current Prediction Importance"].astype(float)
        features_table["Current Prediction Importance"] = features_table["Current Prediction Importance"].map(lambda x: f"{x:.2%}")

        if search_query:
            mask = features_table.apply(
                lambda row: row.astype(str).str.contains(search_query, case=False, na=False).any(),
                axis=1,
            )
            display_features = features_table[mask].reset_index(drop=True)
        else:
            display_features = features_table

        st.dataframe(display_features, use_container_width=True, hide_index=True)

        # Download CSV for features table
        csv_bytes = display_features.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Features CSV",
            csv_bytes,
            file_name="gbr_model_features.csv",
            mime="text/csv",
            key="gbr_features_download",
        )

    st.markdown("---")

    # ----- Hyperparameter Definitions Table -----
    st.markdown("### GBR Hyperparameters")
    if not hyper_defs:
        st.warning("No hyperparameter definitions available.")
    else:
        hyper_df = pd.DataFrame(
            [
                {"Hyperparameter": name, "Definition": desc}
                for name, desc in hyper_defs.items()
            ]
        )
        if search_query:
            mask = hyper_df.apply(
                lambda row: row.astype(str).str.contains(search_query, case=False, na=False).any(),
                axis=1,
            )
            display_hyper = hyper_df[mask].reset_index(drop=True)
        else:
            display_hyper = hyper_df
        st.dataframe(display_hyper, use_container_width=True, hide_index=True)

        # Download CSV for hyperparameters table
        hyper_csv_bytes = display_hyper.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Hyperparameters CSV",
            hyper_csv_bytes,
            file_name="gbr_hyperparameters.csv",
            mime="text/csv",
            key="gbr_hyperparams_download",
        )

    st.markdown("---")
    if st.button("Back to Dashboard", key="back_from_gbr_details"):
        st.session_state.page = "home"
        safe_rerun()

if __name__ == "__main__":
    main()