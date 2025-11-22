"""
FUREcast - Simulated Data & Tool Functions

This module provides simulated versions of the core tools that will eventually
connect to real models, APIs, and databases. It demonstrates the data structures
and interfaces the LLM will interact with.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional
import os
from openai import OpenAI


def get_openai_client():
    """Initialize OpenAI client with API key from environment"""
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(__file__), '../..', '.env'))
    except Exception:
        pass
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get('OPENAI_API_KEY', '')
        except Exception:
            api_key = ''
    return OpenAI(api_key=api_key or None)


def fetch_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """
    Simulate fetching historical price data for given tickers.
    
    In production, this would call yfinance or Alpha Vantage.
    """
    start_date = datetime.strptime(start, '%Y-%m-%d')
    end_date = datetime.strptime(end, '%Y-%m-%d')
    
    dates = pd.date_range(start_date, end_date, freq='B')  # Business days
    
    data = []
    for ticker in tickers:
        # Simulate realistic price movement
        np.random.seed(hash(ticker) % 2**32)  # Consistent per ticker
        base_price = np.random.uniform(50, 500)
        returns = np.random.normal(0.0005, 0.015, len(dates))  # Daily returns
        prices = base_price * np.exp(np.cumsum(returns))
        
        for date, price in zip(dates, prices):
            data.append({
                'ticker': ticker,
                'date': date,
                'open': price * (1 + np.random.uniform(-0.01, 0.01)),
                'high': price * (1 + np.random.uniform(0, 0.02)),
                'low': price * (1 - np.random.uniform(0, 0.02)),
                'close': price,
                'volume': int(np.random.uniform(1e6, 10e6))
            })
    
    return pd.DataFrame(data)


def compute_risk(df: pd.DataFrame, metric: str = 'volatility', window: int = 60) -> Dict[str, float]:
    """
    Simulate risk metric calculations.
    
    Returns a dictionary of ticker -> metric value.
    """
    results = {}
    
    for ticker in df['ticker'].unique():
        ticker_data = df[df['ticker'] == ticker].sort_values('date')
        returns = ticker_data['close'].pct_change().dropna()
        
        if metric == 'volatility':
            # Annualized volatility
            vol = returns.tail(window).std() * np.sqrt(252)
            results[ticker] = vol
        elif metric == 'sharpe':
            # Simplified Sharpe (assuming 2% risk-free rate)
            excess_return = returns.tail(window).mean() * 252 - 0.02
            vol = returns.tail(window).std() * np.sqrt(252)
            results[ticker] = excess_return / vol if vol > 0 else 0
        elif metric == 'drawdown':
            # Max drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = ((cumulative - running_max) / running_max).min()
            results[ticker] = abs(drawdown)
    
    return results


def predict_splg(use_real_model: bool = True) -> Dict[str, Any]:
    """
    Generate GBR model prediction for SPLG next-day return.
    
    Args:
        use_real_model: If True, use trained GBR model; if False, use simulated prediction
    
    Returns:
        Dictionary with prediction results:
        - predicted_return: float
        - direction: 'up', 'down', or 'neutral'
        - confidence: float
        - top_features: list of {name, importance} dicts
    """
    if use_real_model:
        try:
            # Import from pred_model package using relative import
            from pred_model.predict import load_model, predict_with_explanation
            from pred_model.get_latest_features import get_latest_features
            
            # Load model
            model_bundle = load_model()
            
            # Get latest features
            latest_features = get_latest_features(1)
            
            # Make prediction with CORRECT argument order: (features, model_bundle)
            result = predict_with_explanation(latest_features, model_bundle, top_n=5)
            
            # DEBUG
            print(f'INFO - Using GBR prediction model (not dummy model)')
            
            # Format for app consumption
            return {
                'predicted_return': result['predicted_return'],
                'direction': result['direction'],
                'confidence': result['confidence'],
                'top_features': result['top_features'],
                'model_source': 'real'
            }
            
        except FileNotFoundError as e:
            # Model not trained yet, fall back to simulated
            print(f"⚠️ Trained model not found: {e}")
            print("   To train: cd pred_model && python scripts/train_gbr_model.py --quick")
            use_real_model = False
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")
            use_real_model = False
    
    if not use_real_model:
        # Simulated prediction (fallback)
        client = get_openai_client()
        
        prompt = """Generate a realistic stock market prediction for SPLG ETF (S&P 500 Large Cap).
Return a JSON object with:
- predicted_return: float between -0.02 and 0.02 (next day % return)
- direction: "up", "down", or "neutral" (based on return)
- confidence: float between 0.5 and 0.95
- top_features: array of 5 objects with {name: string, importance: float 0-1}

Make the prediction realistic based on current market conditions. Features should be technical indicators like MA_20, RSI, Volatility_5d, etc."""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            # Ensure direction matches return sign
            pred_return = result.get('predicted_return', 0)
            if pred_return > 0.002:
                result['direction'] = 'up'
            elif pred_return < -0.002:
                result['direction'] = 'down'
            else:
                result['direction'] = 'neutral'
                
            # Tag as simulated so UI can conditionally show refresh button
            result['model_source'] = 'simulated'
            return result
        except Exception as e:
            # Fallback to static data if API fails
            return {
                'predicted_return': 0.0035,
                'direction': 'up',
                'confidence': 0.72,
                'top_features': [
                    {'name': 'MA_20_deviation', 'importance': 0.23},
                    {'name': 'RSI_14', 'importance': 0.18},
                    {'name': 'Volatility_5d', 'importance': 0.15},
                    {'name': 'MACD_signal', 'importance': 0.12},
                    {'name': 'Volume_ratio', 'importance': 0.09}
                ],
                'model_source': 'simulated'
            }
def create_price_chart(metric, start_date, end_date, show_events: bool = True):
    # Mapping between display names and DataFrame columns
    metric_mapping = {
        "Closing": "close",
        "Opening": "open",
        "Daily High": "high",
        "Daily Low": "low",
        "Daily Current": "current_price"
    }
    
    # Get the actual column name from the mapping, or use the metric as-is if not in mapping (for backward compatibility)
    df_column = metric_mapping.get(metric, metric)
    
    # Resolve data path relative to this file so it works on Streamlit Cloud
    data_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'rich_features_SPLG_history_full.csv')
    )
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    df = pd.read_csv(data_path)
    # Robust datetime parsing: handle "YYYY-MM-DD" and "YYYY-MM-DD HH:MM:SS" and similar
    try:
        # pandas >= 2.0 supports format='mixed'
        df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce', utc=False)
    except TypeError:
        # fallback for older pandas: try generic parse then trim to date if needed
        df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=False)
        # If any still NaT, try slicing first 10 chars (YYYY-MM-DD)
        mask = df['date'].isna()
        if mask.any():
            df.loc[mask, 'date'] = pd.to_datetime(
                df.loc[mask, 'date'].astype(str).str.slice(0, 10),
                errors='coerce',
                utc=False
            )

    # Drop rows we couldn't parse
    before = len(df)
    df = df.dropna(subset=['date']).copy()
    if len(df) < before:
        # optional: avoid noisy logs, but keeps app stable
        pass

    # Ensure all dates are Timestamps
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Get the maximum date from the dataset
    max_available_date = df['date'].max()
    
    # Adjust end_date if it exceeds the maximum available date
    if end_date > max_available_date:
        # print(f"Warning: Requested end date {end_date.date()} exceeds available data. Using maximum available date: {max_available_date.date()}")
        end_date = max_available_date
    
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

    if df.empty:
        raise ValueError("No data available for the selected date range.")

    # Use the display name for the title and label, or map the column name to a proper display name
    display_names = {
        "close": "Closing",
        "open": "Opening",
        "high": "Daily High",
        "low": "Daily Low",
        "current_price": "Daily Current"
    }
    display_name = metric if metric in metric_mapping else display_names.get(metric, metric.capitalize())

    # --- Integrate Financial Crises events for hover display (CSV or XLSX) ---
    events_df = None
    event_label = 'Event'
    category_label = 'Category'
    impact_label = 'Impact'

    if show_events:
        data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
        # try to find a file that looks like the financial crises timeline (either csv or xlsx)
        events_file = None
        try:
            for f in os.listdir(data_dir):
                if 'financial' in f.lower() and ('crisis' in f.lower() or 'crises' in f.lower() or 'timeline' in f.lower()):
                    if f.lower().endswith(('.csv', '.xlsx', '.xls')):
                        events_file = os.path.join(data_dir, f)
                        break

            if events_file is None:
                raise FileNotFoundError("No financial events file found in data directory")

            if events_file.lower().endswith('.csv'):
                raw_events = pd.read_csv(events_file, dtype=str, encoding='utf-8', low_memory=False)
            else:
                raw_events = pd.read_excel(events_file, dtype=str)

            # Normalize column names to lowercase for detection
            cols_lower = {c: c.lower() for c in raw_events.columns}

            # find a date column (common names include 'date' or the verbose CSV header)
            date_col = None
            for orig, low in cols_lower.items():
                if 'date' in low or 'year' in low or 'day' in low or 'period' in low:
                    date_col = orig
                    break
            if date_col is None:
                # fallback to first column if detection fails
                date_col = raw_events.columns[0]

            # detect event / category / impact columns (prefer exact headers in your CSV)
            def find_col_like(key_words):
                for orig, low in cols_lower.items():
                    for kw in key_words:
                        if kw in low:
                            return orig
                return None

            event_col_name = find_col_like(['event', 'description', 'title', 'name'])
            category_col_name = find_col_like(['category', 'type'])
            impact_col_name = find_col_like(['impact', 'economic', 'market', 'effect', 's&p'])

            # Use the detected original headers as labels for hover text
            event_label = event_col_name if event_col_name is not None else 'Event'
            category_label = category_col_name if category_col_name is not None else 'Category'
            impact_label = impact_col_name if impact_col_name is not None else 'Impact'

            # Build normalized events_df with predictable column names
            events_df = pd.DataFrame()
            events_df['event_date'] = pd.to_datetime(raw_events[date_col].astype(str), errors='coerce')
            events_df['event'] = raw_events[event_col_name].fillna('').astype(str) if event_col_name is not None else ''
            events_df['category'] = raw_events[category_col_name].fillna('').astype(str) if category_col_name is not None else ''
            events_df['impact'] = raw_events[impact_col_name].fillna('').astype(str) if impact_col_name is not None else ''

            events_df = events_df.dropna(subset=['event_date']).sort_values('event_date').reset_index(drop=True)
        except Exception:
            events_df = None
            event_label = event_label or 'Event'
            category_label = category_label or 'Category'
            impact_label = impact_label or 'Impact'

    # Merge events onto price df: prefer exact date matches, fall back to nearest within 3 days
    if show_events and events_df is not None and not events_df.empty:
        df_sorted = df.sort_values('date').reset_index(drop=True)
        ev_sorted = events_df.sort_values('event_date').reset_index(drop=True).rename(columns={'event_date': 'date'})

        # exact merge first
        merged = df_sorted.merge(ev_sorted, on='date', how='left')

        # where exact match is missing, fill from nearest within 3 days using merge_asof
        if merged[['event', 'category', 'impact']].isna().any().any():
            asof = pd.merge_asof(df_sorted, ev_sorted, on='date', direction='nearest', tolerance=pd.Timedelta(days=3))
            for col in ['event', 'category', 'impact']:
                merged[col] = merged[col].fillna(asof[col])

        # Ensure no NaNs (use empty string)
        merged[['event', 'category', 'impact']] = merged[['event', 'category', 'impact']].fillna('').astype(str)
        plot_df = merged
    else:
        # no events or user disabled: create empty columns for consistent plotting
        plot_df = df.copy()
        plot_df['event'] = ''
        plot_df['category'] = ''
        plot_df['impact'] = ''
        # labels remain default

    # Build customdata arrays for hovertemplate
    # Base trace: include next-day return for hover (% format)
    customdata_returns = plot_df[['target_return_t1']].values
    # Event trace: include return plus event metadata
    customdata_all = plot_df[['target_return_t1', 'event', 'category', 'impact']].astype(object).values

    # Boolean mask for rows that have an event
    has_event = plot_df['event'].astype(str).str.strip() != ''

    y_series = plot_df[df_column]

    # y values for event-only trace (non-event -> NaN so Plotly won't draw/hover)
    y_for_events = y_series.where(has_event, np.nan)

    # Base figure: thin muted line for all dates (no event details)
    fig = go.Figure()

    # make base line darker for all selected metrics
    base_line_color = 'dimgray'
    base_line_width = 1.6

    fig.add_trace(go.Scatter(
        x=plot_df['date'],
        y=y_series,
        mode='lines',
        name=display_name,
        line=dict(width=base_line_width, color=base_line_color),
        customdata=customdata_returns,
        hoverinfo='x+y',
        hovertemplate=(
            "Date: %{x|%Y-%m-%d}<br>"
            f"{display_name} Price: $%{{y:.2f}}<br>"
            "Next-Day Return: %{customdata[0]:.2%}<extra></extra>"
        )
    ))

    # Overlay: bold line+markers only on event dates, show event details in hover
    # add overlay only if show_events is True
    if show_events:
        fig.add_trace(go.Scatter(
            x=plot_df['date'],
            y=y_for_events,
            mode='lines+markers',
            name='Event (highlight)',
            line=dict(width=2, color='crimson'),  # adjust thickness here
            marker=dict(size=6, color='crimson'),
            customdata=customdata_all,
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>"
                f"{display_name} Price: $%{{y:.2f}}<br>"
                "Next-Day Return: %{customdata[0]:.2%}<br>"
                f"{event_label}: %{{customdata[1]}}<br>"
                f"{category_label}: %{{customdata[2]}}<br>"
                f"{impact_label}: %{{customdata[3]}}<extra></extra>"
            )
        ))

    # place title slightly higher so "Closing Price from..." sits up one line
    fig.update_layout(
        title=dict(
            text=f"{display_name} Price from {start_date.date()} to {end_date.date()}",
            x=0.01,
            xanchor='left',
            y=0.995,
            yanchor='top',
            font=dict(size=16)
        ),
        xaxis_title='Date',
        yaxis_title=f'{display_name} Price ($)',
        template='plotly_white',
        plot_bgcolor='white',
        legend_title_text="Click on a line below to show/unshow on chart",
        # reduced top margin since annotations were removed
        margin=dict(t=80),
        legend=dict(
            orientation='h',
            y=1.03,
            x=0,
            xanchor='left',
            yanchor='bottom'
        )
    )

    # Darker gridlines (increase opacity + slightly thicker)
    dark_grid = 'rgba(80,80,80,0.18)'   # darker/more visible grid color
    dark_zeroline = 'rgba(0,0,0,0.06)'

    fig.update_xaxes(
        showgrid=True,
        gridcolor=dark_grid,
        gridwidth=1.2,
        zeroline=True,
        zerolinecolor=dark_zeroline
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=dark_grid,
        gridwidth=1.2,
        zeroline=True,
        zerolinecolor=dark_zeroline
    )
    return fig


def create_sector_risk_treemap() -> go.Figure:
    """Create treemap showing sector risk levels"""
    sectors = {
        'Technology': {'size': 28.5, 'volatility': 0.185},
        'Healthcare': {'size': 13.2, 'volatility': 0.142},
        'Financial': {'size': 13.8, 'volatility': 0.208},
        'Consumer Discretionary': {'size': 10.3, 'volatility': 0.195},
        'Industrial': {'size': 8.7, 'volatility': 0.178},
        'Communications': {'size': 8.4, 'volatility': 0.165},
        'Consumer Staples': {'size': 6.8, 'volatility': 0.118},
        'Energy': {'size': 4.2, 'volatility': 0.295},
        'Utilities': {'size': 2.9, 'volatility': 0.138},
        'Real Estate': {'size': 2.4, 'volatility': 0.225},
        'Materials': {'size': 2.8, 'volatility': 0.201}
    }
    
    df = pd.DataFrame([
        {'sector': k, 'size': v['size'], 'volatility': v['volatility']}
        for k, v in sectors.items()
    ])

    # Add volatility in percent so we can format it easily in hovertemplate
    df['volatility_pct'] = df['volatility'] * 100
    
    fig = px.treemap(
        df,
        path=['sector'],
        values='size',
        color='volatility',
        color_continuous_scale='RdYlGn_r',
        title='SPLG Sector Risk Map (by Volatility)',
        custom_data=['volatility_pct']  # make volatility percent available to the trace
    )

    fig.update_traces(
        # display label inside tiles (e.g. "Technology")
        textposition='middle center',
        textfont=dict(size=14),
        texttemplate='%{label}',
        # hover shows sector, size with percent sign, and volatility as percent
        hovertemplate=(
            'sector=%{label}<br>'
            'size=%{value:.1f}%<br>'
            'volatility=%{customdata[0]:.1f}%<br>'
            'Size of block indicates market share and color of box indicates volatility<extra></extra>'
        )
    )
    
    fig.update_layout(
        height=500,
        coloraxis_colorbar=dict(title="Volatility (%)", tickformat='.1%')
    )
    
    return fig


def create_sector_holdings_treemap(color_metric: str = 'DailyChangePct') -> go.Figure:
    """
    Create drill-down treemap showing Sector → Company holdings with real SPLG data.
    
    Args:
        color_metric: Metric to use for color ('DailyChangePct', 'PE', 'Beta', 'DividendYield')
    
    Returns:
        Plotly treemap figure
    """
    # Load the real treemap data
    csv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'treemap_nodes.csv')
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        # Fallback to simulated data if file not found
        return create_sector_risk_treemap()
    
    # Prepare hover data
    hover_cols = ['Weight (%)', 'DailyChangePct', 'PE', 'Beta', 'DividendYield']
    
    # Set color scale and midpoint based on metric
    if color_metric == 'DailyChangePct':
        color_scale = 'RdYlGn'
        range_center = 0
    else:
        color_scale = 'Viridis'
        range_center = None
    
    # Create treemap with Sector → Company drill-down
    fig = px.treemap(
        df,
        path=['Sector', 'Company'],
        values='Weight (%)',
        color=color_metric,
        color_continuous_midpoint=range_center,
        color_continuous_scale=color_scale,
        hover_data=hover_cols,
        title='SPLG Sector → Company Treemap (Real Holdings)'
    )
    
    fig.update_traces(
        textposition='middle center',
        textfont=dict(size=12)
    )
    
    fig.update_layout(
        height=600,
        coloraxis_colorbar=dict(title=color_metric)
    )
    
    return fig


def get_sector_summary() -> pd.DataFrame:
    """
    Get sector-level summary statistics from the treemap data.
    
    Returns:
        DataFrame with aggregated sector metrics
    """
    csv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'treemap_nodes.csv')
    
    try:
        df = pd.read_csv(csv_path)
        
        # Aggregate by sector
        summary = df.groupby('Sector').agg({
            'Weight (%)': 'sum',
            'DailyChangePct': 'mean',
            'PE': 'mean',
            'Beta': 'mean',
            'DividendYield': 'mean'
        }).reset_index().sort_values('Weight (%)', ascending=False)
        
        # Round for display
        summary['Weight (%)'] = summary['Weight (%)'].round(2)
        summary['DailyChangePct'] = summary['DailyChangePct'].round(3)
        summary['PE'] = summary['PE'].round(2)
        summary['Beta'] = summary['Beta'].round(3)
        summary['DividendYield'] = summary['DividendYield'].round(2)
        
        return summary
    except FileNotFoundError:
        # Return empty dataframe if file not found
        return pd.DataFrame()


def create_feature_importance_chart(features: List[Dict[str, Any]]) -> go.Figure:
    """Create bar chart of feature importances"""
    df = pd.DataFrame(features)
    df = df.sort_values('importance', ascending=True)
    
    fig = go.Figure(go.Bar(
        x=df['importance'],
        y=df['name'],
        orientation='h',
        marker=dict(
            color=df['importance'],
            colorscale='Blues',
            showscale=False
        )
    ))
    
    fig.update_layout(
        title='Top Model Features (by Importance)',
        xaxis_title='Importance Score',
        yaxis_title='Feature',
        height=300,
        template='plotly_white',
        margin=dict(l=150)
    )
    
    return fig


def create_sector_comparison_chart(sectors: List[str]) -> go.Figure:
    """Create line chart comparing sector performance"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # Map sector names to ETF tickers
    sector_tickers = {
        'Technology': 'XLK',
        'Healthcare': 'XLV',
        'Financials': 'XLF',
        'Consumer Discretionary': 'XLY',
        'Industrials': 'XLI',
        'Communications': 'XLC',
        'Consumer Staples': 'XLP',
        'Energy': 'XLE',
        'Utilities': 'XLU',
        'Real Estate': 'XLRE',
        'Materials': 'XLB'
    }
    
    tickers = [sector_tickers.get(s, 'XLK') for s in sectors[:3]]  # Limit to 3
    df = fetch_prices(tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    
    fig = go.Figure()
    
    for ticker in tickers:
        ticker_data = df[df['ticker'] == ticker].sort_values('date')
        # Normalize to 100 at start
        normalized = (ticker_data['close'] / ticker_data['close'].iloc[0]) * 100
        
        fig.add_trace(go.Scatter(
            x=ticker_data['date'],
            y=normalized,
            mode='lines',
            name=ticker,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title='Sector Performance Comparison (Normalized)',
        xaxis_title='Date',
        yaxis_title='Normalized Price (Base = 100)',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig


def viz_from_spec(spec: Dict[str, Any]) -> go.Figure:
    """
    Create visualization from specification.
    
    This demonstrates the planned tool interface where LLM can request
    charts by providing a structured spec.
    """
    chart_type = spec.get('type', 'line')
    
    if chart_type == 'treemap':
        return create_sector_risk_treemap()
    elif chart_type == 'bar' and 'features' in spec:
        return create_feature_importance_chart(spec['features'])
    elif chart_type == 'line' and 'sectors' in spec:
        return create_sector_comparison_chart(spec['sectors'])
    elif chart_type == 'price':
        # Build a reasonable date range and let create_price_chart clamp to dataset max
        days = int(spec.get('days', 180))
        end_date = pd.to_datetime('today').normalize()
        start_date = end_date - pd.Timedelta(days=days)
        # default to 'close' metric; create_price_chart handles mapping
        return create_price_chart('close', start_date, end_date, show_events=True)
    else:
        # Default to a price chart
        days = 180
        end_date = pd.to_datetime('today').normalize()
        start_date = end_date - pd.Timedelta(days=days)
        return create_price_chart('close', start_date, end_date, show_events=True)
