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
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '../..', '.env'))
    api_key = os.getenv('OPENAI_API_KEY')
    return OpenAI(api_key=api_key)


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
            # Import prediction module
            import sys
            from pathlib import Path
            pred_model_path = Path(__file__).parent / "pred_model"
            if str(pred_model_path) not in sys.path:
                sys.path.insert(0, str(pred_model_path))
            
            from predict import load_model, predict_with_explanation  # type: ignore
            from get_latest_features import get_latest_features  # type: ignore
            
            # Load model
            model_bundle = load_model()
            
            # Get latest features
            latest_features = get_latest_features(1)
            
            # Make prediction with CORRECT argument order: (features, model_bundle)
            result = predict_with_explanation(latest_features, model_bundle, top_n=5)
            
            # DEBUG
            print(f'USING PREDICTION MODEL (not dummy model)')
            # Format for app consumption
            return {
                'predicted_return': result['predicted_return'],
                'direction': result['direction'],
                'confidence': result['confidence'],
                'top_features': result['top_features']
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
            ]
        }
def merge_timeline_into_prices(df_prices: pd.DataFrame, timeline_path: str) -> pd.DataFrame:
    """
    Merge timeline events into price DataFrame.

    For each date in the prices DataFrame, this function will attach any events
    found in the timeline file. Multiple events on the same day will be
    aggregated into a single string (newline-separated). The returned DataFrame
    will have new optional columns: 'timeline_events' (str) and
    'timeline_event_count' (int).

    Args:
        df_prices: DataFrame with a 'date' column (datetime64)
        timeline_path: absolute path to the Excel file containing timeline

    Returns:
        DataFrame with merged timeline columns. If timeline file is missing or
        empty, returns the original df_prices unchanged.
    """
    try:
        tl = pd.read_excel(timeline_path)
    except FileNotFoundError:
        # No timeline available — return prices untouched
        return df_prices
    except Exception:
        # If excel can't be read for any reason, return prices untouched
        return df_prices

    if tl is None or tl.empty:
        return df_prices

    # Robust column detection
    col_candidates = {c.lower(): c for c in tl.columns}
    # date column
    date_col = None
    for name in ('date', 'day', 'event date', 'event_date'):
        if name in col_candidates:
            date_col = col_candidates[name]
            break
    if date_col is None:
        # Try to find any datetime-like column
        for c in tl.columns:
            if np.issubdtype(tl[c].dtype, np.datetime64):
                date_col = c
                break
    if date_col is None:
        # Can't find a date column; give up
        return df_prices

    # Event description column
    event_col = None
    for name in ('event', 'description', 'event description', 'details'):
        if name in col_candidates:
            event_col = col_candidates[name]
            break
    # Category and impact
    category_col = None
    impact_col = None
    for name in ('category', 'type'):
        if name in col_candidates:
            category_col = col_candidates[name]
            break
    for name in ('impact', 'market impact'):
        if name in col_candidates:
            impact_col = col_candidates[name]
            break

    # Normalize timeline date column
    tl[date_col] = pd.to_datetime(tl[date_col]).dt.normalize()

    # Build a single textual description per event row
    def make_event_row(r):
        parts = []
        if event_col and pd.notna(r.get(event_col, None)):
            parts.append(str(r[event_col]))
        if category_col and pd.notna(r.get(category_col, None)):
            parts.append(f"Category: {r[category_col]}")
        if impact_col and pd.notna(r.get(impact_col, None)):
            parts.append(f"Impact: {r[impact_col]}")
        return " — ".join(parts) if parts else None

    tl['__event_text'] = tl.apply(make_event_row, axis=1)

    # Aggregate multiple events per day
    agg = tl.groupby(date_col)['__event_text'].apply(lambda rows: "\n".join([r for r in rows if r]))
    agg = agg.reset_index().rename(columns={date_col: 'date', '__event_text': 'timeline_events'})
    agg['date'] = pd.to_datetime(agg['date']).dt.normalize()
    agg['timeline_event_count'] = agg['timeline_events'].str.count('\n').fillna(0).astype(int) + (agg['timeline_events'] != '').astype(int)

    # Normalize price df date
    df = df_prices.copy()
    df['date'] = pd.to_datetime(df['date']).dt.normalize()

    # Merge
    merged = pd.merge(df, agg, on='date', how='left')

    # Fill NaNs
    merged['timeline_events'] = merged['timeline_events'].fillna("")
    merged['timeline_event_count'] = merged.get('timeline_event_count', 0).fillna(0).astype(int)

    return merged


def create_price_chart(metric, start_date, end_date):
    """
    Create a Plotly line chart for a given price metric and overlay timeline events.

    This function resolves file paths relative to this module. If the primary
    CSV is missing it falls back to simulated prices (using fetch_prices).
    """
    # Mapping between display names and DataFrame columns
    metric_mapping = {
        "Closing": "close",
        "Opening": "open",
        "Daily High": "high",
        "Daily Low": "low",
        "Daily Current": "current_price"
    }

    # Get the actual column name from the mapping, or use the metric as-is if not in mapping
    df_column = metric_mapping.get(metric, metric)

    # Resolve data paths relative to this file
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    prices_path = os.path.join(base_dir, 'data', 'rich_features_SPLG_history_full.csv')
    timeline_path = os.path.join(base_dir, 'data', 'Financial Crisis Timeline by Day (since 2005).xlsx')

    # Ensure dates are timestamps
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Load prices (with fallback to simulated data)
    if os.path.exists(prices_path):
        df = pd.read_csv(prices_path)
        df['date'] = pd.to_datetime(df['date'])
    else:
        # Fallback: generate simulated daily business prices for SPLG
        sim = fetch_prices(['SPLG'], start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        df = sim[sim['ticker'] == 'SPLG'].rename(columns={'close': 'close'})

    # Clip end_date to available data
    max_available_date = df['date'].max()
    if pd.isna(max_available_date):
        raise ValueError("Price dataset contains no dates")
    if end_date > max_available_date:
        end_date = max_available_date

    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()

    if df.empty:
        raise ValueError("No data available for the selected date range.")

    # Merge timeline events (if available)
    df_merged = merge_timeline_into_prices(df, timeline_path)

    # Display name for title
    display_names = {
        "close": "Closing",
        "open": "Opening",
        "high": "Daily High",
        "low": "Daily Low",
        "current_price": "Daily Current"
    }
    display_name = metric if metric in metric_mapping else display_names.get(metric, metric.capitalize())

    # Base line chart
    fig = px.line(
        df_merged,
        x='date',
        y=df_column,
        title=f"{display_name} Price from {start_date.date()} to {end_date.date()}",
        labels={'date': 'Date', df_column: f'{display_name} Price ($)'}
    )

    # Add event markers for dates that have timeline events
    # Use a safe boolean mask: if the column exists build mask, otherwise empty mask
    if 'timeline_event_count' in df_merged.columns:
        mask = df_merged['timeline_event_count'] > 0
    else:
        mask = pd.Series(False, index=df_merged.index)
    events_df = df_merged[mask]
    if not events_df.empty:
        # Build hover text combining price info and timeline events
        hover_texts = []
        for _, r in events_df.iterrows():
            price_val = r.get(df_column, '')
            events = r.get('timeline_events', '')
            events_html = events.replace('\n', '<br>')
            # Add a bold heading for the events section and bold labels for date/price
            text = (
                f"<b>Date:</b> {r['date'].date()}<br>"
                f"<b>{display_name}:</b> {price_val}<br><br>"
                f"<b>Events:</b><br>{events_html}"
            )
            hover_texts.append(text)

        fig.add_trace(go.Scatter(
            x=events_df['date'],
            y=events_df[df_column],
            mode='markers',
            marker=dict(size=8, color='red', symbol='triangle-up'),
            name='Timeline Events',
            hoverinfo='text',
            hovertext=hover_texts
        ))

    # Ensure chart has reasonable height and bottom margin so UI elements
    # (like the 'How to Use This Tool' info box) don't visually overlap.
    fig.update_layout(template='plotly_white', height=480, margin=dict(t=60, b=80))
    return fig


def create_sector_risk_treemap() -> go.Figure:
    """Create treemap showing sector risk levels"""
    sectors = {
        'Technology': {'size': 28.5, 'volatility': 0.185},
        'Healthcare': {'size': 13.2, 'volatility': 0.142},
        'Financials': {'size': 13.8, 'volatility': 0.208},
        'Consumer Discretionary': {'size': 10.3, 'volatility': 0.195},
        'Industrials': {'size': 8.7, 'volatility': 0.178},
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
    
    fig = px.treemap(
        df,
        path=['sector'],
        values='size',
        color='volatility',
        color_continuous_scale='RdYlGn_r',
        title='SPLG Sector Risk Map (by Volatility)'
    )
    
    fig.update_traces(
        textinfo='label+value',
        textposition='middle center',
        textfont=dict(size=14)
    )
    
    fig.update_layout(
        height=500,
        coloraxis_colorbar=dict(title="Volatility")
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
        return create_price_chart(spec.get('ticker', 'SPLG'), spec.get('days', 180))
    else:
        # Default to price chart
        return create_price_chart()
