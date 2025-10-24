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


def predict_splg() -> Dict[str, Any]:
    """
    Simulate GBR model prediction for SPLG next-day return.
    
    Uses OpenAI to generate realistic prediction with explanation.
    Returns structured prediction data.
    """
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
def create_price_chart(metric, start_date, end_date):
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
    
    df = pd.read_csv("../../data/rich_features_SPLG_history_full.csv")
    print("CSV loaded:", df.head())  # Debugging line

    df['date'] = pd.to_datetime(df['date'])
    
    # Ensure all dates are Timestamps
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Get the maximum date from the dataset
    max_available_date = df['date'].max()
    
    # Adjust end_date if it exceeds the maximum available date
    if end_date > max_available_date:
        print(f"Warning: Requested end date {end_date.date()} exceeds available data. Using maximum available date: {max_available_date.date()}")
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
    
    fig = px.line(
        df,
        x='date',
        y=df_column,
        title=f"{display_name} Price from {start_date.date()} to {end_date.date()}",
        labels={'date': 'Date', df_column: f'{display_name} Price ($)'}
    )
    fig.update_layout(template='plotly_white')
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
