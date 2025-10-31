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

def fetch_prices(tickers: List[str], start: str = None, end: str = None) -> pd.DataFrame:
    """
    Load SPLG historical data from rich_features_SPLG_history.csv.
    Filters by ticker(s) and optional date range.
    """
    # Construct path to your data file
    csv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'rich_features_SPLG_history_full.csv')

    # Load CSV
    df = pd.read_csv(csv_path)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(r'[\s\-]+', '_', regex=True)

    # Ensure expected columns exist
    rename_map = {
        'date': 'date',
        'ticker': 'ticker',
        'close': 'close',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'volume': 'volume'
    }
    df = df.rename(columns=rename_map)

    # Parse dates
    df['date'] = pd.to_datetime(df['date'])

    # Filter by tickers
    if tickers:
        df = df[df['ticker'].isin(tickers)]

    # Filter by date range
    if start:
        df = df[df['date'] >= pd.to_datetime(start)]
    if end:
        df = df[df['date'] <= pd.to_datetime(end)]

    # Sort by date
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)

    return df


def compute_risk(metric: str = 'volatility', window: int = 60) -> Dict[str, float]:
    """
    Compute risk metrics for SPLG using data from rich_features_SPLG_history.csv.
    Supported metrics: 'volatility', 'sharpe', 'drawdown'.
    """
    # Load SPLG historical data
    csv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'rich_features_SPLG_history.csv')
    df = pd.read_csv(csv_path)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(r'[\s\-]+', '_', regex=True)
    df = df.rename(columns={
        'date': 'date',
        'ticker': 'ticker',
        'close': 'close'
    })

    # Parse dates and filter to SPLG
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['ticker'].str.upper() == 'SPLG'].sort_values('date')

    # Calculate daily returns
    df['return'] = df['close'].pct_change().dropna()

    results = {}

    # Compute selected risk metric
    if metric == 'volatility':
        vol = df['return'].tail(window).std() * np.sqrt(252)
        results['SPLG'] = vol

    elif metric == 'sharpe':
        # Simplified Sharpe ratio assuming 2% annual risk-free rate
        mean_return = df['return'].tail(window).mean() * 252
        vol = df['return'].tail(window).std() * np.sqrt(252)
        rf = 0.02
        results['SPLG'] = (mean_return - rf) / vol if vol > 0 else 0

    elif metric == 'drawdown':
        cumulative = (1 + df['return']).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = ((cumulative - running_max) / running_max).min()
        results['SPLG'] = abs(drawdown)

    else:
        raise ValueError("Invalid metric. Use 'volatility', 'sharpe', or 'drawdown'.")

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


def create_price_chart(ticker: str = 'SPLG', days: int = 180) -> go.Figure:
    """Create SPLG price chart with moving averages"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    df = fetch_prices([ticker], start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    df = df.sort_values('date')
    
    # Calculate moving averages
    df['MA_20'] = df['close'].rolling(20).mean()
    df['MA_50'] = df['close'].rolling(50).mean()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['close'],
        mode='lines',
        name='Close Price',
        line=dict(color='#2E86AB', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['MA_20'],
        mode='lines',
        name='20-Day MA',
        line=dict(color='#A23B72', width=1, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['MA_50'],
        mode='lines',
        name='50-Day MA',
        line=dict(color='#F18F01', width=1, dash='dot')
    ))
    
    fig.update_layout(
        title=f'{ticker} Price with Moving Averages',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig


def create_sector_risk_treemap() -> go.Figure:
    """
    Create treemap showing sector risk levels using holdings-with-sector.csv.
    - 'weight' is used as size.
    - 'volatility' is used as color.
    - Automatically groups by sector.
    """

    csv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'holdings-with-sectors.csv')

    # --- Load and clean data ---
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV not found at: {csv_path}")

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(r'[\s\-]+', '_', regex=True)

    # Check expected columns
    required_cols = {'sector', 'weight'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Optional: volatility may be estimated if not present
    if 'volatility' not in df.columns:
        # Estimate volatility using rolling variance of weights (simple proxy)
        df['volatility'] = df['weight'].rolling(window=5, min_periods=1).std().fillna(df['weight'].std())

    # --- Aggregate by sector ---
    df_sector = (
        df.groupby('sector', as_index=False)
          .agg({'weight': 'sum', 'volatility': 'mean'})
          .rename(columns={'weight': 'size'})
    )

    # --- Build treemap ---
    fig = px.treemap(
        df_sector,
        path=['sector'],
        values='size',
        color='volatility',
        color_continuous_scale='RdYlGn_r',
        title='SPLG Sector Weight / Risk-Volatility Map'
    )

    # --- Styling ---
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
