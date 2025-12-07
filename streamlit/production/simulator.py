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
from typing import Dict, List, Any, Optional, Union
import os
from openai import OpenAI


def get_holdings_top_n(n: int = 20) -> pd.DataFrame:
    """Load SPLG holdings and return the top-n by weight.

    This helper is used by the Ask FUREcast plan executor when the
    LLM requests a custom visualization such as "top holdings table".

    It prefers the treemap nodes file (which already has derived metrics
    like DailyChangePct, PE, DividendYield, Beta) and falls back to the
    raw holdings-with-sectors file if needed.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))

    # First preference: treemap_nodes.csv (used by treemap visualizations)
    treemap_path = os.path.join(base_dir, "treemap_nodes.csv")
    holdings_path = os.path.join(base_dir, "holdings-with-sectors.csv")

    df = None
    if os.path.exists(treemap_path):
        try:
            df = pd.read_csv(treemap_path)
            # Normalize weight column name for consistency
            if "Weight (%)" in df.columns:
                df = df.rename(columns={"Weight (%)": "weight"})
        except Exception:
            df = None

    # Fallback: raw holdings CSV
    if df is None and os.path.exists(holdings_path):
        try:
            df = pd.read_csv(holdings_path)
        except Exception:
            df = None

    if df is None or df.empty:
        raise FileNotFoundError(
            "Unable to load SPLG holdings data from treemap_nodes.csv or holdings-with-sectors.csv"
        )

    # Standardize column names that we care about for the top-holdings table.
    cols = {c.lower(): c for c in df.columns}

    # Determine holding name column
    holding_col = None
    for candidate in ["company", "name"]:
        if candidate in cols:
            holding_col = cols[candidate]
            break

    # Determine weight column
    weight_col = None
    for candidate in ["weight", "weight (%)", "weight %"]:
        if candidate in cols:
            weight_col = cols[candidate]
            break

    if weight_col is None:
        raise ValueError("Holdings data is missing a recognizable weight column.")

    # Sort by weight descending and keep top-n
    df_sorted = df.sort_values(by=weight_col, ascending=False).head(n).copy()

    # Build a lean DataFrame with standardized column names used by the UI
    result = pd.DataFrame()

    if holding_col is not None:
        result["Holding"] = df_sorted[holding_col].astype(str)
    else:
        # As a last resort, try ticker
        ticker_col = cols.get("ticker")
        if ticker_col is not None:
            result["Holding"] = df_sorted[ticker_col].astype(str)
        else:
            result["Holding"] = df_sorted.index.astype(str)

    # Percentage from weight column
    result["Percentage"] = df_sorted[weight_col].astype(float)

    # Market Value: we do not have explicit market value in treemap_nodes,
    # but the holdings-with-sectors file does not include it either. For now
    # we expose a placeholder using weight as a proxy so the UI has a column
    # to show; this keeps the table structure consistent with the System Plan.
    if "Market Value" in df_sorted.columns:
        result["Market Value"] = df_sorted["Market Value"]
    else:
        result["Market Value"] = (result["Percentage"] / 100.0)  # proxy

    return result


def _rich_features_path() -> str:
    """Return absolute path to the engineered SPLG feature dataset."""
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'rich_features_SPLG_history_full.csv')
    )


def load_rich_features_dataset() -> pd.DataFrame:
    """Load the engineered SPLG dataset with parsed dates.

    The dataset is relatively small (<6000 rows), so we read it fresh per request
    instead of caching to keep the implementation straightforward.
    """
    data_path = _rich_features_path()
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    return df


def _treemap_nodes_path() -> str:
    return os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'treemap_nodes.csv')


def load_treemap_nodes() -> pd.DataFrame:
    """Load treemap_nodes.csv with basic validation."""
    csv_path = _treemap_nodes_path()
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Treemap data not found at {csv_path}")
    df = pd.read_csv(csv_path)
    required_cols = {'Sector', 'Weight (%)'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"treemap_nodes.csv missing columns: {', '.join(sorted(missing))}")
    return df


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    mask = (~values.isna()) & (~weights.isna())
    if not mask.any():
        return float('nan')
    return float(np.average(values[mask], weights=weights[mask]))


def query_rich_features(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    columns: Optional[List[str]] = None,
    limit: int = 500,
    include_summary: bool = True,
) -> Dict[str, Any]:
    """Filter rows from rich_features_SPLG_history_full.csv for LLM tool use.

    Args:
        start_date: YYYY-MM-DD string; defaults to 180 days before the latest record.
        end_date: YYYY-MM-DD string; defaults to the latest available date.
        columns: Optional list of columns to return. Missing columns are ignored with warnings.
        limit: Maximum number of rows to return (most recent rows after filtering).
        include_summary: Whether to include lightweight aggregate info in the response.

    Returns:
        Dict with `records`, `metadata`, `warnings`, and optional `summary` keys.
    """

    dataset = load_rich_features_dataset()
    warnings: List[str] = []

    if dataset.empty:
        return {
            'records': [],
            'metadata': {'row_count': 0, 'columns': [], 'source': _rich_features_path()},
            'warnings': ['rich_features dataset is empty.']
        }

    min_date = dataset['date'].min()
    max_date = dataset['date'].max()

    def _parse_date(value: Optional[str], fallback: pd.Timestamp) -> pd.Timestamp:
        if not value:
            return fallback
        parsed = pd.to_datetime(value, errors='coerce')
        if pd.isna(parsed):
            warnings.append(f"Could not parse date '{value}'. Using fallback {fallback.date()} instead.")
            return fallback
        return parsed

    default_start = max_date - pd.Timedelta(days=180)
    start_ts = _parse_date(start_date, default_start)
    end_ts = _parse_date(end_date, max_date)

    if start_ts < min_date:
        warnings.append(f"start_date clipped to dataset minimum {min_date.date()}.")
        start_ts = min_date
    if end_ts > max_date:
        warnings.append(f"end_date clipped to dataset maximum {max_date.date()}.")
        end_ts = max_date
    if start_ts > end_ts:
        warnings.append("start_date exceeds end_date; swapping values.")
        start_ts, end_ts = end_ts, start_ts

    filtered = dataset[(dataset['date'] >= start_ts) & (dataset['date'] <= end_ts)].copy()

    if columns:
        normalized_cols = [col.strip() for col in columns if isinstance(col, str)]
        available = [c for c in normalized_cols if c in filtered.columns]
        missing = sorted(set(normalized_cols) - set(available))
        if missing:
            warnings.append(f"Columns not found and skipped: {', '.join(missing)}")
        # Always include date for context
        if 'date' not in available:
            available.insert(0, 'date')
        filtered = filtered[available]

    filtered = filtered.sort_values('date')
    if limit and limit > 0:
        filtered = filtered.tail(limit)

    records = filtered.to_dict(orient='records')

    metadata = {
        'row_count': len(filtered),
        'columns': list(filtered.columns),
        'start_date': start_ts.strftime('%Y-%m-%d'),
        'end_date': end_ts.strftime('%Y-%m-%d'),
        'source': _rich_features_path()
    }

    summary: Optional[Dict[str, Any]] = None
    if include_summary and not filtered.empty:
        latest_row = filtered.iloc[-1]
        summary = {
            'latest_date': latest_row['date'].strftime('%Y-%m-%d') if 'date' in filtered.columns else None,
            'latest_close': float(latest_row['close']) if 'close' in filtered.columns else None,
            'avg_return_5d': float(filtered['return_5d'].mean()) if 'return_5d' in filtered.columns else None,
            'avg_volume': float(filtered['volume'].mean()) if 'volume' in filtered.columns else None,
        }

    response: Dict[str, Any] = {
        'records': records,
        'metadata': metadata,
        'warnings': warnings
    }

    if summary is not None:
        response['summary'] = summary

    return response


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


def create_sector_risk_treemap(metric: str = 'Beta') -> go.Figure:
    """Create treemap showing sector risk levels derived from treemap_nodes.csv."""
    try:
        holdings_df = load_treemap_nodes()
    except Exception:
        return _create_static_sector_risk_treemap()

    if metric not in holdings_df.columns:
        raise ValueError(f"Metric '{metric}' not available in treemap data.")

    summary = (
        holdings_df.groupby('Sector')
        .apply(lambda g: pd.Series({
            'Weight (%)': g['Weight (%)'].sum(),
            metric: _weighted_average(g[metric], g['Weight (%)'])
        }))
        .reset_index()
        .rename(columns={'Sector': 'sector'})
        .sort_values('Weight (%)', ascending=False)
    )

    fig = px.treemap(
        summary,
        path=['sector'],
        values='Weight (%)',
        color=metric,
        color_continuous_scale='RdYlGn_r',
        title=f'SPLG Sector Risk Map ({metric})',
    )

    fig.update_traces(
        textposition='middle center',
        textfont=dict(size=14),
        texttemplate='%{label}',
        hovertemplate=(
            'sector=%{label}<br>'
            'weight=%{value:.2f}%<br>'
            f'{metric}=%{{color:.3f}}<extra></extra>'
        )
    )

    fig.update_layout(
        height=500,
        coloraxis_colorbar=dict(title=metric)
    )

    return fig


def _create_static_sector_risk_treemap() -> go.Figure:
    """Fallback treemap when treemap_nodes.csv is unavailable."""
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
    df['volatility_pct'] = df['volatility'] * 100

    fig = px.treemap(
        df,
        path=['sector'],
        values='size',
        color='volatility',
        color_continuous_scale='RdYlGn_r',
        title='SPLG Sector Risk Map (Simulated)',
        custom_data=['volatility_pct']
    )

    fig.update_traces(
        textposition='middle center',
        textfont=dict(size=14),
        texttemplate='%{label}',
        hovertemplate=(
            'sector=%{label}<br>'
            'size=%{value:.1f}%<br>'
            'volatility=%{customdata[0]:.1f}%<br>'
            '<extra></extra>'
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


def get_sector_risk_table(top_n: Optional[int] = None) -> pd.DataFrame:
    """Return a holdings-level table supporting the sector risk treemap.

    The table surfaces key per-company metrics used by the SPLG Sector Risk Map
    so users can view the underlying data. Rows are sorted by SPLG weight.
    """
    try:
        df = load_treemap_nodes()
    except FileNotFoundError:
        return pd.DataFrame()

    # Columns of interest; only keep ones that are present in the CSV
    preferred_cols = [
        'Company',
        'Sector',
        'Weight (%)',
        'PE',
        'Beta',
        'DividendYield',
        'DailyChangePct',
    ]
    cols = [c for c in preferred_cols if c in df.columns]
    if not cols:
        return pd.DataFrame()

    table = df[cols].copy()
    if 'Weight (%)' in table.columns:
        table = table.sort_values('Weight (%)', ascending=False)
        table['Weight (%)'] = table['Weight (%)'].round(2)
    if 'PE' in table.columns:
        table['PE'] = table['PE'].round(2)
    if 'Beta' in table.columns:
        table['Beta'] = table['Beta'].round(3)
    if 'DividendYield' in table.columns:
        table['DividendYield'] = table['DividendYield'].round(2)
    if 'DailyChangePct' in table.columns:
        table['DailyChangePct'] = table['DailyChangePct'].round(3)

    if top_n:
        table = table.head(int(top_n))

    return table.reset_index(drop=True)


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


def create_sector_bar_chart(
    metric: str = 'DailyChangePct',
    sectors: Optional[List[str]] = None,
    top_n: Optional[int] = None,
    title: str = 'Sector Metric Comparison'
) -> go.Figure:
    """Create a weighted bar chart of sector-level metrics using treemap data."""
    df = load_treemap_nodes()
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not available in treemap data.")

    grouped = (
        df.groupby('Sector')
        .apply(lambda g: pd.Series({
            'Weight (%)': g['Weight (%)'].sum(),
            metric: _weighted_average(g[metric], g['Weight (%)'])
        }))
        .reset_index()
        .rename(columns={'Sector': 'sector'})
        .sort_values('Weight (%)', ascending=False)
    )

    if sectors:
        sectors_upper = {s.lower() for s in sectors}
        grouped = grouped[grouped['sector'].str.lower().isin(sectors_upper)]

    if top_n:
        grouped = grouped.head(top_n)

    if grouped.empty:
        raise ValueError('No sector data available for the requested filters.')

    fig = go.Figure(go.Bar(
        x=grouped['sector'],
        y=grouped[metric],
        marker_color='steelblue'
    ))
    fig.update_layout(
        title=title,
        xaxis_title='Sector',
        yaxis_title=metric,
        template='plotly_white',
        height=420
    )
    return fig


def create_sector_comparison_chart(sectors: List[str], metric: str = 'DailyChangePct') -> go.Figure:
    """Compatibility wrapper: compare sectors using real treemap metrics."""
    return create_sector_bar_chart(metric=metric, sectors=sectors, title='Sector Comparison')


def _records_to_dataframe(result: Optional[Any]) -> Optional[pd.DataFrame]:
    if isinstance(result, dict):
        if 'records' in result and isinstance(result['records'], list):
            return pd.DataFrame(result['records'])
        if 'data' in result and isinstance(result['data'], list):
            return pd.DataFrame(result['data'])
    elif isinstance(result, list):
        return pd.DataFrame(result)
    return None


def create_generic_chart(
    data: pd.DataFrame,
    chart_type: str,
    x: str,
    y: Union[str, List[str]],
    group: Optional[str] = None,
    title: Optional[str] = None,
    yaxis_title: Optional[str] = None,
) -> go.Figure:
    if data is None or data.empty:
        raise ValueError('No data available for visualization.')
    if x not in data.columns:
        raise ValueError(f"Column '{x}' not found in data.")

    series = y if isinstance(y, list) else [y]
    for col in series:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data.")

    fig = go.Figure()
    if chart_type == 'line':
        if group and isinstance(y, str) and group in data.columns:
            for group_name, subset in data.groupby(group):
                fig.add_trace(go.Scatter(
                    x=subset[x],
                    y=subset[y],
                    mode='lines',
                    name=str(group_name)
                ))
        else:
            for col in series:
                fig.add_trace(go.Scatter(x=data[x], y=data[col], mode='lines', name=str(col)))
    elif chart_type == 'bar':
        if group and isinstance(y, str) and group in data.columns:
            grouped = data.groupby([x, group])[y].mean().reset_index()
            for group_name in grouped[group].unique():
                subset = grouped[grouped[group] == group_name]
                fig.add_trace(go.Bar(x=subset[x], y=subset[y], name=str(group_name)))
        else:
            for col in series:
                fig.add_trace(go.Bar(x=data[x], y=data[col], name=str(col)))
    else:
        raise ValueError(f"Unsupported generic chart type '{chart_type}'.")

    fig.update_layout(
        title=title or '',
        xaxis_title=x,
        yaxis_title=yaxis_title or ', '.join(series),
        template='plotly_white',
        height=400
    )
    return fig


def viz_from_spec(spec: Dict[str, Any], tool_results: Optional[Dict[str, Any]] = None) -> Optional[go.Figure]:
    """Render a visualization described by the normalized plan spec."""
    chart_type = spec.get('type', 'price')
    specs = spec.get('specs', {}) or {}
    data_key = spec.get('data_key')
    data_df = _records_to_dataframe(tool_results.get(data_key)) if tool_results and data_key else None

    if chart_type == 'treemap':
        if specs.get('level', 'sector') == 'holdings':
            color_metric = specs.get('color_metric', 'DailyChangePct')
            return create_sector_holdings_treemap(color_metric=color_metric)
        metric = specs.get('metric', 'Beta')
        return create_sector_risk_treemap(metric=metric)

    if chart_type == 'feature_importance':
        features = specs.get('features')
        if not features and data_df is not None:
            features = data_df.to_dict(orient='records')
        if not features and tool_results and data_key:
            raw = tool_results.get(data_key)
            if isinstance(raw, dict) and isinstance(raw.get('top_features'), list):
                features = raw['top_features']
        if not features:
            raise ValueError('Feature importance chart requires features data.')
        return create_feature_importance_chart(features)

    if chart_type == 'bar':
        if specs.get('source') == 'sectors' or specs.get('sectors'):
            return create_sector_bar_chart(
                metric=specs.get('metric', 'DailyChangePct'),
                sectors=specs.get('sectors'),
                top_n=specs.get('top_n'),
                title=specs.get('title', 'Sector Comparison'),
            )
        if data_df is not None:
            return create_generic_chart(
                data_df,
                'bar',
                x=specs.get('x', 'label'),
                y=specs.get('y', ['value']),
                group=specs.get('group'),
                title=specs.get('title'),
                yaxis_title=specs.get('y_label'),
            )
        raise ValueError('Bar chart requires sectors or tool data.')

    if chart_type == 'line':
        if data_df is None:
            raise ValueError('Line chart requires tool data via data_key.')
        return create_generic_chart(
            data_df,
            'line',
            x=specs.get('x', 'date'),
            y=specs.get('y', ['value']),
            group=specs.get('group'),
            title=specs.get('title'),
            yaxis_title=specs.get('y_label'),
        )

    if chart_type == 'price':
        metric = specs.get('metric', 'close')
        days = int(specs.get('days', 180) or 180)
        start = specs.get('start_date')
        end = specs.get('end_date')
        end_date = pd.to_datetime(end) if end else pd.to_datetime('today').normalize()
        start_date = pd.to_datetime(start) if start else end_date - pd.Timedelta(days=days)
        return create_price_chart(metric, start_date, end_date, show_events=specs.get('show_events', True))

    if chart_type == 'table':
        # Tables are rendered directly in the UI layer
        return None

    # Default: return price chart fallback
    default_end = pd.to_datetime('today').normalize()
    default_start = default_end - pd.Timedelta(days=180)
    return create_price_chart('close', default_start, default_end, show_events=True)
