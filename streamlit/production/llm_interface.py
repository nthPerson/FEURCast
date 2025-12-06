"""
FUREcast - LLM Interface Layer

This module handles the LLM orchestration: routing queries, planning tool execution,
and composing natural language responses.
"""

import json
from typing import Dict, List, Any, Optional
import os
from datetime import datetime, timedelta

import pandas as pd
from openai import OpenAI

from simulator import (
    predict_splg,
    fetch_prices,
    compute_risk,
    query_rich_features,
    get_sector_summary,
    get_holdings_top_n,
)

PLAN_VERSION = 1
ALLOWED_VIS_TYPES = {"price", "line", "bar", "treemap", "feature_importance", "table"}
ALLOWED_TOOLS = {
    "predict_splg",
    "query_rich_features",
    "get_sector_summary",
    "get_holdings_top_n",
    "compute_risk",
    "fetch_prices",
}

DATA_SOURCES = [
    "Financial Crisis Timeline by Day (since 2005).csv",
    "investment_glossary.csv",
    "holdings-with-sectors.csv",
    "rich_features_SPLG_history_full.csv",
    "treemap_nodes.csv",
]


def get_openai_client():
    """Initialize OpenAI client with API key from environment"""
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(__file__), '../../', '.env'))
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


def route_query(query: str) -> Dict[str, Any]:
    """
    Classify user query intent and determine which tools to invoke.
    
    Returns a structured plan with intent classification and tool sequence.
    """
    client = get_openai_client()

    system_prompt = f"""You are a financial analytics query router for FUREcast, an educational SPLG ETF analysis tool.

Given a user query, classify the intent and create a tool execution plan that only uses the following local CSV data sources:
- {'; '.join(DATA_SOURCES)}

Available intents:
- market_outlook: User wants prediction or investment timing advice
- sector_analysis: Sector risk, performance, or comparisons
- feature_explanation: Explain model drivers
- custom_visualization: Specific chart/table requests
- general_question: Educational questions about SPLG/market terms

Available tools (each MUST include a unique "result_key"):
- predict_splg -> no args required; returns model prediction
- query_rich_features -> args: start_date, end_date, columns, limit
- get_sector_summary -> no args; returns sector rollups from treemap_nodes.csv
- get_holdings_top_n -> args: n (default 20)
- compute_risk -> args: data_key (points to prior result), metric ('volatility' | 'sharpe' | 'drawdown'), window (int)
- fetch_prices -> **SPLG only**; args: start, end (YYYY-MM-DD). Use the engineered dataset when possible; only fall back here when explicitly asked for synthetic comparisons.

Visualization constraints:
- Allowed types: price, line, bar, treemap, feature_importance, table
- Visualizations must reference actual data via "data_key" that maps to a tool's result_key
- specs may include: metric, start_date, end_date, x, y, group, sectors, columns, top_n, title
- If the request cannot be satisfied with available data, plan a textual response and set visualization.type to "table" with an empty data_key

Return strict JSON:
{{
    "plan_version": {PLAN_VERSION},
    "intent": "intent_category",
    "tools": [
        {{
            "name": "tool_name",
            "result_key": "unique_key",
            "args": {{ ... }}
        }}
    ],
    "visualization": {{
        "type": "chart_type",
        "data_key": "result_key or '' if not needed",
        "specs": {{ ... }}
    }}
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        raw_plan = json.loads(response.choices[0].message.content)
        return normalize_plan(raw_plan, query)
    except Exception:
        return normalize_plan(None, query)


def normalize_plan(plan: Optional[Dict[str, Any]], query: str) -> Dict[str, Any]:
    """Ensure the LLM plan conforms to the schema and falls back safely."""
    normalized = _default_plan(query)

    if not isinstance(plan, dict):
        return normalized

    normalized['intent'] = plan.get('intent', normalized['intent'])
    normalized['plan_version'] = PLAN_VERSION
    normalized['user_query'] = query

    raw_tools = plan.get('tools', [])
    parsed_tools = []
    if isinstance(raw_tools, list):
        for idx, tool in enumerate(raw_tools):
            if not isinstance(tool, dict):
                continue
            name = tool.get('name')
            if name not in ALLOWED_TOOLS:
                continue
            args = tool.get('args') if isinstance(tool.get('args'), dict) else {}
            result_key = tool.get('result_key') or f"{name}_{idx}"
            parsed_tools.append({
                'name': name,
                'args': args,
                'result_key': result_key
            })

    if parsed_tools:
        normalized['tools'] = parsed_tools

    raw_viz = plan.get('visualization')
    viz_dict = normalized['visualization']
    if isinstance(raw_viz, dict):
        viz_type = raw_viz.get('type', viz_dict['type'])
        if viz_type not in ALLOWED_VIS_TYPES:
            viz_type = viz_dict['type']
        data_key = raw_viz.get('data_key')
        if not data_key and normalized['tools']:
            data_key = normalized['tools'][-1]['result_key']
        specs = raw_viz.get('specs') if isinstance(raw_viz.get('specs'), dict) else {}
        viz_dict = {
            'type': viz_type,
            'data_key': data_key or '',
            'specs': specs
        }
    normalized['visualization'] = viz_dict

    return normalized


def _default_plan(query: str) -> Dict[str, Any]:
    return {
        'plan_version': PLAN_VERSION,
        'intent': 'market_outlook',
        'user_query': query,
        'tools': [
            {'name': 'predict_splg', 'args': {}, 'result_key': 'prediction'},
            {
                'name': 'query_rich_features',
                'args': {'limit': 180, 'columns': ['date', 'close', 'open', 'high', 'low', 'volume']},
                'result_key': 'splg_history'
            }
        ],
        'visualization': {
            'type': 'price',
            'data_key': 'splg_history',
            'specs': {'metric': 'close', 'title': 'Recent SPLG performance'}
        }
    }


def compose_answer(query: str, tool_results: Dict[str, Any], plan: Dict[str, Any]) -> str:
    """
    Synthesize tool results into a natural language response.
    
    This is where the LLM explains predictions, interprets metrics,
    and provides educational context.
    """
    client = get_openai_client()
    
    system_prompt = """You are FUREcast, an educational financial analytics assistant.

Your role is to:
1. Explain model predictions and market data in clear, accessible language
2. Provide educational context about investment concepts
3. Always emphasize this is for learning, not financial advice
4. Ground explanations in the data and model results provided

Tone: Knowledgeable but approachable, like a helpful teaching assistant.

ALWAYS include this disclaimer in your response:
"⚠️ This analysis is for educational purposes only and does not constitute financial advice."
"""

    # Format tool results for context
    results_summary = json.dumps(tool_results, indent=2)
    
    user_prompt = f"""User Query: {query}

Tool Results:
{results_summary}

Provide a helpful, educational response that:
1. Directly answers the user's question
2. Explains what the data/predictions mean
3. Offers relevant context or insights
4. Includes the educational disclaimer

Keep your response concise (3-5 paragraphs maximum)."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"""I encountered an error processing your query. Here's what I found from the data:

{json.dumps(tool_results, indent=2)}

⚠️ This analysis is for educational purposes only and does not constitute financial advice."""


def generate_tool_plan(query: str) -> Dict[str, Any]:
    """
    High-level function that combines routing and planning.
    Returns executable tool plan.
    """
    return route_query(query)


def execute_tool_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Execute each tool specified in the normalized plan and return keyed outputs."""
    results: Dict[str, Any] = {}
    for tool in plan.get('tools', []):
        name = tool.get('name')
        result_key = tool.get('result_key') or name
        args = tool.get('args', {}) or {}
        if name not in ALLOWED_TOOLS:
            results[result_key] = {'error': f"Tool '{name}' is not supported."}
            continue
        try:
            results[result_key] = _invoke_tool(name, args, results)
        except Exception as exc:
            results[result_key] = {'error': str(exc)}
    return results


def _invoke_tool(name: str, args: Dict[str, Any], prior_results: Dict[str, Any]) -> Any:
    """Map planner tool names to simulator helpers with input validation."""
    if name == 'predict_splg':
        return predict_splg(use_real_model=args.get('use_real_model', True))

    if name == 'query_rich_features':
        return query_rich_features(
            start_date=args.get('start_date'),
            end_date=args.get('end_date'),
            columns=args.get('columns'),
            limit=int(args.get('limit', 500) or 500),
            include_summary=args.get('include_summary', True),
        )

    if name == 'get_sector_summary':
        df = get_sector_summary()
        return _wrap_dataframe(df, 'treemap_nodes.csv')

    if name == 'get_holdings_top_n':
        df = get_holdings_top_n(int(args.get('n', 20) or 20))
        return _wrap_dataframe(df, 'treemap_nodes.csv')

    if name == 'fetch_prices':
        tickers = args.get('tickers') or ['SPLG']
        unique_tickers = {str(t).upper() for t in tickers}
        if unique_tickers != {'SPLG'}:
            raise ValueError("fetch_prices currently supports SPLG only, based on available CSV data.")
        return query_rich_features(
            start_date=args.get('start'),
            end_date=args.get('end'),
            columns=args.get('columns') or ['date', 'open', 'high', 'low', 'close', 'volume', 'ticker'],
            limit=int(args.get('limit', 500) or 500),
            include_summary=True,
        )

    if name == 'compute_risk':
        metric = args.get('metric', 'volatility')
        window = int(args.get('window', 60) or 60)
        source_key = args.get('data_key')
        df = _records_to_dataframe(prior_results.get(source_key)) if source_key else None
        if df is None:
            slice_resp = query_rich_features(
                start_date=args.get('start_date'),
                end_date=args.get('end_date'),
                columns=['date', 'close', 'ticker'],
                limit=int(args.get('limit', 252) or 252),
                include_summary=False,
            )
            df = _records_to_dataframe(slice_resp)
        if df is None or df.empty:
            raise ValueError('compute_risk requires price data (date, close, ticker).')
        if 'ticker' not in df.columns:
            df['ticker'] = 'SPLG'
        return compute_risk(df, metric=metric, window=window)

    raise ValueError(f"Unsupported tool '{name}'.")


def _records_to_dataframe(result: Optional[Any]) -> Optional[pd.DataFrame]:
    if isinstance(result, dict):
        if 'records' in result and isinstance(result['records'], list):
            return pd.DataFrame(result['records'])
        if all(k in result for k in ('date', 'close')):
            return pd.DataFrame([result])
    elif isinstance(result, list):
        return pd.DataFrame(result)
    return None


def _wrap_dataframe(df: pd.DataFrame, source: str) -> Dict[str, Any]:
    return {
        'records': df.to_dict(orient='records'),
        'metadata': {
            'row_count': int(len(df)),
            'columns': list(df.columns),
            'source': source
        }
    }


def execute_tools_and_compose(query: str) -> Dict[str, Any]:
    """Convenience helper to plan, execute, and compose an answer."""
    plan = generate_tool_plan(query)
    tool_results = execute_tool_plan(plan)
    answer = compose_answer(query, tool_results, plan)
    return {
        'plan': plan,
        'tool_results': tool_results,
        'answer': answer
    }


def explain_prediction(prediction: Dict[str, Any]) -> str:
    """
    Generate plain-language explanation of a GBR prediction.
    """
    client = get_openai_client()
    
    direction = prediction.get('direction', 'neutral')
    confidence = prediction.get('confidence', 0.5)
    return_pct = prediction.get('predicted_return', 0) * 100
    features = prediction.get('top_features', [])
    
    prompt = f"""Explain this stock prediction in 2-3 sentences for a student learning about predictive modeling:

Prediction: {direction.upper()} ({return_pct:+.2f}%)
Confidence: {confidence:.1%}
Top Features: {', '.join([f['name'] for f in features[:3]])}

Focus on what these features suggest about market conditions and why the model might be predicting this direction.
Keep it educational and accessible."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"The model predicts a {direction} movement of {return_pct:+.2f}% with {confidence:.1%} confidence based on recent technical indicators."
