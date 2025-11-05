"""
FUREcast - LLM Interface Layer

This module handles the LLM orchestration: routing queries, planning tool execution,
and composing natural language responses.
"""

import json
from typing import Dict, List, Any, Optional
import os
from openai import OpenAI


def get_openai_client():
    """Initialize OpenAI client with API key from environment"""
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '../../', '.env'))
    api_key = os.getenv('OPENAI_API_KEY')
    return OpenAI(api_key=api_key)


def route_query(query: str) -> Dict[str, Any]:
    """
    Classify user query intent and determine which tools to invoke.
    
    Returns a structured plan with intent classification and tool sequence.
    """
    client = get_openai_client()
    
    system_prompt = """You are a financial analytics query router for FUREcast, an educational SPLG ETF analysis tool.

Given a user query, classify the intent and create a tool execution plan.

Available intents:
- market_outlook: User wants prediction or investment timing advice
- sector_analysis: User wants sector risk, performance, or comparison
- feature_explanation: User wants to understand what drives predictions
- custom_visualization: User requests specific charts or data views
- general_question: General financial education questions

Available tools:
- predict_splg: Get GBR model prediction for SPLG
- fetch_prices: Get historical price data
- compute_risk: Calculate risk metrics
- viz_from_spec: Create visualizations

Return JSON with:
{
  "intent": "intent_category",
  "tools": [
    {"name": "tool_name", "args": {...}}
  ],
  "visualization": {
    "type": "chart_type",
    "specs": {...}
  }
}"""

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
        
        plan = json.loads(response.choices[0].message.content)
        return plan
    except Exception as e:
        # Fallback plan
        return {
            "intent": "market_outlook",
            "tools": [{"name": "predict_splg", "args": {}}],
            "visualization": {"type": "price", "specs": {"ticker": "SPLG", "days": 180}}
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


def execute_tools_and_compose(query: str, tool_results: Dict[str, Any]) -> str:
    """
    High-level function that composes answer from tool results.
    """
    plan = route_query(query)
    return compose_answer(query, tool_results, plan)


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
