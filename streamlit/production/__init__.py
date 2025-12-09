"""
FUREcast Production Package

Contains the Streamlit application, tools, and LLM interface.
"""

from .tools import (
    predict_splg,
    create_price_chart,
    create_sector_risk_treemap,
    create_sector_holdings_treemap,
    get_sector_summary,
    create_feature_importance_chart,
    create_sector_comparison_chart,
)

from .llm_interface import (
    route_query,
    compose_answer,
    explain_prediction,
)

__all__ = [
    # Simulator/tool functions
    'predict_splg',
    'create_price_chart',
    'create_sector_risk_treemap',
    'create_sector_holdings_treemap',
    'get_sector_summary',
    'create_feature_importance_chart',
    'create_sector_comparison_chart',
    # LLM interface functions
    'route_query',
    'compose_answer',
    'explain_prediction',
]
