"""
FUREcast Prediction Model Package

This package contains the trained GBR model and data management utilities.
"""

from .data_updater import get_latest_date_in_dataset, check_for_updates

__all__ = [
    'get_latest_date_in_dataset',
    'check_for_updates',
]
