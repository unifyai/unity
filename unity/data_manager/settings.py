"""
unity/data_manager/settings.py
==============================

Settings for the DataManager module.

DataManager is a utility module that provides canonical data operations
(filter, search, reduce, join, vectorize, plot) on any Unify context.
It does not have simulated implementations - it's a foundational layer
used by other managers and available for composition by the Actor.
"""

from pydantic import Field
from pydantic_settings import BaseSettings


class DataSettings(BaseSettings):
    """DataManager configuration.

    DataManager only has a 'real' implementation - no simulated mode.
    """

    IMPL: str = Field(
        default="real",
        description="DataManager implementation: 'real' only.",
    )
