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
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataSettings(BaseSettings):
    """DataManager configuration.

    Supports both 'real' and 'simulated' implementations.
    """

    IMPL: str = Field(
        default="real",
        description="DataManager implementation: 'real' or 'simulated'.",
    )

    model_config = SettingsConfigDict(
        env_prefix="UNITY_DATA_",
        case_sensitive=True,
        extra="ignore",
    )
