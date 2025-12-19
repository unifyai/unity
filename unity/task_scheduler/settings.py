"""
TaskScheduler-specific settings.

These settings are composed into the global ProductionSettings.
Environment variables use the prefix UNITY_TASK_.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class TaskSettings(BaseSettings):
    """TaskScheduler settings.

    Attributes:
        IMPL: Implementation type - "real" or "simulated".
        ROUTER_TIMEOUT_SECONDS: Timeout for routing operations in seconds.
        LOCAL_VIEW_OFF: Disable local task view caching.
        SIM_ACTOR_DURATION: Duration for simulated actor in seconds.
    """

    IMPL: str = "real"
    ROUTER_TIMEOUT_SECONDS: float = 60.0
    LOCAL_VIEW_OFF: bool = False
    SIM_ACTOR_DURATION: float = 20.0

    model_config = SettingsConfigDict(
        env_prefix="UNITY_TASK_",
        case_sensitive=True,
        extra="ignore",
    )
