"""
Actor-specific settings.

These settings are composed into the global ProductionSettings.
Environment variables use the prefix UNITY_ACTOR_.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class ActorSettings(BaseSettings):
    """Actor settings.

    Attributes:
        IMPL: Implementation type - "hierarchical", "single_function", "code_act", or "simulated".
        SIMULATED_STEPS: Number of steps before auto-completion for simulated actor.
        ANTICAPTCHA_KEY: API key for AntiCaptcha service.
    """

    IMPL: str = "hierarchical"
    SIMULATED_STEPS: int | None = 1
    ANTICAPTCHA_KEY: str = ""

    model_config = SettingsConfigDict(
        env_prefix="UNITY_ACTOR_",
        case_sensitive=True,
        extra="ignore",
    )
