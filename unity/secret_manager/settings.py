"""
SecretManager-specific settings.

These settings are composed into the global ProductionSettings.
Environment variables use the prefix UNITY_SECRET_.
"""

from typing import Final

from pydantic_settings import BaseSettings, SettingsConfigDict

# ────────────────────────────────────────────────────────────────────────────
# Structural context table names
# ────────────────────────────────────────────────────────────────────────────
# Single source of truth for the three tables the SecretManager owns.
# Every TableContext declaration, ContextRegistry lookup, and cross-body
# path construction imports from here instead of sprinkling literals.

SECRETS_TABLE: Final[str] = "Secrets"
"""Shared credential vault.

Hive-scoped via :data:`unity.common.context_registry._HIVE_SCOPED_TABLES`:
Hive members share ``Hives/{hive_id}/Secrets``; solo bodies read the
per-body ``{user}/{assistant}/Secrets`` path.
"""

SECRET_BINDING_TABLE: Final[str] = "SecretBinding"
"""Per-body overlay selecting which credential this body uses per integration."""

SECRET_DEFAULT_TABLE: Final[str] = "SecretDefault"
"""Hive-wide default credential selection per integration.

Also Hive-scoped — solo bodies never read or write this table; they
fall back to ``Secret.name == integration`` instead.
"""

OAUTH_TOKENS_TABLE: Final[str] = "OAuthTokens"
"""Per-body OAuth material synced from Orchestra after each callback.

Held on the per-body path so Hive-shared :data:`SECRETS_TABLE` never
carries another body's Google or Microsoft tokens.
"""


class SecretSettings(BaseSettings):
    """SecretManager settings.

    Attributes:
        ENABLED: Whether SecretManager is enabled.
        IMPL: Implementation type - "real" or "simulated".
        DOTENV_PATH: Path to the .env file for secret storage.
    """

    ENABLED: bool = False
    IMPL: str = "real"
    DOTENV_PATH: str = ""

    model_config = SettingsConfigDict(
        env_prefix="UNITY_SECRET_",
        case_sensitive=True,
        extra="ignore",
    )
