"""Runtime helpers for refresh-token backed OAuth credentials.

SecretManager owns storage and synchronization: it mirrors allowlisted assistant
secrets from Orchestra into the local ``Secrets`` context, ``.env``, and
``os.environ``.  This module owns the runtime interpretation of those mirrored
values: provider aliases, access-token/expiry secret names, freshness checks,
and the sandbox helper exposed to actor-written Python.

The split is deliberate.  ``get_oauth_access_token(...)`` is not a
``primitives.secrets`` tool and does not expose arbitrary secrets; it behaves
like ``reason(...)``/``notify(...)`` as a Python runtime helper for code paths
that must pass an explicit OAuth access token to an SDK/client/request.  Code
that can rely on provider SDK/default environment credential behavior should
continue to do so; env overlays below keep rotating OAuth env vars fresh for
venv and shell backends.
"""

import inspect
import os
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class OAuthProviderMetadata:
    """Runtime metadata for a refresh-token backed OAuth provider."""

    canonical_name: str
    aliases: tuple[str, ...]
    access_token_secret: str
    refresh_token_secret: str | None = None
    expiry_secret: str | None = None
    granted_scopes_secret: str | None = None
    docs_label: str = ""

    @property
    def secret_names(self) -> frozenset[str]:
        return frozenset(
            name
            for name in (
                self.access_token_secret,
                self.refresh_token_secret,
                self.expiry_secret,
                self.granted_scopes_secret,
            )
            if name
        )


_OAUTH_PROVIDER_METADATA: dict[str, OAuthProviderMetadata] = {
    "google": OAuthProviderMetadata(
        canonical_name="google",
        aliases=("google", "gmail", "google_workspace", "drive"),
        access_token_secret="GOOGLE_ACCESS_TOKEN",
        refresh_token_secret="GOOGLE_REFRESH_TOKEN",
        expiry_secret="GOOGLE_TOKEN_EXPIRES_AT",
        granted_scopes_secret="GOOGLE_GRANTED_SCOPES",
        docs_label="Google APIs",
    ),
    "microsoft": OAuthProviderMetadata(
        canonical_name="microsoft",
        aliases=("microsoft", "msft", "ms365", "microsoft_365", "graph"),
        access_token_secret="MICROSOFT_ACCESS_TOKEN",
        refresh_token_secret="MICROSOFT_REFRESH_TOKEN",
        expiry_secret="MICROSOFT_TOKEN_EXPIRES_AT",
        granted_scopes_secret="MICROSOFT_GRANTED_SCOPES",
        docs_label="Microsoft Graph",
    ),
}
_OAUTH_PROVIDER_ALIASES: dict[str, str] = {
    alias.strip().lower().replace("-", "_"): metadata.canonical_name
    for metadata in _OAUTH_PROVIDER_METADATA.values()
    for alias in metadata.aliases
}


def _resolve_oauth_provider(provider: str) -> OAuthProviderMetadata:
    if not isinstance(provider, str) or not provider.strip():
        supported = ", ".join(sorted(_OAUTH_PROVIDER_METADATA))
        raise ValueError(
            "A refresh-token OAuth provider name is required. "
            f"Supported providers: {supported}",
        )
    normalized = provider.strip().lower().replace("-", "_")
    canonical = _OAUTH_PROVIDER_ALIASES.get(normalized, normalized)
    metadata = _OAUTH_PROVIDER_METADATA.get(canonical)
    if metadata is None:
        supported = ", ".join(sorted(_OAUTH_PROVIDER_METADATA))
        raise ValueError(
            f"Unknown refresh-token OAuth provider {provider!r}. "
            f"Supported providers: {supported}",
        )
    return metadata


def refresh_token_oauth_secret_names() -> frozenset[str]:
    names: set[str] = set()
    for metadata in _OAUTH_PROVIDER_METADATA.values():
        names.update(metadata.secret_names)
    return frozenset(names)


def _get_secret_manager():
    from unity.manager_registry import ManagerRegistry

    return ManagerRegistry.get_secret_manager()


def _get_secret_value(secret_manager, name: str) -> str | None:
    getter = getattr(secret_manager, "_get_secret_value", None)
    if callable(getter):
        value = getter(name)
        if isinstance(value, str) and value:
            return value
    value = os.environ.get(name)
    return value if value else None


def _parse_expiry(value: str) -> datetime:
    if value.isdigit():
        return datetime.fromtimestamp(int(value), tz=timezone.utc)
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _token_expires_within(
    secret_manager,
    metadata: OAuthProviderMetadata,
    min_ttl_seconds: int,
) -> bool:
    if metadata.expiry_secret is None:
        return False
    expiry_value = _get_secret_value(secret_manager, metadata.expiry_secret)
    if not expiry_value:
        return True
    try:
        expiry = _parse_expiry(expiry_value)
    except ValueError:
        return True
    remaining = (expiry - datetime.now(timezone.utc)).total_seconds()
    return remaining <= min_ttl_seconds


def get_oauth_access_token(provider: str, *, min_ttl_seconds: int = 300) -> str:
    """
    Return a current OAuth access token for a refresh-token backed provider.

    Use this runtime helper inside generated Python code when a provider SDK,
    client, or direct HTTP request requires an explicit access token. Prefer
    provider SDK/default credential behavior when it can read credentials from
    the runtime environment directly; Unity keeps rotating OAuth env vars
    synced separately for that path.

    Parameters
    ----------
    provider:
        Provider name or alias. Built-in aliases include ``"microsoft"``,
        ``"graph"``, ``"google"``, ``"gmail"``, and ``"drive"``.
    min_ttl_seconds:
        Minimum acceptable token lifetime. If the current token is missing or
        expires within this many seconds, the parent runtime forces an
        assistant-secret sync from Orchestra before returning a token.

    Examples
    --------
    Multiple providers can be used in one sandbox; request each explicitly::

        microsoft_token = get_oauth_access_token("microsoft")
        google_token = get_oauth_access_token("google")

    For direct OAuth2 HTTP APIs such as Microsoft Graph, provider docs commonly
    show the access token in an ``Authorization: Bearer ...`` header. Other SDKs
    may require a credential object or may read environment variables directly,
    so follow the provider's SDK/API docs for how to apply the token.

    Anti-patterns
    -------------
    - Do not print, log, return, or store the token value.
    - Do not save concrete token values in FunctionManager functions or
      GuidanceManager guidance.
    - Do not read rotating OAuth access-token env vars directly when this
      helper is available and an explicit access token is required.
    """
    metadata = _resolve_oauth_provider(provider)
    secret_manager = _get_secret_manager()
    token = _get_secret_value(secret_manager, metadata.access_token_secret)
    needs_force_sync = token is None or _token_expires_within(
        secret_manager,
        metadata,
        min_ttl_seconds,
    )
    secret_manager.sync_assistant_secrets_if_stale(
        ttl_seconds=60.0,
        force=needs_force_sync,
        reason=f"oauth_access_token:{metadata.canonical_name}",
    )
    token = _get_secret_value(secret_manager, metadata.access_token_secret)
    if not token:
        raise ValueError(
            f"No access token is available for refresh-token OAuth provider "
            f"{metadata.canonical_name!r}.",
        )
    if _token_expires_within(secret_manager, metadata, min_ttl_seconds):
        raise ValueError(
            f"The access token for refresh-token OAuth provider "
            f"{metadata.canonical_name!r} is expired or near expiry after sync.",
        )
    return token


def get_refresh_token_oauth_env_overlay() -> dict[str, str]:
    """Return fresh rotating OAuth env vars for subprocess execution backends.

    Venv and persistent shell sessions can outlive the parent process's last
    environment update, so they cannot rely solely on the environment copied at
    process start.  This helper performs the debounced assistant-secret sync,
    then returns only the built-in refresh-token OAuth variables that should be
    overlaid into those subprocesses before execution.
    """
    secret_manager = _get_secret_manager()
    secret_manager.sync_assistant_secrets_if_stale(
        ttl_seconds=60.0,
        reason="oauth_env_overlay",
    )
    overlay: dict[str, str] = {}
    for name in refresh_token_oauth_secret_names():
        value = _get_secret_value(secret_manager, name)
        if value:
            overlay[name] = value
    return overlay


def get_oauth_prompt_context() -> str:
    """Return actor-facing documentation for OAuth runtime helpers."""
    doc = inspect.getdoc(get_oauth_access_token) or ""
    signature = (
        f"def {get_oauth_access_token.__name__}"
        f"{inspect.signature(get_oauth_access_token)}"
    )
    return (
        "### OAuth Access Token Helper: `get_oauth_access_token(...)`\n\n"
        "`get_oauth_access_token(...)` is available inside `execute_code` "
        "Python sessions and stored Python functions. It is a normal sandbox "
        "helper, not a JSON tool call.\n\n"
        f"```python\n{signature}\n```\n\n"
        f"{doc}"
    )
