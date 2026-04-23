from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

UNASSIGNED_SECRET_ID = -1


class Secret(BaseModel):
    """Fixed schema for a credential row stored in the shared secret vault.

    One :class:`Secret` represents a single named credential — a vault
    entry keyed on its human-friendly ``name`` (for example
    ``"Salesforce Admin"``) and identified across the system by the
    integer ``secret_id``. Multiple rows with distinct names may exist
    for the same external service, and which one an individual body
    uses for a given integration is expressed by a
    :class:`SecretBinding` overlay row, not by the vault itself.
    """

    secret_id: int = Field(
        default=UNASSIGNED_SECRET_ID,
        ge=UNASSIGNED_SECRET_ID,
        description=(
            "Stable integer identifier for the credential row "
            "(auto-counted). Safe to surface to LLMs; values are looked "
            "up through this id via ``get_secret_for_integration``."
        ),
    )
    name: str = Field(
        description=(
            "Human-friendly identifier for the credential (for example "
            '``"Salesforce Admin"``). Used as the placeholder name in '
            "``${name}`` substitution and as the lookup key on solo "
            "bodies that have no :class:`SecretBinding`."
        ),
    )
    value: str = Field(
        description=(
            "The raw credential value. Never exposed to LLMs: every "
            "read surface redacts this field."
        ),
    )
    description: str = Field(
        default="",
        description="Human-readable description of the credential's purpose.",
    )
    description_emb: List[float] = Field(
        default_factory=list,
        description="Vector embedding of the description for semantic search.",
    )
    authoring_assistant_id: Optional[int] = Field(
        default=None,
        description=(
            "Assistant id that authored this credential. Stamped at "
            "create time by the authoring body and never rewritten by "
            "subsequent updates — credential provenance stays pinned "
            "to its original author for the lifetime of the row."
        ),
    )


class SecretBinding(BaseModel):
    """Per-body overlay selecting which shared credential a body uses.

    Each body (Hive member or solo) maintains its own
    ``{user}/{assistant}/SecretBinding`` overlay. One row per
    *integration* expresses which :class:`Secret` row in the shared
    vault this body should use when the integration runs. The overlay
    is the body-local steering layer over the shared
    ``Hives/{hive_id}/Secrets`` table (or the per-body ``Secrets``
    table when the body is solo).

    Resolution order for
    :meth:`SecretManager.get_secret_for_integration`:

    1. The body's :class:`SecretBinding` row for *integration* (if any).
    2. The Hive-wide :class:`SecretDefault` row for *integration* (if
       any) — only meaningful for Hive members.
    3. Solo bodies fall back to a :class:`Secret` whose ``name``
       equals *integration* — on a solo body the integration name
       doubles as the credential name.
    4. Raise ``LookupError`` when nothing matches.
    """

    integration: str = Field(
        description=(
            'Canonical integration name (for example ``"salesforce"`` '
            'or ``"github"``) the binding applies to. Matched verbatim '
            "against the integration argument of "
            "``get_secret_for_integration``."
        ),
    )
    secret_id: int = Field(
        description=(
            "FK to the shared :class:`Secret` row this body should use "
            "for *integration*. Stored as an id rather than a name so "
            "renaming a vault row does not silently re-target every "
            "binding."
        ),
        ge=0,
    )
    authoring_assistant_id: Optional[int] = Field(
        default=None,
        description=(
            "Assistant id that authored this binding. Stamped at "
            "create time and preserved across subsequent updates; makes "
            "it possible to attribute a binding to the body that chose "
            "the credential without scraping event history."
        ),
    )

    model_config = {"extra": "allow"}


class SecretDefault(BaseModel):
    """Hive-wide default credential selection for an integration.

    Lives under ``Hives/{hive_id}/SecretDefault`` so every member of
    the Hive reads the same defaults. A member that does not yet have
    its own :class:`SecretBinding` for an integration falls back to
    the default expressed here. Solo bodies never consult
    :class:`SecretDefault` — they fall back to matching
    ``Secret.name == integration`` instead.
    """

    integration: str = Field(
        description=(
            "Canonical integration name the Hive-wide default applies "
            "to. One row per integration; the (integration) column is "
            "the natural key."
        ),
    )
    secret_id: int = Field(
        description="FK to the shared :class:`Secret` row used as the Hive default.",
        ge=0,
    )
    authoring_assistant_id: Optional[int] = Field(
        default=None,
        description=(
            "Assistant id that authored this default. Write-once audit "
            "field — updates keep the original stamp so the first body "
            "to install the default remains attributed."
        ),
    )

    model_config = {"extra": "allow"}


class OAuthToken(BaseModel):
    """Per-body OAuth material synced from Orchestra after each callback.

    OAuth access/refresh tokens are identity material bound to the
    *body* that completed the OAuth flow — they must not leak across
    bodies in a Hive. This model lives on a per-body context
    (``{user}/{assistant}/OAuthTokens``) independent of the shared
    :class:`Secret` vault, so a Hive-shared ``Secrets`` table never
    carries another body's Google or Microsoft tokens.
    """

    name: str = Field(
        description=(
            'Token name (for example ``"GOOGLE_ACCESS_TOKEN"``). '
            "Drawn from "
            ":attr:`SecretManager.OAUTH_SECRET_ALLOWLIST` and used as "
            "the environment-variable key when the token is exported "
            "to ``os.environ``."
        ),
    )
    value: str = Field(
        description="The raw OAuth token value. Never exposed to LLMs.",
    )
    description: str = Field(
        default="",
        description="Human-readable description of the token (provenance note).",
    )

    model_config = {"extra": "allow"}
