"""``SecretManager.get_secret_for_integration`` resolution order.

The resolver backs every integration that reads a credential from the
environment (the LLM tool layer, pre-call adapters, venv-spawned
functions) and picks which ``Secret`` row a body should use for a given
integration name. The order is:

1. The body's own :class:`SecretBinding` row for the integration.
2. The Hive-wide :class:`SecretDefault` for the integration (Hive
   members only).
3. A vault row whose ``name`` equals the integration (solo bodies
   only, matching the legacy single-body convention).
4. :class:`LookupError` otherwise.

These tests drive the real ``SecretManager`` against the real Unify
backend (each test gets its own isolated context via the
``secret_manager_context`` fixture). Hive behaviour is simulated by
patching ``SESSION_DETAILS.hive_id`` after construction — ``_is_hive_member``
reads the field lazily on every call, so the resolver sees a Hive
member while ``_binding_ctx`` / ``_default_ctx`` still point at the
test's per-body contexts. That keeps the test focused on the resolver
branch logic without having to stand up a real ``Hives/{id}`` root.
"""

from __future__ import annotations

import pytest

from unity.secret_manager.secret_manager import SecretManager
from unity.session_details import SESSION_DETAILS

# ─────────────────────────── Solo body paths ────────────────────────────


def test_binding_hit_returns_bound_secret(secret_manager_context):
    """A body's own binding wins over any other resolution tier."""
    sm = SecretManager()
    sm._create_secret(
        name="Salesforce Admin",
        value="token-admin-123",
        description="admin creds",
    )

    sm._bind_secret(integration="salesforce", secret_name="Salesforce Admin")

    resolved = sm.get_secret_for_integration("salesforce")

    assert resolved.name == "Salesforce Admin"
    assert resolved.value == "token-admin-123"


def test_solo_body_name_fallback_matches_integration(secret_manager_context):
    """Solo bodies without a binding fall back to ``name == integration``.

    This preserves the single-body convention where the integration
    name doubles as the credential name — upgrading a body into a Hive
    later just layers bindings on top.
    """
    sm = SecretManager()
    sm._create_secret(
        name="openai",
        value="sk-openai-solo",
        description="solo key",
    )

    resolved = sm.get_secret_for_integration("openai")

    assert resolved.name == "openai"
    assert resolved.value == "sk-openai-solo"


def test_binding_preferred_over_name_fallback(secret_manager_context):
    """Binding takes priority over a ``name == integration`` match.

    Both rows exist; the resolver must pick the binding's target. This
    prevents a solo body's legacy ``name == integration`` entry from
    shadowing an explicit binding the owner installed later.
    """
    sm = SecretManager()
    sm._create_secret(
        name="openai",
        value="sk-legacy",
        description="legacy match by name",
    )
    sm._create_secret(
        name="OpenAI Production",
        value="sk-production",
        description="production key",
    )

    sm._bind_secret(integration="openai", secret_name="OpenAI Production")

    resolved = sm.get_secret_for_integration("openai")

    assert resolved.name == "OpenAI Production"
    assert resolved.value == "sk-production"


def test_solo_body_without_match_raises_lookup_error(secret_manager_context):
    """Solo body, no binding, no matching vault row → ``LookupError``.

    Callers should treat this as a configuration issue rather than a
    transient failure; the resolver doesn't silently return ``None``.
    """
    sm = SecretManager()

    with pytest.raises(LookupError, match="salesforce"):
        sm.get_secret_for_integration("salesforce")


def test_empty_integration_name_raises_value_error(secret_manager_context):
    """An empty integration is a programmer error, not a lookup miss."""
    sm = SecretManager()

    with pytest.raises(ValueError):
        sm.get_secret_for_integration("")


# ─────────────────────────── Hive body paths ────────────────────────────


def test_hive_member_falls_back_to_hive_default(
    secret_manager_context,
    monkeypatch,
):
    """Hive body, no per-body binding → the Hive default supplies the row."""
    sm = SecretManager()
    sm._create_secret(
        name="Stripe Shared",
        value="sk-stripe-team",
        description="shared key",
    )

    monkeypatch.setattr(SESSION_DETAILS, "hive_id", 42, raising=False)

    sm._set_hive_default(integration="stripe", secret_name="Stripe Shared")

    resolved = sm.get_secret_for_integration("stripe")

    assert resolved.name == "Stripe Shared"
    assert resolved.value == "sk-stripe-team"


def test_hive_member_binding_preferred_over_hive_default(
    secret_manager_context,
    monkeypatch,
):
    """A per-body binding overrides the Hive-wide default for that integration.

    One teammate needs a different credential than the team default;
    installing a binding on their body must pre-empt the default the
    rest of the Hive shares.
    """
    sm = SecretManager()
    sm._create_secret(
        name="Stripe Team",
        value="sk-team",
        description="team default",
    )
    sm._create_secret(
        name="Stripe Alice",
        value="sk-alice",
        description="alice override",
    )

    monkeypatch.setattr(SESSION_DETAILS, "hive_id", 7, raising=False)

    sm._set_hive_default(integration="stripe", secret_name="Stripe Team")
    sm._bind_secret(integration="stripe", secret_name="Stripe Alice")

    resolved = sm.get_secret_for_integration("stripe")

    assert resolved.name == "Stripe Alice"
    assert resolved.value == "sk-alice"


def test_hive_member_without_binding_or_default_raises_lookup_error(
    secret_manager_context,
    monkeypatch,
):
    """Hive member without a binding or default → ``LookupError``.

    Critically, a Hive member does **not** fall back to
    ``name == integration`` — that would silently let every body in
    the Hive share a credential just because someone named it after
    the integration. The Hive must opt in explicitly via a default.
    """
    sm = SecretManager()
    sm._create_secret(
        name="salesforce",
        value="sk-would-match-solo",
        description="row that would match a solo fallback",
    )

    monkeypatch.setattr(SESSION_DETAILS, "hive_id", 42, raising=False)

    with pytest.raises(LookupError):
        sm.get_secret_for_integration("salesforce")


def test_set_hive_default_on_solo_body_raises(secret_manager_context):
    """Solo bodies can't install a Hive-wide default; the guard must fire.

    ``SecretDefault`` is Hive-scoped state; a solo body writing one
    would land it on the per-body context where no Hive peer can ever
    read it. The public surface rejects the call outright so an agent
    never silently produces dead data.
    """
    sm = SecretManager()
    sm._create_secret(
        name="Stripe Solo",
        value="sk-solo",
        description="solo key",
    )

    assert SESSION_DETAILS.hive_id is None

    with pytest.raises(RuntimeError, match="Hive-wide"):
        sm._set_hive_default(integration="stripe", secret_name="Stripe Solo")


def test_unbind_falls_back_to_default_for_hive_member(
    secret_manager_context,
    monkeypatch,
):
    """Removing a binding on a Hive member exposes the Hive default again.

    This pins the two-tier contract: a binding is an override, not a
    pin; unbinding must let the default re-surface rather than leaving
    the integration unresolvable.
    """
    sm = SecretManager()
    sm._create_secret(
        name="Team Default",
        value="sk-team",
        description="team default",
    )
    sm._create_secret(
        name="Alice Override",
        value="sk-alice",
        description="alice override",
    )

    monkeypatch.setattr(SESSION_DETAILS, "hive_id", 9, raising=False)

    sm._set_hive_default(integration="api", secret_name="Team Default")
    sm._bind_secret(integration="api", secret_name="Alice Override")

    assert sm.get_secret_for_integration("api").name == "Alice Override"

    sm._unbind_secret(integration="api")

    assert sm.get_secret_for_integration("api").name == "Team Default"
