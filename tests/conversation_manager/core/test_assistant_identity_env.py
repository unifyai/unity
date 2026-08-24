"""Tests for CM eval assistant identity env overrides."""

from __future__ import annotations

import os

import pytest

from tests.conversation_manager.assistant_identity_env import (
    TEST_ASSISTANT_EMAIL,
    TEST_ASSISTANT_NUMBER,
    TEST_ASSISTANT_WHATSAPP_NUMBER,
    ensure_test_assistant_identity_env,
)

pytestmark = pytest.mark.no_unify_context


def test_ensure_overrides_blank_assistant_identity(monkeypatch):
    monkeypatch.setenv("ASSISTANT_EMAIL", "")
    monkeypatch.setenv("ASSISTANT_NUMBER", "   ")
    monkeypatch.setenv("ASSISTANT_WHATSAPP_NUMBER", "")

    ensure_test_assistant_identity_env()

    assert os.environ["ASSISTANT_EMAIL"] == TEST_ASSISTANT_EMAIL
    assert os.environ["ASSISTANT_NUMBER"] == TEST_ASSISTANT_NUMBER
    assert os.environ["ASSISTANT_WHATSAPP_NUMBER"] == TEST_ASSISTANT_WHATSAPP_NUMBER


def test_ensure_preserves_explicit_assistant_identity(monkeypatch):
    monkeypatch.setenv("ASSISTANT_EMAIL", "custom@test.example.com")
    monkeypatch.setenv("ASSISTANT_NUMBER", "+15559998888")
    monkeypatch.setenv("ASSISTANT_WHATSAPP_NUMBER", "+15557776666")

    ensure_test_assistant_identity_env()

    assert os.environ["ASSISTANT_EMAIL"] == "custom@test.example.com"
    assert os.environ["ASSISTANT_NUMBER"] == "+15559998888"
    assert os.environ["ASSISTANT_WHATSAPP_NUMBER"] == "+15557776666"


def test_ensure_blanks_ambient_platform_identity(monkeypatch):
    """A hosted assistant pinned in the machine's .env must not leak into CM
    test boots: populate_from_env skips blank values, so agent_id stays None
    and the boot's ownership binding never consults the platform record."""
    monkeypatch.setenv("ASSISTANT_ID", "524")
    monkeypatch.setenv("OWNER_TEAM_ID", "7")

    ensure_test_assistant_identity_env()

    assert os.environ["ASSISTANT_ID"] == ""
    assert os.environ["OWNER_TEAM_ID"] == ""
