"""Authorship metadata contracts for shared data rows."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from unity.blacklist_manager.types.blacklist import BlackList
from unity.common.authorship import (
    AUTHORING_ASSISTANT_ID_FIELD,
    fields_with_authoring,
    is_shared_authored_context,
    stamp_authoring_assistant_id,
    strip_authoring_assistant_id,
)
from unity.common.model_to_fields import model_to_fields
from unity.contact_manager.types.contact import Contact
from unity.dashboard_manager.types.dashboard import DashboardRecordRow
from unity.dashboard_manager.types.tile import TileRecordRow
from unity.file_manager.types.file import Document, FileRecord
from unity.function_manager.types.function import Function
from unity.function_manager.types.meta import FunctionsMeta
from unity.function_manager.types.venv import VirtualEnv
from unity.guidance_manager.types.guidance import Guidance
from unity.image_manager.types.image import Image
from unity.secret_manager.types import Secret
from unity.session_details import SESSION_DETAILS
from unity.task_scheduler.types.task import Task
from unity.transcript_manager.types.exchange import Exchange
from unity.transcript_manager.types.message import Message


@pytest.fixture(autouse=True)
def reset_session_details() -> None:
    SESSION_DETAILS.reset()
    yield
    SESSION_DETAILS.reset()


def test_insert_stamp_uses_active_assistant_and_rejects_spoofed_value() -> None:
    """Insert payloads use the runtime assistant as the immutable author."""

    SESSION_DETAILS.populate(agent_id=42, user_id="author-user")

    stamped = stamp_authoring_assistant_id(
        {"name": "shared row", AUTHORING_ASSISTANT_ID_FIELD: 999},
    )

    assert stamped == {"name": "shared row", AUTHORING_ASSISTANT_ID_FIELD: 42}


def test_update_payloads_cannot_change_authoring_assistant_id() -> None:
    """Update payload cleaning removes caller-controlled authorship fields."""

    cleaned = strip_authoring_assistant_id(
        {"title": "renamed", AUTHORING_ASSISTANT_ID_FIELD: 999},
    )

    assert cleaned == {"title": "renamed"}


@pytest.mark.parametrize(
    "model",
    [
        Task,
        Contact,
        Secret,
        Guidance,
        Function,
        FunctionsMeta,
        VirtualEnv,
        FileRecord,
        Document,
        BlackList,
        DashboardRecordRow,
        TileRecordRow,
        Message,
        Exchange,
        Image,
    ],
)
def test_shared_row_models_register_immutable_authoring_field(
    model: type[BaseModel],
) -> None:
    """Static shared-row schemas expose authoring metadata as non-mutable."""

    field = model_to_fields(model)[AUTHORING_ASSISTANT_ID_FIELD]

    assert field["type"] == "int"
    assert field["mutable"] is False


@pytest.mark.parametrize(
    "context",
    [
        "default/42/Data/Forecasts",
        "default/42/FileRecords/Local",
        "Spaces/7/Knowledge/Competitors",
        "Spaces/7/Files/Contracts/12/Content",
        "default/42/Functions/Compositional",
    ],
)
def test_dynamic_shared_contexts_receive_authoring_field(context: str) -> None:
    """Dynamic tables under shared roots inherit the authoring column."""

    assert is_shared_authored_context(context) is True
    fields = fields_with_authoring({"name": {"type": "str", "mutable": True}})

    assert fields[AUTHORING_ASSISTANT_ID_FIELD]["type"] == "int"
    assert fields[AUTHORING_ASSISTANT_ID_FIELD]["mutable"] is False


def test_unrelated_contexts_are_not_treated_as_shared_data() -> None:
    """Authorship registration stays scoped to shared data tables."""

    assert (
        is_shared_authored_context("default/42/AssistantJobs/startup_events") is False
    )
    assert is_shared_authored_context("default/42/Tasks/Activations") is False
    assert is_shared_authored_context("default/42/Tasks/Runs") is False
