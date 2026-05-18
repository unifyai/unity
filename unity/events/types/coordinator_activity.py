"""Payload model for product-facing Coordinator setup activity."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CoordinatorActivityEntityType = Literal[
    "human",
    "colleague",
    "workspace",
    "credential",
    "task",
    "knowledge",
    "guidance",
    "dashboard",
    "function",
    "data",
    "validation",
]

CoordinatorActivityPhase = Literal[
    "started",
    "progress",
    "needs_input",
    "blocked",
    "completed",
    "failed",
]

CoordinatorActivityStage = Literal[
    "discovery",
    "requirements",
    "proposal",
    "confirmation",
    "implementation",
    "integration_setup",
    "validation",
    "handoff",
]

CoordinatorActivitySurface = Literal[
    "authorized_humans",
    "colleagues",
    "workspaces",
    "membership",
    "invitation",
    "credentials",
    "tasks",
    "memory",
    "guidance",
    "dashboards",
    "functions",
    "data",
    "communications",
]


class CoordinatorActivityEntity(BaseModel):
    """One user-visible object referenced by a Coordinator activity card."""

    model_config = ConfigDict(extra="forbid")

    type: CoordinatorActivityEntityType
    id: str | None = None
    name: str


class CoordinatorActivityPayload(BaseModel):
    """User-facing progress event for Coordinator setup work.

    The payload intentionally carries renderable card fields rather than raw
    tool arguments, model reasoning, SDK responses, or secret-bearing details.
    """

    model_config = ConfigDict(extra="forbid")

    activity_id: str
    phase: CoordinatorActivityPhase
    stage: CoordinatorActivityStage
    surfaces: list[CoordinatorActivitySurface] = Field(default_factory=list)
    title: str
    summary: str | None = None
    checklist_item_id: int | None = None
    related_entities: list[CoordinatorActivityEntity] = Field(default_factory=list)
    chat_prompt: str | None = None
    chat_prompt_label: str | None = None
    correlation_id: str | None = None
    occurred_at: datetime
    status: Literal["ok", "error"] = "ok"
    error: str | None = None
