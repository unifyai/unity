"""Shared audit stamping helpers.

Every Hive-shared table carries an ``authoring_assistant_id`` column that
records which body authored the row. The stamp is written once at create
time and preserved across updates and merges so reviewers can always
answer "which body wrote this?" even when the row is shared across an
entire Hive. Solo bodies write the same column with their own id to keep
the contract uniform.

Concrete managers call :func:`authoring_assistant_id` when building the
``entries`` dict passed to ``unity.create_logs`` / ``unity_log``; the
column never appears in an update allowlist so the stamp is structurally
unreachable from the mutation surface.
"""

from __future__ import annotations

from typing import Final, Optional

from unity.session_details import SESSION_DETAILS


AUTHORING_COLUMN: Final[str] = "authoring_assistant_id"
"""Canonical column name for the audit stamp on every Hive-shared row.

Writers inject this column with :func:`authoring_assistant_id` at create
time; update paths strip it from the caller-supplied payload so the stamp
is immutable. Pydantic models that ship a typed shared-row schema declare
the same field on their model; dynamic-schema managers (Knowledge, Data)
use this constant to stamp the column on raw row dicts.
"""


def authoring_assistant_id() -> Optional[int]:
    """Return the active body's assistant id for audit stamping, if available.

    Returns ``None`` when ``SESSION_DETAILS`` has not been populated —
    typically offline utilities or early startup paths where no body is
    active yet. Managers propagate the ``None`` into the created row so
    the column is present for every write even when the runtime has no
    body identity to stamp.
    """
    if not SESSION_DETAILS.is_initialized:
        return None
    return SESSION_DETAILS.assistant.agent_id
