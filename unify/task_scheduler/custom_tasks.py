"""
Collection of custom task definitions from ``tasks.jsonl`` files.

Source entries live in per-deployment directories as one JSON object per
line. Collection helpers accept explicit paths so sync can target different
source trees across org -> user -> assistant cascade layers.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

from ..common.prompt_helpers import get_assistant_timezone
from .types.trigger import Trigger

logger = logging.getLogger(__name__)

TASKS_JSONL_FILENAME = "tasks.jsonl"

TASK_SYNC_FIELDS = (
    "name",
    "description",
    "schedule",
    "trigger",
    "deadline",
    "max_runtime_seconds",
    "repeat",
    "priority",
    "tags",
    "response_policy",
    "entrypoint_function",
    "offline",
    "requires_filesystem",
    "requires_computer",
)


class CustomTaskSourceEntry(BaseModel):
    """One source-defined task row from ``tasks.jsonl``."""

    key: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    schedule: Optional[Dict[str, Any]] = None
    trigger: Optional[Dict[str, Any]] = None
    deadline: Optional[str] = None
    max_runtime_seconds: Optional[int] = None
    repeat: Optional[List[Dict[str, Any]]] = None
    priority: str = "normal"
    tags: Optional[List[str]] = None
    response_policy: Optional[str] = None
    entrypoint_function: Optional[str] = None
    offline: bool = False
    requires_filesystem: bool = False
    requires_computer: bool = False
    destination: str = "personal"
    auto_sync: bool = True


def _stable_json(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, sort_keys=True, default=str)


def _compute_task_hash(
    *,
    key: str,
    destination: str,
    fields: Dict[str, Any],
) -> str:
    components = [key, destination or "personal"]
    for field_name in TASK_SYNC_FIELDS:
        value = fields.get(field_name)
        if field_name in {"offline", "requires_filesystem", "requires_computer"}:
            components.append(str(bool(value)))
        elif field_name in {"schedule", "trigger", "repeat"}:
            components.append(_stable_json(value))
        else:
            components.append("" if value is None else str(value))
    combined = "\n".join(components)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def _normalize_task_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(payload)
    if "task_name" in normalized and "name" not in normalized:
        normalized["name"] = normalized.pop("task_name")
    if "task_description" in normalized and "description" not in normalized:
        normalized["description"] = normalized.pop("task_description")
    if "enabled" in normalized and "auto_sync" not in normalized:
        normalized["auto_sync"] = bool(normalized.pop("enabled"))
    execution_mode = normalized.pop("execution_mode", None)
    if execution_mode is not None and "offline" not in normalized:
        normalized["offline"] = str(execution_mode).strip().lower() == "offline"
    return normalized


def _resolve_jsonl_path(path: Path) -> Optional[Path]:
    if path.is_file() and path.suffix == ".jsonl":
        return path
    if path.is_dir():
        candidate = path / TASKS_JSONL_FILENAME
        if candidate.is_file():
            return candidate
    return None


def _parse_jsonl_file(jsonl_path: Path) -> List[CustomTaskSourceEntry]:
    entries: List[CustomTaskSourceEntry] = []
    for line_no, raw_line in enumerate(jsonl_path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        # A malformed line must fail the collection, not be skipped: a
        # silently dropped key reads as "removed from the source" and the
        # reconcile prunes the live task it still owns.
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("task row must be a JSON object")
            entry = CustomTaskSourceEntry.model_validate(
                _normalize_task_payload(payload),
            )
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            raise ValueError(
                f"Invalid tasks.jsonl line {jsonl_path}:{line_no}: {exc}",
            ) from exc
        if entry.schedule is not None and entry.trigger is not None:
            raise ValueError(
                f"Invalid tasks.jsonl line {jsonl_path}:{line_no}: "
                "schedule and trigger are mutually exclusive",
            )
        if entry.trigger is not None:
            try:
                entry = entry.model_copy(
                    update={
                        "trigger": Trigger.model_validate(entry.trigger).model_dump(
                            mode="json",
                        ),
                    },
                )
            except ValidationError as exc:
                raise ValueError(
                    f"Invalid tasks.jsonl line {jsonl_path}:{line_no}: "
                    f"invalid trigger: {exc}",
                ) from exc
        if not entry.auto_sync:
            logger.debug("Skipping %s: auto_sync=False", entry.key)
            continue
        entries.append(entry)
    return entries


def _localize_repeat(
    repeat: Optional[List[Dict[str, Any]]],
) -> Optional[List[Dict[str, Any]]]:
    """Anchor a bundle's wall-clock slots to the assistant's own zone.

    A bundle is universal, so it cannot name a zone: "08:30" in a shelf
    manifest means *the installer's* half past eight, not UTC's. Resolved
    here, at the moment a bundle becomes one assistant's planted task, which
    is the first point where whose morning it is is known.

    Left alone when the assistant has no zone (behaviour is then exactly what
    it was, UTC) or when the source named one explicitly, which an author
    would only do for a schedule that genuinely is absolute.

    The resolved zone lands in the task's ``custom_hash`` along with the rest
    of ``repeat``, so an assistant that later changes timezone re-plants on
    the next reconcile rather than keeping the hours of a country it left.
    """

    if not repeat:
        return repeat
    zone = get_assistant_timezone()
    if not zone:
        return repeat

    localized: List[Dict[str, Any]] = []
    for pattern in repeat:
        if not isinstance(pattern, dict):
            localized.append(pattern)
            continue
        if pattern.get("time_of_day") and not pattern.get("timezone"):
            if zone.upper() == "UTC":
                # Stamping UTC is indistinguishable from not stamping at all,
                # and for a bundle slot it is almost certainly wrong: the
                # manifest wrote a human hour and this is the moment that
                # becomes a real instant. Said out loud, because the symptom
                # otherwise appears days later as a task firing at a sensible
                # hour in the wrong place.
                logger.warning(
                    "Anchoring %s to UTC: the assistant declares no local "
                    "timezone, so every bare wall-clock slot in this source "
                    "means UTC's hour rather than the installer's.",
                    pattern.get("time_of_day"),
                )
            pattern = {**pattern, "timezone": zone}
        localized.append(pattern)
    return localized


def collect_custom_tasks(
    path: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """Load ``tasks.jsonl`` from a directory or direct file path."""
    if path is None:
        logger.debug("Custom tasks path is None, nothing to collect")
        return {}

    jsonl_path = _resolve_jsonl_path(Path(path))
    if jsonl_path is None:
        logger.debug("No tasks.jsonl found at %s", path)
        return {}

    tasks: Dict[str, Dict[str, Any]] = {}
    for entry in _parse_jsonl_file(jsonl_path):
        destination = entry.destination or "personal"
        requires_computer = entry.requires_computer
        fields = {
            "name": entry.name,
            "description": entry.description,
            "schedule": entry.schedule,
            "trigger": entry.trigger,
            "deadline": entry.deadline,
            "max_runtime_seconds": entry.max_runtime_seconds,
            "repeat": _localize_repeat(entry.repeat),
            "priority": entry.priority,
            "tags": entry.tags,
            "response_policy": entry.response_policy,
            "entrypoint_function": entry.entrypoint_function,
            "offline": entry.offline,
            "requires_filesystem": entry.requires_filesystem,
            "requires_computer": requires_computer,
        }
        custom_hash = _compute_task_hash(
            key=entry.key,
            destination=destination,
            fields=fields,
        )
        tasks[entry.key] = {
            "custom_key": entry.key,
            "custom_hash": custom_hash,
            "destination": destination,
            **fields,
        }

    logger.debug(
        "Collected %d custom task entries from %s",
        len(tasks),
        jsonl_path,
    )
    return tasks


def compute_custom_tasks_hash(
    source_tasks: Optional[Dict[str, Dict[str, Any]]] = None,
    *,
    entrypoint_resolution: Optional[Dict[str, Optional[int]]] = None,
) -> str:
    """Aggregate fingerprint of the custom task entries and their derived
    entrypoint resolution.

    Per-entry ``custom_hash`` covers authored source fields only, so it is
    blind to the functions store being re-registered under new ids. Folding
    the ``entrypoint_function`` → ``function_id`` resolution into the
    aggregate makes a function renumbering invalidate the stored hash,
    which forces the per-key pass whose ``derived_stale`` check re-points
    dangling rows.
    """
    tasks = source_tasks if source_tasks is not None else {}
    if not tasks:
        return ""

    parts = [tasks[key]["custom_hash"] for key in sorted(tasks.keys())]
    parts += [
        f"{name}={fid}" for name, fid in sorted((entrypoint_resolution or {}).items())
    ]
    combined = "|".join(parts)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def collect_tasks_from_directories(
    directories: List[Path],
) -> Dict[str, Dict[str, Any]]:
    """Collect custom tasks from multiple directories and merge."""
    merged: Dict[str, Dict[str, Any]] = {}
    for directory in directories:
        merged.update(collect_custom_tasks(path=directory))
    return merged
