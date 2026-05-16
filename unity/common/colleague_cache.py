"""Shared helpers for assistant display names and colleague attribution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import unify
from unify.utils.http import RequestError

from unity.session_details import SESSION_DETAILS

FORMER_COLLEAGUE_LABEL = "a former colleague"
UNKNOWN_COLLEAGUE_LABEL = "a colleague"
CURRENT_ASSISTANT_FALLBACK_LABEL = "you"


def display_name(*, first_name: object, surname: object | None = None) -> str | None:
    """Return a trimmed full name assembled from first/surname parts."""

    name = " ".join(str(part or "").strip() for part in (first_name, surname)).strip()
    return name or None


def assistant_display_name(assistant: dict[str, Any]) -> str | None:
    """Return a normalized assistant display name from API row variants."""

    first_name = assistant.get("first_name") or assistant.get("firstName")
    surname = (
        assistant.get("surname")
        or assistant.get("last_name")
        or assistant.get("lastName")
    )
    return display_name(first_name=first_name, surname=surname)


@dataclass
class ColleagueNameCache:
    """Lazy cache for resolving assistant ids to human-readable names."""

    missing_label: str = FORMER_COLLEAGUE_LABEL
    _name_by_assistant_id: dict[int, str] = field(default_factory=dict)
    _missing_assistant_ids: set[int] = field(default_factory=set)
    _session_cache_key: tuple[int | None, str] | None = None

    def clear(self) -> None:
        """Reset all cached assistant-name lookups."""

        self._name_by_assistant_id.clear()
        self._missing_assistant_ids.clear()
        self._session_cache_key = None

    def _ensure_session_scope(self) -> None:
        """Invalidate cached lookups when org/key scope changes."""

        current_key = (SESSION_DETAILS.org_id, SESSION_DETAILS.unify_key)
        if self._session_cache_key == current_key:
            return
        self._name_by_assistant_id.clear()
        self._missing_assistant_ids.clear()
        self._session_cache_key = current_key

    def resolve(self, authoring_assistant_id: int | None) -> str | None:
        """Resolve an authoring assistant id to a stable display label."""

        if authoring_assistant_id is None:
            return None

        self._ensure_session_scope()
        assistant_id = int(authoring_assistant_id)
        current_assistant_id = SESSION_DETAILS.assistant.agent_id
        if current_assistant_id is not None and assistant_id == int(
            current_assistant_id,
        ):
            return (
                display_name(
                    first_name=SESSION_DETAILS.assistant.first_name,
                    surname=SESSION_DETAILS.assistant.surname,
                )
                or CURRENT_ASSISTANT_FALLBACK_LABEL
            )

        cached = self._name_by_assistant_id.get(assistant_id)
        if cached is not None:
            return cached
        if assistant_id in self._missing_assistant_ids:
            return self.missing_label

        try:
            assistants = unify.list_assistants(
                agent_id=assistant_id,
                list_all_org=SESSION_DETAILS.org_id is not None,
                api_key=SESSION_DETAILS.unify_key,
            )
        except RequestError:
            self._name_by_assistant_id[assistant_id] = UNKNOWN_COLLEAGUE_LABEL
            return UNKNOWN_COLLEAGUE_LABEL
        except Exception:
            self._name_by_assistant_id[assistant_id] = UNKNOWN_COLLEAGUE_LABEL
            return UNKNOWN_COLLEAGUE_LABEL

        if not assistants:
            self._missing_assistant_ids.add(assistant_id)
            return self.missing_label

        name = assistant_display_name(assistants[0]) or UNKNOWN_COLLEAGUE_LABEL
        self._name_by_assistant_id[assistant_id] = name
        return name
