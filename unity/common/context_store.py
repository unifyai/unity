from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import unify
from unify.utils.http import RequestError as _UnifyRequestError

logger = logging.getLogger(__name__)

# Private fields injected by log_utils wrappers
_PRIVATE_FIELDS: Dict[str, str] = {
    "_user": "str",
    "_user_id": "str",
    "_assistant": "str",
    "_assistant_id": "str",
    "_org": "str",
    "_org_id": "int",
}

_CREATE_CONTEXT_MAX_ATTEMPTS = 3
_CREATE_CONTEXT_BACKOFF_SECS = (0.5, 1.5)


def _is_transient(exc: _UnifyRequestError) -> bool:
    """Return True if the RequestError is likely transient and worth retrying."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is None:
        return True
    return status == 429 or status >= 500


def _create_context_with_retry(
    name: str,
    *,
    unique_keys: Optional[Dict[str, str]] = None,
    auto_counting: Optional[Dict[str, Optional[str]]] = None,
    description: Optional[str] = None,
    foreign_keys: Optional[list[Dict[str, Any]]] = None,
    project: Optional[str] = None,
) -> None:
    """Call ``unify.create_context`` with retry on transient failures.

    Retries up to ``_CREATE_CONTEXT_MAX_ATTEMPTS`` times with exponential
    backoff for transient HTTP errors (5xx, 429, network).  Non-transient
    errors (4xx) are raised immediately (except "already exists" which the
    SDK handles via ``exist_ok=True``).
    """
    last_exc: Optional[Exception] = None
    for attempt in range(_CREATE_CONTEXT_MAX_ATTEMPTS):
        try:
            unify.create_context(
                name,
                unique_keys=unique_keys,
                auto_counting=auto_counting,
                description=description,
                foreign_keys=foreign_keys,
                project=project,
            )
            return
        except _UnifyRequestError as exc:
            if not _is_transient(exc):
                raise
            last_exc = exc
            if attempt < _CREATE_CONTEXT_MAX_ATTEMPTS - 1:
                delay = _CREATE_CONTEXT_BACKOFF_SECS[
                    min(attempt, len(_CREATE_CONTEXT_BACKOFF_SECS) - 1)
                ]
                logger.warning(
                    "create_context(%r) attempt %d/%d failed (status %s), "
                    "retrying in %.1fs",
                    name,
                    attempt + 1,
                    _CREATE_CONTEXT_MAX_ATTEMPTS,
                    getattr(
                        getattr(exc, "response", None),
                        "status_code",
                        "?",
                    ),
                    delay,
                )
                time.sleep(delay)
    raise last_exc  # type: ignore[misc]


class TableStore:
    """
    Idempotent context/field provisioner with safe accessors.

    Guarantees that a given ``(project, context)`` exists with the required
    fields before read/write operations. Falls back to an ensure→retry path when
    encountering a backend 404 due to races or eventual consistency.
    """

    # Process-local memo to avoid repeated ensures in the same run
    _ENSURED: set[tuple[str, str]] = set()

    def __init__(
        self,
        context: str,
        *,
        unique_keys: Optional[Dict[str, str]] = None,
        auto_counting: Optional[Dict[str, Optional[str]]] = None,
        description: Optional[str] = None,
        fields: Optional[Dict[str, Any]] = None,
        foreign_keys: Optional[list[Dict[str, Any]]] = None,
    ) -> None:
        self._ctx = context
        self._project = unify.active_project()
        self._unique_keys = dict(unique_keys or {})
        self._auto_counting = dict(auto_counting or {})
        self._description = description or ""
        self._fields = dict(fields or {})
        self._foreign_keys = list(foreign_keys or [])

    # ──────────────────────────────────────────────────────────────────────
    # Provisioning
    # ──────────────────────────────────────────────────────────────────────
    def _all_contexts(self) -> List[str]:
        """
        Derive aggregation contexts for this user/assistant-scoped context.

        Returns two contexts for cross-assistant and cross-user aggregation:
          - {user_id}/All/{suffix} - all assistants for this user
          - All/{suffix}           - all users, all assistants

        Example: "42/7/Contacts" returns:
          - "42/All/Contacts"
          - "All/Contacts"

        Returns empty list if context doesn't have user_id/assistant_id prefix.
        """
        parts = self._ctx.split("/")
        if len(parts) < 3:
            return []

        # Handle test contexts: tests/.../{default_user_id}/{default_assistant_id}/Suffix
        # Scope aggregations to the test root to avoid cross-test contamination.
        if parts[0] == "tests":
            from unity.session_details import UNASSIGNED_USER_CONTEXT

            try:
                user_idx = parts.index(UNASSIGNED_USER_CONTEXT)
            except ValueError:
                return []

            # Need at least User/Assistant/Suffix after the test root
            if user_idx + 2 >= len(parts):
                return []

            test_root = "/".join(parts[:user_idx])
            user_ctx = parts[user_idx]
            suffix = "/".join(parts[user_idx + 2 :])
            return [
                f"{test_root}/{user_ctx}/All/{suffix}",
                f"{test_root}/All/{suffix}",
            ]

        # Production path: User/Assistant/Suffix
        user_ctx = parts[0]
        suffix = "/".join(parts[2:])  # Everything after user_id/assistant_id
        return [
            f"{user_ctx}/All/{suffix}",
            f"All/{suffix}",
        ]

    def _ensure_all_contexts(self, all_ctxs: List[str]) -> None:
        """Ensure aggregation contexts exist for cross-assistant / cross-user queries.

        These contexts mirror the source context's fields (plus private fields)
        but have no unique_keys or auto_counting.
        """
        for all_ctx in all_ctxs:
            key = (self._project, all_ctx)
            if key in self._ENSURED:
                continue

            if all_ctx.startswith("All/"):
                description = f"Global aggregation of {self._ctx.split('/')[-1]} across all users and assistants"
            else:
                description = f"Aggregation of {self._ctx.split('/')[-1]} across all assistants for this user"

            _create_context_with_retry(all_ctx, description=description)

            fields_with_private = dict(self._fields)
            fields_with_private.update(_PRIVATE_FIELDS)

            if fields_with_private:
                try:
                    unify.create_fields(fields_with_private, context=all_ctx)
                except Exception:
                    pass

            self._ENSURED.add(key)

    def ensure_context(self) -> None:
        """Create the context (and its fields / aggregation siblings) in Orchestra.

        Uses ``_create_context_with_retry`` so transient HTTP errors (5xx / 429 /
        network) are retried with backoff.  Non-transient errors propagate
        immediately — a missing context is fatal for the owning manager, so
        callers must not silently swallow the exception.
        """
        key = (self._project, self._ctx)
        if key in self._ENSURED:
            return

        _create_context_with_retry(
            self._ctx,
            unique_keys=self._unique_keys or None,
            auto_counting=self._auto_counting or None,
            description=self._description,
            foreign_keys=self._foreign_keys or None,
        )

        if self._fields:
            try:
                unify.create_fields(self._fields, context=self._ctx)
            except Exception:
                pass

        self._ENSURED.add(key)

        all_ctxs = self._all_contexts()
        if all_ctxs:
            self._ensure_all_contexts(all_ctxs)

    # ──────────────────────────────────────────────────────────────────────
    # Accessors with 404→ensure→retry
    # ──────────────────────────────────────────────────────────────────────
    def get_columns(self) -> Dict[str, str]:
        """Return {column_name: column_type} for this context.

        If the backend returns 404 (missing context), run ``ensure_context``
        once and retry with a tiny backoff. Normalises to a single string
        label per field, preferring 'data_type' then 'type'.
        """
        import time as _time

        def _normalize_fields(raw: Any) -> Dict[str, str]:
            if not isinstance(raw, dict):
                return {}
            out: Dict[str, str] = {}
            for k, v in raw.items():
                try:
                    if isinstance(v, dict):
                        out[str(k)] = (
                            str(v.get("data_type") or v.get("type") or "")
                        ).strip() or "unknown"
                    else:
                        out[str(k)] = str(v)
                except Exception:
                    # Extremely defensive – field schemas should never break callers.
                    out[str(k)] = "unknown"
            return out

        # First attempt
        try:
            data = unify.get_fields(project=self._project, context=self._ctx)
            return _normalize_fields(data)
        except _UnifyRequestError as e:
            # 404: context missing (race / test teardown / eventual consistency).
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status != 404:
                raise

        # Ensure then retry a few times (handles eventual consistency after creation).
        self.ensure_context()
        last_exc: Exception | None = None
        for delay in (0.0, 0.05, 0.15):
            if delay:
                _time.sleep(delay)
            try:
                data = unify.get_fields(project=self._project, context=self._ctx)
                return _normalize_fields(data)
            except _UnifyRequestError as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status == 404:
                    last_exc = e
                    continue
                raise
            except Exception as e:
                last_exc = e
                break
        if last_exc is not None:
            raise last_exc
        return {}
