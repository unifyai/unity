from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import unisdk
from unisdk.utils.http import RequestError as _UnifyRequestError

from unify.common.authorship import fields_with_authoring, is_shared_authored_context

logger = logging.getLogger(__name__)

# Private fields injected by log_utils wrappers
_CREATE_CONTEXT_MAX_ATTEMPTS = 3
_CREATE_CONTEXT_BACKOFF_SECS = (0.5, 1.5)


def _is_transient(exc: _UnifyRequestError) -> bool:
    """Return True if the RequestError is likely transient and worth retrying."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is None:
        return True
    return status == 429 or status >= 500


def _is_already_exists_context_error(exc: _UnifyRequestError) -> bool:
    """Return whether the backend reported an idempotent context-exists conflict."""
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status != 400:
        return False
    text = (getattr(response, "text", "") or "").lower()
    return "already exists" in text and "context" in text


class ContextIdentityError(RuntimeError):
    """A live context is missing identity configuration its manager declares."""


def _verify_context_identity(
    name: str,
    *,
    unique_keys: Dict[str, str],
    auto_counting: Dict[str, Optional[str]],
    project: Optional[str],
) -> None:
    """Raise when the live context lacks the declared identity configuration.

    ``create_context`` reports success both for a fresh create and for a
    pre-existing context, and a pre-existing context keeps whatever
    configuration it was first created with — including none at all when a
    row write reached the backend first and auto-created it bare. A bare
    context assigns no ids to inserted rows, and no API can retrofit the
    configuration, so provisioning time is the only place the corruption is
    caught before unaddressable rows accumulate.
    """
    live = unisdk.get_context(name, project=project)
    missing_keys = set(unique_keys) - set(live.get("unique_keys") or [])
    missing_counters = set(auto_counting) - set(live.get("auto_counting") or {})
    if missing_keys or missing_counters:
        raise ContextIdentityError(
            f"Context {name!r} is live without its declared identity "
            f"configuration (missing unique_keys: {sorted(missing_keys)}, "
            f"missing auto_counting: {sorted(missing_counters)}). It was "
            "first created without them — typically implicitly, by a row "
            "write that reached the backend before provisioning — and the "
            "backend cannot retrofit them. Delete and re-provision the "
            "context to repair.",
        )


def _create_context_with_retry(
    name: str,
    *,
    unique_keys: Optional[Dict[str, str]] = None,
    auto_counting: Optional[Dict[str, Optional[str]]] = None,
    description: Optional[str] = None,
    foreign_keys: Optional[list[Dict[str, Any]]] = None,
    owner_scope: Optional[str] = None,
    owner_id: Optional[int] = None,
    project: Optional[str] = None,
) -> None:
    """Call ``unisdk.create_context`` with retry on transient failures.

    Retries up to ``_CREATE_CONTEXT_MAX_ATTEMPTS`` times with exponential
    backoff for transient HTTP errors (5xx, 429, network).  Non-transient
    errors (4xx) are raised immediately (except "already exists" which the
    SDK handles via ``exist_ok=True``).

    ``owner_scope`` / ``owner_id`` make the context's ownership explicit (the
    unit of O(owner) bulk deletion); when omitted the backend infers it from
    the context name.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(_CREATE_CONTEXT_MAX_ATTEMPTS):
        try:
            unisdk.create_context(
                name,
                unique_keys=unique_keys,
                auto_counting=auto_counting,
                description=description,
                foreign_keys=foreign_keys,
                owner_scope=owner_scope,
                owner_id=owner_id,
                project=project,
            )
            # ``create_context`` reports success for a pre-existing context
            # too (and its insert races resolve via on-conflict-do-nothing),
            # so a successful call does not prove the live context carries
            # the identity configuration just declared. Verify it does.
            if unique_keys or auto_counting:
                _verify_context_identity(
                    name,
                    unique_keys=unique_keys or {},
                    auto_counting=auto_counting or {},
                    project=project,
                )
            return
        except _UnifyRequestError as exc:
            if _is_already_exists_context_error(exc):
                if unique_keys or auto_counting:
                    _verify_context_identity(
                        name,
                        unique_keys=unique_keys or {},
                        auto_counting=auto_counting or {},
                        project=project,
                    )
                return
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
        self._project = unisdk.active_project()
        self._unique_keys = dict(unique_keys or {})
        self._auto_counting = dict(auto_counting or {})
        self._description = description or ""
        self._fields = dict(fields or {})
        if is_shared_authored_context(context):
            self._fields = fields_with_authoring(self._fields)
        self._foreign_keys = list(foreign_keys or [])

    # ──────────────────────────────────────────────────────────────────────
    # Provisioning
    # ──────────────────────────────────────────────────────────────────────
    def ensure_context(self) -> None:
        """Create the context (and its fields) in Orchestra.

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
                unisdk.create_fields(self._fields, context=self._ctx)
            except Exception:
                pass

        self._ENSURED.add(key)

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
            data = unisdk.get_fields(project=self._project, context=self._ctx)
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
                data = unisdk.get_fields(project=self._project, context=self._ctx)
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
