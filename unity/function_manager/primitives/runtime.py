"""
Runtime primitives interface for state managers.

This module provides:
- `ComputerPrimitives` - Computer use (web/desktop) control capabilities
- `Primitives` - Scoped runtime interface for accessing state manager primitives
- `_AsyncPrimitiveWrapper` - Async wrapper for sync managers

All manager configuration (aliases, excluded methods, class paths) is defined in
`unity.function_manager.primitives.registry`. This module only handles runtime instantiation
and async wrapping.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import threading
from typing import Any, Callable, Optional, TYPE_CHECKING

from unity.function_manager.primitives.scope import (
    PrimitiveScope,
    VALID_MANAGER_ALIASES,
)
from unity.function_manager.primitives.registry import (
    get_registry,
    _CLASS_PATH_TO_ALIAS,
)
from unity.manager_registry import SingletonABCMeta

if TYPE_CHECKING:
    from unity.comms.primitives import CommsPrimitives
    from unity.function_manager.computer_backends import ComputerBackend
    from unity.contact_manager.contact_manager import ContactManager
    from unity.transcript_manager.transcript_manager import TranscriptManager
    from unity.knowledge_manager.knowledge_manager import KnowledgeManager
    from unity.task_scheduler.task_scheduler import TaskScheduler
    from unity.secret_manager.secret_manager import SecretManager
    from unity.web_searcher.web_searcher import WebSearcher

logger = logging.getLogger(__name__)


# =============================================================================
# ComputerPrimitives - Computer Use (Web/Desktop) Control
# =============================================================================


# Default agent-service URL for local development
DEFAULT_AGENT_SERVER_URL = "http://localhost:3000"

# Gate that blocks lazy MagnitudeBackend initialization until the managed VM is
# confirmed ready.  Set immediately for localhost / mock (no VM to wait for),
# or by _startup_sequence after log_job_startup confirms VM readiness.
_vm_ready = threading.Event()


_COMPUTER_METHODS = (
    "act",
    "observe",
    "query",
    "navigate",
    "get_links",
    "get_content",
    "get_screenshot",
    # Low-level actions (bypass LLM planning)
    "click",
    "double_click",
    "right_click",
    "drag",
    "scroll",
    "type_text",
    "press_enter",
    "press_tab",
    "press_backspace",
    "select_all",
    "press_key",
    "switch_tab",
    "close_tab",
    "new_tab",
    "go_back",
    "wait_for",
    "save_browser_state",
    "execute_actions",
)

_DESKTOP_METHODS = tuple(name for name in _COMPUTER_METHODS if name != "get_content")
_WEB_SESSION_METHODS = _COMPUTER_METHODS


def _publish_desktop_invoked(method_name: str) -> None:
    """Fire-and-forget EventBus publish for desktop primitive invocations."""
    try:
        from unity.events.event_bus import EVENT_BUS, Event

        asyncio.get_running_loop().create_task(
            EVENT_BUS.publish(
                Event(type="DesktopPrimitiveInvoked", payload={"method": method_name}),
            ),
        )
    except Exception:
        pass


def _publish_computer_act_completed(instruction: str, result: "ActResult") -> None:
    """Fire-and-forget EventBus publish when a visible session's act() completes."""
    try:
        from unity.events.event_bus import EVENT_BUS, Event

        asyncio.get_running_loop().create_task(
            EVENT_BUS.publish(
                Event(
                    type="ComputerActCompleted",
                    payload={
                        "instruction": instruction,
                        "summary": result.summary,
                    },
                ),
            ),
        )
    except Exception:
        pass


def _is_dead_session_error(e) -> bool:
    """Return True if the error indicates the session/browser is permanently gone."""
    from unity.function_manager.computer_backends import ComputerAgentError

    if not isinstance(e, ComputerAgentError):
        return False
    if e.error_type == "session_not_found":
        return True
    msg = (e.message or "").lower()
    return any(
        pattern in msg
        for pattern in (
            "browser has been closed",
            "context has been closed",
            "page has been closed",
        )
    )


def _make_session_method(
    method_name: str,
    owner: "ComputerPrimitives",
    session_resolver,
    *,
    mode: str = "",
    on_session_dead=None,
):
    """Build a wrapped async method that routes through a session.

    ``session_resolver`` is an async callable returning a ``ComputerSession``.
    Shared by ``_ComputerNamespace`` (lazy singleton) and ``WebSessionHandle``
    (bound instance).

    ``on_session_dead`` is an optional callback invoked when a request fails
    with a terminal session error (session removed, browser closed).
    """
    from unity.function_manager.computer_backends import ComputerSession

    is_desktop = mode == "desktop"

    if method_name == "get_screenshot":

        async def screenshot_wrapper(*args, **kwargs):
            kwargs.pop("_clarification_up_q", None)
            kwargs.pop("_clarification_down_q", None)
            if not _vm_ready.is_set():
                ready = await asyncio.to_thread(_vm_ready.wait, 300)
                if not ready:
                    raise RuntimeError(
                        "Managed VM did not become ready within 5 minutes",
                    )
            import base64, io
            from PIL import Image as _Image

            session = await session_resolver()
            try:
                b64 = await session.get_screenshot()
            except Exception as e:
                if on_session_dead and _is_dead_session_error(e):
                    on_session_dead()
                raise
            if is_desktop:
                _publish_desktop_invoked(method_name)
            return _Image.open(io.BytesIO(base64.b64decode(b64)))

        screenshot_wrapper.__name__ = method_name
        from unity.function_manager.computer_backends import ComputerBackend

        screenshot_wrapper.__doc__ = (
            getattr(ComputerBackend, method_name, None).__doc__
            or getattr(ComputerSession, method_name, None).__doc__
        )
        return screenshot_wrapper

    async def wrapper(*args, **kwargs):
        import time as _w_time
        import logging as _w_logging

        _w_t0 = _w_time.perf_counter()
        _w_log = _w_logging.getLogger("unity")

        def _w_ms():
            return f"{(_w_time.perf_counter() - _w_t0) * 1000:.0f}ms"

        _w_log.debug(
            f"⏱️ [desktop.{method_name} +{_w_ms()}] entered",
        )
        kwargs.pop("_clarification_up_q", None)
        kwargs.pop("_clarification_down_q", None)
        if not _vm_ready.is_set():
            _w_log.debug(f"⏱️ [desktop.{method_name} +{_w_ms()}] waiting for _vm_ready")
            ready = await asyncio.to_thread(_vm_ready.wait, 300)
            _w_log.debug(
                f"⏱️ [desktop.{method_name} +{_w_ms()}] _vm_ready resolved (ready={ready})",
            )
            if not ready:
                raise RuntimeError("Managed VM did not become ready within 5 minutes")
        else:
            _w_log.debug(f"⏱️ [desktop.{method_name} +{_w_ms()}] _vm_ready already set")
        if method_name in owner._SECRET_INJECTED_METHODS and args:
            _w_log.debug(f"⏱️ [desktop.{method_name} +{_w_ms()}] resolving secrets")
            resolved = await owner.secret_manager.from_placeholder(args[0])
            args = (resolved,) + args[1:]
            _w_log.debug(f"⏱️ [desktop.{method_name} +{_w_ms()}] secrets resolved")
        _w_log.debug(f"⏱️ [desktop.{method_name} +{_w_ms()}] session_resolver start")
        session = await session_resolver()
        _w_log.debug(
            f"⏱️ [desktop.{method_name} +{_w_ms()}] session resolved (id={getattr(session, '_session_id', '?')})",
        )
        _w_log.debug(
            f"⏱️ [desktop.{method_name} +{_w_ms()}] calling session.{method_name}",
        )
        try:
            result = await getattr(session, method_name)(*args, **kwargs)
        except Exception as e:
            if on_session_dead and _is_dead_session_error(e):
                on_session_dead()
            raise
        _w_log.debug(
            f"⏱️ [desktop.{method_name} +{_w_ms()}] session.{method_name} returned",
        )
        if is_desktop:
            _publish_desktop_invoked(method_name)
        is_visible = getattr(session, "_mode", "") in ("desktop", "web-vm")
        if is_visible and method_name == "act":
            _publish_computer_act_completed(args[0] if args else "", result)
        return result

    wrapper.__name__ = method_name
    # Prefer the rich docstrings from ComputerBackend ABC over ComputerSession's terse ones
    from unity.function_manager.computer_backends import ComputerBackend

    wrapper.__doc__ = (
        getattr(ComputerBackend, method_name, None).__doc__
        or getattr(ComputerSession, method_name, None).__doc__
    )
    return wrapper


class _ComputerNamespace:
    """Thin wrapper that routes method calls to a lazily-created singleton session.

    Used for ``primitives.computer.desktop``.
    """

    def __init__(self, owner: "ComputerPrimitives", mode: str):
        self._owner = owner
        self._mode = mode

        async def _resolve():
            return await owner.backend.get_session(mode)

        methods = _DESKTOP_METHODS if mode == "desktop" else _WEB_SESSION_METHODS
        for name in methods:
            setattr(self, name, _make_session_method(name, owner, _resolve, mode=mode))


class WebSessionHandle:
    """Wrapped browser session returned by ``primitives.computer.web.new_session()``.

    Exposes ``act``, ``observe``, ``query``, ``navigate``, ``get_links``,
    ``get_content``, and ``get_screenshot`` plus ``stop()`` for explicit
    lifecycle management.  Unlike ``primitives.computer.desktop``, web
    sessions retain ``get_content()`` because they operate on real browser
    pages rather than the noVNC viewer surface.
    """

    def __init__(
        self,
        session: "ComputerSession",
        owner: "ComputerPrimitives",
        session_id: int,
    ):
        self._session = session
        self._owner = owner
        self._session_id = session_id
        self._agent_session_id: str = session._session_id
        self._label = f"Web {session_id}"
        self._active = True

        def _mark_inactive():
            self._active = False

        async def _resolve():
            return self._session

        for name in _WEB_SESSION_METHODS:
            setattr(
                self,
                name,
                _make_session_method(
                    name,
                    owner,
                    _resolve,
                    on_session_dead=_mark_inactive,
                ),
            )

    @property
    def session_id(self) -> int:
        """Numeric session identifier (0, 1, 2, ...)."""
        return self._session_id

    @property
    def label(self) -> str:
        """Human-readable label shown as a visual badge in the browser."""
        return self._label

    @property
    def visible(self) -> bool:
        """Whether this session renders on the VM desktop (``web-vm`` mode)."""
        return self._session._mode == "web-vm"

    @property
    def active(self) -> bool:
        """Whether this session is still running (``stop()`` has not been called)."""
        return self._active

    async def stop(self):
        """Stop the browser session and release resources."""
        self._active = False
        await self._session.stop()


class _WebSessionFactory:
    """Factory for independent browser sessions.

    Accessed as ``primitives.computer.web``.  Has no default session --
    every browser session is explicitly created via ``new_session()`` and
    must be stopped when no longer needed.
    """

    def __init__(self, owner: "ComputerPrimitives"):
        self._owner = owner
        self._handles: list[WebSessionHandle] = []
        self._next_id: int = 0

    async def new_session(self, visible: bool = True) -> WebSessionHandle:
        """Create a new independent browser session.

        Each call spawns a fresh Chromium process with its own browsing
        context (cookies, storage, etc.).  Multiple sessions can run in
        parallel without interfering with each other.

        Parameters
        ----------
        visible : bool, default True
            If True, the browser window renders on the VM desktop (visible
            via noVNC) but is controlled entirely via CDP -- no mouse or
            keyboard involvement.  The user can un-minimize the window in
            noVNC to observe the session in real time.

            If False, the browser runs headless on the host machine for
            fast background lookups where visibility is unnecessary.

        Returns
        -------
        WebSessionHandle
            Session handle with methods: ``act``, ``observe``, ``query``,
            ``navigate``, ``get_links``, ``get_content``, ``get_screenshot``,
            and ``stop``.  Call ``stop()`` when done to release resources.
        """
        sid = self._next_id
        self._next_id += 1
        label = f"Web {sid}"
        mode = "web-vm" if visible else "web"
        session = await self._owner.backend.create_session(mode, label=label)
        handle = WebSessionHandle(session, self._owner, session_id=sid)
        self._handles.append(handle)
        return handle

    def list_sessions(
        self,
        visible_only: bool = False,
        active_only: bool = False,
    ) -> list[WebSessionHandle]:
        """List web sessions created across the entire system.

        Returns the actual ``WebSessionHandle`` objects from a global
        registry shared by all actors (``ComputerPrimitives`` is a
        singleton).  Use this for cross-actor coordination — e.g. to
        check how many browser sessions are currently active before
        creating new ones.

        Parameters
        ----------
        visible_only : bool, default False
            When True, only return sessions running on the VM desktop
            (``visible=True`` at creation time).  Excludes headless
            background sessions.
        active_only : bool, default False
            When True, only return sessions where ``stop()`` has not
            been called.

        Returns
        -------
        list[WebSessionHandle]
            Matching session handles.  Each handle can be used to call
            ``act``, ``observe``, ``navigate``, ``stop``, etc.
        """
        result = self._handles
        if visible_only:
            result = [h for h in result if h.visible]
        if active_only:
            result = [h for h in result if h.active]
        return list(result)

    def get_session(
        self,
        session_id: int,
        *,
        active_only: bool = True,
    ) -> WebSessionHandle:
        """Return an existing web session handle by numeric ID.

        This is the ergonomic reattachment API for browser sessions created in
        earlier turns or by other actors.  The ``session_id`` matches the
        handle's numeric ``session_id`` property and the IDs shown in
        ``<active_web_sessions>`` snapshots.

        Parameters
        ----------
        session_id : int
            Numeric session identifier (0, 1, 2, ...).
        active_only : bool, default True
            When True, only return sessions that have not been stopped.

        Returns
        -------
        WebSessionHandle
            The existing matching session handle.

        Raises
        ------
        ValueError
            If no matching session exists, or if the matching session is
            inactive while ``active_only`` is True.
        """
        for handle in self._handles:
            if handle.session_id != session_id:
                continue
            if active_only and not handle.active:
                raise ValueError(
                    f"Web session {session_id} exists but is inactive. "
                    "Use list_sessions(active_only=True) to discover reusable sessions.",
                )
            return handle

        raise ValueError(
            f"No web session with id {session_id}. "
            "Use list_sessions() to inspect the available handles.",
        )

    async def list_sessions_with_metadata(
        self,
        visible_only: bool = False,
        active_only: bool = False,
    ) -> list[dict]:
        """Like ``list_sessions`` but includes URL metadata for each session.

        Returns a list of dicts with keys ``handle``, ``url``, ``label``,
        and ``session_id``.
        """
        sessions = self.list_sessions(
            visible_only=visible_only,
            active_only=active_only,
        )
        result = []
        for h in sessions:
            url = ""
            try:
                url = await h._session.get_current_url()
            except Exception:
                pass
            result.append(
                {
                    "handle": h,
                    "session_id": h.session_id,
                    "label": h.label,
                    "url": url,
                },
            )
        return result


class ComputerPrimitives(metaclass=SingletonABCMeta):
    """Multi-mode computer control interface.

    Two attributes:

    - ``primitives.computer.desktop`` -- singleton namespace for full desktop
      control (mouse/keyboard via noVNC).  Methods: ``act``, ``observe``,
      ``query``, ``navigate``, ``get_links``, and ``get_screenshot``.
    - ``primitives.computer.web`` -- factory for independent browser sessions.
      Call ``new_session(visible=True/False)`` to create a session handle with
      the desktop method set plus ``get_content()`` and ``stop()``.  Use
      ``get_session(session_id)`` or ``list_sessions()`` to reattach to an
      existing browser session.

    Singleton via ``SingletonABCMeta`` / ``ManagerRegistry``.  All actors
    (including nested sub-agents) share the same backend connection.
    """

    _DYNAMIC_METHODS = (
        "act",
        "observe",
        "query",
        "navigate",
        "get_links",
        "get_content",
    )
    _LOW_LEVEL_METHODS = (
        "click",
        "double_click",
        "right_click",
        "drag",
        "scroll",
        "type_text",
        "press_enter",
        "press_tab",
        "press_backspace",
        "select_all",
        "press_key",
        "switch_tab",
        "close_tab",
        "new_tab",
        "go_back",
        "wait_for",
        "save_browser_state",
        "execute_actions",
    )
    _PRIMITIVE_METHODS = _DYNAMIC_METHODS + ("get_screenshot",) + _LOW_LEVEL_METHODS
    _SECRET_INJECTED_METHODS = frozenset({"act", "observe", "type_text"})

    @staticmethod
    def _resolve_container_url(explicit_url: str | None) -> str | None:
        """Resolve the agent-service container URL.

        Returns ``None`` when the managed VM hasn't been confirmed ready yet
        (``_vm_ready`` is unset), signalling that no URL is available.  Once
        ``_vm_ready`` is set the method returns the real desktop-derived URL
        or falls back to ``DEFAULT_AGENT_SERVER_URL`` for local/mock setups.
        """
        if explicit_url is not None and explicit_url != DEFAULT_AGENT_SERVER_URL:
            return explicit_url
        try:
            from unity.session_details import SESSION_DETAILS

            if SESSION_DETAILS.assistant.desktop_url:
                from urllib.parse import urlparse

                parsed = urlparse(SESSION_DETAILS.assistant.desktop_url)
                return f"{parsed.scheme}://{parsed.netloc}/api"
            if not _vm_ready.is_set():
                return None
        except Exception:
            pass
        return DEFAULT_AGENT_SERVER_URL

    def __init__(
        self,
        computer_mode: str = "magnitude",
        *,
        container_url: str | None = None,
        local_url: str | None = None,
        connect_now: bool = False,
        # Legacy compat: callers that pass agent_server_url get it mapped to container_url
        agent_server_url: str | None = None,
        **_kwargs,
    ):
        resolved_container = container_url or self._resolve_container_url(
            agent_server_url,
        )
        logger.info(
            f"ComputerPrimitives init: container_url={resolved_container}, mode={computer_mode}",
        )

        self._computer_mode = computer_mode
        self._computer_kwargs_map = {
            "magnitude": {
                "container_url": resolved_container,
                "local_url": local_url,
            },
            "mock": {},
        }

        self._secret_manager = None
        self._backend = None
        self._desktop_ns: Optional[_ComputerNamespace] = None
        self._web_factory: Optional[_WebSessionFactory] = None
        self._pending_url_mappings: dict[str, str] | None = None

        self._interject_queues: set[asyncio.Queue] = set()
        self._user_remote_control_active: bool = False

        if computer_mode == "mock":
            _vm_ready.set()

        if connect_now:
            _ = self.backend

    @property
    def secret_manager(self):
        if self._secret_manager is None:
            from unity.manager_registry import ManagerRegistry

            self._secret_manager = ManagerRegistry.get_secret_manager()
        return self._secret_manager

    @property
    def backend(self) -> "ComputerBackend":
        if self._backend is None:
            from unity.function_manager.computer_backends import (
                MagnitudeBackend,
                MockComputerBackend,
            )

            fresh_url = self._resolve_container_url(None)
            if fresh_url is not None and fresh_url != DEFAULT_AGENT_SERVER_URL:
                self._computer_kwargs_map["magnitude"]["container_url"] = fresh_url

            effective_url = self._computer_kwargs_map["magnitude"].get("container_url")
            logger.info(
                f"Creating {self._computer_mode} backend: fresh_url={fresh_url}, "
                f"container_url={effective_url}",
            )

            if self._computer_mode == "magnitude":
                self._backend = MagnitudeBackend(
                    **self._computer_kwargs_map["magnitude"],
                )
            elif self._computer_mode == "mock":
                self._backend = MockComputerBackend(
                    **self._computer_kwargs_map.get("mock", {}),
                )
            else:
                raise ValueError(f"Unknown computer_mode: '{self._computer_mode}'.")
            self._backend._on_session_closed = self._invalidate_web_session
            if self._pending_url_mappings is not None:
                self._backend._url_mappings = self._pending_url_mappings
        return self._backend

    @property
    def url_mappings(self) -> dict[str, str] | None:
        if self._backend is not None:
            return self._backend._url_mappings
        return self._pending_url_mappings

    @url_mappings.setter
    def url_mappings(self, mappings: dict[str, str] | None) -> None:
        self._pending_url_mappings = mappings
        if self._backend is not None:
            self._backend._url_mappings = mappings

    @staticmethod
    def mark_ready() -> None:
        """Signal that the agent-service is available and ready for requests.

        In production this is called by the VM startup sequence after the
        managed container boots.  Tests or sandboxes that manage their own
        agent-service should call this after the service is listening.
        """
        _vm_ready.set()

    def _invalidate_web_session(self, agent_session_id: str) -> None:
        """Mark the WebSessionHandle matching the agent-service UUID as inactive."""
        if self._web_factory is None:
            return
        for h in self._web_factory._handles:
            if h._agent_session_id == agent_session_id:
                h._active = False

    # ── Sub-namespace properties ─────────────────────────────────────────

    @property
    def desktop(self) -> _ComputerNamespace:
        """Desktop control namespace (mouse/keyboard via noVNC)."""
        if self._desktop_ns is None:
            self._desktop_ns = _ComputerNamespace(self, "desktop")
        return self._desktop_ns

    @property
    def web(self) -> _WebSessionFactory:
        """Factory for independent browser sessions.

        Call ``new_session(visible=True/False)`` to create a session, or
        ``get_session(session_id)`` / ``list_sessions()`` to reattach to an
        existing one.
        """
        if self._web_factory is None:
            self._web_factory = _WebSessionFactory(self)
        return self._web_factory

    # ── Steering control (not exposed as actor tools) ────────────────────

    async def pause(self) -> None:
        """Pause the underlying browser agent's action loop.

        Called programmatically by the actor's steering mechanism when the
        CodeActActor is paused.  This is not an actor tool — it is never
        listed in ``_DYNAMIC_METHODS`` or ``_PRIMITIVE_METHODS``.
        """
        await self.backend.pause()

    async def resume(self) -> None:
        """Resume the underlying browser agent's action loop.

        Called programmatically by the actor's steering mechanism when the
        CodeActActor is resumed.
        """
        await self.backend.resume()

    # ── Interject queue registry ─────────────────────────────────────────

    _REMOTE_CONTROL_STARTED_MSG = (
        "The user has taken remote control of the desktop. They are now "
        "able to operate the mouse and keyboard directly, so the screen "
        "state may diverge from what you last observed. This is not cause "
        "for alarm — the user may be collaborating, demonstrating "
        "something, or performing a task themselves. Use your judgement: "
        "if you are unsure whether to proceed with computer actions, "
        "request clarification rather than guessing."
    )
    _REMOTE_CONTROL_STOPPED_MSG = (
        "The user has released remote control of the desktop. The screen "
        "state may have changed since your last observation. Before "
        "continuing any computer-related work, take a fresh screenshot "
        "to re-orient yourself."
    )

    @staticmethod
    def _build_interjection(base_msg: str, context: str | None) -> dict:
        msg = base_msg
        if context:
            msg = f"{base_msg}\n\nRecent conversation context:\n{context}"
        return {"message": msg}

    def register_interject_queue(self, queue: asyncio.Queue) -> None:
        """Register an actor's interject queue for environmental broadcasts.

        If the user currently has remote control, the queue immediately
        receives the corresponding interjection so late-arriving actors
        are informed of the current state.
        """
        self._interject_queues.add(queue)
        if self._user_remote_control_active:
            queue.put_nowait(
                self._build_interjection(self._REMOTE_CONTROL_STARTED_MSG, None),
            )

    def deregister_interject_queue(self, queue: asyncio.Queue) -> None:
        """Remove an actor's interject queue from the registry."""
        self._interject_queues.discard(queue)

    def set_user_remote_control(
        self,
        active: bool,
        conversation_context: str | None = None,
    ) -> None:
        """Update remote-control state and broadcast to all registered actors.

        Parameters
        ----------
        active:
            Whether the user currently has remote control.
        conversation_context:
            Optional snippet of recent conversation to include in the
            interjection, giving actors immediate context on *why* the
            user is taking or releasing control.
        """
        self._user_remote_control_active = active
        base = (
            self._REMOTE_CONTROL_STARTED_MSG
            if active
            else self._REMOTE_CONTROL_STOPPED_MSG
        )
        payload = self._build_interjection(base, conversation_context)
        for q in self._interject_queues:
            q.put_nowait(payload)


# =============================================================================
# Async Wrapper for Sync Managers
# =============================================================================


class _AsyncPrimitiveWrapper:
    """
    Wrapper that provides async versions of sync manager methods.

    Delegates to the original manager without modifying it,
    ensuring internal code using the manager synchronously continues to work.

    Uses asyncio.to_thread() for sync methods to avoid blocking the event loop.
    """

    def __init__(self, manager: Any, manager_alias: str):
        """
        Initialize the wrapper.

        Args:
            manager: The original sync manager instance.
            manager_alias: The manager alias to look up primitive methods.
        """
        object.__setattr__(self, "_wrapped_manager", manager)
        object.__setattr__(self, "_manager_alias", manager_alias)
        # Get primitive methods from registry
        registry = get_registry()
        object.__setattr__(
            self,
            "_primitive_methods",
            set(registry.primitive_methods(manager_alias=manager_alias)),
        )

    def __getattr__(self, name: str) -> Any:
        """
        Get an attribute - returns async wrapper for primitive methods, else delegates.
        """
        attr = getattr(self._wrapped_manager, name)

        # Only wrap methods that are in our primitive methods set
        if name not in self._primitive_methods:
            return attr

        # Non-callable attributes pass through directly
        if not callable(attr):
            return attr

        # Create async wrapper that uses to_thread for sync methods
        @functools.wraps(attr)
        async def async_method_wrapper(*args, **kwargs):
            if asyncio.iscoroutinefunction(attr):
                return await attr(*args, **kwargs)
            else:
                return await asyncio.to_thread(attr, *args, **kwargs)

        return async_method_wrapper


def _create_async_wrapper(manager: Any, manager_alias: str) -> _AsyncPrimitiveWrapper:
    """
    Create an async wrapper for a sync manager.

    Args:
        manager: The original sync manager instance.
        manager_alias: The manager alias for registry lookup.

    Returns:
        An async wrapper around the manager.
    """
    return _AsyncPrimitiveWrapper(manager, manager_alias)


# =============================================================================
# Manager Registry Key Mapping
# =============================================================================

# Maps manager_alias to ManagerRegistry getter method name.
# Empty string means direct construction (e.g. singleton via metaclass).
_ALIAS_TO_GETTER: dict[str, str] = {
    "comms": "",
    "contacts": "get_contact_manager",
    "dashboards": "get_dashboard_manager",
    "data": "get_data_manager",
    "transcripts": "get_transcript_manager",
    "knowledge": "get_knowledge_manager",
    "tasks": "get_task_scheduler",
    "secrets": "get_secret_manager",
    "web": "get_web_searcher",
    "files": "get_file_manager",
    "computer": "",
    "actor": "",
}

# Managers that need async wrapping (sync implementations)
_SYNC_MANAGERS: frozenset[str] = frozenset({"dashboards", "data", "files"})


# =============================================================================
# Primitives Runtime Class (Scoped)
# =============================================================================


class Primitives:
    """
    Scoped runtime interface to all primitives (state managers and computer).

    Only managers in the provided `primitive_scope` are accessible.
    Attempting to access an out-of-scope manager raises AttributeError.

    Most managers are obtained via ManagerRegistry typed methods to respect
    IMPL settings (real vs simulated). ComputerPrimitives is constructed
    directly (singleton via metaclass).

    Sync managers (DataManager, FileManager) are wrapped with async interfaces
    for consistency - the LLM can safely use `await` on all primitives.

    Usage:
        scope = PrimitiveScope(scoped_managers=frozenset({"files", "contacts"}))
        primitives = Primitives(primitive_scope=scope)

        # Accessible:
        await primitives.files.describe(file_path="...")
        await primitives.contacts.ask(text="...")

        # Raises AttributeError:
        primitives.tasks  # not in scope
    """

    def __init__(self, *, primitive_scope: Optional[PrimitiveScope] = None) -> None:
        """
        Initialize primitives with the given scope.

        Args:
            primitive_scope: Defines which managers are accessible.
                           If None, all managers are accessible.
        """
        self._primitive_scope = primitive_scope or PrimitiveScope.all_managers()
        # Lazy-initialized manager instances
        self._managers: dict[str, Any] = {}

    @property
    def primitive_scope(self) -> PrimitiveScope:
        """The scope controlling which managers are accessible."""
        return self._primitive_scope

    def _get_manager(self, alias: str) -> Any:
        """
        Get or create a manager instance by alias.

        Raises AttributeError if alias is not in scope.
        """
        if alias not in self._primitive_scope.scoped_managers:
            available = sorted(self._primitive_scope.scoped_managers)
            raise AttributeError(
                f"primitives.{alias} is not available in this scope. "
                f"Available managers: {available}",
            )

        if alias in self._managers:
            return self._managers[alias]

        getter_name = _ALIAS_TO_GETTER.get(alias)
        if getter_name is None:
            raise AttributeError(f"Unknown manager alias: {alias}")

        if getter_name == "":
            # Direct construction via primitive_class_path from the registry.
            from unity.function_manager.primitives.registry import _MANAGER_BY_ALIAS

            spec = _MANAGER_BY_ALIAS.get(alias)
            if spec is None:
                raise AttributeError(f"No ManagerSpec for alias: {alias}")
            cls = get_registry()._load_manager_class(spec.primitive_class_path)
            if cls is None:
                raise AttributeError(
                    f"Could not load class for alias {alias!r}: "
                    f"{spec.primitive_class_path}",
                )
            manager = cls()
        else:
            from unity.manager_registry import ManagerRegistry

            getter = getattr(ManagerRegistry, getter_name)
            manager = getter()

        # Wrap sync managers with async interface
        if alias in _SYNC_MANAGERS:
            manager = _create_async_wrapper(manager, alias)

        self._managers[alias] = manager
        return manager

    def __getattr__(self, name: str) -> Any:
        """Attribute access for manager retrieval."""
        if name in VALID_MANAGER_ALIASES:
            return self._get_manager(name)

        raise AttributeError(f"'Primitives' object has no attribute '{name}'")

    # Convenience properties for type hints (IDE support)
    # These are optional and just provide better autocomplete

    @property
    def comms(self) -> "CommsPrimitives":
        """Assistant-owned communication primitives."""
        return self._get_manager("comms")

    @property
    def contacts(self) -> "ContactManager":
        """Contact management primitives (ask, update)."""
        return self._get_manager("contacts")

    @property
    def dashboards(self) -> "_AsyncPrimitiveWrapper":
        """Dashboard primitives (create_tile, create_dashboard, etc.)."""
        return self._get_manager("dashboards")

    @property
    def data(self) -> "_AsyncPrimitiveWrapper":
        """Data operations primitives (filter, search, reduce, join, etc.)."""
        return self._get_manager("data")

    @property
    def transcripts(self) -> "TranscriptManager":
        """Transcript management primitives (ask)."""
        return self._get_manager("transcripts")

    @property
    def knowledge(self) -> "KnowledgeManager":
        """Knowledge management primitives (ask, update, refactor)."""
        return self._get_manager("knowledge")

    @property
    def tasks(self) -> "TaskScheduler":
        """Task scheduling primitives (ask, update, execute)."""
        return self._get_manager("tasks")

    @property
    def secrets(self) -> "SecretManager":
        """Secret management primitives (ask, update)."""
        return self._get_manager("secrets")

    @property
    def web(self) -> "WebSearcher":
        """Web search primitives (ask)."""
        return self._get_manager("web")

    @property
    def files(self) -> "_AsyncPrimitiveWrapper":
        """File management primitives (describe, reduce, filter_files, etc.)."""
        return self._get_manager("files")

    @property
    def computer(self) -> "ComputerPrimitives":
        """Computer use primitives (act, navigate, observe, query, etc.)."""
        return self._get_manager("computer")

    @property
    def actor(self) -> Any:
        """Actor delegation primitives (run)."""
        return self._get_manager("actor")


# =============================================================================
# Primitive Callable Resolution (for FunctionManager execution)
# =============================================================================


def get_primitive_callable(
    primitive_data: dict[str, Any],
    *,
    primitives: Optional[Primitives] = None,
) -> Optional[Callable]:
    """
    Resolve a primitive metadata dict to its actual callable.

    Uses the ``primitives`` instance (which handles all aliases uniformly)
    when available, falling back to a default-scoped Primitives instance.

    Args:
        primitive_data: Primitive metadata with primitive_class and primitive_method.
        primitives: Scoped Primitives instance for resolution.

    Returns:
        The callable method, or None if resolution fails.
    """
    class_path = primitive_data.get("primitive_class")
    method_name = primitive_data.get("primitive_method")

    if not class_path or not method_name:
        return None

    manager_alias = _CLASS_PATH_TO_ALIAS.get(class_path)
    if not manager_alias:
        return None

    # Use provided primitives instance, or construct a default-scoped one.
    if primitives is None:
        primitives = Primitives()

    manager = getattr(primitives, manager_alias, None)
    if manager is None:
        return None
    return getattr(manager, method_name, None)
