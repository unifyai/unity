"""
unity/__init__.py
==================

Package initialization for the Unity AI Assistant framework.

The runtime must be explicitly initialized via init() before using managers:

    import unity
    unity.init()  # Activates Unify project, selects assistant, starts EventBus

For code that may run before or after init(), use ensure_initialised() which
is a no-op if already initialized.

LLM I/O logging is now handled directly in the unillm package. Enable it via:
  - UNILLM_IO_LOG=true (to enable logging)
  - UNILLM_LOG_DIR=/path/to/logs (to set the output directory)

Logging is configured centrally in unity.logger (imported below).
"""

from typing import Optional

from unity.common.context_registry import ContextRegistry

# Attempt to import the external 'unify' SDK. If unavailable, provide a minimal
# no-op shim so importing the 'unity' package does not require extra installs.
try:  # pragma: no cover - simple import guard
    import unify  # type: ignore
except Exception:  # ImportError or others

    class _UnifyShim:

        def active_project(self) -> bool:
            return False

        def activate(self, *_args, **_kwargs) -> None:
            pass

        def set_context(self, *_args, **_kwargs) -> None:
            pass

        def get_active_context(self) -> dict:
            return {}

    unify = _UnifyShim()  # type: ignore


# Logging is configured entirely in unity.logger — import it so that
# the module-level setup (handler, formatter, library muting) runs once.
import unity.logger  # noqa: F401

# ---------------------------------------------------------------------------
# Lazy runtime initialisation
# ---------------------------------------------------------------------------

from unity.session_details import SESSION_DETAILS

_INITIALISED = False


def _list_all_assistants() -> list[dict]:
    """Return the list of assistants available to the current account.

    The helper mirrors the *list_all_assistants* REST call documented in
    the Unify API.  On any network / authentication error an **empty** list
    is returned so that offline test-suites continue to operate.
    """
    try:
        return unify.list_assistants()
    except Exception:
        # Offline / stubbed environments fall back to an empty list so that
        # the rest of the initialisation sequence can proceed with a dummy
        # assistant record (created later by ContactManager).
        return []


def init(
    project_name: str = "Assistants",
    assistant_id: Optional[int] = None,
    overwrite: bool = False,
    assistant_record: dict | None = None,
) -> None:  # noqa: D401 – imperative name
    """Initialise the *unity* runtime.

    This performs two steps **once** per interpreter session:

    1. Activate the given *project_name* in the Unify SDK (unless a project is
       already active).
    2. Construct and wire-up the global :pydata:`unity.events.event_bus.EVENT_BUS`
       singleton.  Until this function is called attempts to use
       ``EVENT_BUS`` raise a :class:`RuntimeError`.

    Parameters
    ----------
    assistant_record : dict | None
        The authoritative assistant dict (from the startup event). When provided,
        used directly as SESSION_DETAILS.assistant_record. When absent, the record
        is looked up via *assistant_id* from the Unify API — a matching assistant
        **must** exist or a ValueError is raised.
    """

    global _INITIALISED
    if _INITIALISED:
        return

    # 0. Validate LLM provider credentials are present
    from unity.settings import SETTINGS as _SETTINGS

    _SETTINGS.validate_llm_providers()

    # 1. Ensure Unify project is active
    if not unify.active_project():
        unify.activate(project_name, overwrite)

    # ── assistant validation & context selection ─────────────────────────
    if assistant_record:
        SESSION_DETAILS.assistant_record = assistant_record
    elif assistant_id is not None:
        assistants = _list_all_assistants()
        filtered = [a for a in assistants if a["agent_id"] == str(assistant_id)]
        if not filtered:
            raise ValueError(
                f"No assistant with agent_id={assistant_id} found among "
                f"{len(assistants)} assistants. Pass assistant_record explicitly "
                f"or ensure the assistant exists.",
            )
        SESSION_DETAILS.assistant_record = filtered[0]
    else:
        # No assistant specified — only acceptable in test/offline environments
        # where SESSION_DETAILS.assistant.id is already populated (provides
        # the fallback context path via assistant_context property).
        pass

    # 2. Set the Unify context using user_id/assistant_id (e.g., "42/7")
    full_ctx = f"{SESSION_DETAILS.user_context}/{SESSION_DETAILS.assistant_context}"

    # Idempotent context setup: tolerate concurrent creation from parallel processes
    # (e.g., pytest-xdist workers, CI parallelism, multi-instance deployments)
    try:
        unify.set_context(full_ctx)
    except Exception as e:
        if "already exists" in str(e).lower():
            unify.set_context(full_ctx, skip_create=True)
        else:
            raise

    ContextRegistry.setup()

    # 3. Bring up the global EventBus
    from .events import event_bus as _event_bus_mod

    _event_bus_mod._initialize_event_bus()

    # 4. Wire up LLM event hook to publish unillm events to EventBus
    from .events.llm_event_hook import install_llm_event_hook

    install_llm_event_hook()

    # 5. Wire up spending limit check hook
    from .spending_limits import install_limit_check_hook

    install_limit_check_hook()

    _INITIALISED = True


def ensure_initialised(
    project_name: str = "Assistants",
    assistant_id: Optional[int] = None,
    overwrite: bool = False,
    assistant_record: dict | None = None,
) -> None:
    """Ensure the runtime is initialised if no active read/write contexts exist.

    If both read and write contexts are already configured, this is a no-op.
    Otherwise, it calls :pyfunc:`init` to select an assistant and set a
    consistent context (e.g. "{user_id}/{assistant_id}") before any manager
    constructs its own sub-context (like "{user_id}/{assistant_id}/Contacts").
    """
    try:
        ctxs = unify.get_active_context()
        read_ctx = ctxs.get("read") if isinstance(ctxs, dict) else None
        write_ctx = ctxs.get("write") if isinstance(ctxs, dict) else None
    except Exception:
        read_ctx = write_ctx = None

    if read_ctx and write_ctx:
        return

    # Defer to the canonical initialiser which picks assistant + sets context
    init(
        project_name=project_name,
        assistant_id=assistant_id,
        overwrite=overwrite,
        assistant_record=assistant_record,
    )


# What the package exports at top-level
__all__ = ["init"]
