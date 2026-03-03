"""
unity/__init__.py
==================

Package initialization for the Unity AI Assistant framework.

The runtime must be explicitly initialized via init() before using managers:

    import unity
    unity.init()  # Activates Unify project, sets context, starts EventBus

For code that may run before or after init(), use ensure_initialised() which
is a no-op if already initialized.

LLM I/O logging is now handled directly in the unillm package. Enable it via:
  - UNILLM_IO_LOG=true (to enable logging)
  - UNILLM_LOG_DIR=/path/to/logs (to set the output directory)

Logging is configured centrally in unity.logger (imported below).
"""

try:
    import onnxruntime as _ort

    _ort.set_default_logger_severity(
        4,
    )  # FATAL — suppress thread affinity noise in containers
except Exception:
    pass

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


def init(
    project_name: str = "Assistants",
    overwrite: bool = False,
) -> None:  # noqa: D401 – imperative name
    """Initialise the *unity* runtime.

    Reads SESSION_DETAILS.assistant.agent_id (set by the startup event) for
    the context path. All assistant identity and profile data lives on
    SESSION_DETAILS — this function only handles project activation,
    context setup, EventBus, and hooks.
    """

    global _INITIALISED
    if _INITIALISED:
        return

    from unity.settings import SETTINGS as _SETTINGS

    _SETTINGS.validate_llm_providers()

    if not unify.active_project():
        unify.activate(project_name, overwrite)

    # Set the Unify context using user_id/assistant_id (e.g., "42/7")
    full_ctx = f"{SESSION_DETAILS.user_context}/{SESSION_DETAILS.assistant_context}"

    # Idempotent context setup: tolerate concurrent creation from parallel processes
    try:
        unify.set_context(full_ctx)
    except Exception as e:
        if "already exists" in str(e).lower():
            unify.set_context(full_ctx, skip_create=True)
        else:
            raise

    ContextRegistry.setup()

    from .events import event_bus as _event_bus_mod

    _event_bus_mod._initialize_event_bus()

    from .events.llm_event_hook import install_llm_event_hook

    install_llm_event_hook()

    from .spending_limits import install_limit_check_hook

    install_limit_check_hook()

    _INITIALISED = True


def ensure_initialised(
    project_name: str = "Assistants",
    overwrite: bool = False,
) -> None:
    """Ensure the runtime is initialised if no active read/write contexts exist.

    If both read and write contexts are already configured, this is a no-op.
    Otherwise, it calls :pyfunc:`init` to set up project, context, and EventBus.
    """
    try:
        ctxs = unify.get_active_context()
        read_ctx = ctxs.get("read") if isinstance(ctxs, dict) else None
        write_ctx = ctxs.get("write") if isinstance(ctxs, dict) else None
    except Exception:
        read_ctx = write_ctx = None

    if read_ctx and write_ctx:
        return

    init(project_name=project_name, overwrite=overwrite)


# What the package exports at top-level
__all__ = ["init"]
