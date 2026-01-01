"""
unity/__init__.py
==================

Package initialization for the Unity AI Assistant framework.

Importing this package performs one-time setup:
  - Configures logging to filter to unity.* loggers only

The runtime must be explicitly initialized via init() before using managers:

    import unity
    unity.init()  # Activates Unify project, selects assistant, starts EventBus

For code that may run before or after init(), use ensure_initialised() which
is a no-op if already initialized.

LLM I/O logging is now handled directly in the unillm package. Enable it via:
  - UNILLM_IO_LOG=true (to enable logging)
  - UNILLM_LOG_DIR=/path/to/logs (to set the output directory)
"""

from typing import Optional

from unity.common.context_registry import ContextRegistry

# Attempt to import the external 'unify' SDK. If unavailable, provide a minimal
# no-op shim so importing the 'unity' package does not require extra installs.
try:  # pragma: no cover - simple import guard
    import unify  # type: ignore
except Exception:  # ImportError or others

    class _UnifyShim:
        def set_client_direct_mode(self, *_args, **_kwargs) -> None:
            pass

        def active_project(self) -> bool:
            return False

        def activate(self, *_args, **_kwargs) -> None:
            pass

        def set_context(self, *_args, **_kwargs) -> None:
            pass

        def get_active_context(self) -> dict:
            return {}

    unify = _UnifyShim()  # type: ignore


# Set direct mode to True to avoid the overhead of the Unify API.
unify.set_client_direct_mode(True)


# ---------------------------------------------------------------------------
# Default logging hygiene
# ---------------------------------------------------------------------------

from unity.settings import SETTINGS as _SETTINGS


def _configure_default_logging() -> None:
    """Apply safe, idempotent default logging rules.

    Mutes verbose HTTP client libraries and filters to only show unity.* logs.
    """
    if getattr(_configure_default_logging, "_done", False):
        return

    try:
        import logging

        # 1) Keep our project logs visible
        logging.getLogger("unity").setLevel(logging.INFO)

        # 2) Mute common HTTP client libraries and LLM SDKs
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)
        logging.getLogger("LiteLLM Proxy").setLevel(logging.WARNING)
        logging.getLogger("LiteLLM Router").setLevel(logging.WARNING)

        # 3) Only show logs from unity.* loggers
        class _OnlyProject(logging.Filter):
            def filter(self, record: "logging.LogRecord") -> bool:  # type: ignore[name-defined]
                name = record.name or ""
                return name == "unity" or name.startswith("unity.")

        root = logging.getLogger()
        root.addFilter(_OnlyProject())
        for h in list(root.handlers):
            try:
                h.addFilter(_OnlyProject())
            except Exception:
                pass
    except Exception:
        # Never let logging setup crash imports
        pass

    _configure_default_logging._done = True  # type: ignore[attr-defined]


_configure_default_logging()


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
    default_assistant: dict | None = None,
) -> None:  # noqa: D401 – imperative name
    """Initialise the *unity* runtime.

    This performs two steps **once** per interpreter session:

    1. Activate the given *project_name* in the Unify SDK (unless a project is
       already active).
    2. Construct and wire-up the global :pydata:`unity.events.event_bus.EVENT_BUS`
       singleton.  Until this function is called attempts to use
       ``EVENT_BUS`` raise a :class:`RuntimeError`.
    """

    global _INITIALISED
    if _INITIALISED:
        return

    # 0. Validate LLM provider credentials are present
    _SETTINGS.validate_llm_providers()

    # 1. Ensure Unify project is active
    if not unify.active_project():
        unify.activate(project_name, overwrite)

    # ── assistant validation & context selection ─────────────────────────
    # Determine which assistant record to use and store in SESSION_DETAILS
    assistants = _list_all_assistants()

    if assistants:
        if not default_assistant:
            if assistant_id is None:
                SESSION_DETAILS.assistant_record = assistants[0]
            else:
                filtered_assistants = [
                    assistant
                    for assistant in assistants
                    if assistant["agent_id"] == str(assistant_id)
                ]
                SESSION_DETAILS.assistant_record = (
                    filtered_assistants[0] if filtered_assistants else None
                )
        else:
            SESSION_DETAILS.assistant_record = default_assistant
    else:
        # No assistants returned or explicitly passed (offline)
        SESSION_DETAILS.assistant_record = default_assistant

    # 2. Set the Unify context name using computed properties from SESSION_DETAILS
    # Context is now UserName/AssistantName (e.g., "JohnDoe/MyAssistant")
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

    _INITIALISED = True


def ensure_initialised(
    project_name: str = "Assistants",
    assistant_id: Optional[int] = None,
    overwrite: bool = False,
    default_assistant: dict | None = None,
) -> None:
    """Ensure the runtime is initialised if no active read/write contexts exist.

    If both read and write contexts are already configured, this is a no-op.
    Otherwise, it calls :pyfunc:`init` to select an assistant and set a
    consistent context (e.g. "{UserName}/{AssistantName}") before any manager
    constructs its own sub-context (like "{UserName}/{AssistantName}/Contacts").
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
        default_assistant=default_assistant,
    )


# What the package exports at top-level
__all__ = ["init"]
