import os
from typing import Optional

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


def _truthy(env: str, default: bool = True) -> bool:
    v = os.getenv(env)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _configure_default_logging() -> None:
    """Apply safe, idempotent default logging rules.

    Defaults:
      - Show `unity` logs at INFO.
      - Hide noisy third-party HTTP clients (httpx/urllib3/openai) unless opted-in.
      - By default, include-only project logs (unity*) unless overridden via env.

    Env flags (all optional):
      - UNITY_SILENCE_HTTPX=true|false (default true)
      - UNITY_SILENCE_URLLIB3=true|false (default true)
      - UNITY_SILENCE_OPENAI=true|false (default true)
      - UNITY_LOG_ONLY_PROJECT=true|false (default true)
      - UNITY_LOG_INCLUDE_PREFIXES="unity,unify_requests" (used when UNITY_LOG_ONLY_PROJECT=true)
    """
    if getattr(_configure_default_logging, "_done", False):
        return

    try:
        import logging

        # 1) Keep our project logs visible
        logging.getLogger("unity").setLevel(logging.INFO)

        # 2) Mute common HTTP client libraries by default
        if _truthy("UNITY_SILENCE_HTTPX", True):
            logging.getLogger("httpx").setLevel(logging.WARNING)
        if _truthy("UNITY_SILENCE_URLLIB3", True):
            logging.getLogger("urllib3").setLevel(logging.WARNING)
        if _truthy("UNITY_SILENCE_OPENAI", True):
            logging.getLogger("openai").setLevel(logging.WARNING)

        # 3) Optional include-only filter (default: enabled per request)
        if _truthy("UNITY_LOG_ONLY_PROJECT", True):
            allow_raw = os.getenv("UNITY_LOG_INCLUDE_PREFIXES", "unity")
            allow = tuple(s.strip() for s in allow_raw.split(",") if s.strip())

            class _OnlyProject(logging.Filter):
                def filter(self, record: "logging.LogRecord") -> bool:  # type: ignore[name-defined]
                    name = record.name or ""
                    # exact match or child logger of any allowed prefix
                    return any(name == p or name.startswith(p + ".") for p in allow)

            root = logging.getLogger()
            # Attach to root so future handlers are also filtered
            root.addFilter(_OnlyProject())
            # And to any already-present handlers to be thorough
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

_INITIALISED = False
ASSISTANT = None  # Will hold the selected assistant record once init() runs


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

    global _INITIALISED, ASSISTANT
    if _INITIALISED:
        return

    # 1. Ensure Unify project is active
    if not unify.active_project():
        unify.activate(project_name, overwrite)

    # ── assistant validation & context selection ─────────────────────────
    assistants = _list_all_assistants()

    if assistants:
        if not default_assistant:
            if assistant_id is None:
                ASSISTANT = assistants[0]
            else:
                filtered_assistants = [
                    assistant
                    for assistant in assistants
                    if assistant["agent_id"] == str(assistant_id)
                ]
                ASSISTANT = filtered_assistants[0] if filtered_assistants else None
        else:
            ASSISTANT = default_assistant
        first_name = "".join(
            [chnk.capitalize() for chnk in ASSISTANT["first_name"].split(" ")],
        )
        surname = "".join(
            [chnk.capitalize() for chnk in ASSISTANT["surname"].split(" ")],
        )
        ctx = first_name + surname
    else:
        # No assistants returned or explicitly passed (offline)
        ASSISTANT = default_assistant
        ctx = "Assistant"

    # 2. Set the assistant context *after* validation
    unify.set_context(ctx)

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
    consistent context (e.g. "{AssistantName}") before any manager constructs
    its own sub-context (like "{AssistantName}/Contacts").
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
__all__ = ["init", "ASSISTANT"]
