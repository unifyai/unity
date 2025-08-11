import os
import requests
import unify
from .helpers import _handle_exceptions
from typing import Optional

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
        url = f"{os.environ['UNIFY_BASE_URL']}/assistant"
        headers = {"Authorization": f"Bearer {os.environ['UNIFY_KEY']}"}
        response = requests.request("GET", url, headers=headers)
        _handle_exceptions(response)
        data = response.json()
        return data.get("info", []) if isinstance(data, dict) else []
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
