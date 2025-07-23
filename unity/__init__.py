import os
import requests
import unify
from .helpers import _handle_exceptions

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
        url = f"{os.environ['UNIFY_BASE_URL']}/assistant?"
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
    assistant_id: int = 0,
    overwrite: bool = False,
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
        if not (0 <= assistant_id < len(assistants)):
            raise ValueError(
                f"assistant_id {assistant_id} out of range – "
                f"{len(assistants)} assistants available.",
            )
        ASSISTANT = assistants[assistant_id]
    else:
        # No assistants returned (offline / stub environment) – leave None.
        ASSISTANT = None

    # 2. Set the assistant context *after* validation
    unify.set_context(str(assistant_id))

    # 3. Bring up the global EventBus
    from .events import event_bus as _event_bus_mod

    _event_bus_mod._initialize_event_bus()

    _INITIALISED = True


# What the package exports at top-level
__all__ = ["init", "ASSISTANT"]
