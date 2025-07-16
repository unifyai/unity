import unify

# ---------------------------------------------------------------------------
# Lazy runtime initialisation
# ---------------------------------------------------------------------------

_INITIALISED = False


def init(project_name: str = "Assistants") -> None:  # noqa: D401 – imperative name
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

    # 1. Ensure Unify project is active ------------------------------------
    if not unify.active_project():
        unify.activate(project_name)

    # 2. Bring up the global EventBus --------------------------------------
    from .events import event_bus as _event_bus_mod

    _event_bus_mod._initialize_event_bus()

    _INITIALISED = True


# What the package exports at top-level
__all__ = ["init"]
