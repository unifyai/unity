from enum import StrEnum


class ActivatedBy(StrEnum):
    """
    Reason a task instance transitioned into the active state.

    This value is set automatically at activation time and never edited directly.
    """

    schedule = "schedule"  # start_at reached (one-off or recurring)
    queue = "queue"  # previous task completed; next in queue activated
    trigger = "trigger"  # inbound trigger event occurred
    explicit = "explicit"  # user explicitly requested immediate start
