from enum import Enum


class StateManager(Enum):
    """Enumeration of managers that support a clear() operation.

    Values correspond to Conductor attribute suffixes (without leading underscore).
    """

    CONTACTS = "contact_manager"
    TRANSCRIPTS = "transcript_manager"
    KNOWLEDGE = "knowledge_manager"
    TASKS = "task_scheduler"
    WEB_SEARCH = "web_searcher"

    # Planned integrations – exposed here for forward compatibility
    GUIDANCE = "guidance_manager"
    SECRETS = "secret_manager"
