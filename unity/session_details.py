"""
unity/session_details.py
=========================

Runtime details for the active assistant session.

This differs from unity.settings (SETTINGS) in a key way:
  - SETTINGS: Static configuration from environment/.env, frozen at import time
  - SESSION_DETAILS: Dynamic runtime state, populated when a session starts

Populated dynamically when the ConversationManager receives a StartupEvent.

Usage:
    from unity.session_details import SESSION_DETAILS

    # Check if initialized
    if SESSION_DETAILS.is_initialized:
        print(SESSION_DETAILS.assistant.first_name)

    # All fields have sensible defaults, so no `or "fallback"` is needed
    name = SESSION_DETAILS.assistant.name  # computed from first_name + surname
"""

import os
from dataclasses import dataclass, field

# ─────────────────────────────────────────────────────────────────────────────
# Unassigned Identity Sentinels
# Used by idle containers that haven't been assigned a real assistant/user yet.
# ─────────────────────────────────────────────────────────────────────────────
UNASSIGNED_USER_ID = "default"

# ─────────────────────────────────────────────────────────────────────────────
# Placeholder Contact Details
# Used in tests/offline mode when no real profile exists.
# ─────────────────────────────────────────────────────────────────────────────
PLACEHOLDER_ASSISTANT_FIRST_NAME = "Default"
PLACEHOLDER_ASSISTANT_SURNAME = "Assistant"
PLACEHOLDER_ASSISTANT_EMAIL = "assistant@unify.ai"
PLACEHOLDER_ASSISTANT_PHONE = "+10000000000"
PLACEHOLDER_ASSISTANT_BIO = "Your helpful AI assistant."
PLACEHOLDER_USER_FIRST_NAME = "Default"
PLACEHOLDER_USER_SURNAME = "User"
PLACEHOLDER_USER_EMAIL = "user@example.com"

# ─────────────────────────────────────────────────────────────────────────────
# Context Path Defaults (for Unify context hierarchy)
# Format: {user_id}/{agent_id}/... e.g., "default/0/Contacts"
# ─────────────────────────────────────────────────────────────────────────────
UNASSIGNED_USER_CONTEXT = UNASSIGNED_USER_ID
UNASSIGNED_ASSISTANT_CONTEXT = "0"

# ─────────────────────────────────────────────────────────────────────────────
# Voice Defaults
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_VOICE_PROVIDER = "cartesia"
DEFAULT_VOICE_MODE = "tts"


def _encode_int_csv(values: list[int]) -> str:
    """Encode integer ids for env vars that can only carry plain strings."""
    return ",".join(str(value) for value in values) if values else ""


def _decode_int_csv(value: str) -> list[int]:
    """Decode an integer-id CSV env var."""
    return [int(item) for item in value.split(",") if item.strip()]


@dataclass
class AssistantDetails:
    """Details about the assistant."""

    agent_id: int | None = None
    binding_id: str = ""
    first_name: str = ""
    surname: str = ""
    age: str = ""
    nationality: str = ""
    timezone: str = ""  # IANA timezone identifier (e.g., "America/New_York")
    about: str = ""
    job_title: str = ""
    number: str = ""
    email: str = ""
    email_provider: str = "google_workspace"
    whatsapp_number: str = ""
    discord_bot_id: str = ""
    contact_id: int = 0  # Contact ID in Contacts table
    desktop_mode: str = "ubuntu"  # "ubuntu" or "windows" - determines VM type
    desktop_url: str | None = None  # URL for managed VM desktop access
    user_desktop_mode: str | None = (
        None  # "ubuntu", "windows", or "macos" if user has own desktop
    )
    user_desktop_filesys_sync: bool = False  # Whether to sync files to user's desktop
    user_desktop_url: str | None = (
        None  # URL for user's own desktop (not the managed VM)
    )
    space_ids: list[int] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"{self.first_name} {self.surname}".strip()


@dataclass
class UserDetails:
    """Details about the user (boss)."""

    id: str = UNASSIGNED_USER_ID
    first_name: str = ""
    surname: str = ""
    number: str = ""
    email: str = ""
    whatsapp_number: str = ""
    contact_id: int = 1  # Contact ID in Contacts table

    @property
    def name(self) -> str:
        return f"{self.first_name} {self.surname}".strip()


@dataclass
class OrgDetails:
    """Details about the organization (when in org context).

    None/empty values indicate personal (non-org) context.
    """

    id: int | None = None  # Organization ID, None for personal context
    name: str = ""  # Organization name


@dataclass
class TeamDetails:
    """Details about team memberships within the current org.

    A user can belong to multiple teams.  ``ids`` contains all team IDs
    the user is a member of (empty list for personal / no-team context).
    """

    ids: list[int] = field(default_factory=list)


@dataclass
class VoiceConfig:
    """Voice configuration for the session."""

    provider: str = DEFAULT_VOICE_PROVIDER
    id: str = ""
    mode: str = DEFAULT_VOICE_MODE


@dataclass
class VoiceCallConfig:
    """Runtime configuration for an active voice call.

    Populated by configure_from_cli() in medium_scripts/common.py
    when a call agent subprocess is launched.
    """

    outbound: bool = False
    channel: str = ""  # "phone" or "meet"
    contact_json: str = "{}"  # JSON-serialized contact dict
    boss_json: str = "{}"  # JSON-serialized boss dict


@dataclass
class SessionDetails:
    """Runtime details populated by ConversationManager on startup.

    Hierarchical structure with sub-containers for assistant, user, and voice.
    All fields have sensible defaults so callers never need `or "fallback"` patterns.

    API keys (`unify_key`, `shared_unify_key`) are lazy properties that fall back
    to environment variables if not explicitly set. This mirrors how the `unify`
    package handles UNIFY_KEY and provides automatic access without requiring
    explicit `populate_from_env()` calls.
    """

    assistant: AssistantDetails = field(default_factory=AssistantDetails)
    user: UserDetails = field(default_factory=UserDetails)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    voice_call: VoiceCallConfig = field(default_factory=VoiceCallConfig)
    _unify_key: str = field(default="", repr=False)
    _shared_unify_key: str = field(default="", repr=False)

    # Organization context (None id for personal/non-org context)
    org: OrgDetails = field(default_factory=OrgDetails)

    # Team memberships within the org
    team: TeamDetails = field(default_factory=TeamDetails)

    _initialized: bool = field(default=False, repr=False)

    @property
    def assistant_context(self) -> str:
        """The assistant's agent_id as the context path component.

        Used for Unify context paths like '{user_id}/{agent_id}/...'.
        """
        if self.assistant.agent_id is not None:
            return str(self.assistant.agent_id)
        return UNASSIGNED_ASSISTANT_CONTEXT

    @property
    def user_context(self) -> str:
        """The user's ID used as the context path component.

        Used for Unify context paths like '{user_id}/{assistant_id}/...'.
        """
        return self.user.id or UNASSIGNED_USER_CONTEXT

    @property
    def is_initialized(self) -> bool:
        """Returns True if populate() has been called."""
        return self._initialized

    @property
    def user_id(self) -> str:
        """Shortcut to user.id for convenient access."""
        return self.user.id

    @property
    def org_id(self) -> int | None:
        """Shortcut to org.id for convenient access."""
        return self.org.id

    @org_id.setter
    def org_id(self, value: int | None) -> None:
        self.org.id = value

    @property
    def org_name(self) -> str:
        """Shortcut to org.name for convenient access."""
        return self.org.name

    @org_name.setter
    def org_name(self, value: str) -> None:
        self.org.name = value

    @property
    def team_ids(self) -> list[int]:
        """Shortcut to team.ids for convenient access."""
        return self.team.ids

    @team_ids.setter
    def team_ids(self, value: list[int]) -> None:
        self.team.ids = value

    @property
    def space_ids(self) -> list[int]:
        """Shortcut to assistant.space_ids for convenient access."""
        return self.assistant.space_ids

    @space_ids.setter
    def space_ids(self, value: list[int]) -> None:
        self.assistant.space_ids = value

    @property
    def unify_key(self) -> str:
        """API key for Unify services.

        Falls back to UNIFY_KEY environment variable if not explicitly set.
        This mirrors how the `unify` package handles API key resolution.
        """
        if self._unify_key:
            return self._unify_key
        return os.environ.get("UNIFY_KEY", "")

    @unify_key.setter
    def unify_key(self, value: str) -> None:
        self._unify_key = value

    @property
    def shared_unify_key(self) -> str:
        """Shared API key for cross-assistant operations.

        Falls back to SHARED_UNIFY_KEY environment variable if not explicitly set.
        """
        if self._shared_unify_key:
            return self._shared_unify_key
        return os.environ.get("SHARED_UNIFY_KEY", "")

    @shared_unify_key.setter
    def shared_unify_key(self, value: str) -> None:
        self._shared_unify_key = value

    def populate(
        self,
        *,
        agent_id: int | None = None,
        assistant_first_name: str = "",
        assistant_surname: str = "",
        assistant_age: str = "",
        assistant_nationality: str = "",
        assistant_timezone: str = "",
        assistant_about: str = "",
        assistant_job_title: str = "",
        assistant_number: str = "",
        assistant_email: str = "",
        assistant_email_provider: str = "google_workspace",
        assistant_whatsapp_number: str = "",
        assistant_discord_bot_id: str = "",
        assistant_contact_id: int = 0,
        user_id: str = "",
        user_first_name: str = "",
        user_surname: str = "",
        user_number: str = "",
        user_email: str = "",
        user_whatsapp_number: str = "",
        org_id: int | None = None,
        org_name: str = "",
        team_ids: list[int] | None = None,
        space_ids: list[int] | None = None,
        voice_provider: str = "",
        voice_id: str = "",
        binding_id: str = "",
        desktop_mode: str = "ubuntu",
        user_desktop_mode: str | None = None,
        user_desktop_filesys_sync: bool = False,
        user_desktop_url: str | None = None,
    ) -> None:
        """Populate the session with runtime values.

        Called by ConversationManager when a StartupEvent is received.
        """
        self.assistant.agent_id = agent_id
        self.assistant.first_name = assistant_first_name
        self.assistant.surname = assistant_surname
        self.assistant.age = assistant_age
        self.assistant.nationality = assistant_nationality
        self.assistant.timezone = assistant_timezone
        self.assistant.about = assistant_about
        self.assistant.job_title = assistant_job_title
        self.assistant.number = assistant_number
        self.assistant.email = assistant_email
        self.assistant.email_provider = assistant_email_provider
        self.assistant.whatsapp_number = assistant_whatsapp_number
        self.assistant.discord_bot_id = assistant_discord_bot_id
        self.assistant.contact_id = assistant_contact_id
        self.assistant.binding_id = binding_id
        self.assistant.desktop_mode = desktop_mode
        self.assistant.user_desktop_mode = user_desktop_mode
        self.assistant.user_desktop_filesys_sync = user_desktop_filesys_sync
        self.assistant.user_desktop_url = user_desktop_url
        self.user.id = user_id
        self.user.first_name = user_first_name
        self.user.surname = user_surname
        self.user.number = user_number
        self.user.email = user_email
        self.user.whatsapp_number = user_whatsapp_number
        self.org.id = org_id
        self.org.name = org_name
        self.team.ids = team_ids or []
        self.assistant.space_ids = space_ids or []
        self.voice.provider = voice_provider
        self.voice.id = voice_id
        self._initialized = True

    def reset(self) -> None:
        """Reset to default state (useful for tests)."""
        self.assistant = AssistantDetails()
        self.user = UserDetails()
        self.org = OrgDetails()
        self.team = TeamDetails()
        self.voice = VoiceConfig()
        self.voice_call = VoiceCallConfig()
        self._unify_key = ""
        self._shared_unify_key = ""
        self._initialized = False

    def export_to_env(self) -> None:
        """Export current values to environment variables.

        Called after populate() to ensure subprocesses can inherit values.
        """
        os.environ["ASSISTANT_ID"] = (
            str(self.assistant.agent_id) if self.assistant.agent_id is not None else ""
        )
        os.environ["ASSISTANT_FIRST_NAME"] = self.assistant.first_name
        os.environ["ASSISTANT_SURNAME"] = self.assistant.surname
        os.environ["ASSISTANT_NAME"] = self.assistant.name
        os.environ["ASSISTANT_AGE"] = self.assistant.age
        os.environ["ASSISTANT_NATIONALITY"] = self.assistant.nationality
        os.environ["ASSISTANT_TIMEZONE"] = self.assistant.timezone
        os.environ["ASSISTANT_ABOUT"] = self.assistant.about
        os.environ["ASSISTANT_JOB_TITLE"] = self.assistant.job_title
        os.environ["ASSISTANT_NUMBER"] = self.assistant.number
        os.environ["ASSISTANT_EMAIL"] = self.assistant.email
        os.environ["ASSISTANT_EMAIL_PROVIDER"] = self.assistant.email_provider
        os.environ["ASSISTANT_WHATSAPP_NUMBER"] = self.assistant.whatsapp_number
        os.environ["ASSISTANT_DESKTOP_MODE"] = self.assistant.desktop_mode
        os.environ["ASSISTANT_DESKTOP_URL"] = self.assistant.desktop_url or ""
        os.environ["ASSISTANT_USER_DESKTOP_MODE"] = (
            self.assistant.user_desktop_mode or ""
        )
        os.environ["ASSISTANT_USER_DESKTOP_FILESYS_SYNC"] = str(
            self.assistant.user_desktop_filesys_sync,
        )
        os.environ["ASSISTANT_USER_DESKTOP_URL"] = self.assistant.user_desktop_url or ""
        os.environ["USER_ID"] = self.user.id
        os.environ["USER_FIRST_NAME"] = self.user.first_name
        os.environ["USER_SURNAME"] = self.user.surname
        os.environ["USER_NAME"] = self.user.name
        os.environ["USER_NUMBER"] = self.user.number
        os.environ["USER_EMAIL"] = self.user.email
        os.environ["USER_WHATSAPP_NUMBER"] = self.user.whatsapp_number
        os.environ["ORG_ID"] = str(self.org.id) if self.org.id is not None else ""
        os.environ["ORG_NAME"] = self.org.name
        os.environ["TEAM_IDS"] = _encode_int_csv(self.team.ids)
        self.export_space_ids_to_env()
        os.environ["VOICE_PROVIDER"] = self.voice.provider
        os.environ["VOICE_ID"] = self.voice.id
        os.environ["VOICE_MODE"] = self.voice.mode
        # Voice call config (for call agent subprocesses)
        os.environ["OUTBOUND"] = str(self.voice_call.outbound)
        os.environ["CHANNEL"] = self.voice_call.channel
        os.environ["CONTACT"] = self.voice_call.contact_json
        os.environ["BOSS"] = self.voice_call.boss_json
        os.environ["UNIFY_KEY"] = self.unify_key

    def export_space_ids_to_env(self) -> None:
        """Export current shared-space memberships to the subprocess env shape."""
        os.environ["SPACE_IDS"] = _encode_int_csv(self.assistant.space_ids)

    def populate_from_env(self) -> None:
        """Populate from environment variables.

        Useful for subprocesses that inherit env vars from parent process.
        Only sets fields if the corresponding env var is non-empty.
        """
        if val := os.environ.get("ASSISTANT_ID"):
            try:
                self.assistant.agent_id = int(val)
            except (ValueError, TypeError):
                pass
        if val := os.environ.get("ASSISTANT_FIRST_NAME"):
            self.assistant.first_name = val
        if val := os.environ.get("ASSISTANT_SURNAME"):
            self.assistant.surname = val
        if val := os.environ.get("ASSISTANT_AGE"):
            self.assistant.age = val
        if val := os.environ.get("ASSISTANT_NATIONALITY"):
            self.assistant.nationality = val
        if val := os.environ.get("ASSISTANT_TIMEZONE"):
            self.assistant.timezone = val
        if val := os.environ.get("ASSISTANT_ABOUT"):
            self.assistant.about = val
        if val := os.environ.get("ASSISTANT_JOB_TITLE"):
            self.assistant.job_title = val
        if val := os.environ.get("ASSISTANT_NUMBER"):
            self.assistant.number = val
        if val := os.environ.get("ASSISTANT_EMAIL"):
            self.assistant.email = val
        if val := os.environ.get("ASSISTANT_EMAIL_PROVIDER"):
            self.assistant.email_provider = val
        if val := os.environ.get("ASSISTANT_WHATSAPP_NUMBER"):
            self.assistant.whatsapp_number = val
        if val := os.environ.get("ASSISTANT_DISCORD_BOT_ID"):
            self.assistant.discord_bot_id = val
        if val := os.environ.get("ASSISTANT_CONTACT_ID"):
            try:
                self.assistant.contact_id = int(val)
            except ValueError:
                pass
        if val := os.environ.get("ASSISTANT_DESKTOP_MODE"):
            self.assistant.desktop_mode = val
        if val := os.environ.get("ASSISTANT_DESKTOP_URL"):
            self.assistant.desktop_url = val if val else None
        if val := os.environ.get("ASSISTANT_USER_DESKTOP_MODE"):
            self.assistant.user_desktop_mode = val if val else None
        if val := os.environ.get("ASSISTANT_USER_DESKTOP_FILESYS_SYNC"):
            self.assistant.user_desktop_filesys_sync = val == "True"
        if val := os.environ.get("ASSISTANT_USER_DESKTOP_URL"):
            self.assistant.user_desktop_url = val if val else None
        if val := os.environ.get("USER_ID"):
            self.user.id = val
        if val := os.environ.get("USER_FIRST_NAME"):
            self.user.first_name = val
        if val := os.environ.get("USER_SURNAME"):
            self.user.surname = val
        if val := os.environ.get("USER_NUMBER"):
            self.user.number = val
        if val := os.environ.get("USER_EMAIL"):
            self.user.email = val
        if val := os.environ.get("USER_WHATSAPP_NUMBER"):
            self.user.whatsapp_number = val
        if val := os.environ.get("ORG_ID"):
            try:
                self.org.id = int(val)
            except ValueError:
                pass
        if val := os.environ.get("ORG_NAME"):
            self.org.name = val
        if val := os.environ.get("TEAM_IDS"):
            try:
                self.team.ids = _decode_int_csv(val)
            except (ValueError, TypeError):
                pass
        if val := os.environ.get("SPACE_IDS"):
            try:
                self.assistant.space_ids = _decode_int_csv(val)
            except (ValueError, TypeError):
                pass
        if val := os.environ.get("VOICE_PROVIDER"):
            self.voice.provider = val
        if val := os.environ.get("VOICE_ID"):
            self.voice.id = val
        if val := os.environ.get("VOICE_MODE"):
            self.voice.mode = val
        # Voice call config (for call agent subprocesses)
        if val := os.environ.get("OUTBOUND"):
            self.voice_call.outbound = val == "True"
        if val := os.environ.get("CHANNEL"):
            self.voice_call.channel = val
        if val := os.environ.get("CONTACT"):
            self.voice_call.contact_json = val
        if val := os.environ.get("BOSS"):
            self.voice_call.boss_json = val
        # General config
        if val := os.environ.get("SHARED_UNIFY_KEY"):
            self.shared_unify_key = val
        if val := os.environ.get("UNIFY_KEY"):
            self.unify_key = val
        self._initialized = True

    def get_subprocess_env(self, **overrides: str) -> dict[str, str]:
        """Get a copy of the current environment for subprocess use.

        This is the only approved way to access os.environ for subprocess
        communication. First exports SESSION_DETAILS values, then returns
        a copy with any overrides applied.

        Args:
            **overrides: Key-value pairs to add/override in the env dict.

        Returns:
            A dict suitable for passing to subprocess.run(env=...).
        """
        self.export_to_env()
        env = dict(os.environ)
        env.update(overrides)
        return env

    @staticmethod
    def get_impl_setting(name: str, default: str = "real") -> str:
        """Get implementation setting from environment for test-time override.

        SETTINGS is instantiated at import time, before test conftests can set
        environment variables. This helper reads directly from os.environ to
        allow test_conversation_manager/conftest.py to set UNITY_*_IMPL=simulated
        and have it take effect.

        This is intentionally a static method since it doesn't depend on session
        state and is only used during manager initialization.

        Args:
            name: The env var name (e.g., "UNITY_CONTACT_IMPL").
            default: Default value if env var is not set.

        Returns:
            The implementation type ("real" or "simulated").
        """
        return os.environ.get(name, default)


# Global singleton instance
SESSION_DETAILS = SessionDetails()
