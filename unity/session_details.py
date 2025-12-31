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
        print(SESSION_DETAILS.assistant.name)

    # All fields have sensible defaults, so no `or "fallback"` is needed
    name = SESSION_DETAILS.assistant.name  # empty string until populated
"""

import os
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────────────
# Default Assistant Identity
# Used when no real assistant is configured (offline mode, tests)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_ASSISTANT_ID = "default-assistant"
DEFAULT_ASSISTANT_FIRST_NAME = "Default"
DEFAULT_ASSISTANT_SURNAME = "Assistant"
DEFAULT_ASSISTANT_EMAIL = "assistant@unify.ai"
DEFAULT_ASSISTANT_PHONE = "+10000000000"
DEFAULT_ASSISTANT_BIO = "Your helpful AI assistant."

# ─────────────────────────────────────────────────────────────────────────────
# Default User Identity
# Used when no real user is configured (offline mode, tests)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_USER_ID = "default"
DEFAULT_USER_FIRST_NAME = "Default"
DEFAULT_USER_SURNAME = "User"
DEFAULT_USER_EMAIL = "user@example.com"

# ─────────────────────────────────────────────────────────────────────────────
# Context Path Defaults (for Unify context hierarchy)
# Format: {UserContext}/{AssistantContext}/... e.g., "DefaultUser/DefaultAssistant/Contacts"
# Values derived from {FirstName}{Surname} for consistency
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_USER_CONTEXT = "DefaultUser"
DEFAULT_ASSISTANT_CONTEXT = "DefaultAssistant"

# ─────────────────────────────────────────────────────────────────────────────
# Voice Defaults
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_VOICE_PROVIDER = "cartesia"
DEFAULT_VOICE_MODE = "tts"


@dataclass
class AssistantDetails:
    """Details about the assistant."""

    id: str = DEFAULT_ASSISTANT_ID
    name: str = ""  # Populated from StartupEvent; empty = use defaults
    age: str = ""
    nationality: str = ""
    timezone: str = ""  # IANA timezone identifier (e.g., "America/New_York")
    about: str = ""
    number: str = ""
    email: str = ""
    contact_id: int = 0  # Contact ID in Contacts table


@dataclass
class UserDetails:
    """Details about the user (boss)."""

    id: str = DEFAULT_USER_ID
    name: str = ""
    number: str = ""
    email: str = ""
    contact_id: int = 1  # Contact ID in Contacts table


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
    """

    assistant: AssistantDetails = field(default_factory=AssistantDetails)
    user: UserDetails = field(default_factory=UserDetails)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    voice_call: VoiceCallConfig = field(default_factory=VoiceCallConfig)
    unify_key: str = ""
    shared_unify_key: str = ""

    # Raw assistant record from Unify API (for contexts that need the full dict)
    assistant_record: dict | None = field(default=None, repr=False)

    _initialized: bool = field(default=False, repr=False)

    @property
    def assistant_context(self) -> str:
        """Derived context string for the assistant (e.g., 'JohnSmith').

        Used for Unify context paths like '{user_context}/{assistant_context}/...'.
        """
        # Prefer deriving from assistant_record if available (has first_name/surname)
        if self.assistant_record:
            first = self.assistant_record.get("first_name") or ""
            surname = self.assistant_record.get("surname") or ""
            if first or surname:
                first_part = "".join(chunk.capitalize() for chunk in first.split())
                surname_part = "".join(chunk.capitalize() for chunk in surname.split())
                return first_part + surname_part
        # Fall back to assistant.name if populated
        name = self.assistant.name
        if name:
            return "".join(chunk.capitalize() for chunk in name.split())
        return DEFAULT_ASSISTANT_CONTEXT

    @property
    def user_context(self) -> str:
        """Derived context string for the user (e.g., 'JohnDoe').

        Used for Unify context paths like '{user_context}/{assistant_context}/...'.
        """
        name = self.user.name
        if name:
            return "".join(chunk.capitalize() for chunk in name.split())
        return DEFAULT_USER_CONTEXT

    @property
    def is_initialized(self) -> bool:
        """Returns True if populate() has been called."""
        return self._initialized

    def populate(
        self,
        *,
        assistant_id: str = "",
        assistant_name: str = "",
        assistant_age: str = "",
        assistant_nationality: str = "",
        assistant_timezone: str = "",
        assistant_about: str = "",
        assistant_number: str = "",
        assistant_email: str = "",
        assistant_contact_id: int = 0,
        user_id: str = "",
        user_name: str = "",
        user_number: str = "",
        user_email: str = "",
        voice_provider: str = "",
        voice_id: str = "",
        voice_mode: str = "",
    ) -> None:
        """Populate the session with runtime values.

        Called by ConversationManager when a StartupEvent is received.
        """
        self.assistant.id = assistant_id
        self.assistant.name = assistant_name
        self.assistant.age = assistant_age
        self.assistant.nationality = assistant_nationality
        self.assistant.timezone = assistant_timezone
        self.assistant.about = assistant_about
        self.assistant.number = assistant_number
        self.assistant.email = assistant_email
        self.assistant.contact_id = assistant_contact_id
        self.user.id = user_id
        self.user.name = user_name
        self.user.number = user_number
        self.user.email = user_email
        self.voice.provider = voice_provider
        self.voice.id = voice_id
        self.voice.mode = voice_mode
        self._initialized = True

    def reset(self) -> None:
        """Reset to default state (useful for tests)."""
        self.assistant = AssistantDetails()
        self.user = UserDetails()
        self.voice = VoiceConfig()
        self.voice_call = VoiceCallConfig()
        self.unify_key = ""
        self.shared_unify_key = ""
        self.assistant_record = None
        self._initialized = False

    def export_to_env(self) -> None:
        """Export current values to environment variables.

        Called after populate() to ensure subprocesses can inherit values.
        """
        os.environ["ASSISTANT_ID"] = self.assistant.id
        os.environ["ASSISTANT_NAME"] = self.assistant.name
        os.environ["ASSISTANT_AGE"] = self.assistant.age
        os.environ["ASSISTANT_NATIONALITY"] = self.assistant.nationality
        os.environ["ASSISTANT_TIMEZONE"] = self.assistant.timezone
        os.environ["ASSISTANT_ABOUT"] = self.assistant.about
        os.environ["ASSISTANT_NUMBER"] = self.assistant.number
        os.environ["ASSISTANT_EMAIL"] = self.assistant.email
        os.environ["USER_ID"] = self.user.id
        os.environ["USER_NAME"] = self.user.name
        os.environ["USER_NUMBER"] = self.user.number
        os.environ["USER_EMAIL"] = self.user.email
        os.environ["VOICE_PROVIDER"] = self.voice.provider
        os.environ["VOICE_ID"] = self.voice.id
        os.environ["VOICE_MODE"] = self.voice.mode
        # Voice call config (for call agent subprocesses)
        os.environ["OUTBOUND"] = str(self.voice_call.outbound)
        os.environ["CHANNEL"] = self.voice_call.channel
        os.environ["CONTACT"] = self.voice_call.contact_json
        os.environ["BOSS"] = self.voice_call.boss_json
        os.environ["UNIFY_KEY"] = self.unify_key

    def populate_from_env(self) -> None:
        """Populate from environment variables.

        Useful for subprocesses that inherit env vars from parent process.
        Only sets fields if the corresponding env var is non-empty.
        """
        if val := os.environ.get("ASSISTANT_ID"):
            self.assistant.id = val
        if val := os.environ.get("ASSISTANT_NAME"):
            self.assistant.name = val
        if val := os.environ.get("ASSISTANT_AGE"):
            self.assistant.age = val
        if val := os.environ.get("ASSISTANT_NATIONALITY"):
            self.assistant.nationality = val
        if val := os.environ.get("ASSISTANT_TIMEZONE"):
            self.assistant.timezone = val
        if val := os.environ.get("ASSISTANT_ABOUT"):
            self.assistant.about = val
        if val := os.environ.get("ASSISTANT_NUMBER"):
            self.assistant.number = val
        if val := os.environ.get("ASSISTANT_EMAIL"):
            self.assistant.email = val
        if val := os.environ.get("ASSISTANT_CONTACT_ID"):
            try:
                self.assistant.contact_id = int(val)
            except ValueError:
                pass
        if val := os.environ.get("USER_ID"):
            self.user.id = val
        if val := os.environ.get("USER_NAME"):
            self.user.name = val
        if val := os.environ.get("USER_NUMBER"):
            self.user.number = val
        if val := os.environ.get("USER_EMAIL"):
            self.user.email = val
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
