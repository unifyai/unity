"""
unity/session_details.py
=========================

Runtime details for the active assistant session.

Populated dynamically when the ConversationManager receives a StartupEvent.

Usage:
    from unity.session_details import SESSION_DETAILS

    # Check if initialized
    if SESSION_DETAILS.is_initialized:
        print(SESSION_DETAILS.assistant.name)

    # All fields have sensible defaults, so no `or "fallback"` is needed
    name = SESSION_DETAILS.assistant.name  # defaults to "assistant"
"""

import os
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────────────
# Default values - single source of truth
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_ASSISTANT_ID = "default-assistant"
DEFAULT_ASSISTANT_NAME = "assistant"
DEFAULT_USER_ID = "default"
DEFAULT_VOICE_PROVIDER = "cartesia"
DEFAULT_VOICE_MODE = "tts"


@dataclass
class AssistantDetails:
    """Details about the assistant."""

    id: str = DEFAULT_ASSISTANT_ID
    name: str = DEFAULT_ASSISTANT_NAME
    age: str = ""
    nationality: str = ""
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
    whatsapp_number: str = ""
    email: str = ""


@dataclass
class VoiceConfig:
    """Voice configuration for the session."""

    provider: str = DEFAULT_VOICE_PROVIDER
    id: str = ""
    mode: str = DEFAULT_VOICE_MODE


@dataclass
class SessionDetails:
    """Runtime details populated by ConversationManager on startup.

    Hierarchical structure with sub-containers for assistant, user, and voice.
    All fields have sensible defaults so callers never need `or "fallback"` patterns.
    """

    assistant: AssistantDetails = field(default_factory=AssistantDetails)
    user: UserDetails = field(default_factory=UserDetails)
    voice: VoiceConfig = field(default_factory=VoiceConfig)

    _initialized: bool = field(default=False, repr=False)

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
        assistant_about: str = "",
        assistant_number: str = "",
        assistant_email: str = "",
        assistant_contact_id: int = 0,
        user_id: str = "",
        user_name: str = "",
        user_number: str = "",
        user_whatsapp_number: str = "",
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
        self.assistant.about = assistant_about
        self.assistant.number = assistant_number
        self.assistant.email = assistant_email
        self.assistant.contact_id = assistant_contact_id
        self.user.id = user_id
        self.user.name = user_name
        self.user.number = user_number
        self.user.whatsapp_number = user_whatsapp_number
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
        self._initialized = False

    def export_to_env(self) -> None:
        """Export current values to environment variables.

        Called after populate() to ensure subprocesses can inherit values.
        """
        os.environ["ASSISTANT_ID"] = self.assistant.id
        os.environ["ASSISTANT_NAME"] = self.assistant.name
        os.environ["ASSISTANT_AGE"] = self.assistant.age
        os.environ["ASSISTANT_NATIONALITY"] = self.assistant.nationality
        os.environ["ASSISTANT_ABOUT"] = self.assistant.about
        os.environ["ASSISTANT_NUMBER"] = self.assistant.number
        os.environ["ASSISTANT_EMAIL"] = self.assistant.email
        os.environ["USER_ID"] = self.user.id
        os.environ["USER_NAME"] = self.user.name
        os.environ["USER_NUMBER"] = self.user.number
        os.environ["USER_WHATSAPP_NUMBER"] = self.user.whatsapp_number
        os.environ["USER_EMAIL"] = self.user.email
        os.environ["VOICE_PROVIDER"] = self.voice.provider
        os.environ["VOICE_ID"] = self.voice.id
        os.environ["VOICE_MODE"] = self.voice.mode

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
        if val := os.environ.get("USER_WHATSAPP_NUMBER"):
            self.user.whatsapp_number = val
        if val := os.environ.get("USER_EMAIL"):
            self.user.email = val
        if val := os.environ.get("VOICE_PROVIDER"):
            self.voice.provider = val
        if val := os.environ.get("VOICE_ID"):
            self.voice.id = val
        if val := os.environ.get("VOICE_MODE"):
            self.voice.mode = val
        self._initialized = True


# Global singleton instance
SESSION_DETAILS = SessionDetails()
