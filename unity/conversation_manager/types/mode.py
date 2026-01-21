from enum import StrEnum


class Mode(StrEnum):
    """
    Enumeration of ConversationManager operational modes.

    Modes determine how the ConversationManager handles communication:
    - CALL: Phone call voice mode
    - UNIFY_MEET: Unify Meet voice/video mode
    - TEXT: Asynchronous text-based communication (SMS, email, Unify messages)
    """

    CALL = "call"
    UNIFY_MEET = "unify_meet"
    TEXT = "text"

    @classmethod
    def voice_modes(cls) -> tuple["Mode", ...]:
        """Return all voice-based modes."""
        return (cls.CALL, cls.UNIFY_MEET)

    @property
    def is_voice(self) -> bool:
        """Check if this mode is a voice mode."""
        return self in self.voice_modes()


# Export valid values for validation/random selection
VALID_MODES: tuple[str, ...] = tuple(m.value for m in Mode)
