"""
Guided Learning Module

Provides keyframe capture and activity tracking for guided learning scenarios
where a user demonstrates a workflow while narrating their actions.
"""

from .types import GuidedLearningStep, SpeechSegment, KeyframeEvent
from .guided_learning_manager import (
    GuidedLearningManager,
    GuidedLearningSettings,
    FrameCaptureMode,
    KeyframeSelectionMode,
)
from .input_listener import (
    InputEventListener,
    InputListenerSettings,
    InputEvent,
    InputEventType,
    Point,
)

__all__ = [
    "GuidedLearningManager",
    "GuidedLearningStep",
    "SpeechSegment",
    "KeyframeEvent",
    "GuidedLearningSettings",
    "FrameCaptureMode",
    "KeyframeSelectionMode",
    "InputEventListener",
    "InputListenerSettings",
    "InputEvent",
    "InputEventType",
    "Point",
]
