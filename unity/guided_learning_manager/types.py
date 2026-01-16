"""
Type definitions for the Guided Learning module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

from unity.image_manager.image_manager import ImageHandle

if TYPE_CHECKING:
    from unity.image_manager.types import AnnotatedImageRef


@dataclass
class SpeechSegment:
    """A single continuous speech segment from the user."""

    text: str
    start_time: float
    end_time: float

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class KeyframeEvent:
    """
    A visually significant frame detected during the activity window.

    This represents a moment where something meaningful changed on screen.
    """

    timestamp: float
    image_handle: ImageHandle

    # Optional: metrics that triggered the detection (for debugging)
    ssim_score: Optional[float] = None
    change_ratio: Optional[float] = None
    detection_reason: str = "visual_change"


@dataclass
class GuidedLearningStep:
    """
    A single logical step in the guided learning flow.

    The (transcript, keyframes) tuple should be LOSSLESS - meaning someone
    could reconstruct exactly what happened by reading the transcript and
    viewing the frames in order.

    Examples:
    ---------
    1. Commentary only (no visual change):
       transcript: "This is the admin dashboard"
       keyframes: [current_screen]  # Just context

    2. Simple action:
       transcript: "Now I click submit"
       keyframes: [before_click, after_click]

    3. Multi-step action:
       transcript: "I drag this file to documents, then rename it"
       keyframes: [initial, dragging, dropped, rename_dialog, final]

    4. Multi-utterance (natural pauses within one logical action):
       transcript: "First we... then... and finally..."
       keyframes: [frame1, frame2, frame3]
    """

    # Concatenated speech from all utterances in this step
    transcript: str

    # All visually significant frames, in chronological order
    # Could be empty (off-screen commentary), 1 (context), 2 (before/after), or N (multi-step)
    keyframes: List[KeyframeEvent] = field(default_factory=list)

    # Individual speech segments (for fine-grained analysis if needed)
    speech_segments: List[SpeechSegment] = field(default_factory=list)

    # Time boundaries of this step
    start_time: float = 0.0
    end_time: float = 0.0

    # Metadata flags
    has_visual_changes: bool = False
    is_commentary_only: bool = False

    # Optional: context frame captured at step start (even if no visual changes)
    context_frame: Optional[KeyframeEvent] = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def num_keyframes(self) -> int:
        return len(self.keyframes)

    def _get_annotation(self, kf: "KeyframeEvent") -> str:
        """
        Get the best annotation for a keyframe.

        Priority:
        1. LLM-generated detection_reason (if from LLM mode)
        2. ImageHandle's auto-caption (fallback for algorithmic mode)
        3. Generic detection_reason (last resort)
        """
        # If detection_reason has semantic content (from LLM mode), use it
        if kf.detection_reason and "llm_selected" in kf.detection_reason:
            return f"[t={kf.timestamp:.1f}s] {kf.detection_reason}"

        # Otherwise, try to use the ImageHandle's auto-generated caption
        caption = kf.image_handle.caption
        if caption:
            return f"[t={kf.timestamp:.1f}s] {caption}"

        # Last resort: use generic detection_reason
        return f"[t={kf.timestamp:.1f}s] {kf.detection_reason}"

    def to_actor_interject_args(self) -> tuple[str, List["AnnotatedImageRef"]]:
        """
        Returns (transcript, annotated_images) ready for HierarchicalActorHandle.interject().

        Uses the best available annotation for each keyframe:
        - LLM mode: Uses the semantic detection_reason from LLM keyframe selection
        - Algorithmic mode: Uses ImageHandle's auto-generated caption

        Example:
            step = await guided_learning_manager.get_next_step()
            transcript, images = step.to_actor_interject_args()
            await actor_handle.interject(transcript, images=images)

        Returns:
            Tuple of (transcript, list of AnnotatedImageRef)
        """
        from unity.image_manager.types import AnnotatedImageRef, RawImageRef

        annotated: List[AnnotatedImageRef] = []

        # Include context frame if no keyframes (commentary only)
        if self.context_frame and not self.keyframes:
            annotated.append(
                AnnotatedImageRef(
                    raw_image_ref=RawImageRef(
                        image_id=self.context_frame.image_handle.image_id,
                    ),
                    annotation=self._get_annotation(self.context_frame),
                ),
            )

        # Add all keyframes with their best available annotations
        for kf in self.keyframes:
            annotated.append(
                AnnotatedImageRef(
                    raw_image_ref=RawImageRef(image_id=kf.image_handle.image_id),
                    annotation=self._get_annotation(kf),
                ),
            )

        return self.transcript, annotated
