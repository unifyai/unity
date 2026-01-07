from __future__ import annotations

"""Shared timeout values for steerability tests.

These tests run async tool loops (often with LLM-cached execution), so we prefer
consistent, slightly generous timeouts rather than many ad-hoc literals.
"""

# Wait for expected handles to register in the `SteerableToolPane`.
HANDLE_REGISTRATION_TIMEOUT = 60.0

# Wait for a `steering_applied` pane event to appear (pause/resume/interject/etc).
STEERING_EVENT_TIMEOUT = 60.0

# Timeout for `HierarchicalActorHandle.interject(...)` to return.
INTERJECT_TIMEOUT = 120.0

# Timeout for `pause()` / `resume()` calls to return.
PAUSE_RESUME_TIMEOUT = 30.0

# Wait for clarification questions/answers to flow through queues and pane.
CLARIFICATION_TIMEOUT = 60.0

# Timeout for the plan to complete after gates are released.
PLAN_COMPLETION_TIMEOUT = 180.0
