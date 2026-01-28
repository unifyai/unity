"""
ConversationManager initialization helpers for the sandbox.

This module is responsible for starting a ConversationManager instance in-process
for sandbox use:
- default: simulated comms (no CommsManager; outbound actions mocked)
- optional: real comms (`--real-comms`) with an explicit safety confirmation layer
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from sandboxes.conversation_manager.sandbox_simulated_actor import SandboxSimulatedActor
from unity.conversation_manager.event_broker import get_event_broker, reset_event_broker
from unity.conversation_manager.main import run_conversation_manager
from unity.conversation_manager.domains.managers_utils import init_conv_manager
from sandboxes.conversation_manager.real_comms_safety import (
    SafetyConfig,
    apply_real_comms_safety,
)
from sandboxes.conversation_manager.simulated_comms import apply_simulated_comms

LG = logging.getLogger("conversation_manager_sandbox")

_SIMULATION_GUIDANCE = (
    "Return actionable results (found/not-found/need-identifier). "
    "Do not claim side effects (sending SMS/email/calls) unless explicitly requested. "
    "If key details are missing (recipient, identifier, time), ask for them succinctly. "
    "Summarize the next step clearly."
)


async def initialize_cm(*, args: Any):
    """
    Initialize ConversationManager for the sandbox.

    Returns:
        ConversationManager instance
    """
    # Ensure global broker is fresh; CM internals use get_event_broker() in places.
    try:
        reset_event_broker()
    except Exception:
        pass
    event_broker = get_event_broker()
    # Sandbox resets the global broker so each run starts clean. Some CM modules
    # cache the broker at import time (e.g. `managers_utils.event_broker`), so
    # repoint those caches to the freshly reset instance to keep all event
    # publications visible to the sandbox subscriber.
    try:  # sandbox-only best-effort patching
        from unity.conversation_manager.domains import managers_utils as _managers_utils

        _managers_utils.event_broker = event_broker  # type: ignore[attr-defined]
    except Exception:
        pass

    real_comms = bool(getattr(args, "real_comms", False))

    # Start CM in-process.
    cm = await run_conversation_manager(
        project_name=args.project_name,
        event_broker=event_broker,
        stop_event=asyncio.Event(),
        enable_comms_manager=True if real_comms else False,
        apply_test_mocks=False if real_comms else True,
    )

    if real_comms:
        # Apply safety layer AFTER CM startup so comms_utils imports are initialized.
        apply_real_comms_safety(
            config=SafetyConfig(
                auto_confirm=bool(getattr(args, "auto_confirm", False)),
            ),
        )

        # In real-comms mode, CM will usually be initialized by StartupEvent.
        # However, for local dev it's often desirable to have managers ready.
        # init_conv_manager without an injected actor uses default Actor (not desired here),
        # so we keep SimulatedActor injection even in real-comms mode per epic constraint.
        sim_actor = SandboxSimulatedActor(
            steps=None,
            duration=8.0,
            log_mode="print",
            simulation_guidance=_SIMULATION_GUIDANCE,
        )
        await init_conv_manager(cm, actor=sim_actor)
    else:
        # Simulated mode should not depend on COMMS_URL, GCP Pub/Sub, or provisioned numbers.
        apply_simulated_comms()

        # Inject SimulatedActor; run_conversation_manager doesn't init managers when mocks are enabled.
        sim_actor = SandboxSimulatedActor(
            steps=None,
            duration=8.0,
            log_mode="print",
            simulation_guidance=_SIMULATION_GUIDANCE,
        )
        await init_conv_manager(cm, actor=sim_actor)

    LG.info(
        "ConversationManager initialized (%s, SimulatedActor injected)",
        "real-comms" if real_comms else "simulated",
    )
    return cm


async def shutdown_cm(cm) -> None:
    """Best-effort shutdown of the running CM instance."""
    try:
        cm.stop.set()
    except Exception:
        pass
    try:
        await asyncio.wait_for(cm.cleanup(), timeout=10.0)
    except Exception:
        pass
