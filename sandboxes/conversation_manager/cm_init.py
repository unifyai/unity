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
from typing import Any, Callable, Optional

from sandboxes.conversation_manager.actor_factory import ActorFactory
from sandboxes.conversation_manager.config_manager import ActorConfig
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


async def initialize_cm(
    *,
    args: Any,
    progress_callback: Optional[Callable[[str], None]] = None,
):
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
    cfg: ActorConfig = getattr(
        args,
        "_actor_config",
        ActorConfig(actor_type="simulated"),
    )
    use_real_managers = cfg.managers_mode == "real"

    # Start CM in-process.
    cm = await run_conversation_manager(
        project_name=args.project_name,
        event_broker=event_broker,
        stop_event=asyncio.Event(),
        enable_comms_manager=True if real_comms else False,
        # Simulated-only mode uses CM test mocks; real-managers mode must not.
        apply_test_mocks=(False if (real_comms or use_real_managers) else True),
    )

    if real_comms:
        # Apply safety layer AFTER CM startup so comms_utils imports are initialized.
        apply_real_comms_safety(
            config=SafetyConfig(
                auto_confirm=bool(getattr(args, "auto_confirm", False)),
            ),
        )

        # In real-comms mode, CM will usually be initialized by StartupEvent. For local dev
        # it's often desirable to have managers ready, so we eagerly init with the selected actor.
        actor = ActorFactory.create_actor(
            cfg,
            args=args,
            progress_callback=progress_callback or print,
        ).actor
        await init_conv_manager(cm, actor=actor)
    else:
        # Simulated mode should not depend on COMMS_URL, GCP Pub/Sub, or provisioned numbers.
        apply_simulated_comms()

        # Inject selected actor; run_conversation_manager doesn't init managers when mocks are enabled.
        actor = ActorFactory.create_actor(
            cfg,
            args=args,
            progress_callback=progress_callback or print,
        ).actor
        await init_conv_manager(cm, actor=actor)

        # Ensure the system user contact (contact_id=1) has basic identity details
        # populated in the Contacts table.
        #
        # In sandbox/simulated contexts, ContactManager provisions system contacts from
        # `SESSION_DETAILS` / API. When those aren't initialized (common locally), the
        # default user contact may be missing `phone_number` / `email_address`, which
        # makes brain actions like `send_sms` / `send_email` fail with an Error event.
        try:
            from sandboxes.conversation_manager.event_publisher import (
                get_simulated_user_contact,
            )

            u = get_simulated_user_contact()
            if getattr(cm, "contact_manager", None) is not None:
                cm.contact_manager.update_contact(  # type: ignore[union-attr]
                    contact_id=1,
                    first_name=u.get("first_name") or None,
                    surname=u.get("surname") or None,
                    phone_number=u.get("phone_number") or None,
                    email_address=u.get("email_address") or None,
                    should_respond=True,
                )
        except Exception:
            pass

    LG.info(
        "ConversationManager initialized (%s, actor=%s)",
        "real-comms" if real_comms else "simulated-comms",
        cfg.actor_type,
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
