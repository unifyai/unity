"""
Actor + primitives factory for the ConversationManager sandbox.

This is sandbox-only wiring glue. It centralizes:
- selecting the correct actor type for the chosen sandbox configuration
- configuring manager implementations (simulated vs real) via env vars
- constructing a CodeActActor with explicit environments and a computer backend

Important:
- We keep the existing `--real-comms` flag orthogonal to actor configuration.
  Real comms is guarded by explicit safety confirmations elsewhere.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

from sandboxes.conversation_manager.computer_activity import (
    ComputerActivity,
    install_computer_activity_hooks,
)
from sandboxes.conversation_manager.config_manager import ActorConfig
from sandboxes.conversation_manager.sandbox_simulated_actor import SandboxSimulatedActor
from unity.actor.code_act_actor import CodeActActor
from unity.actor.environments import ComputerEnvironment, StateManagerEnvironment
from unity.function_manager.computer_backends import VALID_MOCK_SCREENSHOT_PNG
from unity.function_manager.primitives import ComputerPrimitives, Primitives
from unity.manager_registry import ManagerRegistry

LG = logging.getLogger("conversation_manager_sandbox")

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class ActorFactoryResult:
    actor: Any
    primitives: Optional[Primitives]
    computer_primitives: Optional[ComputerPrimitives]


class ActorFactory:
    """Create actors for sandbox configurations."""

    # Keep this guidance consistent with `cm_init._SIMULATION_GUIDANCE`.
    _SIMULATION_GUIDANCE = (
        "Return actionable results (found/not-found/need-identifier). "
        "Do not claim side effects (sending SMS/email/calls) unless explicitly requested. "
        "If key details are missing (recipient, identifier, time), ask for them succinctly. "
        "Summarize the next step clearly."
    )

    @classmethod
    def create_actor(
        cls,
        config: ActorConfig,
        *,
        args: Any,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> ActorFactoryResult:
        """
        Create the actor instance for the selected sandbox configuration.

        Returns the actor plus (for CodeAct) the constructed primitives objects that
        back the actor environments.
        """
        progress = progress_callback or (lambda _m: None)

        # Configure manager IMPL selection for all modes (including simulated).
        # Without this, ManagerRegistry defaults to "real" implementations,
        # which require full Orchestra connectivity for system contact sync.
        cls._apply_manager_impl_env(config.managers_mode)
        # Ensure no stale singleton managers leak across sandbox restarts/switches.
        try:
            ManagerRegistry.clear()
        except Exception:
            pass

        if config.actor_type == "simulated":
            actor = SandboxSimulatedActor(
                steps=None,
                duration=8.0,
                log_mode="print",
                simulation_guidance=cls._SIMULATION_GUIDANCE,
            )
            return ActorFactoryResult(
                actor=actor,
                primitives=None,
                computer_primitives=None,
            )

        progress("[init] Loading configuration...")
        progress(f"✓ Actor selected: {config.actor_type}")

        primitives = cls.build_primitives(
            mode=config.managers_mode,
            progress_callback=progress,
        )

        computer_primitives = cls.create_computer_backend(
            mode=config.computer_backend_mode,
            args=args,
            progress_callback=progress,
        )

        # Keep `primitives.computer` consistent with the dedicated `computer_primitives`
        # environment. This avoids accidentally creating two separate computer backends.
        if computer_primitives is not None:
            try:
                primitives._computer = computer_primitives  # type: ignore[attr-defined]
            except Exception:
                pass

        envs = [StateManagerEnvironment(primitives)]
        if computer_primitives is not None:
            envs.append(ComputerEnvironment(computer_primitives))

        # Pass explicit environments to avoid implicit computer dependencies.
        actor = CodeActActor(
            environments=envs,
            computer_primitives=computer_primitives,
            # If we passed computer_primitives, other computer params are ignored.
            computer_mode=(
                "mock" if config.computer_backend_mode == "mock" else "magnitude"
            ),
            agent_server_url=getattr(args, "agent_server_url", None),
            headless=bool(getattr(args, "headless", False)),
        )

        return ActorFactoryResult(
            actor=actor,
            primitives=primitives,
            computer_primitives=computer_primitives,
        )

    @staticmethod
    def build_primitives(
        *,
        mode: Literal["simulated", "real"],
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Primitives:
        """
        Return a Primitives instance configured for the selected manager mode.

        Implementation approach:
        - We configure manager IMPL selection via env vars (so CM initialization and
          CodeAct share the same singleton instances from ManagerRegistry).
        - We do not eagerly instantiate every manager here; CM startup will do that.
        """
        progress = progress_callback or (lambda _m: None)
        progress(f"[init] Managers mode: {mode}")
        return Primitives()

    @staticmethod
    def create_computer_backend(
        *,
        mode: Literal["none", "mock", "real"],
        args: Any,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Optional[ComputerPrimitives]:
        """
        Create the computer backend for the selected mode.

        - none: no computer tools
        - mock: MockComputerBackend (no external deps)
        - real: Magnitude backend via agent-service (connect now for startup feedback)
        """
        progress = progress_callback or (lambda _m: None)

        if mode == "none":
            progress("[computer] Disabled")
            return None

        if mode == "mock":
            progress("[computer] Using mock backend")
            # No external dependencies; connect lazily.
            cp = ComputerPrimitives(
                computer_mode="mock",
                connect_now=False,
            )
            activity = ComputerActivity()
            setattr(args, "_computer_activity", activity)
            install_computer_activity_hooks(
                computer_primitives=cp,
                activity=activity,
                emit_line=getattr(args, "_computer_log_sink", None),
            )
            return cp

        # real
        agent_server_url = getattr(args, "agent_server_url", None)
        progress("[computer] Connecting to agent-service...")
        cp = ComputerPrimitives(
            headless=bool(getattr(args, "headless", False)),
            computer_mode="magnitude",
            agent_mode=getattr(args, "agent_mode", "web"),
            agent_server_url=str(agent_server_url),
            connect_now=True,
        )
        activity = ComputerActivity()
        setattr(args, "_computer_activity", activity)
        # We only know "connected" if `connect_now` succeeded.
        activity.mark_connected_sync(True)
        install_computer_activity_hooks(
            computer_primitives=cp,
            activity=activity,
            emit_line=getattr(args, "_computer_log_sink", None),
        )
        progress("✓ Computer ready (agent-service connected)")
        return cp

    @staticmethod
    def _apply_manager_impl_env(mode: Literal["simulated", "real"]) -> None:
        """
        Configure manager IMPL selection for sandbox runs.

        This controls what `ManagerRegistry.get_*()` returns when CM initializes.
        """
        impl = "simulated" if mode == "simulated" else "real"

        # State managers
        os.environ["UNITY_CONTACT_IMPL"] = impl
        os.environ["UNITY_TRANSCRIPT_IMPL"] = impl
        os.environ["UNITY_TASK_IMPL"] = impl
        os.environ["UNITY_KNOWLEDGE_IMPL"] = impl
        os.environ["UNITY_GUIDANCE_IMPL"] = impl
        os.environ["UNITY_SECRET_IMPL"] = impl
        os.environ["UNITY_WEB_IMPL"] = impl
        os.environ["UNITY_FILE_IMPL"] = impl

        # Support managers commonly used by CM / primitives
        os.environ["UNITY_DATA_IMPL"] = impl
        os.environ["UNITY_FUNCTION_IMPL"] = impl
        os.environ["UNITY_CONVERSATION_IMPL"] = impl

        # Memory is optional; keep it aligned so behavior is predictable.
        os.environ["UNITY_MEMORY_IMPL"] = impl

        # If the mock computer backend is used, ensure a safe default screenshot exists.
        # (No-op for real mode.)
        os.environ.setdefault(
            "UNITY_COMPUTER_DEFAULT_SCREENSHOT",
            VALID_MOCK_SCREENSHOT_PNG,
        )
