"""
Actor Integration Manager for Guided Learning.

This module provides the orchestration layer that manages Actor initialization,
interjection processing, clarification handling, and execution coordination.
It ties together the Actor, mocking infrastructure, and display layer.

Usage:
    from sandboxes.guided_learning_manager.actor_integration import (
        ActorIntegrationManager,
        ActorIntegrationConfig,
    )

    aim = ActorIntegrationManager()
    await aim.initialize(ActorIntegrationConfig(enabled=True))

    # Process demonstration steps
    for step in steps:
        plan_state = await aim.process_step(step)
        print(plan_state.summary)

    # Execute the learned plan
    async for notification in aim.execute_plan():
        print(notification)

    await aim.cleanup()
"""

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Coroutine,
    List,
    Optional,
    TYPE_CHECKING,
)

from sandboxes.guided_learning_manager.mocks import (
    SimpleMockVerificationClient,
)
from sandboxes.guided_learning_manager.plan_display import (
    PlanDisplayFormatter,
    PlanDisplayState,
)

if TYPE_CHECKING:
    from unity.actor.hierarchical_actor import (
        HierarchicalActor,
        HierarchicalActorHandle,
    )
    from unity.guided_learning_manager import GuidedLearningStep


logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class ActorIntegrationConfig:
    """
    Configuration for Actor integration.

    Attributes:
        enabled: Whether Actor integration is enabled
        execute_plan: If True, plan will be executed after learning (requires agent-service)
        debug_mode: If True, display full plan code instead of tree view
        computer_mode: Computer backend mode ("mock" for testing, "magnitude" for real agent-service)
        environments: List of environment names to enable
        connect_now: If True, connect to agent-service immediately
        mock_verification: If True, use SimpleMockVerificationClient
        mock_state_managers: If True, use simulated state managers
        headless: If True, run web in headless mode (no visible window)
    """

    enabled: bool = False
    execute_plan: bool = False
    debug_mode: bool = False
    computer_mode: str = "mock"
    environments: List[str] = field(
        default_factory=lambda: ["computer_primitives", "primitives"],
    )
    connect_now: bool = False
    mock_verification: bool = True
    mock_state_managers: bool = True
    enable_course_correction: bool = False
    headless: bool = False


# ────────────────────────────────────────────────────────────────────────────
# ActorIntegrationManager
# ────────────────────────────────────────────────────────────────────────────


class ActorIntegrationManager:
    """
    Orchestrates Actor integration for the guided learning sandbox.

    This manager handles:
    - Actor initialization (learning vs execution mode)
    - Interjection processing with progress indicators
    - Clarification handling via callbacks
    - Execution flow coordination

    Usage:
        aim = ActorIntegrationManager()
        await aim.initialize(ActorIntegrationConfig(enabled=True))

        # During learning
        plan_state = await aim.process_step(step)

        # At the end
        async for notification in aim.execute_plan():
            print(notification)

        await aim.cleanup()
    """

    def __init__(self):
        self.actor: Optional["HierarchicalActor"] = None
        self.actor_handle: Optional["HierarchicalActorHandle"] = None
        self.plan_formatter: PlanDisplayFormatter = PlanDisplayFormatter()
        self.config: Optional[ActorIntegrationConfig] = None

        # Clarification handling
        self.clarification_handler_task: Optional[asyncio.Task] = None
        self.on_clarification_callback: Optional[
            Callable[[str], Coroutine[Any, Any, str]]
        ] = None

        # Statistics
        self.step_count: int = 0
        self.successful_steps: int = 0
        self.failed_steps: int = 0
        self.total_time: float = 0.0
        self._start_time: Optional[float] = None

        # Execution state
        self._initialized: bool = False

    async def initialize(self, config: ActorIntegrationConfig) -> None:
        """
        Initialize Actor with correct mode and mocking.

        Steps:
        1. Create HierarchicalActor with appropriate computer_mode
        2. Mock primitives using mocking infrastructure
        3. Create HierarchicalActorHandle with goal=None, persist=True
        4. Cancel auto-started execution task
        5. Set plan_source_code = "" to skip initial generation
        6. Inject SimpleMockVerificationClient
        7. Start concurrent clarification handler task

        Args:
            config: Configuration for Actor integration

        Raises:
            RuntimeError: If agent-service is not available in execution mode
        """
        from unity.actor.hierarchical_actor import (
            HierarchicalActor,
            HierarchicalActorHandle,
        )
        from unity.actor.environments import (
            ComputerEnvironment,
            StateManagerEnvironment,
        )
        from unity.function_manager.primitives import ComputerPrimitives, Primitives

        self.config = config

        if not config.enabled:
            logger.info("Actor integration disabled.")
            return

        logger.info(
            f"Initializing Actor integration: mode={'execution' if config.execute_plan else 'learning'}, "
            f"computer_mode={config.computer_mode}",
        )

        # Step 1: Create ComputerPrimitives
        # MockComputerBackend is used when computer_mode="mock"
        # MagnitudeBackend is used when computer_mode="magnitude" (requires agent-service)
        computer_primitives = ComputerPrimitives(
            headless=config.headless,
            computer_mode=config.computer_mode,
            connect_now=config.connect_now,
        )
        # MockComputerBackend provides safe, canned responses for all methods

        # Step 2: Create Primitives (state managers) with simulated managers
        primitives = Primitives()

        # Apply simulated managers for safe, side-effect-free execution
        if config.mock_state_managers:
            from unity.contact_manager.simulated import SimulatedContactManager
            from unity.knowledge_manager.simulated import SimulatedKnowledgeManager
            from unity.task_scheduler.simulated import SimulatedTaskScheduler
            from unity.transcript_manager.simulated import SimulatedTranscriptManager
            from unity.guidance_manager.simulated import SimulatedGuidanceManager
            from unity.secret_manager.simulated import SimulatedSecretManager
            from unity.web_searcher.simulated import SimulatedWebSearcher

            primitives._contacts = SimulatedContactManager(
                description="A sandbox contact database for guided learning.",
            )
            primitives._knowledge = SimulatedKnowledgeManager(
                description="A sandbox knowledge base for guided learning.",
            )
            primitives._tasks = SimulatedTaskScheduler(
                description="A sandbox task scheduler for guided learning.",
            )
            primitives._transcripts = SimulatedTranscriptManager(
                description="A sandbox transcript manager for guided learning.",
            )
            primitives._guidance = SimulatedGuidanceManager(
                description="A sandbox guidance manager for guided learning.",
            )
            primitives._secrets = SimulatedSecretManager(
                description="A sandbox secret manager for guided learning.",
            )
            primitives._web_search = SimulatedWebSearcher(
                description="A sandbox web searcher for guided learning.",
            )

        # Step 3: Create HierarchicalActor with both environments
        # Use exposed_managers to exclude 'files' which has resolution issues
        state_manager_env = StateManagerEnvironment(
            primitives,
            exposed_managers={
                "contacts",
                "knowledge",
                "tasks",
                "transcripts",
                "guidance",
                "secrets",
                "web",
            },
        )

        self.actor = HierarchicalActor(
            headless=config.headless,
            computer_mode=config.computer_mode,
            connect_now=config.connect_now,
            enable_course_correction=config.enable_course_correction,
            environments=[
                ComputerEnvironment(computer_primitives),
                state_manager_env,
            ],
        )

        # Step 3: Create HierarchicalActorHandle
        # Using a learning goal so interjections have context for plan building.
        # With persist=True, after the initial plan completes, the Actor transitions
        # to PAUSED_FOR_INTERJECTION state where it waits for further instructions.
        self.actor_handle = HierarchicalActorHandle(
            actor=self.actor,
            goal="The user is demonstrating a workflow which you must replicate. Start with an empty plan and build it gradually as you receive more demonstrations from the user. Each demonstration should build the plan incrementally.",
            persist=True,  # Transitions to PAUSED_FOR_INTERJECTION after initial execution
        )

        # Step 4: Wait for initial plan to complete and enter PAUSED_FOR_INTERJECTION
        # The Actor will generate a simple plan based on the goal, execute it (no-op),
        # and then transition to PAUSED_FOR_INTERJECTION ready for user demonstrations.
        await self.actor_handle.awaiting_next_instruction()

        # Step 6: Inject SimpleMockVerificationClient
        if config.mock_verification:
            self.actor_handle.verification_client = SimpleMockVerificationClient()

        # Step 7: Start concurrent clarification handler task
        self.clarification_handler_task = asyncio.create_task(
            self._clarification_handler(),
            name="ClarificationHandler",
        )

        self._initialized = True
        self._start_time = time.time()
        logger.info("Actor integration initialized successfully.")

    async def process_step(self, step: "GuidedLearningStep") -> PlanDisplayState:
        """
        Process a captured step via Actor interjection.

        Steps:
        1. Format interjection using step.to_actor_interject_args()
        2. Call actor_handle.interject() (blocking)
        3. Extract plan from clean_function_source_map
        4. Use PlanDisplayFormatter to format display
        5. Return PlanDisplayState

        Args:
            step: The GuidedLearningStep to process

        Returns:
            PlanDisplayState with updated plan information

        Raises:
            RuntimeError: If manager is not initialized
        """
        if not self._initialized or self.actor_handle is None:
            raise RuntimeError("ActorIntegrationManager not initialized")

        self.step_count += 1
        step_start = time.time()

        try:
            # Step 1: Format interjection
            transcript, images = step.to_actor_interject_args()

            logger.info(
                f"Processing step {self.step_count}: transcript='{transcript[:50]}...'",
            )

            # Step 2: Call actor interjection (blocking)
            await self.actor_handle.interject(transcript, images=images)

            # Step 3 & 4: Extract and format plan
            plan_state = self.plan_formatter.parse_plan_for_display(self.actor_handle)

            # Update mode based on config
            mode = (
                "execution" if self.config and self.config.execute_plan else "learning"
            )
            plan_state = PlanDisplayState(
                step_number=plan_state.step_number,
                functions=plan_state.functions,
                summary=plan_state.summary,
                mode=mode,
                new_count=plan_state.new_count,
                modified_count=plan_state.modified_count,
                removed_count=plan_state.removed_count,
                git_diff=plan_state.git_diff,
            )

            self.successful_steps += 1
            logger.info(f"Step {self.step_count} processed successfully")

            return plan_state

        except Exception as e:
            self.failed_steps += 1
            logger.error(f"Failed to process step {self.step_count}: {e}")
            raise

        finally:
            self.total_time += time.time() - step_start

    async def get_plan_display(self) -> PlanDisplayState:
        """
        Extract and format current plan for display.

        Returns:
            Current PlanDisplayState

        Raises:
            RuntimeError: If manager is not initialized
        """
        if not self._initialized or self.actor_handle is None:
            raise RuntimeError("ActorIntegrationManager not initialized")

        plan_state = self.plan_formatter.parse_plan_for_display(self.actor_handle)
        mode = "execution" if self.config and self.config.execute_plan else "learning"
        return PlanDisplayState(
            step_number=plan_state.step_number,
            functions=plan_state.functions,
            summary=plan_state.summary,
            mode=mode,
            new_count=plan_state.new_count,
            modified_count=plan_state.modified_count,
            removed_count=plan_state.removed_count,
            git_diff=plan_state.git_diff,
        )

    def get_full_plan(self) -> str:
        """
        Get the full plan source code.

        Returns:
            Complete Python source code as a string

        Raises:
            RuntimeError: If manager is not initialized
        """
        if not self._initialized or self.actor_handle is None:
            raise RuntimeError("ActorIntegrationManager not initialized")

        return self.plan_formatter.format_full_plan(self.actor_handle)

    async def execute_plan(self) -> AsyncIterator[str]:
        """
        Execute the learned plan (execution mode only).

        Yields notifications as the plan executes, then returns the final result.

        Yields:
            Notification messages during execution

        Returns:
            The final result string (accessed via `async for` loop)

        Raises:
            RuntimeError: If manager is not initialized or not in execution mode
        """
        if not self._initialized or self.actor_handle is None:
            raise RuntimeError("ActorIntegrationManager not initialized")

        if self.config and not self.config.execute_plan:
            logger.warning("execute_plan() called but execute_plan=False in config")

        logger.info("Starting plan execution...")

        # Start result retrieval
        result_task = asyncio.create_task(self.actor_handle.result())

        # Listen for notifications during execution
        while not result_task.done():
            try:
                notification = await asyncio.wait_for(
                    self.actor_handle.next_notification(),
                    timeout=0.5,
                )
                if notification and "message" in notification:
                    yield notification["message"]
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.debug(f"Notification error (expected if done): {e}")
                break

        # Final result
        try:
            result = await result_task
            logger.info(f"Plan execution complete: {result}")
            yield f"RESULT: {result}"
        except Exception as e:
            logger.error(f"Plan execution failed: {e}")
            yield f"ERROR: {e}"

    async def cleanup(self) -> None:
        """Clean up Actor resources."""
        logger.info("Cleaning up ActorIntegrationManager...")

        # Cancel clarification handler
        if self.clarification_handler_task:
            self.clarification_handler_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.clarification_handler_task
            self.clarification_handler_task = None

        # Stop actor handle
        if self.actor_handle and not self.actor_handle.done():
            try:
                await self.actor_handle.stop("cleanup")
            except Exception as e:
                logger.debug(f"Error stopping actor handle: {e}")

        # Close actor
        if self.actor:
            try:
                await self.actor.close()
            except Exception as e:
                logger.debug(f"Error closing actor: {e}")
            self.actor = None

        self.actor_handle = None
        self._initialized = False
        logger.info("ActorIntegrationManager cleanup complete.")

    @property
    def num_functions(self) -> int:
        """Number of functions in the current plan."""
        if self.actor_handle:
            return len(getattr(self.actor_handle, "clean_function_source_map", {}))
        return 0

    @property
    def num_lines(self) -> int:
        """Number of lines in the current plan."""
        if self.actor_handle:
            source_map = getattr(self.actor_handle, "clean_function_source_map", {})
            return sum(code.count("\n") + 1 for code in source_map.values())
        return 0

    @property
    def total_steps(self) -> int:
        """Total number of steps processed."""
        return self.step_count

    # ─────────────────────────────────────────────────────────────────────────
    # Private Methods
    # ─────────────────────────────────────────────────────────────────────────

    async def _clarification_handler(self) -> None:
        """
        Concurrent task that handles clarification requests.

        Polls next_clarification() with timeout and invokes the callback
        when a clarification is requested.
        """
        if not self.actor_handle:
            return

        logger.debug("Clarification handler started")

        while not self.actor_handle.done():
            try:
                # Poll for clarifications (non-blocking with timeout)
                clar = await asyncio.wait_for(
                    self.actor_handle.next_clarification(),
                    timeout=0.5,
                )

                # Notify sandbox via callback
                if self.on_clarification_callback and clar:
                    question = clar.get("question", str(clar))
                    logger.info(f"Clarification requested: {question}")

                    answer = await self.on_clarification_callback(question)

                    await self.actor_handle.answer_clarification(
                        clar.get("call_id", ""),
                        answer,
                    )
                    logger.info(f"Clarification answered: {answer[:50]}...")

            except asyncio.TimeoutError:
                continue  # No clarification pending
            except asyncio.CancelledError:
                logger.debug("Clarification handler cancelled")
                break
            except Exception as e:
                logger.error(f"Clarification handler error: {e}")
                break

        logger.debug("Clarification handler stopped")
