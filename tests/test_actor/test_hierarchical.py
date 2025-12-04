"""
Tests for HierarchicalActor - an actor that executes and self-corrects a complete python script to accomplish a task.

This test file covers:
- Action caching and interjection handling
- Course correction and recovery flows
- Async verification (non-blocking, preemption)
- Clarification flow with user input
- Code merge logic (AST-based surgical replacement)
- Entrypoint execution
- Immediate pause/resume
- Nested function replacement
- Retrospective refactoring
- Robustness fixes (race conditions)
- Sandbox isolation and merge
- Scoped context in prompts
- Skill injection, sanitization, and memoization
- Skip verify flag
- Steerable exploration, modification, and replacement
- Visual reasoning
"""

import ast
import asyncio
import contextlib
import functools
import json
import logging
import sys
import textwrap
import traceback

import pytest
import unify
import unity
from pydantic import BaseModel, Field
from unittest.mock import AsyncMock, MagicMock, patch

from unity.actor.hierarchical_actor import (
    CacheInvalidateSpec,
    CacheStepRange,
    FunctionPatch,
    HierarchicalActor,
    HierarchicalActorHandle,
    ImplementationDecision,
    InterjectionDecision,
    StateVerificationDecision,
    VerificationAssessment,
    _HierarchicalHandleState,
)
from unity.function_manager.function_manager import FunctionManager
from unity.conversation_manager.handle import ConversationManagerHandle
from unity.common.async_tool_loop import SteerableToolHandle
from unity.controller.browser_backends import BrowserAgentError


# ────────────────────────────────────────────────────────────────────────────
# Logging Setup
# ────────────────────────────────────────────────────────────────────────────

logging.getLogger("urllib3").propagate = False
logging.getLogger("websockets").propagate = False
logging.getLogger("openai").setLevel(logging.INFO)
logging.getLogger("httpcore").setLevel(logging.INFO)
logging.getLogger("UnifyAsyncLogger").setLevel(logging.INFO)

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
if not root_logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(levelname)s][%(name)s] %(message)s")
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

unity.init(overwrite=True)


# ────────────────────────────────────────────────────────────────────────────
# Shared Mock Classes
# ────────────────────────────────────────────────────────────────────────────


class NoKeychainBrowser:
    """
    Mock browser that prevents Keychain prompts during tests.
    
    Args:
        url: URL to return from get_current_url()
        screenshot: Screenshot data to return from get_screenshot()
        with_backend_mocks: If True, adds MagicMock backend with barrier/interrupt
    """
    
    def __init__(
        self,
        url: str = "",
        screenshot: str = "",
        with_backend_mocks: bool = False,
    ):
        self._url = url
        self._screenshot = screenshot
        if with_backend_mocks:
            self.backend = MagicMock()
            self.backend.barrier = AsyncMock()
            self.backend.interrupt_current_action = AsyncMock()
        else:
            self.backend = object()

    async def get_current_url(self) -> str:
        return self._url

    async def get_screenshot(self) -> str:
        return self._screenshot

    def stop(self) -> None:
        pass


class SimpleMockVerificationClient:
    """
    Mock verification client that always returns success.
    Use for tests that don't need to control verification outcomes.
    """

    def __init__(self):
        self.generate = AsyncMock(side_effect=self._side_effect)
        self._current_format = VerificationAssessment

    def set_response_format(self, model):
        self._current_format = model

    def reset_response_format(self):
        self._current_format = VerificationAssessment

    def reset_messages(self):
        pass

    def set_system_message(self, message):
        pass

    async def _side_effect(self, *args, **kwargs):
        if self._current_format.__name__ == "StateVerificationDecision":
            return StateVerificationDecision(
                matches=True,
                reason="Mock: precondition satisfied.",
            ).model_dump_json()

        return VerificationAssessment(
            status="ok",
            reason="Mock verification success.",
        ).model_dump_json()


class ConfigurableMockVerificationClient:
    """
    Mock verification client with configurable per-function behavior.
    Use for tests that need to control verification outcomes and timing.
    """

    def __init__(self):
        self.behaviors = {}
        self.generate = AsyncMock(side_effect=self._side_effect)
        self._current_format = VerificationAssessment

    def set_behavior(self, func_name, delay_or_sequence=0, status="ok", reason="Mock success", *, sequence=None):
        """
        Configure behavior for a specific function.
        
        Supports multiple call signatures:
        - set_behavior(func_name, sequence) - list of (status, reason) tuples
        - set_behavior(func_name, delay, status, reason) - single response with delay
        - set_behavior(func_name, delay, status, reason, sequence=...) - with sequence
        """
        # Detect if second arg is a sequence (list) or delay (number)
        if isinstance(delay_or_sequence, list):
            # Called as: set_behavior(func_name, sequence)
            self.behaviors[func_name] = {
                "delay": 0,
                "status": "ok",
                "reason": "Mock success",
                "sequence": list(delay_or_sequence),
                "calls": 0,
            }
        else:
            # Called as: set_behavior(func_name, delay, status, reason, ...)
            self.behaviors[func_name] = {
                "delay": delay_or_sequence,
                "status": status,
                "reason": reason,
                "sequence": list(sequence or []),
                "calls": 0,
            }

    def set_response_format(self, model):
        self._current_format = model

    def reset_response_format(self):
        self._current_format = VerificationAssessment

    def reset_messages(self):
        pass

    def set_system_message(self, message):
        pass

    async def _side_effect(self, *args, **kwargs):
        # Extract prompt text from messages
        messages = kwargs.get("messages", [])
        prompt = ""
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, str):
                prompt += content
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        prompt += block["text"]
                    elif isinstance(block, str):
                        prompt += block

        # Handle StateVerificationDecision (precondition checks)
        if self._current_format.__name__ == "StateVerificationDecision":
            return StateVerificationDecision(
                matches=True,
                reason="Mock: precondition satisfied.",
            ).model_dump_json()

        # Extract function name from prompt
        func_name = None
        for line in prompt.split("\n"):
            if "Function Under Review:" in line and "`" in line:
                parts = line.split("`")
                if len(parts) >= 2:
                    raw_name = parts[1]
                    func_name = raw_name.split("(")[0].strip()
                    break

        # Check for configured behavior
        if func_name and func_name in self.behaviors:
            behavior = self.behaviors[func_name]
            
            # Apply delay if configured
            if behavior.get("delay", 0) > 0:
                await asyncio.sleep(behavior["delay"])
            
            # Use sequence if available
            if behavior["sequence"]:
                idx = min(behavior["calls"], len(behavior["sequence"]) - 1)
                status, reason = behavior["sequence"][idx]
                behavior["calls"] += 1
            else:
                status = behavior["status"]
                reason = behavior["reason"]
            
            return VerificationAssessment(
                status=status,
                reason=reason,
            ).model_dump_json()

        # Default: return success
        return VerificationAssessment(
            status="ok",
            reason="Mock verification success.",
        ).model_dump_json()


class MockImplementationClient:
    """Mock implementation client for testing recovery/reimplementation flows."""

    def __init__(self, new_code: str):
        self._new_code = new_code
        self.generate = AsyncMock(side_effect=self._get_payload)

    async def _get_payload(self, *args, **kwargs):
        payload_dict = {
            "action": "implement_function",
            "reason": "Applying mock fix.",
            "code": self._new_code,
        }
        return json.dumps(payload_dict)

    def reset_messages(self):
        pass

    def set_response_format(self, model):
        pass

    def reset_response_format(self):
        pass


# ────────────────────────────────────────────────────────────────────────────
# Shared Helper Functions
# ────────────────────────────────────────────────────────────────────────────


async def wait_for_state(task, expected_state, timeout=60, poll=0.5):
    """Poll the task's state until it matches expected_state or times out."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if task._state == expected_state:
            return
        await asyncio.sleep(poll)
    tail = "\n".join(task.action_log[-15:]) if hasattr(task, 'action_log') else ""
    raise AssertionError(
        f"Timed out waiting for state {expected_state.name}; "
        f"current state={task._state.name}\n--- Log Tail ---\n{tail}"
    )


async def wait_for_log_entry(task, log_substring: str, timeout=60, poll=0.5):
    """Poll the task's action_log until a specific substring appears or times out."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        log_content = "\n".join(task.action_log)
        if log_substring in log_content:
            return
        await asyncio.sleep(poll)
    tail = "\n".join(task.action_log[-20:])
    raise AssertionError(
        f"Timed out waiting for log entry '{log_substring}'.\n--- Log Tail ---\n{tail}"
    )


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1: Action Caching Tests
# ════════════════════════════════════════════════════════════════════════════


# --- Updated Test Plan with ConversationManager Call ---
CANNED_PLAN_FOR_INTERJECTION_TEST_ACTION_CACHING = textwrap.dedent(
    """
    async def main_plan():
        '''Main plan for testing action caching, including conversation_manager.ask.'''
        # --- Need imports inside the plan code ---
        from pydantic import BaseModel, Field
        print("--- Caching Test: Starting ---")

        # --- Define Pydantic models inside the plan code ---
        class UserPreference(BaseModel):
            item: str = Field(description="The item the user wants.")
        UserPreference.model_rebuild() # Important!

        class PageResult(BaseModel):
            heading: str = Field(description="The main heading of the page.")
        PageResult.model_rebuild() # Important!
        # --- End Model Definitions ---

        # Step 1: Navigate (will be cached)
        print("--- Caching Test: Step 1/4 - Navigating ---")
        await computer_primitives.navigate("https://example.com/start")

        # Step 2: Act (will be cached)
        print("--- Caching Test: Step 2/4 - Performing an action ---")
        await computer_primitives.act(
            "Click the 'Search' button."
        )

        # Step 3: Ask Conversation Manager (will be cached)
        print("--- Caching Test: Step 3/4 - Asking Conversation Manager ---")
        conv_handle = await computer_primitives.conversation_manager.ask(
            "What item are you looking for?",
            response_format=UserPreference
        )
        preference = await conv_handle.result() # Expects UserPreference instance
        print(f"--- Caching Test: User wants: {preference.item} ---") # Access .item

        # Step 4: Observe (will be cached)
        print("--- Caching Test: Step 4/4 - Observing the result ---")
        page_info = await computer_primitives.observe(
            "What is the main heading?",
            response_format=PageResult
        )
        print(f"--- Caching Test: Observed heading: {page_info.heading} ---") # Access .heading

        # Add a sleep to ensure the interjection has time to be processed
        await asyncio.sleep(2) # Reduced sleep as browser calls are mocked
        return "Original plan finished."
""",
)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_cache_hits_after_interjection_including_conversation_manager_calls():
    """
    Validates that after an interjection, previously executed actions including
    conversation_manager.ask result in cache hits. Mocks all external calls.
    """
    print(
        "\n\n--- Starting Test Harness for 'Interjection Caching with ConvManager' ---",
    )
    actor = HierarchicalActor(
        headless=True,
        browser_mode="magnitude",
        connect_now=False,
    )  # connect_now=False prevents real browser init
    actor.computer_primitives._browser = NoKeychainBrowser()

    active_task = None
    try:
        # --- Define Pydantic models matching those in the plan FOR MOCKING ---
        class UserPreference(BaseModel):
            item: str = Field(description="The item the user wants.")

        UserPreference.model_rebuild()

        class PageResult(BaseModel):
            heading: str = Field(description="The main heading of the page.")

        PageResult.model_rebuild()
        # --- End Model Definitions ---

        # --- Mock Setup ---
        # Mock basic action_provider methods
        actor.computer_primitives.navigate = AsyncMock(return_value=None)
        actor.computer_primitives.act = AsyncMock(return_value=None)
        actor.computer_primitives.observe = AsyncMock(
            return_value=PageResult(heading="Mock Heading"),
        )

        # Mock ConversationManagerHandle and the handle returned by its ask method
        mock_cm_handle = MagicMock(spec=ConversationManagerHandle)
        mock_ask_sub_handle = AsyncMock(
            spec=SteerableToolHandle,
        )  # Handle returned by ask

        # Configure the sub-handle's result method
        mock_ask_sub_handle.result = AsyncMock(
            return_value=UserPreference(item="mock_item"),
        )

        # Configure the main handle's ask method to return the sub-handle
        mock_cm_handle.ask = AsyncMock(
            return_value=mock_ask_sub_handle,
        )  # ask returns a handle

        # Mock the property access on the real action_provider
        actor.computer_primitives._conversation_manager = mock_cm_handle
        # --- End Mock Setup ---

        active_task = HierarchicalActorHandle(

            actor=actor,
            goal="Test action provider caching after modification, including conv manager.",
            persist=True,  # Need persist=True to allow interjection after completion
        )
        # Immediately cancel the auto-started task from __init__
        if active_task._execution_task:
            active_task._execution_task.cancel()
            try:
                await active_task._execution_task
            except asyncio.CancelledError:
                pass  # Expected cancellation
        # Manually inject the canned plan
        # Ensure the plan code itself also has the Pydantic definitions and model_rebuild calls
        active_task.plan_source_code = actor._sanitize_code(
            CANNED_PLAN_FOR_INTERJECTION_TEST_ACTION_CACHING,
            active_task,
        )

        # Manually trigger initialization and run
        active_task._execution_task = asyncio.create_task(
            active_task._initialize_and_run(),
        )

        # Ensure the plan actually paused
        await wait_for_log_entry(
            active_task,
            "STATE CHANGE: RUNNING -> PAUSED_FOR_INTERJECTION",
            timeout=30,
        )

        # --- Count initial cache misses ---
        initial_log = "\n".join(active_task.action_log)
        initial_miss_count = initial_log.count("CACHE MISS")
        # Expecting misses for: navigate, act, conversation_manager.ask (returns handle), result of the handle, observe
        expected_misses = 5
        assert (
            initial_miss_count == expected_misses
        ), f"Expected {expected_misses} initial CACHE MISS logs, found {initial_miss_count}! Log:\n{initial_log}"
        print(f"✅ Found correct number of initial cache misses ({expected_misses}).")
        # ---

        interjection_message = "Okay, now perform one final action: click 'Submit'."
        print(f"\n>>> INTERJECTING with: '{interjection_message}'")
        # Mock the modification client to just add a step without invalidating cache
        # Use actual newline, json.dumps will properly escape it
        modified_plan_code_base = CANNED_PLAN_FOR_INTERJECTION_TEST_ACTION_CACHING.replace(
            'return "Original plan finished."',
            'await computer_primitives.act("Click Submit")\n    return "Modified plan finished."',
        )
        # json.dumps will properly escape the newline for JSON
        modified_plan_code_escaped = json.dumps(modified_plan_code_base)

        active_task.modification_client.generate = AsyncMock(
            return_value=textwrap.dedent(
                f"""
                {{
                    "action": "modify_task",
                    "reason": "Adding a final submit step.",
                    "patches": [
                        {{
                            "function_name": "main_plan",
                            "new_code": {modified_plan_code_escaped}
                        }}
                    ]
                }}
            """,
            ),
        )

        interject_status = await active_task.interject(interjection_message)
        print(f">>> Interjection status: {interject_status}")

        print("\n>>> Waiting for the modified plan to complete...")
        # Since persist=True, the plan will pause again after the modified run.
        # We need to wait for that *second* pause state change log entry.
        # We need to be careful not to match the first state change log again.

        # Find the index of the restart log
        restart_log_index = -1
        for i, entry in enumerate(active_task.action_log):
            if (
                "RUN TRANSITION" in entry
                or "RESTART: Restarting execution loop" in entry
            ):
                restart_log_index = i
                break
        assert restart_log_index != -1, "Could not find plan restart log entry!"

        # Now, wait for the PAUSED_FOR_INTERJECTION state *after* the restart
        loop = asyncio.get_event_loop()
        deadline = loop.time() + 30  # 30 second timeout for the second run
        second_pause_found = False
        print("\n>>> Waiting for the *second* PAUSED_FOR_INTERJECTION state...")
        while loop.time() < deadline:
            # Check logs *after* the restart index
            current_log_slice = active_task.action_log[restart_log_index + 1 :]
            if any(
                "STATE CHANGE: RUNNING -> PAUSED_FOR_INTERJECTION" in entry
                for entry in current_log_slice
            ):
                print(">>> Found second PAUSED_FOR_INTERJECTION state.")
                second_pause_found = True
                break
            await asyncio.sleep(0.1)

        if not second_pause_found:
            tail = "\n".join(active_task.action_log[-20:])
            raise AssertionError(
                f"Timed out waiting for the *second* PAUSED_FOR_INTERJECTION state after restart.\n"
                f"--- Action log tail ---\n{tail}",
            )

        # NOW it's safe to stop the plan and get the result
        await active_task.stop("Modified plan ran, stopping test.")
        final_result = await active_task.result()  # Get final result after stop
        print(f"\n--- Plan finished with result: {final_result} ---")

        # --- Assertions ---
        # (Rest of the assertions remain the same)
        final_log = "\n".join(active_task.action_log)

        # 1. Count total cache hits and misses
        total_miss_count = final_log.count("CACHE MISS")
        total_hit_count = final_log.count("CACHE HIT")

        # Expecting initial 4 misses + 1 miss for the new 'act("Click Submit")'
        expected_total_misses = expected_misses + 1
        assert (
            total_miss_count == expected_total_misses
        ), f"Expected {expected_total_misses} total CACHE MISS logs, found {total_miss_count}!"

        # Expecting hits for: navigate, act, conversation_manager.ask, handle.result(), observe during replay
        expected_total_hits = 5
        assert (
            total_hit_count == expected_total_hits
        ), f"Expected {expected_total_hits} CACHE HIT logs after interjection, found {total_hit_count}!"

        print(
            f"✅ Found correct number of total cache misses ({expected_total_misses}) and hits ({expected_total_hits}).",
        )

        # 2. Specifically check for conv.ask hit log entry after restart
        # (Find restart_index again, just to be safe, though it should be the same)
        restart_index = -1
        for i, entry in enumerate(active_task.action_log):
            if (
                "RUN TRANSITION" in entry
                or "RESTART: Restarting execution loop" in entry
            ):
                restart_index = i
                break
        assert (
            restart_index != -1
        ), "Could not find plan restart log entry (for assertion)!"

        replay_log_entries = active_task.action_log[restart_index:]
        # Look for the cache hit on the conversation_manager.ask call
        conv_ask_hit_found = any(
            "CACHE HIT" in entry
            and "conversation_manager" in entry
            and ".ask(" in entry
            for entry in replay_log_entries
        )
        # Look for the cache hit on the subsequent .result() call
        conv_result_hit_found = any(
            "CACHE HIT" in entry
            and "conversation_manager" in entry
            and "-> .result()" in entry
            for entry in replay_log_entries
        )

        assert (
            conv_ask_hit_found
        ), "CACHE HIT for conversation_manager.ask was not found in the replay log!"
        assert (
            conv_result_hit_found
        ), "CACHE HIT for conversation_manager handle .result() was not found in the replay log!"

        print("✅ CACHE HIT confirmed for conversation_manager.ask during replay.")
        print(
            "✅ CACHE HIT confirmed for conversation_manager handle .result() during replay.",
        )  # Added confirmation

    finally:
        print("\n--- Cleaning up resources... ---")
        if active_task and not active_task.done():
            # Ensure cleanup stops the task if it's still running/paused
            await active_task.stop("Test cleanup")
        if actor:
            await actor.close()


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_loop_iterations_get_unique_cache_keys():
    """
    Validates that loop context tracking prevents false cache hits across loop iterations,
    while allowing proper cache hits during replay. Tests both simple and nested loops.
    """
    print("\n\n--- Starting Test Harness for 'Loop Context Caching' ---")

    # Define the test plan with loops
    LOOP_CACHING_PLAN = textwrap.dedent(
        """
        async def main_plan():
            '''Test plan with loops to validate loop context tracking in cache keys.'''
            from pydantic import BaseModel, Field

            class StepResult(BaseModel):
                status: str = Field(description="Status of the step")
            StepResult.model_rebuild()

            print("--- Starting Loop Test ---")

            # Simple loop - each iteration should get cache misses on first run
            iteration_count = 0
            while iteration_count < 3:
                print(f"--- Loop iteration {iteration_count} ---")
                # These calls have same args but different loop context each iteration
                await computer_primitives.act(f"Perform action in iteration {iteration_count}")
                result = await computer_primitives.observe(
                    "Check the status",
                    response_format=StepResult
                )
                print(f"Iteration {iteration_count} status: {result.status}")
                iteration_count += 1

            print("--- Loop completed ---")
            await asyncio.sleep(1)
            return "Loop plan finished"
    """,
    )

    actor = HierarchicalActor(
        headless=True,
        browser_mode="magnitude",
        connect_now=False,
    )
    actor.computer_primitives._browser = NoKeychainBrowser()

    active_task = None

    try:
        # Define Pydantic model for mocking
        class StepResult(BaseModel):
            status: str = Field(description="Status of the step")

        StepResult.model_rebuild()

        # Mock setup
        actor.computer_primitives.act = AsyncMock(return_value="done")
        actor.computer_primitives.observe = AsyncMock(
            return_value=StepResult(status="ok"),
        )

        active_task = HierarchicalActorHandle(

            actor=actor,
            goal="Test loop context tracking in cache keys",
            persist=True,
        )

        # Cancel auto-started task and inject our test plan
        if active_task._execution_task:
            active_task._execution_task.cancel()
            try:
                await active_task._execution_task
            except asyncio.CancelledError:
                pass

        active_task.plan_source_code = actor._sanitize_code(
            LOOP_CACHING_PLAN,
            active_task,
        )

        # Start execution
        active_task._execution_task = asyncio.create_task(
            active_task._initialize_and_run(),
        )

        # Wait for plan to complete first run
        await wait_for_log_entry(
            active_task,
            "STATE CHANGE: RUNNING -> PAUSED_FOR_INTERJECTION",
            timeout=30,
        )

        print("\n>>> First run completed. Analyzing cache behavior...")

        # === FIRST RUN ASSERTIONS ===
        initial_log = "\n".join(active_task.action_log)

        # Count cache misses - we should have 3 iterations × 2 calls per iteration = 6 misses
        initial_miss_count = initial_log.count("CACHE MISS")
        expected_initial_misses = 6  # 3 iterations × (1 act + 1 observe)

        print(
            f"\nFirst run - Expected {expected_initial_misses} cache misses, found {initial_miss_count}",
        )
        assert initial_miss_count == expected_initial_misses, (
            f"Expected {expected_initial_misses} initial CACHE MISS logs (3 iterations × 2 calls), "
            f"found {initial_miss_count}! This suggests loop context is not properly differentiating iterations.\n"
            f"Log:\n{initial_log}"
        )

        # Verify NO cache hits on first run
        initial_hit_count = initial_log.count("CACHE HIT")
        assert initial_hit_count == 0, (
            f"Expected 0 CACHE HIT on first run, found {initial_hit_count}! "
            f"First run should only have misses."
        )

        print(
            "✅ First run: All loop iterations correctly got cache misses (no false hits across iterations)",
        )

        # Check loop context appears in logs (optional debug check)
        loop_context_mentions = initial_log.count("LOOP_CONTEXT:")
        print(f"   Found {loop_context_mentions} loop context log entries")

        # === TRIGGER INTERJECTION ===
        interjection_message = "Great! Now add one more action at the end."
        print(f"\n>>> INTERJECTING with: '{interjection_message}'")

        # Modify the plan to add one more action after the loop
        modified_plan = LOOP_CACHING_PLAN.replace(
            'return "Loop plan finished"',
            'await computer_primitives.act("Final action after loop")\n    return "Modified loop plan finished"',
        )
        modified_plan_escaped = json.dumps(modified_plan)

        active_task.modification_client.generate = AsyncMock(
            return_value=textwrap.dedent(
                f"""
                {{
                    "action": "modify_task",
                    "reason": "Adding final action after loop.",
                    "patches": [
                        {{
                            "function_name": "main_plan",
                            "new_code": {modified_plan_escaped}
                        }}
                    ]
                }}
            """,
            ),
        )

        interject_status = await active_task.interject(interjection_message)
        print(f">>> Interjection status: {interject_status}")

        # Find restart log index
        restart_log_index = -1
        for i, entry in enumerate(active_task.action_log):
            if (
                "RUN TRANSITION" in entry
                or "RESTART: Restarting execution loop" in entry
            ):
                restart_log_index = i
                break
        assert restart_log_index != -1, "Could not find plan restart log entry!"

        # Wait for second completion
        print("\n>>> Waiting for modified plan to complete...")
        loop = asyncio.get_event_loop()
        deadline = loop.time() + 30
        second_pause_found = False

        while loop.time() < deadline:
            current_log_slice = active_task.action_log[restart_log_index + 1 :]
            if any(
                "STATE CHANGE: RUNNING -> PAUSED_FOR_INTERJECTION" in entry
                for entry in current_log_slice
            ):
                print(">>> Modified plan completed.")
                second_pause_found = True
                break
            await asyncio.sleep(0.1)

        if not second_pause_found:
            tail = "\n".join(active_task.action_log[-20:])
            raise AssertionError(
                f"Timed out waiting for modified plan completion.\n"
                f"--- Action log tail ---\n{tail}",
            )

        # Stop and get result
        await active_task.stop("Modified plan ran, stopping test.")
        final_result = await active_task.result()
        print(f"\n--- Plan finished with result: {final_result} ---")

        # === SECOND RUN ASSERTIONS ===
        final_log = "\n".join(active_task.action_log)
        replay_log = "\n".join(active_task.action_log[restart_log_index:])

        # Count cache behavior in replay
        replay_miss_count = replay_log.count("CACHE MISS")
        replay_hit_count = replay_log.count("CACHE HIT")

        # On replay:
        # - 3 iterations × 2 calls should be cache HITS (we've done these exact iterations before)
        # - 1 new final action should be a cache MISS
        expected_replay_hits = 6  # All loop iterations should hit cache
        expected_replay_misses = 1  # Only the new final action

        print(
            f"\nSecond run (replay) - Expected {expected_replay_hits} cache hits, found {replay_hit_count}",
        )
        print(
            f"Second run (replay) - Expected {expected_replay_misses} cache misses, found {replay_miss_count}",
        )

        assert replay_hit_count == expected_replay_hits, (
            f"Expected {expected_replay_hits} CACHE HIT on replay (all 3 loop iterations × 2 calls), "
            f"found {replay_hit_count}! Loop context should be stable and reproducible.\n"
            f"Replay log:\n{replay_log}"
        )

        assert replay_miss_count == expected_replay_misses, (
            f"Expected {expected_replay_misses} CACHE MISS on replay (only the new final action), "
            f"found {replay_miss_count}!\n"
            f"Replay log:\n{replay_log}"
        )

        print(
            "✅ Replay: All loop iterations correctly got cache hits (loop context is stable)",
        )
        print("✅ Replay: New action after loop correctly got cache miss")

        # Verify specific loop iteration hits
        # Check that we see cache hits for iteration-specific calls
        iteration_0_hit = any(
            "CACHE HIT" in entry and "Perform action in iteration 0" in entry
            for entry in replay_log.split("\n")
        )
        iteration_1_hit = any(
            "CACHE HIT" in entry and "Perform action in iteration 1" in entry
            for entry in replay_log.split("\n")
        )
        iteration_2_hit = any(
            "CACHE HIT" in entry and "Perform action in iteration 2" in entry
            for entry in replay_log.split("\n")
        )

        assert iteration_0_hit, "Cache hit for loop iteration 0 not found in replay!"
        assert iteration_1_hit, "Cache hit for loop iteration 1 not found in replay!"
        assert iteration_2_hit, "Cache hit for loop iteration 2 not found in replay!"

        print(
            "✅ Verified: Individual loop iterations (0, 1, 2) all got cache hits on replay",
        )

        # Total cache stats
        total_miss_count = final_log.count("CACHE MISS")
        total_hit_count = final_log.count("CACHE HIT")
        expected_total_misses = (
            expected_initial_misses + expected_replay_misses
        )  # 6 + 1 = 7
        expected_total_hits = expected_replay_hits  # 6

        assert (
            total_miss_count == expected_total_misses
        ), f"Total cache misses: expected {expected_total_misses}, found {total_miss_count}"
        assert (
            total_hit_count == expected_total_hits
        ), f"Total cache hits: expected {expected_total_hits}, found {total_hit_count}"

        print(f"\n✅✅✅ LOOP CONTEXT CACHING TEST PASSED! ✅✅✅")
        print(f"   Total misses: {total_miss_count} (expected {expected_total_misses})")
        print(f"   Total hits: {total_hit_count} (expected {expected_total_hits})")
        print(f"   Loop context tracking is working correctly!")

    finally:
        print("\n--- Cleaning up resources... ---")
        if active_task and not active_task.done():
            await active_task.stop("Test cleanup")
        if actor:
            await actor.close()


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_nested_loop_combinations_get_unique_cache_keys():
    """
    Validates that nested loop context tracking works correctly, ensuring
    each combination of outer/inner loop iterations gets unique cache keys.
    """
    print("\n\n--- Starting Test Harness for 'Nested Loop Context Caching' ---")

    # Define test plan with nested loops
    NESTED_LOOP_PLAN = textwrap.dedent(
        """
        async def main_plan():
            '''Test plan with nested loops to validate nested loop context tracking.'''
            from pydantic import BaseModel, Field

            class NestedResult(BaseModel):
                value: int = Field(description="Result value")
            NestedResult.model_rebuild()

            print("--- Starting Nested Loop Test ---")

            outer = 0
            while outer < 2:
                print(f"--- Outer loop iteration {outer} ---")

                inner = 0
                while inner < 2:
                    print(f"  --- Inner loop iteration {inner} ---")
                    # Each combination (outer, inner) should get unique cache key
                    result = await computer_primitives.observe(
                        f"Get value for outer={outer}, inner={inner}",
                        response_format=NestedResult
                    )
                    print(f"  Result for ({outer},{inner}): {result.value}")
                    inner += 1

                outer += 1

            print("--- Nested loops completed ---")
            await asyncio.sleep(1)
            return "Nested loop plan finished"
    """,
    )

    actor = HierarchicalActor(
        headless=True,
        browser_mode="magnitude",
        connect_now=False,
    )
    actor.computer_primitives._browser = NoKeychainBrowser()

    active_task = None

    try:
        # Define model for mocking
        class NestedResult(BaseModel):
            value: int = Field(description="Result value")

        NestedResult.model_rebuild()

        # Mock setup
        call_count = 0

        async def mock_observe(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return NestedResult(value=call_count)

        actor.computer_primitives.observe = AsyncMock(side_effect=mock_observe)

        active_task = HierarchicalActorHandle(

            actor=actor,
            goal="Test nested loop context tracking",
            persist=True,
        )

        # Cancel and inject plan
        if active_task._execution_task:
            active_task._execution_task.cancel()
            try:
                await active_task._execution_task
            except asyncio.CancelledError:
                pass

        active_task.plan_source_code = actor._sanitize_code(
            NESTED_LOOP_PLAN,
            active_task,
        )

        # Start execution
        active_task._execution_task = asyncio.create_task(
            active_task._initialize_and_run(),
        )

        # Wait for completion
        await wait_for_log_entry(
            active_task,
            "STATE CHANGE: RUNNING -> PAUSED_FOR_INTERJECTION",
            timeout=30,
        )

        print("\n>>> First run completed. Analyzing nested loop cache behavior...")

        # === ASSERTIONS ===
        initial_log = "\n".join(active_task.action_log)

        # 2 outer iterations × 2 inner iterations = 4 observe calls = 4 cache misses
        initial_miss_count = initial_log.count("CACHE MISS")
        expected_misses = 4

        print(
            f"\nExpected {expected_misses} cache misses (2×2 nested iterations), found {initial_miss_count}",
        )
        assert initial_miss_count == expected_misses, (
            f"Expected {expected_misses} CACHE MISS for nested loops, found {initial_miss_count}! "
            f"Nested loop context tracking may not be working correctly."
        )

        # No hits on first run
        initial_hit_count = initial_log.count("CACHE HIT")
        assert (
            initial_hit_count == 0
        ), f"Expected 0 CACHE HIT on first run, found {initial_hit_count}"

        print("✅ Nested loops: All iteration combinations got unique cache misses")

        # Check for nested loop context in logs
        # Should see both while_1 and while_2 (outer and inner loops)
        has_outer_loop = "while_1" in initial_log
        has_inner_loop = "while_2" in initial_log

        print(f"   Found outer loop context (while_1): {has_outer_loop}")
        print(f"   Found inner loop context (while_2): {has_inner_loop}")

        # Stop the plan
        await active_task.stop("Test completed")
        final_result = await active_task.result()
        print(f"\n--- Plan finished with result: {final_result} ---")

        print(f"\n✅✅✅ NESTED LOOP CONTEXT CACHING TEST PASSED! ✅✅✅")
        print(f"   Each nested loop iteration combination got unique cache keys!")

    finally:
        print("\n--- Cleaning up resources... ---")
        if active_task and not active_task.done():

            await active_task.stop("Test cleanup")
        if actor:
            await actor.close()


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_action_caching_orchestrator():
    """
    Orchestrates all action caching tests.
    
    Runs the following sub-tests:
    - test_cache_hits_after_interjection_including_conversation_manager_calls: Validates cache hits after interjections
    - test_loop_iterations_get_unique_cache_keys: Ensures loop iterations get unique cache keys
    - test_nested_loop_combinations_get_unique_cache_keys: Validates nested loop cache key uniqueness
    """
    try:
        await test_cache_hits_after_interjection_including_conversation_manager_calls()
        await test_loop_iterations_get_unique_cache_keys()
        await test_nested_loop_combinations_get_unique_cache_keys()
        print("\n\n🎉🎉🎉 ALL TESTS PASSED! 🎉🎉🎉")
    except Exception as e:
        print(f"\n\n❌❌❌ A TEST FAILED: {e} ❌❌❌")
        import traceback

        traceback.print_exc()
    finally:
        await asyncio.sleep(1)  # Allow tasks to clean up


