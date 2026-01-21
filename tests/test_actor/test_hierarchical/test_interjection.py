import asyncio
import json
import textwrap

import pytest
from pydantic import BaseModel, Field
from unittest.mock import AsyncMock

from unity.actor.hierarchical_actor import (
    CacheInvalidateSpec,
    FunctionPatch,
    HierarchicalActor,
    HierarchicalActorHandle,
    InterjectionDecision,
    _HierarchicalHandleState,
)
from unity.function_manager.computer_backends import (
    MockComputerBackend,
    VALID_MOCK_SCREENSHOT_PNG,
)

from tests.test_actor.test_hierarchical.helpers import (
    SimpleMockVerificationClient,
    wait_for_log_entry,
    wait_for_state,
)


CANNED_PLAN_FOR_INTERJECTION_TEST_ACTION_CACHING = textwrap.dedent(
    """
    async def main_plan():
        '''Main plan for testing action caching with browser primitives.'''
        from pydantic import BaseModel, Field
        import asyncio
        print("--- Caching Test: Starting ---")

        class PageResult(BaseModel):
            heading: str = Field(description="The main heading of the page.")
        PageResult.model_rebuild()

        print("--- Caching Test: Step 1/3 - Navigating ---")
        await computer_primitives.navigate("https://example.com/start")

        print("--- Caching Test: Step 2/3 - Performing an action ---")
        await computer_primitives.act("Click the 'Search' button.")

        print("--- Caching Test: Step 3/3 - Observing the result ---")
        page_info = await computer_primitives.observe(
            "What is the main heading?",
            response_format=PageResult
        )
        print(f"--- Caching Test: Observed heading: {page_info.heading} ---")

        await asyncio.sleep(2)
        return "Original plan finished."
""",
)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_cache_hits_after_interjection_for_browser_primitives():
    """Moved out of the monolith: validates cache hits after modify_task interjection."""
    actor = HierarchicalActor(headless=True, computer_mode="mock", connect_now=False)
    active_task = None
    try:

        class PageResult(BaseModel):
            heading: str = Field(description="The main heading of the page.")

        PageResult.model_rebuild()

        actor.computer_primitives.navigate = AsyncMock(return_value=None)
        actor.computer_primitives.act = AsyncMock(return_value=None)
        actor.computer_primitives.observe = AsyncMock(
            return_value=PageResult(heading="Mock Heading"),
        )

        active_task = HierarchicalActorHandle(
            actor=actor,
            goal="Test action provider caching after modification.",
            persist=True,
        )
        if active_task._execution_task:
            active_task._execution_task.cancel()
            try:
                await active_task._execution_task
            except asyncio.CancelledError:
                pass

        active_task.verification_client = SimpleMockVerificationClient()
        active_task.plan_source_code = actor._sanitize_code(
            CANNED_PLAN_FOR_INTERJECTION_TEST_ACTION_CACHING,
            active_task,
        )
        active_task._execution_task = asyncio.create_task(
            active_task._initialize_and_run(),
        )

        await wait_for_log_entry(
            active_task,
            "STATE CHANGE: RUNNING -> PAUSED_FOR_INTERJECTION",
            timeout=30,
        )

        initial_log = "\n".join(active_task.action_log)
        assert initial_log.count("CACHE MISS") == 3

        interjection_message = "Okay, now perform one final action: click 'Submit'."
        modified_plan_code_base = CANNED_PLAN_FOR_INTERJECTION_TEST_ACTION_CACHING.replace(
            'return "Original plan finished."',
            'await computer_primitives.act("Click Submit")\n    return "Modified plan finished."',
        )
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

        _ = await active_task.interject(interjection_message)

        # Wait for the modified plan to re-run and pause again (persist=True).
        # We avoid relying on sleeps because replay timing varies (cache vs live).
        loop = asyncio.get_event_loop()
        deadline = loop.time() + 30
        while loop.time() < deadline:
            if (
                "\n".join(active_task.action_log).count(
                    "STATE CHANGE: RUNNING -> PAUSED_FOR_INTERJECTION",
                )
                >= 2
            ):
                break
            await asyncio.sleep(0.1)

        # Stop and evaluate cache stats.
        await active_task.stop("Modified plan ran, stopping test.")
        final_log = "\n".join(active_task.action_log)
        assert final_log.count("CACHE MISS") == 4
        assert final_log.count("CACHE HIT") >= 3

    finally:
        if active_task and not active_task.done():
            try:
                await active_task.stop()
            except Exception:
                pass
        try:
            await actor.close()
        except Exception:
            pass


CANNED_PLAN_FOR_REPLACEMENT_STEERABLE_REPLACE = textwrap.dedent(
    """
    async def main_plan():
        '''A simple plan that waits, intended to be replaced.'''
        import asyncio
        await asyncio.sleep(5)
        return "Original plan finished (this should not be reached)."
""",
)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_replace_interjection_discards_plan_and_starts_fresh():
    """Moved out of the monolith: validates replace_task decision surface."""
    actor = HierarchicalActor(headless=True, computer_mode="mock", connect_now=False)
    actor.computer_primitives._computer = MockComputerBackend(
        url="https://mock-url.com",
        screenshot=VALID_MOCK_SCREENSHOT_PNG,
    )
    actor.computer_primitives.navigate = AsyncMock(return_value=None)
    actor.computer_primitives.act = AsyncMock(return_value=None)

    active_task = None
    try:
        active_task = HierarchicalActorHandle(
            actor=actor,
            goal="This goal will be replaced.",
            parent_chat_context=[
                {"role": "user", "content": "Start the original test."},
            ],
            max_escalations=1,
            max_local_retries=1,
        )
        if active_task._execution_task:
            active_task._execution_task.cancel()
            try:
                await active_task._execution_task
            except asyncio.CancelledError:
                pass

        active_task.verification_client = SimpleMockVerificationClient()

        async def mock_modification_generate(*args, **kwargs):
            _ = (args, kwargs)
            response = InterjectionDecision(
                action="replace_task",
                reason="User wants to completely change the task to visit wikipedia",
                patches=[],
                cache=CacheInvalidateSpec(invalidate_steps=[]),
            )
            return response.model_dump_json()

        active_task.modification_client.generate = mock_modification_generate
        active_task.plan_source_code = actor._sanitize_code(
            CANNED_PLAN_FOR_REPLACEMENT_STEERABLE_REPLACE,
            active_task,
        )
        active_task._execution_task = asyncio.create_task(
            active_task._initialize_and_run(),
        )

        await asyncio.sleep(1)
        status = await active_task.interject(
            "Forget that. Please go to wikipedia.org and find the page for 'Asynchronous programming'.",
        )
        assert "re-" in status.lower() or "new goal" in status.lower()

        if not active_task.done():
            await active_task.stop("Test complete - replace_task verified")

    finally:
        if active_task and not active_task.done():
            try:
                await active_task.stop()
            except Exception:
                pass
        try:
            await actor.close()
        except Exception:
            pass


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_interject_with_various_image_formats():
    """Moved out of the monolith: validates interject(images=...) accepts ImageRefs/list formats."""
    from unity.image_manager.image_manager import ImageManager
    from unity.image_manager.types import AnnotatedImageRef, RawImageRef, ImageRefs

    actor = HierarchicalActor(headless=True, computer_mode="mock", connect_now=False)
    actor.computer_primitives._computer = MockComputerBackend(
        url="https://mock-url.com",
        screenshot=VALID_MOCK_SCREENSHOT_PNG,
    )
    actor.computer_primitives.navigate = AsyncMock(return_value=None)
    actor.computer_primitives.act = AsyncMock(return_value=None)

    im = ImageManager()
    [img_id] = im.add_images(
        [{"caption": "test image", "data": VALID_MOCK_SCREENSHOT_PNG}],
    )
    img_id = int(img_id)

    active_task = None
    try:
        active_task = HierarchicalActorHandle(
            actor=actor,
            goal="Test interjection with various image formats",
            persist=True,
        )
        if active_task._execution_task:
            active_task._execution_task.cancel()
            try:
                await active_task._execution_task
            except asyncio.CancelledError:
                pass

        active_task.verification_client = SimpleMockVerificationClient()
        active_task.modification_client.generate = AsyncMock(
            return_value=InterjectionDecision(
                action="modify_task",
                reason="Ack.",
                patches=[],
            ).model_dump_json(),
        )

        active_task.plan_source_code = actor._sanitize_code(
            textwrap.dedent(
                """
                async def main_plan():
                    return "Plan complete"
                """,
            ),
            active_task,
        )
        active_task._execution_task = asyncio.create_task(
            active_task._initialize_and_run(),
        )
        await wait_for_state(
            active_task,
            _HierarchicalHandleState.PAUSED_FOR_INTERJECTION,
            timeout=30,
        )

        list_images = [
            AnnotatedImageRef(
                raw_image_ref=RawImageRef(image_id=img_id),
                annotation="list format",
            ),
        ]
        s1 = await active_task.interject("list format", images=list_images)
        assert "error" not in s1.lower()

        refs_images = ImageRefs(
            [
                AnnotatedImageRef(
                    raw_image_ref=RawImageRef(image_id=img_id),
                    annotation="rootmodel format",
                ),
            ],
        )
        s2 = await active_task.interject("rootmodel format", images=refs_images)
        assert "error" not in s2.lower()

    finally:
        if active_task and not active_task.done():
            try:
                await active_task.stop()
            except Exception:
                pass
        try:
            await actor.close()
        except Exception:
            pass


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_user_interjections_incrementally_build_and_modify_plan():
    """
    Moved out of the monolith: a multi-step teaching session where interjections
    incrementally build up a plan and then the user completes the task.
    """
    # Use a fresh FunctionManager to avoid cross-test coupling and make the
    # "teaching session" deterministic (no accidental skill injections).
    from unity.function_manager.function_manager import FunctionManager

    fm = FunctionManager()
    fm.clear()
    actor = HierarchicalActor(
        headless=True,
        computer_mode="mock",
        connect_now=False,
        function_manager=fm,
        can_store=False,
    )
    actor.computer_primitives.navigate = AsyncMock(return_value=None)
    actor.computer_primitives.act = AsyncMock(return_value=None)

    active_task = None
    interjection_count = 0

    def create_mock_modification_response(
        interjection_num: int,
    ) -> InterjectionDecision:
        if interjection_num == 1:
            return InterjectionDecision(
                action="modify_task",
                reason="User wants to navigate to allrecipes.com",
                patches=[
                    FunctionPatch(
                        function_name="main_plan",
                        new_code=textwrap.dedent(
                            """
                            async def main_plan():
                                '''Navigate to allrecipes.'''
                                await computer_primitives.navigate("https://www.allrecipes.com")
                                return "Navigated to allrecipes.com"
                            """,
                        ),
                    ),
                ],
                cache=CacheInvalidateSpec(invalidate_steps=[]),
            )
        if interjection_num == 2:
            return InterjectionDecision(
                action="modify_task",
                reason="User wants to search for chocolate chip cookies",
                patches=[
                    FunctionPatch(
                        function_name="main_plan",
                        new_code=textwrap.dedent(
                            """
                            async def main_plan():
                                '''Navigate and search.'''
                                await computer_primitives.navigate("https://www.allrecipes.com")
                                await computer_primitives.act("Search for 'chocolate chip cookies'")
                                return "Searched for chocolate chip cookies"
                            """,
                        ),
                    ),
                ],
                cache=CacheInvalidateSpec(invalidate_steps=[]),
            )
        return InterjectionDecision(
            action="complete_task",
            reason="User indicated the session is complete",
            patches=[],
            cache=CacheInvalidateSpec(invalidate_steps=[]),
        )

    async def mock_generate(*args, **kwargs):
        _ = (args, kwargs)
        nonlocal interjection_count
        interjection_count += 1
        response = create_mock_modification_response(interjection_count)
        return response.model_dump_json()

    try:
        active_task = HierarchicalActorHandle(actor=actor, goal=None, persist=True)
        active_task.verification_client = SimpleMockVerificationClient()
        active_task.modification_client.generate = mock_generate

        # Under parallel load, initial plan generation + verification can take time.
        await wait_for_state(
            active_task,
            _HierarchicalHandleState.PAUSED_FOR_INTERJECTION,
            timeout=90,
        )

        _ = await active_task.interject("Navigate to allrecipes.com")
        await wait_for_state(
            active_task,
            _HierarchicalHandleState.PAUSED_FOR_INTERJECTION,
            timeout=30,
        )

        _ = await active_task.interject(
            "Great, now search for 'chocolate chip cookies'.",
        )
        await wait_for_state(
            active_task,
            _HierarchicalHandleState.PAUSED_FOR_INTERJECTION,
            timeout=30,
        )

        _ = await active_task.interject("Perfect, that's all. We're done.")
        final_result = await asyncio.wait_for(active_task.result(), timeout=30)

        assert active_task._state in {
            _HierarchicalHandleState.COMPLETED,
            _HierarchicalHandleState.PAUSED_FOR_INTERJECTION,
        }
        assert not str(final_result).startswith("ERROR")

    finally:
        if active_task and not active_task.done():
            try:
                await active_task.stop()
            except Exception:
                pass
        try:
            await actor.close()
        except Exception:
            pass


CANNED_PLAN_FOR_EXPLORATION_STEERABLE_EXPLORE = textwrap.dedent(
    """
    async def step_one_navigate():
        '''Navigates to the website.'''
        print("--- Main Plan: Navigating to Google.com ---")
        await computer_primitives.navigate("https://www.google.com/")
        print("--- Main Plan: Navigation complete. ---")

    async def step_two_pause():
        '''Pauses execution to allow for a side-quest.'''
        print("--- Main Plan: Pausing for 2 seconds to allow interjection... ---")
        await asyncio.sleep(2)
        print("--- Main Plan: Resuming after pause. ---")

    async def step_three_search():
        '''Performs a search after resuming.'''
        print("--- Main Plan: Executing final step (searching for 'Unity'). ---")
        await computer_primitives.act("Type 'Unity' in the search bar and press Enter")
        print("--- Main Plan: Search complete. ---")

    async def main_plan():
        '''Main entry point for the test plan.'''
        await step_one_navigate()
        await step_two_pause()
        await step_three_search()
        return "Main plan finished successfully after detached exploration."
    """,
)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_explore_interjection_runs_in_detached_sandbox():
    """Moved out of the monolith: validates explore_detached interjection path."""
    from unity.function_manager.function_manager import FunctionManager

    fm = FunctionManager()
    fm.clear()
    actor = HierarchicalActor(
        headless=True,
        computer_mode="mock",
        connect_now=False,
        function_manager=fm,
        can_store=False,
    )
    actor.computer_primitives._computer = MockComputerBackend(
        url="https://mock-url.com",
        screenshot=VALID_MOCK_SCREENSHOT_PNG,
    )
    actor.computer_primitives.navigate = AsyncMock(return_value=None)
    actor.computer_primitives.act = AsyncMock(return_value=None)

    active_task = None
    try:
        active_task = HierarchicalActorHandle(
            actor=actor,
            goal="Test the 'explore_detached' (sandbox) functionality.",
            parent_chat_context=[{"role": "user", "content": "Start the main plan."}],
            persist=False,
        )

        if active_task._execution_task:
            active_task._execution_task.cancel()
            try:
                await active_task._execution_task
            except asyncio.CancelledError:
                pass

        active_task.verification_client = SimpleMockVerificationClient()

        async def mock_modification_generate(*args, **kwargs):
            _ = (args, kwargs)
            return InterjectionDecision(
                action="explore_detached",
                reason="Running detached exploration.",
                patches=[],
                cache=CacheInvalidateSpec(invalidate_steps=[]),
            ).model_dump_json()

        active_task.modification_client.generate = mock_modification_generate

        async def mock_explore_detached(*args, **kwargs):
            _ = (args, kwargs)
            return "The page title is 'Google'"

        actor._run_detached_exploration = mock_explore_detached

        active_task.plan_source_code = actor._sanitize_code(
            CANNED_PLAN_FOR_EXPLORATION_STEERABLE_EXPLORE,
            active_task,
        )
        active_task._execution_task = asyncio.create_task(
            active_task._initialize_and_run(),
        )

        # Poll faster to avoid edge timing where the log line appears right at the timeout boundary.
        await wait_for_log_entry(
            active_task,
            "CACHE MISS: Executing computer_primitives.navigate",
            timeout=30,
            poll=0.05,
        )
        _ = await active_task.interject(
            "Quick question, what is the title of the current page?",
        )

        await wait_for_log_entry(
            active_task,
            "step_three_search",
            timeout=60,
            poll=0.05,
        )

        _ = await asyncio.wait_for(active_task.result(), timeout=30)

        final_log = "\n".join(active_task.action_log)
        assert "step_three_search" in final_log or "step_one_navigate" in final_log

    finally:
        if active_task and not active_task.done():
            try:
                await active_task.stop()
            except Exception:
                pass
        try:
            await actor.close()
        except Exception:
            pass


CANNED_PLAN_FOR_MODIFICATION_STEERABLE_MODIFY = textwrap.dedent(
    """
    async def initial_step():
        '''Navigates to the website and then waits for further instructions.'''
        print("--- Canned Plan: Navigating to allrecipes.com ---")
        await computer_primitives.navigate("https://www.allrecipes.com/")
        print("--- Canned Plan: Navigation complete. Now pausing for 5 seconds... ---")
        for i in range(5):
            print(f"--- Canned Plan: Waiting... ({i+1}/5) ---")
            await asyncio.sleep(1)
        print("--- Canned Plan: Pause complete. Finishing plan. ---")

    async def main_plan():
        '''Main entry point for the test plan.'''
        await initial_step()
        return "Initial plan finished without modification."
    """,
)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_modify_interjection_merges_new_code_into_existing_plan():
    """Moved out of the monolith: validates modify_task merges new code and re-runs."""
    actor = HierarchicalActor(headless=True, computer_mode="mock", connect_now=False)
    actor.computer_primitives._computer = MockComputerBackend(
        url="https://mock-url.com",
        screenshot=VALID_MOCK_SCREENSHOT_PNG,
    )
    actor.computer_primitives.navigate = AsyncMock(return_value=None)
    actor.computer_primitives.act = AsyncMock(return_value=None)

    active_task = None
    try:
        active_task = HierarchicalActorHandle(
            actor=actor,
            goal="Test the 'modify_task' interjection.",
            parent_chat_context=[{"role": "user", "content": "Start the test."}],
            max_escalations=1,
            max_local_retries=1,
        )

        if active_task._execution_task:
            active_task._execution_task.cancel()
            try:
                await active_task._execution_task
            except asyncio.CancelledError:
                pass

        active_task.verification_client = SimpleMockVerificationClient()

        async def mock_modification_generate(*args, **kwargs):
            _ = (args, kwargs)
            return InterjectionDecision(
                action="modify_task",
                reason="User wants to search for vegetarian lasagna",
                patches=[
                    FunctionPatch(
                        function_name="main_plan",
                        new_code=textwrap.dedent(
                            """
                            async def main_plan():
                                '''Modified plan to search for vegetarian lasagna.'''
                                print("--- Modified Plan: Navigating to allrecipes.com ---")
                                await computer_primitives.navigate("https://www.allrecipes.com/")
                                print("--- Modified Plan: Searching for vegetarian lasagna ---")
                                await computer_primitives.act("Type 'vegetarian lasagna' in search bar and click search")
                                return "Modified plan completed - searched for vegetarian lasagna."
                            """,
                        ),
                    ),
                ],
                cache=CacheInvalidateSpec(invalidate_steps=[]),
            ).model_dump_json()

        active_task.modification_client.generate = mock_modification_generate

        active_task.plan_source_code = actor._sanitize_code(
            CANNED_PLAN_FOR_MODIFICATION_STEERABLE_MODIFY,
            active_task,
        )
        active_task._execution_task = asyncio.create_task(
            active_task._initialize_and_run(),
        )

        await wait_for_log_entry(active_task, "allrecipes.com", timeout=15)
        status = await active_task.interject(
            "Great, now that you're on the homepage, type vegetarian lasagna into the search bar and click search.",
        )
        await wait_for_log_entry(active_task, "vegetarian lasagna", timeout=30)

        if not active_task.done():
            await active_task.stop("Test complete")
        if active_task._execution_task:
            try:
                await asyncio.wait_for(active_task._execution_task, timeout=10)
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                pass

        final_log = "\n".join(active_task.action_log)
        assert "vegetarian lasagna" in final_log.lower() or "modify" in status.lower()

    finally:
        if active_task and not active_task.done():
            try:
                await active_task.stop()
            except Exception:
                pass
        try:
            await actor.close()
        except Exception:
            pass
