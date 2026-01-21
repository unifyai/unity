"""Image/visual reasoning tests for HierarchicalActor."""

import asyncio
import textwrap
from unittest.mock import AsyncMock

import pytest

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
    wait_for_state,
)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_actor_extracts_information_from_images_during_execution():
    """Validates that interject(images=...) is handled during execution (mocked)."""
    actor = HierarchicalActor(headless=True, computer_mode="mock", connect_now=False)

    actor.computer_primitives._computer = MockComputerBackend(
        url="https://mock-url.com",
        screenshot=VALID_MOCK_SCREENSHOT_PNG,
    )
    actor.computer_primitives.navigate = AsyncMock(return_value=None)
    actor.computer_primitives.act = AsyncMock(return_value=None)

    active_task = None
    interjection_count = 0

    try:
        from unity.image_manager.image_manager import ImageManager
        from unity.image_manager.types import AnnotatedImageRef, RawImageRef

        im = ImageManager()
        [login_img_id, cell_img_id] = im.add_images(
            [
                {"caption": "login credentials", "data": VALID_MOCK_SCREENSHOT_PNG},
                {"caption": "cell highlight", "data": VALID_MOCK_SCREENSHOT_PNG},
            ],
        )
        login_img_id = int(login_img_id)
        cell_img_id = int(cell_img_id)
        _ = login_img_id

        active_task = HierarchicalActorHandle(
            actor=actor,
            goal="Go to google drive. Sign in with the email 'yusha@unify.ai' and password shown in the image.",
            persist=True,
        )

        if active_task._execution_task:
            active_task._execution_task.cancel()
            try:
                await active_task._execution_task
            except asyncio.CancelledError:
                pass

        active_task.verification_client = SimpleMockVerificationClient()

        def create_mock_modification_response(
            interjection_num: int,
        ) -> InterjectionDecision:
            if interjection_num == 1:
                return InterjectionDecision(
                    action="modify_task",
                    reason="Opening timesheet file",
                    patches=[
                        FunctionPatch(
                            function_name="main_plan",
                            new_code=textwrap.dedent(
                                """
                                async def main_plan():
                                    '''Opens timesheet file.'''
                                    await computer_primitives.navigate("https://drive.google.com")
                                    await computer_primitives.act("Open 'Unify TimeSheet' file")
                                    return "Timesheet opened"
                                """,
                            ),
                        ),
                    ],
                    cache=CacheInvalidateSpec(invalidate_steps=[]),
                )
            if interjection_num == 2:
                return InterjectionDecision(
                    action="modify_task",
                    reason="Filling cell based on image",
                    patches=[
                        FunctionPatch(
                            function_name="main_plan",
                            new_code=textwrap.dedent(
                                """
                                async def main_plan():
                                    '''Calculates and fills total hours.'''
                                    await computer_primitives.act("Calculate total hours for James")
                                    await computer_primitives.act("Fill in the highlighted cell with total")
                                    return "Cell filled with total hours"
                                """,
                            ),
                        ),
                    ],
                    cache=CacheInvalidateSpec(invalidate_steps=[]),
                )
            return InterjectionDecision(
                action="complete_task",
                reason="User indicated task is complete",
                patches=[],
                cache=CacheInvalidateSpec(invalidate_steps=[]),
            )

        async def mock_modification_generate(*args, **kwargs):
            _ = (args, kwargs)
            nonlocal interjection_count
            interjection_count += 1
            return create_mock_modification_response(
                interjection_count,
            ).model_dump_json()

        active_task.modification_client.generate = mock_modification_generate

        active_task.plan_source_code = actor._sanitize_code(
            textwrap.dedent(
                """
                async def main_plan():
                    '''Logs into Google Drive.'''
                    print("--- Navigating to Google Drive ---")
                    await computer_primitives.navigate("https://drive.google.com")
                    print("--- Logging in with credentials from image ---")
                    await computer_primitives.act("Sign in with 'yusha@unify.ai' and password from image")
                    return "Logged into Google Drive"
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

        await active_task.interject(
            "Go to the 'My Drive' folder and Open the 'Unify TimeSheet' file.",
        )
        await wait_for_state(
            active_task,
            _HierarchicalHandleState.PAUSED_FOR_INTERJECTION,
            timeout=30,
        )

        await active_task.interject(
            "Great! Now please calculate the total hrs for james and fill in the total in the cell highlighted in the image.",
            images=[
                AnnotatedImageRef(
                    raw_image_ref=RawImageRef(image_id=cell_img_id),
                    annotation="Cell to be edited is highlighted",
                ),
            ],
        )
        await wait_for_state(
            active_task,
            _HierarchicalHandleState.PAUSED_FOR_INTERJECTION,
            timeout=30,
        )

        await active_task.interject("Perfect, the task is complete. Thank you.")
        _ = await asyncio.wait_for(active_task.result(), timeout=30)

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
