"""
Fixtures for ConversationManager → CodeActActor integration tests.

Key properties:
- Function-scoped ConversationManager and CodeActActor (shared async event loop per test)
- Deterministic waits: no fixed sleeps; explicit timeouts everywhere
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import AsyncIterator

import pytest
import pytest_asyncio

from tests.helpers import scenario_file_lock, get_or_create_contact
from tests.conversation_manager.cm_test_driver import CMStepDriver
from tests.conversation_manager.conftest import (
    BOSS,
    TEST_CONTACTS,
    DEFAULT_RESPONSE_POLICY,
)


def pytest_configure(config) -> None:
    """
    Configure environment variables for CodeActActor integration tests.

    Note: parent CM tests' conftest sets UNITY_ACTOR_IMPL="simulated" and disables
    several optional managers. We inject CodeActActor directly, so we do NOT rely
    on UNITY_ACTOR_IMPL, but we DO override manager enablement as needed.
    """
    os.environ.setdefault("TEST", "true")
    os.environ.setdefault("UNITY_CONVERSATION_JOB_NAME", "test_job")

    # These tests validate direct manager behavior (including fast-path tools),
    # so they need concrete manager implementations.
    os.environ["UNITY_CONTACT_IMPL"] = "real"
    os.environ["UNITY_TRANSCRIPT_IMPL"] = "real"

    # Enable FileManager for attachment/file flows.
    os.environ["UNITY_FILE_ENABLED"] = "true"

    # Keep KnowledgeManager disabled for determinism/performance in this suite.
    os.environ["UNITY_KNOWLEDGE_ENABLED"] = "false"

    # Keep optional managers disabled for focus + determinism.
    os.environ["UNITY_GUIDANCE_ENABLED"] = "false"
    os.environ["UNITY_SECRET_ENABLED"] = "false"
    os.environ["UNITY_SKILL_ENABLED"] = "false"
    os.environ["UNITY_WEB_ENABLED"] = "false"
    os.environ["UNITY_MEMORY_ENABLED"] = "false"

    # Ensure NEW marker comparisons are stable in tests.
    os.environ.setdefault("UNITY_INCREMENTING_TIMESTAMPS", "true")

    # Some Unify log readers assume contexts already exist. These integration tests
    # often run in isolation (fresh project, empty DB), so pre-create contexts to
    # avoid sporadic 404s on GET/PUT /logs for brand-new contexts.
    os.environ.setdefault("UNIFY_PRETEST_CONTEXT_CREATE", "true")


@pytest_asyncio.fixture(autouse=True)
async def _reset_litellm_logging_worker_per_test():
    """
    Pytest-asyncio runs different fixture scopes on different event loops.

    LiteLLM's GLOBAL_LOGGING_WORKER uses an asyncio.Queue bound to the event loop
    that first initialized it. When later used from a different loop, it can raise:
      "Queue ... is bound to a different event loop"

    We reset the worker per test so it always initializes on the current test loop.
    """
    try:
        from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

        try:
            await GLOBAL_LOGGING_WORKER.stop()
        except Exception:
            pass
        # Force re-init on next use.
        try:
            GLOBAL_LOGGING_WORKER._worker_task = None
            GLOBAL_LOGGING_WORKER._running_tasks.clear()
            GLOBAL_LOGGING_WORKER._queue = None
            GLOBAL_LOGGING_WORKER._sem = None
        except Exception:
            pass
    except Exception:
        # If litellm isn't installed/available, ignore.
        pass

    yield


@pytest_asyncio.fixture(scope="function")
async def conversation_manager_codeact() -> AsyncIterator[CMStepDriver]:
    """
    Start ConversationManager in-process for CodeActActor integration tests.

    NOTE: This fixture is function-scoped because ConversationManager spawns asyncio
    tasks (e.g., actor_watch_result) that must run on the same event loop as the
    test's CodeActActor handle. Module-scoped async fixtures run on a different
    loop under pytest-asyncio strict mode, which can prevent ActorResult propagation.
    """
    from unity.conversation_manager.event_broker import reset_event_broker
    from unity.conversation_manager import start_async, stop_async
    from unity.conversation_manager.domains import managers_utils
    from unity.common.prompt_helpers import now as prompt_now

    reset_event_broker()

    cm = await start_async(
        project_name="TestProject",
        enable_comms_manager=False,
        apply_test_mocks=True,
    )

    # Initialize managers once. Actor created here is a placeholder; tests override per-test.
    with scenario_file_lock("cm_integration_codeact"):
        await managers_utils.init_conv_manager(cm)

        # Ensure system contacts are well-formed for tests.
        if cm.contact_manager is not None:
            cm.contact_manager.update_contact(
                contact_id=0,
                first_name="Default",
                surname="Assistant",
                should_respond=True,
            )
            cm.contact_manager.update_contact(
                contact_id=1,
                first_name=BOSS["first_name"],
                surname=BOSS["surname"],
                email_address=BOSS["email_address"],
                phone_number=BOSS["phone_number"],
                should_respond=True,
                response_policy=BOSS["response_policy"],
            )

        # Ensure baseline test contacts exist idempotently (safe under parallel pytest).
        for contact_data in TEST_CONTACTS:
            contact_id = get_or_create_contact(
                cm.contact_manager,
                first_name=contact_data["first_name"],
                surname=contact_data.get("surname"),
                email_address=contact_data.get("email_address"),
                phone_number=contact_data.get("phone_number"),
            )
            if contact_id and cm.contact_manager is not None:
                cm.contact_manager.update_contact(
                    contact_id=contact_id,
                    should_respond=contact_data.get("should_respond", True),
                    response_policy=contact_data.get(
                        "response_policy",
                        DEFAULT_RESPONSE_POLICY,
                    ),
                )

    # Reset last_snapshot to the (possibly patched) prompt_now time.
    cm.last_snapshot = prompt_now(as_string=False)

    driver = CMStepDriver(cm)
    yield driver

    await stop_async()
    reset_event_broker()


@pytest_asyncio.fixture
async def code_act_actor() -> AsyncIterator[object]:
    """Create a fresh primitives-only CodeActActor for each test."""
    from unity.actor.code_act_actor import CodeActActor
    from unity.actor.environments import StateManagerEnvironment
    from unity.function_manager.primitives import Primitives, PrimitiveScope

    scope = PrimitiveScope(
        scoped_managers=frozenset({"contacts", "tasks", "transcripts", "files"}),
    )
    primitives = Primitives(primitive_scope=scope)
    env = StateManagerEnvironment(primitives)

    actor = CodeActActor(environments=[env], function_manager=None)

    # Strip FunctionManager tools for determinism (focus on routing via primitives).
    try:
        act_tools = actor.get_tools("act")
        if "execute_code" in act_tools:
            actor.add_tools("act", {"execute_code": act_tools["execute_code"]})
    except Exception:
        # If tool filtering fails, tests will surface it; don't hard-crash fixture.
        pass

    try:
        yield actor
    finally:
        try:
            await actor.close()
        except Exception:
            pass


@pytest.fixture
def initialized_cm_codeact(
    conversation_manager_codeact: CMStepDriver,
    code_act_actor: object,
) -> CMStepDriver:
    """
    Provide a clean CM state + a fresh CodeActActor bound to cm.actor.

    Clears cross-test state, then injects the per-test actor instance.
    """
    driver = conversation_manager_codeact

    # Clear any conversation state from previous tests.
    driver.contact_index.clear_conversations()
    driver.cm.in_flight_actions.clear()
    driver.cm.completed_actions.clear()
    driver.cm.chat_history.clear()

    # Bind per-test actor.
    driver.cm.actor = code_act_actor

    # Ensure fixture files exist (sanity check).
    fixtures_dir = Path(__file__).parent / "fixtures"
    assert fixtures_dir.exists(), f"Missing fixtures directory: {fixtures_dir}"

    # Reset last_snapshot to use current (possibly patched) prompt time.
    from unity.common.prompt_helpers import now as prompt_now

    driver.cm.last_snapshot = prompt_now(as_string=False)

    return driver


@pytest.fixture(scope="module")
def test_files(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    """
    Return absolute paths for sample fixture files.
    """
    fixtures_dir = Path(__file__).parent / "fixtures"

    return {
        "test_report.pdf": str(fixtures_dir / "test_report.pdf"),
        "test_data.csv": str(fixtures_dir / "test_data.csv"),
    }
