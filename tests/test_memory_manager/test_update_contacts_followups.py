import pytest

from tests.helpers import _handle_project

from unity.memory_manager.memory_manager import MemoryManager
from unity.contact_manager.simulated import SimulatedContactManager
from unity.transcript_manager.simulated import SimulatedTranscriptManager
from unity.knowledge_manager.simulated import SimulatedKnowledgeManager
from unity.task_scheduler.simulated import SimulatedTaskScheduler


# ---------------------------------------------------------------------------
# Helper stubs
# ---------------------------------------------------------------------------


def _patch_start_async_tool_loop(monkeypatch):
    """Replace the heavy LLM tool-loop with a minimal stub that simply
    invokes the `create_contact` tool (if present) and returns a dummy handle
    whose `.result()` coroutine finishes immediately.
    """
    import unity.memory_manager.memory_manager as mm_mod

    def _fake_loop(client, message, tools, *_, **__):  # noqa: D401 – imperative helper
        class _Handle:  # minimal stand-in for AsyncToolLoopHandle
            async def result(self):  # noqa: D401 – imperative helper
                # Simulate *one* new contact creation so the wrapper logic
                # inside `update_contacts` captures the freshly assigned id.
                if "create_contact" in tools:
                    await tools["create_contact"](first_name="Test")
                return "ok"

            # The real handle exposes steering methods but the unit-test never
            # calls them, so we omit them for brevity.

        return _Handle()

    # Apply patch to *the exact symbol* imported in memory_manager.py
    monkeypatch.setattr(mm_mod, "start_async_tool_loop", _fake_loop, raising=True)


def _patch_create_contact(monkeypatch):
    """Provide a synchronous `_create_contact` implementation that returns a
    predictable structure with an incremental `contact_id` so the surrounding
    wrapper records it correctly.
    """

    def _fake_create_contact(self, **__):  # noqa: D401 – imperative helper
        _fake_create_contact.counter += 1  # type: ignore[attr-defined]
        return {
            "outcome": "contact created (stub)",
            "details": {"contact_id": _fake_create_contact.counter},
        }

    _fake_create_contact.counter = 41  # start at 42 on first call

    # SimulatedContactManager does *not* define `_create_contact` by default –
    # we add it dynamically (raise if attribute exists is *not* desired here).
    monkeypatch.setattr(
        SimulatedContactManager,
        "_create_contact",
        _fake_create_contact,
        raising=False,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_update_contacts_triggers_followups(monkeypatch):
    """`update_contacts` should invoke `update_contact_bio` *and*
    `update_contact_rolling_summary` exactly once for a newly created contact
    when the default flags are used.
    """

    # --- counters ---------------------------------------------------------
    counts = {"bio": 0, "rolling": 0, "policy": 0}

    async def _stub_bio(self, *_, **__):  # noqa: D401 – imperative helper
        counts["bio"] += 1
        return "bio-ok"

    async def _stub_roll(self, *_, **__):  # noqa: D401 – imperative helper
        counts["rolling"] += 1
        return "roll-ok"

    async def _stub_policy(self, *_, **__):  # noqa: D401 – imperative helper
        counts["policy"] += 1
        return "policy-ok"

    # Patch lightweight helpers & heavy internals
    monkeypatch.setattr(MemoryManager, "update_contact_bio", _stub_bio, raising=True)
    monkeypatch.setattr(
        MemoryManager,
        "update_contact_rolling_summary",
        _stub_roll,
        raising=True,
    )
    monkeypatch.setattr(
        MemoryManager,
        "update_contact_response_policy",
        _stub_policy,
        raising=True,
    )

    _patch_create_contact(monkeypatch)
    _patch_start_async_tool_loop(monkeypatch)

    # Instantiate MemoryManager with simulated sub-managers so no I/O occurs
    mm = MemoryManager(
        contact_manager=SimulatedContactManager(
            description=(
                "TEST SCENARIO: Update contacts follow-ups. As a simulated ContactManager,"
                " deterministically allow creation and updates; expose predictable ids so"
                " the MemoryManager can trigger follow-up updates."
            ),
        ),
        transcript_manager=SimulatedTranscriptManager(
            description=(
                "TEST SCENARIO: Update contacts follow-ups. Provide straightforward transcript"
                " answers; no external dependencies."
            ),
        ),
        knowledge_manager=SimulatedKnowledgeManager(
            description=(
                "TEST SCENARIO: Update contacts follow-ups. Keep behaviour lightweight and"
                " side-effect free."
            ),
        ),
        task_scheduler=SimulatedTaskScheduler(
            description=(
                "TEST SCENARIO: Update contacts follow-ups. Accept simple updates with minimal"
                " responses; no I/O."
            ),
        ),
    )

    # Run the method under test
    await mm.update_contacts("dummy transcript with new contact")

    # Expectations: exactly one call each
    assert counts["bio"] == 1, "update_contact_bio should run once"
    assert counts["rolling"] == 1, "update_contact_rolling_summary should run once"
    assert counts["policy"] == 1, "update_contact_response_policy should run once"


@pytest.mark.asyncio
@_handle_project
async def test_update_contacts_respects_flags(monkeypatch):
    """When both follow-up flags are disabled, the respective helpers should
    *not* be invoked."""

    counts = {"bio": 0, "rolling": 0, "policy": 0}

    async def _stub_bio(self, *_, **__):  # noqa: D401 – imperative helper
        counts["bio"] += 1
        return "bio-ok"

    async def _stub_roll(self, *_, **__):  # noqa: D401 – imperative helper
        counts["rolling"] += 1
        return "roll-ok"

    async def _stub_policy(self, *_, **__):  # noqa: D401 – imperative helper
        counts["policy"] += 1
        return "policy-ok"

    monkeypatch.setattr(MemoryManager, "update_contact_bio", _stub_bio, raising=True)
    monkeypatch.setattr(
        MemoryManager,
        "update_contact_rolling_summary",
        _stub_roll,
        raising=True,
    )
    monkeypatch.setattr(
        MemoryManager,
        "update_contact_response_policy",
        _stub_policy,
        raising=True,
    )

    _patch_create_contact(monkeypatch)
    _patch_start_async_tool_loop(monkeypatch)

    mm = MemoryManager(
        contact_manager=SimulatedContactManager(
            description=(
                "TEST SCENARIO: Update contacts flags. As a simulated ContactManager, support"
                " deterministic updates; MemoryManager may or may not call follow-ups based"
                " on flags."
            ),
        ),
        transcript_manager=SimulatedTranscriptManager(
            description=(
                "TEST SCENARIO: Update contacts flags. Provide straightforward transcript"
                " answers only."
            ),
        ),
        knowledge_manager=SimulatedKnowledgeManager(
            description=(
                "TEST SCENARIO: Update contacts flags. Lightweight behaviour without external"
                " I/O."
            ),
        ),
        task_scheduler=SimulatedTaskScheduler(
            description=(
                "TEST SCENARIO: Update contacts flags. Accept simple updates with minimal"
                " responses; no I/O."
            ),
        ),
    )

    # Both follow-ups explicitly disabled
    await mm.update_contacts(
        "dummy transcript",
        update_bios=False,
        update_rolling_summaries=False,
        update_response_policies=False,
    )

    # No follow-up helper should have run
    assert counts["bio"] == 0, "update_contact_bio should NOT run when disabled"
    assert (
        counts["rolling"] == 0
    ), "update_contact_rolling_summary should NOT run when disabled"
    assert (
        counts["policy"] == 0
    ), "update_contact_response_policy should NOT run when disabled"
