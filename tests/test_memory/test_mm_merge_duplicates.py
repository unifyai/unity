import pytest

from tests.helpers import _handle_project

from unity.memory_manager.memory_manager import MemoryManager
from unity.contact_manager.simulated import SimulatedContactManager
from unity.transcript_manager.simulated import SimulatedTranscriptManager
from unity.knowledge_manager.simulated import SimulatedKnowledgeManager
from unity.task_scheduler.simulated import SimulatedTaskScheduler


@pytest.mark.asyncio
@_handle_project
async def test_update_contacts_merges_duplicate_contacts(monkeypatch):
    """MemoryManager.update_contacts should call merge_contacts when the
    transcript indicates two contact IDs refer to the *same* person.

    We monkey-patch the heavy LLM tool-loop so it immediately invokes the
    `merge_contacts` tool.  The ContactManager implementation itself is also
    patched with a lightweight stub so no external I/O occurs.  The assertion
    verifies that the merge helper is triggered exactly **once**.
    """

    # ------------------------------------------------------------------
    # 1.  Patch ContactManager._merge_contacts with a lightweight stub
    # ------------------------------------------------------------------
    merge_counter = {"calls": 0}

    def _stub_merge_contacts(
        self,
        *,
        contact_id_1: int,
        contact_id_2: int,
        overrides: dict,
    ):  # noqa: D401 – imperative stub
        merge_counter["calls"] += 1
        # Minimal ToolOutcome-shaped payload
        return {
            "outcome": "contacts merged (stub)",
            "details": {
                "kept_contact_id": contact_id_1,
                "deleted_contact_id": contact_id_2,
            },
        }

    monkeypatch.setattr(
        SimulatedContactManager,
        "_merge_contacts",
        _stub_merge_contacts,
        raising=False,  # method does not exist originally
    )

    # ------------------------------------------------------------------
    # 1b.  Ensure SimulatedContactManager has a _create_contact helper
    #       required by MemoryManager.update_contacts wrapper initialisation.
    # ------------------------------------------------------------------

    def _stub_create_contact(self, **__):  # noqa: D401 – simple stub
        _stub_create_contact.counter += 1  # type: ignore[attr-defined]
        return {
            "outcome": "contact created (stub)",
            "details": {"contact_id": _stub_create_contact.counter},
        }

    _stub_create_contact.counter = 41  # start at 42 on first invocation

    # Add attribute if missing or override if present
    monkeypatch.setattr(
        SimulatedContactManager,
        "_create_contact",
        _stub_create_contact,
        raising=False,
    )

    # ------------------------------------------------------------------
    # 2.  Replace the heavy tool-use loop with a stub that calls merge_contacts
    # ------------------------------------------------------------------
    import unity.memory_manager.memory_manager as mm_mod

    def _fake_tool_loop(client, message, tools, *_, **__):  # noqa: D401 – stub
        class _Handle:
            async def result(self):  # noqa: D401 – imperative stub
                # Simulate the LLM deciding to merge contact 7 into 12
                if "merge_contacts" in tools:
                    await tools["merge_contacts"](
                        contact_id_1=7,
                        contact_id_2=12,
                        overrides={"contact_id": 1},
                    )
                return "ok"

        return _Handle()

    monkeypatch.setattr(
        mm_mod,
        "start_async_tool_use_loop",
        _fake_tool_loop,
        raising=True,
    )

    # ------------------------------------------------------------------
    # 3.  Instantiate MemoryManager wired to simulated managers
    # ------------------------------------------------------------------
    mm = MemoryManager(
        contact_manager=SimulatedContactManager(description="merge-test"),
        transcript_manager=SimulatedTranscriptManager(description="merge-test"),
        knowledge_manager=SimulatedKnowledgeManager(description="merge-test"),
        task_scheduler=SimulatedTaskScheduler(description="merge-test"),
    )

    # ------------------------------------------------------------------
    # 4.  Run update_contacts – the fake tool-loop triggers the merge
    # ------------------------------------------------------------------
    await mm.update_contacts("dummy transcript mentioning duplicate David")

    # ------------------------------------------------------------------
    # 5.  Assertion – merge helper must be called exactly once
    # ------------------------------------------------------------------
    assert merge_counter["calls"] == 1, "Expected exactly one call to _merge_contacts"
