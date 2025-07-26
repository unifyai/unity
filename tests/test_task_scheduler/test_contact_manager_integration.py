import pytest

from unity.task_scheduler.task_scheduler import TaskScheduler


# --------------------------------------------------------------------------- #
#  Ensure ContactManager.ask is exposed as a tool for both ask & update flows #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_ts_tools_expose_contact_manager_ask():
    """TaskScheduler should surface ContactManager.ask inside its tool dictionaries."""

    ts = TaskScheduler()

    cm_tool_key = "ContactManager_ask"

    # ask-side tools
    assert cm_tool_key in ts._ask_tools, (
        f"{cm_tool_key} must be present in _ask_tools to enable the LLM to "
        "resolve contact information during TaskScheduler.ask runs."
    )

    # update-side tools (inherits ask tools + mutating helpers)
    assert cm_tool_key in ts._update_tools, (
        f"{cm_tool_key} must be present in _update_tools so that update() "
        "can translate human names into contact_id values when building triggers, "
        "etc."
    )


# --------------------------------------------------------------------------- #
#  Additional integration checks: verify real calls during ask/update flows   #
# --------------------------------------------------------------------------- #


import asyncio
import functools

from tests.helpers import _handle_project
from unity.contact_manager.contact_manager import ContactManager


# We keep these as *slow* eval-style tests because they spin up an LLM loop.
# Mark with `pytest.mark.eval` so CI can skip or run them selectively.


@pytest.mark.eval
@pytest.mark.asyncio
@_handle_project
async def test_ts_ask_calls_contact_manager_ask(monkeypatch):
    """A contact-oriented question should cause TaskScheduler.ask to call ContactManager.ask exactly once."""

    # --------------------------------------------------------------------
    # 1. Explicitly create an *accountant* contact so the lookup succeeds
    #    without devolving into an empty-result edge-case.  Importantly we
    #    perform this step *before* monkey-patching ContactManager.ask so
    #    any internal helper calls during the ``update`` flow do **not**
    #    inflate the expected call-count.
    # --------------------------------------------------------------------

    cm = ContactManager()

    create_handle = await cm.update(
        "Add a new contact: Sara Smith, email sara.smith@example.com, bio 'Company accountant'. She prefers an *informal* tone",
    )
    await create_handle.result()

    # --------------------------------------------------------------------
    # 2. Spy on ContactManager.ask *after* the contact exists
    # --------------------------------------------------------------------

    calls = {"count": 0}

    original = ContactManager.ask

    @functools.wraps(original)
    async def spy(self, text: str, **kw):  # type: ignore[override]
        calls["count"] += 1
        return await original(self, text, **kw)

    monkeypatch.setattr(ContactManager, "ask", spy, raising=True)

    # --------------------------------------------------------------------
    # 3. Proceed with the original integration scenario
    # --------------------------------------------------------------------

    ts = TaskScheduler()

    # Ensure there is at least one task mentioning the accountant so the scheduler's
    # task search surfaces it yet still lacks enough detail to answer the tone
    # question outright.  Creating the task directly avoids any ContactManager
    # look-ups during setup (keeping the expected call-count at exactly one).
    ts._create_task(
        name="Email our accountant about next week's taxes",
        description="Draft an email to our accountant regarding next week's tax submission.",
    )

    handle = await ts.ask(
        "When we email our accountant next week, should we speak formally or casually?",
    )

    # Give the loop a reasonable amount of time; these eval tests can be slower.
    await asyncio.wait_for(handle.result(), timeout=180)

    assert (
        calls["count"] == 1
    ), "ContactManager.ask should be called exactly once during TaskScheduler.ask"


@pytest.mark.eval
@pytest.mark.asyncio
@_handle_project
async def test_ts_update_calls_contact_manager_ask(monkeypatch):
    """Trigger-based task creation referencing a contact name must query ContactManager.ask exactly once."""

    # --------------------------------------------------------------------
    # 1. Explicitly create a *Sara Smith* contact so the lookup succeeds
    #    without devolving into an empty-result edge-case.  Importantly we
    #    perform this step *before* monkey-patching ContactManager.ask so
    #    any internal helper calls during the ``update`` flow do **not**
    #    inflate the expected call-count.
    # --------------------------------------------------------------------

    cm = ContactManager()

    create_handle = await cm.update(
        "Add a new contact: Sara Smith, email sara.smith@example.com, bio 'Company accountant'. She prefers an *informal* tone",
    )
    await create_handle.result()

    # --------------------------------------------------------------------
    # 2. Spy on ContactManager.ask *after* the contact exists
    # --------------------------------------------------------------------

    calls = {"count": 0}

    original = ContactManager.ask

    @functools.wraps(original)
    async def spy(self, text: str, **kw):  # type: ignore[override]
        calls["count"] += 1
        return await original(self, text, **kw)

    monkeypatch.setattr(ContactManager, "ask", spy, raising=True)

    # --------------------------------------------------------------------
    # 3. Create a task mentioning Sara so the scheduler's task search has
    #    context, yet still lacks enough detail for the trigger-building
    #    question – mirroring the ask-flow test setup.
    # --------------------------------------------------------------------

    ts = TaskScheduler()

    ts._create_task(
        name="Remind Sara about her homework",
        description="Prepare guidance to help Sara with her homework when she calls.",
    )

    # --------------------------------------------------------------------
    # 4. Proceed with the original integration scenario
    # --------------------------------------------------------------------

    cmd = "When Sara phones, please help her with her homework by starting an appropriate task automatically."

    handle = await ts.update(cmd)

    await asyncio.wait_for(handle.result(), timeout=180)

    assert (
        calls["count"] == 1
    ), "ContactManager.ask should be called exactly once during TaskScheduler.update"
