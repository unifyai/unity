import pytest
import unify

from unity.task_scheduler.task_scheduler import TaskScheduler
from tests.helpers import _handle_project


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_search_tasks_single_reference_basic():
    ts = TaskScheduler()

    entries = [
        (
            "Draft project update email",
            "Write a detailed project update email for stakeholders",
        ),
        ("Send quick status text", "Send a short status update via text message"),
        (
            "Prepare weekly reading notes",
            "Compile notes from this week's readings for the team",
        ),
        (
            "Schedule client check-in",
            "Arrange a check-in; client is often hard to reach by phone",
        ),
    ]
    for name, description in entries:
        ts._create_task(name=name, description=description)

    query = "short messages"
    results = ts._search_tasks(references={"description": query}, k=3)

    assert results[0].name == "Send quick status text"

    # Ensure vector column was created for the referenced source
    cols = unify.get_fields(context=ts._ctx)
    assert "_description_emb" in cols


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_search_tasks_multi_columns_json_and_vec_created():
    ts = TaskScheduler()

    # Distribute signal across two columns
    ts._create_task(
        name="Compose detailed email report",
        description="Loves detailed emails and reports",
    )
    ts._create_task(
        name="Log phone issues",
        description="Short notes, hates phone calls",
    )
    ts._create_task(
        name="Set up quick texting",
        description="Prefers texting and quick pings",
    )

    query = "quick text pings"
    # Provide separate references; ranking should still pick the texting task
    refs = {"description": query, "name": "irrelevant"}
    results = ts._search_tasks(references=refs, k=2)

    assert len(results) == 2
    # The task mentioning texting and quick pings – should be the top hit
    assert results[0].name == "Set up quick texting"

    # Ensure vector columns were created for each referenced source
    cols = unify.get_fields(context=ts._ctx)
    assert "_description_emb" in cols
    assert "_name_emb" in cols


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_search_tasks_all_columns_default_derivation():
    ts = TaskScheduler()

    # Populate different fields so the all-columns JSON helps similarity
    ts._create_task(
        name="Summarize reading list",
        description="Prepare a summary of recent reading materials",
    )
    ts._create_task(
        name="Configure email notifications",
        description="System works best when notifications are sent via email",
    )
    ts._create_task(
        name="Draft SMS templates",
        description="Create text message templates for quick outreach",
    )

    # Build a composite expression spanning multiple task fields
    expr = "str({name}) + ' ' + str({description}) + ' ' + str({response_policy}) + ' ' + str({priority})"
    query = "best to emails"
    results = ts._search_tasks(references={expr: query}, k=2)

    assert len(results) == 2
    assert results[0].name == "Configure email notifications"

    # Ensure a derived embedding column exists for the composite expression
    cols = unify.get_fields(context=ts._ctx)
    assert any(k.startswith("_expr_") and k.endswith("_emb") for k in cols.keys())


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_search_tasks_sum_of_cosine_ranking():
    ts = TaskScheduler()

    # A: matches both references
    ts._create_task(
        name="Follow up on training call",
        description="Follow up on the phone call last week about training",
        response_policy="For VIP clients, escalate immediately to the account manager",
    )
    # B: matches only the response_policy reference
    ts._create_task(
        name="Prepare onboarding guide",
        description="Haven't spoken yet",
        response_policy="VIP client handling; escalate to senior support when needed",
    )
    # C: matches only the description reference
    ts._create_task(
        name="Tax consultation follow-up",
        description="Prepare follow-up from the phone call last week regarding taxes",
        response_policy="Standard communication guidelines",
    )

    # Provide multiple references including different fields
    refs = {
        "response_policy": "VIP",
        "description": "phone call last week",
    }
    results = ts._search_tasks(references=refs, k=3)
    assert len(results) == 3
    names = [t.name for t in results]

    # Ensure the task matching both is ranked above the others
    assert names[0] == "Follow up on training call"
    assert names.index("Follow up on training call") < names.index(
        "Prepare onboarding guide",
    )
    assert names.index("Follow up on training call") < names.index(
        "Tax consultation follow-up",
    )

    # Ensure columns and vectors were created
    cols = unify.get_fields(context=ts._ctx)
    assert "response_policy" in cols
    assert "_response_policy_emb" in cols
    assert "_description_emb" in cols
    assert any(k.startswith("_sum_cos_") for k in cols.keys())
