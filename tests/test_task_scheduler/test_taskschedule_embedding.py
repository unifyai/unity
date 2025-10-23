import pytest
from unity.task_scheduler.task_scheduler import TaskScheduler
from tests.helpers import _handle_project


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_tasklist_embedding_search():
    # Start the TaskScheduler thread
    manager = TaskScheduler()

    # Create two tasks semantically related to "searching LinkedIn for contacts"
    id1 = manager._create_task(
        name="connecting with industry professionals",
        description="looking for contacts on a career-oriented site",
    )
    # create a totally different task
    id2 = manager._create_task(
        name="Find product prices",
        description="get the prices of all products",
    )

    # Keyword-based filter search should yield no hits
    filter_results = manager._filter_tasks(filter="'LinkedIn' in description")
    assert filter_results == []

    # Semantic search with k=2 returns both tasks in ascending distance order
    sim_results = manager._search_tasks(
        references={
            "str({name}) + ' ' + str({description})": "searching LinkedIn for contacts",
        },
        k=2,
    )
    assert isinstance(sim_results, list)
    assert len(sim_results) == 2
    assert sim_results[0].name == "connecting with industry professionals"
    assert sim_results[1].name == "Find product prices"

    # Semantic search with k=1 respects the limit and returns only the closest match
    sim_results_k1 = manager._search_tasks(
        references={
            "str({name}) + ' ' + str({description})": "searching LinkedIn for contacts",
        },
        k=1,
    )
    assert len(sim_results_k1) == 1
    # New default: when references is None/empty, return recent tasks only
    recent_only = manager._search_tasks(references=None, k=2)
    assert [t.name for t in recent_only] == [
        "Find product prices",
        "connecting with industry professionals",
    ]
    recent_only_empty = manager._search_tasks(references={}, k=2)
    assert [t.name for t in recent_only_empty] == [
        "Find product prices",
        "connecting with industry professionals",
    ]
    assert sim_results_k1[0].name == "connecting with industry professionals"
