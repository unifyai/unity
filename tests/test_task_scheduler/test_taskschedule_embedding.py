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
    assert (filter_results == []) or (filter_results["tasks"] == [])

    # Semantic search with k=2 returns both tasks in ascending distance order
    res = manager._search_tasks(
        references={
            "str({name}) + ' ' + str({description})": "searching LinkedIn for contacts",
        },
        k=2,
    )
    sim_results = res["tasks"] if isinstance(res, dict) else res
    assert isinstance(sim_results, list)
    assert len(sim_results) == 2
    assert sim_results[0].name == "connecting with industry professionals"
    assert sim_results[1].name == "Find product prices"

    # Semantic search with k=1 respects the limit and returns only the closest match
    res_k1 = manager._search_tasks(
        references={
            "str({name}) + ' ' + str({description})": "searching LinkedIn for contacts",
        },
        k=1,
    )
    sim_results_k1 = res_k1["tasks"] if isinstance(res_k1, dict) else res_k1
    assert len(sim_results_k1) == 1
    # New default: when references is None/empty, return recent tasks only
    recent_only = manager._search_tasks(references=None, k=2)
    recent_only_tasks = (
        recent_only["tasks"] if isinstance(recent_only, dict) else recent_only
    )
    assert [t.name for t in recent_only_tasks] == [
        "Find product prices",
        "connecting with industry professionals",
    ]
    recent_only_empty = manager._search_tasks(references={}, k=2)
    recent_only_empty_tasks = (
        recent_only_empty["tasks"]
        if isinstance(recent_only_empty, dict)
        else recent_only_empty
    )
    assert [t.name for t in recent_only_empty_tasks] == [
        "Find product prices",
        "connecting with industry professionals",
    ]
    assert sim_results_k1[0].name == "connecting with industry professionals"
