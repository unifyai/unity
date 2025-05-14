from tests.helpers import _handle_project
from task_list_manager.task_list_manager import TaskListManager
from task_list_manager.types.schedule import Schedule


# Convenience to make schedules quickly
def _sch(prev_, next_):
    from datetime import datetime, timezone

    return Schedule(
        prev_task=prev_,
        next_task=next_,
        start_time=datetime.now(timezone.utc).isoformat(),
    )


@_handle_project
def test_get_queue_and_reorder():
    tlm = TaskListManager()
    tlm.start()

    # -----  create three queued tasks with an explicit chain  -----
    t0 = tlm._create_task(
        name="T0",
        description="first",
        schedule=_sch(None, 1),
    )
    t1 = tlm._create_task(
        name="T1",
        description="second",
        schedule=_sch(0, 2),
    )
    t2 = tlm._create_task(
        name="T2",
        description="third",
        schedule=_sch(1, None),
    )

    queue = tlm._get_task_queue()
    assert [t.task_id for t in queue] == [0, 1, 2]

    # -----  swap the order (0,2,1)  -----
    tlm._update_task_queue(original=[0, 1, 2], new=[0, 2, 1])

    new_q = tlm._get_task_queue()
    assert [t.task_id for t in new_q] == [0, 2, 1]


@_handle_project
def test_insert_into_queue():
    tlm = TaskListManager()
    tlm.start()

    # base queue with one task
    tlm._create_task(name="base", description="x", schedule=_sch(-1, -1))

    # create a brand-new task that will be inserted
    new_id = tlm._create_task(name="insert-me", description="y")

    tlm._update_task_queue(original=[0], new=[0, new_id])

    q = tlm._get_task_queue()
    assert [t.task_id for t in q] == [0, new_id]
    # also check the linkage of node 0 -> new_id
    assert q[0].schedule.next_task == new_id
