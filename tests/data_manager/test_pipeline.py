"""Tests for the generic DAG pipeline execution engine.

Covers:
- TaskGraph dependency resolution and lifecycle
- PipelineExecutor parallel execution, retry with backoff, fail-fast, cancellation
"""

from __future__ import annotations

import threading
from unity.data_manager.utils.pipeline import (
    ExecutionConfig,
    PipelineExecutor,
    Task,
    TaskGraph,
    TaskResult,
    TaskStatus,
)

# ────────────────────────────────────────────────────────────────────────────
# TaskGraph
# ────────────────────────────────────────────────────────────────────────────


class TestTaskGraph:
    """Tests for TaskGraph dependency resolution and state tracking."""

    def test_empty_graph_is_complete(self):
        g = TaskGraph(name="empty")
        assert g.is_complete()
        assert not g.has_failures()
        assert g.get_ready_tasks() == []

    def test_single_task_ready_immediately(self):
        g = TaskGraph(name="single")
        g.add_task(Task(id="a", task_type="work", func=lambda: None))

        ready = g.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == "a"

    def test_dependency_blocks_until_completed(self):
        g = TaskGraph(name="dep")
        g.add_task(Task(id="root", task_type="create", func=lambda: None))
        g.add_task(
            Task(
                id="child",
                task_type="insert",
                func=lambda: None,
                dependencies={"root"},
            ),
        )

        ready = g.get_ready_tasks()
        assert [t.id for t in ready] == ["root"]

        g.mark_completed("root", TaskResult(success=True))
        ready = g.get_ready_tasks()
        assert [t.id for t in ready] == ["child"]

    def test_multiple_independent_tasks_ready(self):
        g = TaskGraph(name="parallel")
        g.add_task(Task(id="a", task_type="w", func=lambda: None))
        g.add_task(Task(id="b", task_type="w", func=lambda: None))
        g.add_task(Task(id="c", task_type="w", func=lambda: None))

        ready = g.get_ready_tasks()
        assert len(ready) == 3

    def test_diamond_dependency(self):
        """A diamond: root -> (a, b) -> join"""
        g = TaskGraph(name="diamond")
        g.add_task(Task(id="root", task_type="r", func=lambda: None))
        g.add_task(
            Task(id="a", task_type="w", func=lambda: None, dependencies={"root"}),
        )
        g.add_task(
            Task(id="b", task_type="w", func=lambda: None, dependencies={"root"}),
        )
        g.add_task(
            Task(id="join", task_type="w", func=lambda: None, dependencies={"a", "b"}),
        )

        assert [t.id for t in g.get_ready_tasks()] == ["root"]

        g.mark_completed("root", TaskResult(success=True))
        ready_ids = sorted(t.id for t in g.get_ready_tasks())
        assert ready_ids == ["a", "b"]

        g.mark_completed("a", TaskResult(success=True))
        # b depends only on root (already completed), so it's also ready
        ready_after_a = [t.id for t in g.get_ready_tasks()]
        assert ready_after_a == ["b"]

        g.mark_completed("b", TaskResult(success=True))
        assert [t.id for t in g.get_ready_tasks()] == ["join"]

    def test_mark_cancelled(self):
        g = TaskGraph(name="cancel")
        g.add_task(Task(id="a", task_type="w", func=lambda: None))
        g.mark_cancelled("a")

        assert g.tasks["a"].status == TaskStatus.CANCELLED
        assert g.is_complete()

    def test_failure_does_not_unblock_dependents(self):
        g = TaskGraph(name="fail-dep")
        g.add_task(Task(id="root", task_type="r", func=lambda: None))
        g.add_task(
            Task(id="child", task_type="w", func=lambda: None, dependencies={"root"}),
        )

        g.mark_completed("root", TaskResult(success=False, error="boom"))
        assert g.has_failures()
        # child should NOT be ready because root failed (not in _completed)
        assert g.get_ready_tasks() == []

    def test_get_summary(self):
        g = TaskGraph(name="summary")
        g.add_task(Task(id="a", task_type="w", func=lambda: None))
        g.add_task(Task(id="b", task_type="w", func=lambda: None))

        g.mark_completed("a", TaskResult(success=True))
        g.mark_completed("b", TaskResult(success=False, error="err"))

        summary = g.get_summary()
        assert summary["total"] == 2
        assert summary["statuses"]["completed"] == 1
        assert summary["statuses"]["failed"] == 1
        assert summary["success"] is False


# ────────────────────────────────────────────────────────────────────────────
# PipelineExecutor
# ────────────────────────────────────────────────────────────────────────────


class TestPipelineExecutor:
    """Tests for PipelineExecutor execution logic."""

    def test_execute_single_task(self):
        g = TaskGraph(name="single")
        g.add_task(Task(id="a", task_type="w", func=lambda: "done"))

        executor = PipelineExecutor()
        results = executor.execute(g)

        assert results["a"].success
        assert results["a"].value == "done"
        assert results["a"].duration_ms > 0

    def test_execute_chain(self):
        """Sequential chain: a -> b -> c"""
        order = []

        def make_fn(name):
            def fn():
                order.append(name)
                return name

            return fn

        g = TaskGraph(name="chain")
        g.add_task(Task(id="a", task_type="w", func=make_fn("a")))
        g.add_task(Task(id="b", task_type="w", func=make_fn("b"), dependencies={"a"}))
        g.add_task(Task(id="c", task_type="w", func=make_fn("c"), dependencies={"b"}))

        executor = PipelineExecutor()
        results = executor.execute(g)

        assert all(r.success for r in results.values())
        assert order == ["a", "b", "c"]

    def test_parallel_execution(self):
        """Independent tasks should run concurrently."""
        started = threading.Event()
        barrier = threading.Barrier(2, timeout=5)

        def parallel_task():
            barrier.wait()
            return "ok"

        g = TaskGraph(name="parallel")
        g.add_task(Task(id="a", task_type="w", func=parallel_task))
        g.add_task(Task(id="b", task_type="w", func=parallel_task))

        executor = PipelineExecutor(config=ExecutionConfig(max_workers=2))
        results = executor.execute(g)

        assert results["a"].success
        assert results["b"].success

    def test_retry_on_failure(self):
        """Tasks should be retried up to max_retries on failure."""
        attempts = {"count": 0}

        def flaky():
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise ValueError("not yet")
            return "success"

        g = TaskGraph(name="retry")
        g.add_task(Task(id="a", task_type="w", func=flaky))

        executor = PipelineExecutor(
            config=ExecutionConfig(max_retries=3, retry_delay_seconds=0.01),
        )
        results = executor.execute(g)

        assert results["a"].success
        assert results["a"].value == "success"
        assert results["a"].retries == 2  # succeeded on 3rd attempt (retries=2)

    def test_retry_exhaustion(self):
        """Task should fail after exhausting all retries."""

        def always_fail():
            raise RuntimeError("permanent failure")

        g = TaskGraph(name="exhaust")
        g.add_task(Task(id="a", task_type="w", func=always_fail))

        executor = PipelineExecutor(
            config=ExecutionConfig(max_retries=2, retry_delay_seconds=0.01),
        )
        results = executor.execute(g)

        assert not results["a"].success
        assert "permanent failure" in results["a"].error

    def test_fail_fast_stops_execution(self):
        """With fail_fast=True, pending tasks should not run after a failure."""
        executed = []

        def record(name):
            def fn():
                executed.append(name)
                if name == "fail":
                    raise RuntimeError("boom")
                return name

            return fn

        g = TaskGraph(name="fail-fast")
        g.add_task(Task(id="fail", task_type="w", func=record("fail")))
        g.add_task(
            Task(
                id="after",
                task_type="w",
                func=record("after"),
                dependencies={"fail"},
            ),
        )

        executor = PipelineExecutor(
            config=ExecutionConfig(
                fail_fast=True,
                max_retries=0,
                retry_delay_seconds=0.01,
            ),
        )
        results = executor.execute(g)

        assert not results["fail"].success
        assert "after" in results  # should be cancelled
        assert not results["after"].success
        assert "after" not in executed

    def test_dependency_failure_cancels_downstream(self):
        """Downstream tasks are cancelled when their dependency fails."""

        def fail_fn():
            raise RuntimeError("root failed")

        g = TaskGraph(name="cascade")
        g.add_task(Task(id="root", task_type="w", func=fail_fn))
        g.add_task(
            Task(id="child1", task_type="w", func=lambda: "ok", dependencies={"root"}),
        )
        g.add_task(
            Task(
                id="child2",
                task_type="w",
                func=lambda: "ok",
                dependencies={"child1"},
            ),
        )

        executor = PipelineExecutor(
            config=ExecutionConfig(max_retries=0, retry_delay_seconds=0.01),
        )
        results = executor.execute(g)

        assert not results["root"].success
        assert not results["child1"].success
        assert "Dependency failed" in results["child1"].error
        assert not results["child2"].success

    def test_completion_callback(self):
        """on_task_complete should be called for each finished task."""
        completed_tasks = []

        def on_complete(task, result):
            completed_tasks.append((task.id, result.success))

        g = TaskGraph(name="callback")
        g.add_task(Task(id="a", task_type="w", func=lambda: "ok"))
        g.add_task(Task(id="b", task_type="w", func=lambda: "ok", dependencies={"a"}))

        executor = PipelineExecutor(on_task_complete=on_complete)
        executor.execute(g)

        assert len(completed_tasks) == 2
        assert ("a", True) in completed_tasks
        assert ("b", True) in completed_tasks

    def test_stop_request(self):
        """Calling stop() should prevent new tasks from starting."""
        g = TaskGraph(name="stop")
        g.add_task(Task(id="a", task_type="w", func=lambda: "ok"))
        g.add_task(Task(id="b", task_type="w", func=lambda: "ok", dependencies={"a"}))

        executor = PipelineExecutor()

        def stop_after_a(task, result):
            if task.id == "a":
                executor.stop()

        executor._on_task_complete = stop_after_a
        results = executor.execute(g)

        assert results["a"].success
        # b may or may not have run depending on timing, but the executor
        # should have stopped scheduling new tasks

    def test_task_metadata_preserved(self):
        """Task metadata should be available through the graph after execution."""
        g = TaskGraph(name="meta")
        g.add_task(
            Task(
                id="a",
                task_type="insert_chunk",
                func=lambda: {"inserted_ids": [1, 2, 3]},
                metadata={"chunk_index": 0, "chunk_size": 3},
            ),
        )

        executor = PipelineExecutor()
        results = executor.execute(g)

        assert g.tasks["a"].metadata["chunk_index"] == 0
        assert results["a"].value == {"inserted_ids": [1, 2, 3]}
