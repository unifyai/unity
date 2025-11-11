import pytest

from unity.common.async_tool_loop import _nested_steer_on


class _TaskInfoMeta:
    def __init__(self, name: str, call_id: str, handle):
        self.name = name
        self.call_id = call_id
        self.handle = handle


class _TaskContainer:
    def __init__(self, info: dict):
        self.task_info = info


class NodeHandle:
    """Minimal steerable node with nested children via _task.task_info."""

    def __init__(self, name: str):
        self.name = name
        # Optional tool identity used by nested_steer matching (canonical "Class.method")
        # For these tests we treat <Name>.run as the tool id when provided.
        self._loop_id = None
        self.paused = 0
        self.resumed = 0
        self.stopped = 0
        self.interjections: list[str] = []
        self._task = _TaskContainer({})

    # Steering surface
    async def interject(self, message: str, **_):
        self.interjections.append(message)
        return None

    def stop(self, *_, **__):
        self.stopped += 1
        return "stopped"

    def pause(self, *_, **__):
        self.paused += 1
        return "paused"

    def resume(self, *_, **__):
        self.resumed += 1
        return "resumed"


def _wire_children(parent: NodeHandle, mapping: dict[str, NodeHandle]):
    info = {}
    for i, (name, h) in enumerate(mapping.items()):
        # Assign a unique tool identity per wired child so structure-based matching
        # can target branches deterministically.
        h._loop_id = f"{name}.run"
        info[id(h)] = _TaskInfoMeta(name=name, call_id=f"cid-{i}", handle=h)
    parent._task = _TaskContainer(info)


@pytest.mark.asyncio
async def test_conditions_any_full_triggers_parent_then():
    """Interject parent if either A or B fully propagate."""

    root = NodeHandle("Root")
    a = NodeHandle("A")
    b = NodeHandle("B")
    # Deep child for A so A can be FULL
    a_deep = NodeHandle("Deep1")
    _wire_children(a, {"Deep1": a_deep})
    _wire_children(root, {"A": a, "B": b})

    spec = {
        "children": [
            {
                "tool": "A.run",
                "children": [
                    {"tool": "Deep1.run", "steps": [{"method": "pause"}]},
                ],
            },
            {
                "tool": "B.run",
                # No children/steps – remains NONE
            },
        ],
        "conditions": [
            {
                "when": {
                    "any": [
                        {"child": {"tool": "A.run"}, "status": "full"},
                        {"child": {"tool": "B.run"}, "status": "full"},
                    ],
                },
                "then": [{"method": "interject", "args": "children ready"}],
            },
        ],
    }

    res = await _nested_steer_on(root, spec)

    # Parent interjection should have been applied at the root path
    assert any(
        (item.get("method") == "interject") and (item.get("path") == ["NodeHandle"])
        for item in (res.get("applied") or [])
    ), "Expected root interject when any branch is full"
    # Status map should indicate A.run is full at the root level entry
    assert any(
        (v.get("children", {}).get("A.run") == "full")
        for v in res.get("status", {}).values()
    )


@pytest.mark.asyncio
async def test_conditions_all_partial_triggers_parent_pause():
    """Pause parent if both branches are partial (mix of full/none down the specified tree)."""

    root = NodeHandle("Root")
    b1 = NodeHandle("Branch1")
    b2 = NodeHandle("Branch2")
    # Branch1: Deep1 present, Deep2 missing ⇒ partial
    b1_d1 = NodeHandle("Deep1")
    _wire_children(b1, {"Deep1": b1_d1})
    # Branch2: DeepA present, DeepB missing ⇒ partial
    b2_da = NodeHandle("DeepA")
    _wire_children(b2, {"DeepA": b2_da})
    _wire_children(root, {"Branch1": b1, "Branch2": b2})

    spec = {
        "children": [
            {
                "tool": "Branch1.run",
                "children": [
                    {"tool": "Deep1.run", "steps": [{"method": "pause"}]},
                    {"tool": "Deep2.run", "steps": [{"method": "pause"}]},
                ],
            },
            {
                "tool": "Branch2.run",
                "children": [
                    {"tool": "DeepA.run", "steps": [{"method": "pause"}]},
                    {"tool": "DeepB.run", "steps": [{"method": "pause"}]},
                ],
            },
        ],
        "conditions": [
            {
                "when": {
                    "all": [
                        {"child": {"tool": "Branch1.run"}, "status": "partial"},
                        {"child": {"tool": "Branch2.run"}, "status": "partial"},
                    ],
                },
                "then": [{"method": "pause"}],
            },
        ],
    }

    res = await _nested_steer_on(root, spec)

    # Root pause should have been applied as a "then" step at the current node
    assert any(
        (item.get("method") == "pause") and (item.get("path") == ["NodeHandle"])
        for item in (res.get("applied") or [])
    ), "Expected root pause to be applied when both branches are partial"
