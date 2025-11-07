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
        "children": {
            "A": {
                "children": {
                    "Deep1": {"steps": [{"method": "pause"}]},
                },
            },
            "B": {
                # No children/steps – remains NONE
            },
        },
        "conditions": [
            {
                "when": {
                    "any": [
                        {"selector": "A", "status": "full"},
                        {"selector": "B", "status": "full"},
                    ],
                },
                "then": [{"method": "interject", "args": "children ready"}],
            },
        ],
    }

    res = await _nested_steer_on(root, spec)

    # Parent interjection should have been applied
    assert "children ready" in root.interjections
    # Status map should indicate A is full
    key = next(iter(res.get("status", {}).keys()))
    # Locate the root entry (path contains class name or custom label); just ensure A is full in any entry
    assert any(
        (v.get("children", {}).get("A") == "full")
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
        "children": {
            "Branch1": {
                "children": {
                    "Deep1": {"steps": [{"method": "pause"}]},
                    "Deep2": {"steps": [{"method": "pause"}]},
                },
            },
            "Branch2": {
                "children": {
                    "DeepA": {"steps": [{"method": "pause"}]},
                    "DeepB": {"steps": [{"method": "pause"}]},
                },
            },
        },
        "conditions": [
            {
                "when": {
                    "all": [
                        {"selector": "Branch1", "status": "partial"},
                        {"selector": "Branch2", "status": "partial"},
                    ],
                },
                "then": [{"method": "pause"}],
            },
        ],
    }

    await _nested_steer_on(root, spec)

    assert root.paused >= 1
