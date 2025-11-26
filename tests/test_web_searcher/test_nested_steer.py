import asyncio
import pytest

from unity.web_searcher.web_searcher import WebSearcher
from tests.helpers import _handle_project
from tests.test_async_tool_loop.async_helpers import _wait_for_condition


@pytest.mark.asyncio
@_handle_project
async def test_nested_steer_interject_reaches_inner_ask():
    """
    Verify that nested_steer can target the inner WebSearcher.ask loop within
    an update→ask nested configuration and that an interjection is delivered to
    the inner loop (bypassing the outer update loop).
    """
    ws = WebSearcher()

    # Start an update flow that will run an inner ask on the first turn
    h = await ws.update(
        "Add Medium as a gated website (with credentials if present) and summarize the setup.",
    )

    try:
        # Wait deterministically until the nested ask handle has been adopted
        async def _ask_child_adopted():
            try:
                task_info = getattr(getattr(h, "_task", None), "task_info", {})  # type: ignore[attr-defined]
                if isinstance(task_info, dict):
                    return any(
                        getattr(meta, "name", None) == "ask"
                        and getattr(meta, "handle", None) is not None
                        for meta in task_info.values()
                    )
            except Exception:
                return False
            return False

        await _wait_for_condition(_ask_child_adopted, poll=0.01, timeout=60.0)

        # Send an interjection only to the inner ask loop via nested_steer
        msg = "nested-steer interjection for inner ask"
        spec = {
            "children": [
                {
                    "tool": "WebSearcher.ask",
                    "steps": [
                        {"method": "interject", "args": msg},
                    ],
                },
            ],
        }

        await h.nested_steer(spec)  # type: ignore[attr-defined]

        # Locate the adopted inner ask handle
        child_handle = None
        task_info = getattr(getattr(h, "_task", None), "task_info", {})  # type: ignore[attr-defined]
        if isinstance(task_info, dict):
            for meta in task_info.values():
                if (
                    getattr(meta, "name", None) == "ask"
                    and getattr(meta, "handle", None) is not None
                ):
                    child_handle = getattr(meta, "handle", None)
                    break
        assert child_handle is not None, "Expected inner ask handle to be adopted"

        # Assert the interjection is recorded on the inner ask handle's user-visible history
        async def _interjection_visible_on_inner():
            try:
                hist = getattr(child_handle, "_user_visible_history", [])  # type: ignore[attr-defined]
                for m in hist or []:
                    if isinstance(m, dict) and m.get("role") == "user":
                        c = m.get("content")
                        if isinstance(c, str) and c == msg:
                            return True
                        if isinstance(c, dict) and c.get("message") == msg:
                            return True
                return False
            except Exception:
                return False

        await _wait_for_condition(
            _interjection_visible_on_inner,
            poll=0.01,
            timeout=60.0,
        )

        # And confirm it did NOT land on the outer update handle
        outer_hist = getattr(h, "_user_visible_history", [])  # type: ignore[attr-defined]
        assert not any(
            isinstance(m, dict)
            and m.get("role") == "user"
            and (
                (isinstance(m.get("content"), str) and m.get("content") == msg)
                or (
                    isinstance(m.get("content"), dict)
                    and m.get("content", {}).get("message") == msg
                )
            )
            for m in (outer_hist or [])
        ), "Interjection should bypass the outer update loop"
    finally:
        try:
            h.stop("cleanup")  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            await asyncio.wait_for(h.result(), timeout=120)  # type: ignore[attr-defined]
        except Exception:
            pass
