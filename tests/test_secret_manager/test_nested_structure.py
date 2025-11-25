import asyncio
import pytest

from unity.secret_manager.secret_manager import SecretManager
from tests.helpers import _handle_project
from tests.test_async_tool_loop.async_helpers import (
    _wait_for_tool_request,
    _wait_for_condition,
)


@pytest.mark.asyncio
@_handle_project
async def test_nested_structure_flat_ask():
    """
    Verify a flat, in‑flight SecretManager.ask loop reports a minimal structure.
    """
    sm = SecretManager()

    h = await sm.ask("Which secrets are currently stored?")

    try:
        structure = await h.nested_structure()  # type: ignore[attr-defined]
        expected = {
            "handle": "ReadOnlyAskGuardHandle(AsyncToolLoopHandle)",
            "tool": "SecretManager.ask",
            "children": [],
        }
        assert structure == expected
    finally:
        try:
            h.stop("cleanup")  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            await asyncio.wait_for(h.result(), timeout=60)  # type: ignore[attr-defined]
        except Exception:
            pass


@pytest.mark.asyncio
@_handle_project
async def test_nested_structure_flat_update_before_nested(monkeypatch):
    """
    Verify a flat, in‑flight SecretManager.update loop reports a minimal structure
    when the first‑turn nested ask has been requested but not yet adopted.
    """
    gate = asyncio.Event()

    # Gate the manager's ask so it does not return a handle yet (keeps structure flat)
    original_ask = SecretManager.ask

    async def _gated_ask(self, *args, **kwargs):
        await gate.wait()
        # Return a simple string result to avoid creating a nested handle in this test
        return "ok"

    # Ensure the dynamic tool name exposed to the LLM remains exactly "ask"
    _gated_ask.__name__ = "ask"  # type: ignore[attr-defined]
    _gated_ask.__qualname__ = "ask"  # type: ignore[attr-defined]

    monkeypatch.setattr(SecretManager, "ask", _gated_ask, raising=True)

    sm = SecretManager()
    h = await sm.update(
        "Please rotate the token used by the Slack bot.",
    )

    try:
        # Wait until the assistant has requested the first-turn 'ask' tool
        client = getattr(h, "_client", None)  # internal test-only access
        assert (
            client is not None
        ), "Expected AsyncToolLoopHandle to expose its client for tests"
        await _wait_for_tool_request(client, "ask")

        # Ask has been requested but is still blocked by the gate → no nested handle yet
        structure = await h.nested_structure()  # type: ignore[attr-defined]
        expected = {
            "handle": "AsyncToolLoopHandle",
            "tool": "SecretManager.update",
            "children": [],
        }
        assert structure == expected
    finally:
        # Release the gate so the loop can finish cleanly
        try:
            gate.set()
        except Exception:
            pass
        try:
            await asyncio.wait_for(h.result(), timeout=120)  # type: ignore[attr-defined]
        except Exception:
            try:
                h.stop("cleanup")  # type: ignore[attr-defined]
            except Exception:
                pass
        finally:
            # Restore original ask just in case (good hygiene within same process)
            try:
                monkeypatch.setattr(SecretManager, "ask", original_ask, raising=True)
            except Exception:
                pass


@pytest.mark.asyncio
@_handle_project
async def test_nested_structure_update_then_ask_nested():
    """
    Verify a nested structure for SecretManager.update → SecretManager.ask
    (update flow is expected to consult ask on the first turn for discovery/verification).
    """
    sm = SecretManager()
    h = await sm.update(
        "Rotate the token for the secret used by our Slack bot and verify the update.",
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

        structure = await h.nested_structure()  # type: ignore[attr-defined]
        expected = {
            "handle": "AsyncToolLoopHandle",
            "tool": "SecretManager.update",
            "children": [
                {
                    "handle": "ReadOnlyAskGuardHandle(AsyncToolLoopHandle)",
                    "tool": "SecretManager.ask",
                    "children": [],
                },
            ],
        }
        assert structure == expected
    finally:
        try:
            h.stop("cleanup")  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            await asyncio.wait_for(h.result(), timeout=120)  # type: ignore[attr-defined]
        except Exception:
            pass
