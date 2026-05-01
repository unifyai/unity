from __future__ import annotations

from typing import Any

import pytest
import unillm
from pydantic import BaseModel

from unity.common.reasoning import reason
from unity.function_manager import venv_runner
from unity.function_manager.execution_env import (
    ENVIRONMENT_MODULES,
    create_execution_globals,
)


def test_execution_globals_expose_reason_and_unillm():
    globals_dict = create_execution_globals()

    assert globals_dict["reason"] is reason
    assert globals_dict["unillm"] is unillm
    assert "new_llm_client" in globals_dict
    assert "unillm" in ENVIRONMENT_MODULES


@pytest.mark.asyncio
async def test_venv_runner_reason_routes_through_runtime_rpc(
    monkeypatch: pytest.MonkeyPatch,
):
    class Decision(BaseModel):
        category: str
        needs_reply: bool

    calls: list[tuple[str, dict[str, Any]]] = []

    async def fake_rpc_call(path: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        calls.append((path, kwargs))
        return {"category": "scheduling", "needs_reply": True}

    monkeypatch.setattr(venv_runner, "rpc_call_async", fake_rpc_call)

    result = await venv_runner.reason(
        "Classify this email.",
        system="Use the inbox rubric.",
        response_format=Decision,
    )

    assert isinstance(result, Decision)
    assert result.category == "scheduling"
    assert result.needs_reply is True

    assert calls[0][0] == "runtime.reason"
    sent_kwargs = calls[0][1]
    assert sent_kwargs["prompt"] == "Classify this email."
    assert sent_kwargs["system"] == "Use the inbox rubric."
    assert sent_kwargs["temperature"] == 0.0
    assert sent_kwargs["response_format"]["type"] == "json_schema"
    assert sent_kwargs["response_format"]["json_schema"]["name"] == "Decision"
