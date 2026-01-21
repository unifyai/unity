"""
Shared utilities for HierarchicalActor tests.

Keep these helpers narrowly scoped to HierarchicalActor-specific behavior so the
tests remain readable and consistent across the modularized test files.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional
from unittest.mock import AsyncMock

from unity.actor.hierarchical_actor import VerificationAssessment


class SimpleMockVerificationClient:
    """
    Mock verification client that always returns success.

    Use for tests that don't need to control verification outcomes.
    """

    def __init__(self):
        self.generate = AsyncMock(side_effect=self._side_effect)
        self._current_format = VerificationAssessment

    def set_response_format(self, model: Any):
        self._current_format = model

    def reset_response_format(self):
        self._current_format = VerificationAssessment

    def reset_messages(self):
        pass

    def set_system_message(self, message: str):
        _ = message

    async def _side_effect(self, *args, **kwargs):
        _ = (args, kwargs)
        return VerificationAssessment(
            status="ok",
            reason="Mock verification success.",
        ).model_dump_json()


class ConfigurableMockVerificationClient:
    """
    Mock verification client with configurable per-function behavior.

    Useful for verification timing/preemption tests and controlled failure flows.
    """

    def __init__(self):
        self.behaviors: dict[str, dict[str, Any]] = {}
        self.generate = AsyncMock(side_effect=self._side_effect)
        self._current_format = VerificationAssessment

    def set_behavior(
        self,
        func_name: str,
        delay_or_sequence: Any = 0,
        status: str = "ok",
        reason: str = "Mock success",
        *,
        sequence: Optional[list[tuple[str, str]]] = None,
    ):
        # Detect if second arg is a sequence (list) or delay (number)
        if isinstance(delay_or_sequence, list):
            self.behaviors[func_name] = {
                "delay": 0,
                "status": "ok",
                "reason": "Mock success",
                "sequence": list(delay_or_sequence),
                "calls": 0,
            }
        else:
            self.behaviors[func_name] = {
                "delay": delay_or_sequence,
                "status": status,
                "reason": reason,
                "sequence": list(sequence or []),
                "calls": 0,
            }

    def set_response_format(self, model: Any):
        self._current_format = model

    def reset_response_format(self):
        self._current_format = VerificationAssessment

    def reset_messages(self):
        pass

    def set_system_message(self, message: str):
        _ = message

    async def _side_effect(self, *args, **kwargs):
        _ = args

        # Extract prompt text from messages
        messages = kwargs.get("messages", [])
        prompt = ""
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, str):
                prompt += content
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        prompt += block["text"]
                    elif isinstance(block, str):
                        prompt += block

        # Extract function name from prompt
        func_name = None
        for line in prompt.split("\n"):
            if "Function Under Review:" in line and "`" in line:
                parts = line.split("`")
                if len(parts) >= 2:
                    raw_name = parts[1]
                    func_name = raw_name.split("(")[0].strip()
                    break

        if func_name and func_name in self.behaviors:
            behavior = self.behaviors[func_name]
            if behavior.get("delay", 0) > 0:
                await asyncio.sleep(behavior["delay"])

            if behavior["sequence"]:
                idx = min(behavior["calls"], len(behavior["sequence"]) - 1)
                status, reason = behavior["sequence"][idx]
                behavior["calls"] += 1
            else:
                status = behavior["status"]
                reason = behavior["reason"]

            return VerificationAssessment(
                status=status,
                reason=reason,
            ).model_dump_json()

        return VerificationAssessment(
            status="ok",
            reason="Mock verification success.",
        ).model_dump_json()


class MockImplementationClient:
    """Mock implementation client for testing recovery/reimplementation flows."""

    def __init__(self, new_code: str):
        self._new_code = new_code
        self.generate = AsyncMock(side_effect=self._get_payload)

    async def _get_payload(self, *args, **kwargs):
        _ = (args, kwargs)
        payload_dict = {
            "action": "implement_function",
            "reason": "Applying mock fix.",
            "code": self._new_code,
        }
        return json.dumps(payload_dict)

    def reset_messages(self):
        pass

    def set_response_format(self, model: Any):
        _ = model

    def reset_response_format(self):
        pass


async def wait_for_state(task, expected_state, timeout: float = 60, poll: float = 0.1):
    """Poll the handle state until it matches expected_state."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if task._state == expected_state:
            return
        await asyncio.sleep(poll)
    tail = "\n".join(task.action_log[-15:]) if hasattr(task, "action_log") else ""
    raise AssertionError(
        f"Timed out waiting for state {expected_state.name}; "
        f"current state={task._state.name}\n--- Log Tail ---\n{tail}",
    )


async def wait_for_log_entry(
    task,
    log_substring: str,
    timeout: float = 60,
    poll: float = 0.5,
):
    """Poll the handle's action_log until a substring appears."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        log_content = "\n".join(task.action_log)
        if log_substring in log_content:
            return
        await asyncio.sleep(poll)
    # Final check to avoid edge races where logs land at the deadline.
    log_content = "\n".join(task.action_log)
    if log_substring in log_content:
        return
    tail = "\n".join(task.action_log[-20:])
    raise AssertionError(
        f"Timed out waiting for log entry '{log_substring}'.\n--- Log Tail ---\n{tail}",
    )
