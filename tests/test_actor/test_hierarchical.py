import ast
import asyncio
import contextlib
import functools
import json
import logging
import sys
import textwrap
import traceback

import pytest
import unify
import unity
from pydantic import BaseModel, Field
from unittest.mock import AsyncMock, MagicMock, patch

from unity.actor.hierarchical_actor import (
    CacheInvalidateSpec,
    CacheStepRange,
    FunctionPatch,
    HierarchicalActor,
    HierarchicalActorHandle,
    ImplementationDecision,
    InterjectionDecision,
    StateVerificationDecision,
    VerificationAssessment,
    _HierarchicalHandleState,
)
from unity.function_manager.function_manager import FunctionManager
from unity.conversation_manager.handle import ConversationManagerHandle
from unity.common.async_tool_loop import SteerableToolHandle
from unity.controller.browser_backends import BrowserAgentError


# ────────────────────────────────────────────────────────────────────────────
# Logging Setup
# ────────────────────────────────────────────────────────────────────────────

logging.getLogger("urllib3").propagate = False
logging.getLogger("websockets").propagate = False
logging.getLogger("openai").setLevel(logging.INFO)
logging.getLogger("httpcore").setLevel(logging.INFO)
logging.getLogger("UnifyAsyncLogger").setLevel(logging.INFO)

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
if not root_logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(levelname)s][%(name)s] %(message)s")
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

unity.init(overwrite=True)


# ────────────────────────────────────────────────────────────────────────────
# Shared Mock Classes
# ────────────────────────────────────────────────────────────────────────────


class NoKeychainBrowser:
    """
    Mock browser that prevents Keychain prompts during tests.
    
    Args:
        url: URL to return from get_current_url()
        screenshot: Screenshot data to return from get_screenshot()
        with_backend_mocks: If True, adds MagicMock backend with barrier/interrupt
    """
    
    def __init__(
        self,
        url: str = "",
        screenshot: str = "",
        with_backend_mocks: bool = False,
    ):
        self._url = url
        self._screenshot = screenshot
        if with_backend_mocks:
            self.backend = MagicMock()
            self.backend.barrier = AsyncMock()
            self.backend.interrupt_current_action = AsyncMock()
        else:
            self.backend = object()

    async def get_current_url(self) -> str:
        return self._url

    async def get_screenshot(self) -> str:
        return self._screenshot

    def stop(self) -> None:
        pass


class SimpleMockVerificationClient:
    """
    Mock verification client that always returns success.
    Use for tests that don't need to control verification outcomes.
    """

    def __init__(self):
        self.generate = AsyncMock(side_effect=self._side_effect)
        self._current_format = VerificationAssessment

    def set_response_format(self, model):
        self._current_format = model

    def reset_response_format(self):
        self._current_format = VerificationAssessment

    def reset_messages(self):
        pass

    def set_system_message(self, message):
        pass

    async def _side_effect(self, *args, **kwargs):
        if self._current_format.__name__ == "StateVerificationDecision":
            return StateVerificationDecision(
                matches=True,
                reason="Mock: precondition satisfied.",
            ).model_dump_json()

        return VerificationAssessment(
            status="ok",
            reason="Mock verification success.",
        ).model_dump_json()


class ConfigurableMockVerificationClient:
    """
    Mock verification client with configurable per-function behavior.
    Use for tests that need to control verification outcomes and timing.
    """

    def __init__(self):
        self.behaviors = {}
        self.generate = AsyncMock(side_effect=self._side_effect)
        self._current_format = VerificationAssessment

    def set_behavior(self, func_name, delay_or_sequence=0, status="ok", reason="Mock success", *, sequence=None):
        """
        Configure behavior for a specific function.
        
        Supports multiple call signatures:
        - set_behavior(func_name, sequence) - list of (status, reason) tuples
        - set_behavior(func_name, delay, status, reason) - single response with delay
        - set_behavior(func_name, delay, status, reason, sequence=...) - with sequence
        """
        # Detect if second arg is a sequence (list) or delay (number)
        if isinstance(delay_or_sequence, list):
            # Called as: set_behavior(func_name, sequence)
            self.behaviors[func_name] = {
                "delay": 0,
                "status": "ok",
                "reason": "Mock success",
                "sequence": list(delay_or_sequence),
                "calls": 0,
            }
        else:
            # Called as: set_behavior(func_name, delay, status, reason, ...)
            self.behaviors[func_name] = {
                "delay": delay_or_sequence,
                "status": status,
                "reason": reason,
                "sequence": list(sequence or []),
                "calls": 0,
            }

    def set_response_format(self, model):
        self._current_format = model

    def reset_response_format(self):
        self._current_format = VerificationAssessment

    def reset_messages(self):
        pass

    def set_system_message(self, message):
        pass

    async def _side_effect(self, *args, **kwargs):
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

        # Handle StateVerificationDecision (precondition checks)
        if self._current_format.__name__ == "StateVerificationDecision":
            return StateVerificationDecision(
                matches=True,
                reason="Mock: precondition satisfied.",
            ).model_dump_json()

        # Extract function name from prompt
        func_name = None
        for line in prompt.split("\n"):
            if "Function Under Review:" in line and "`" in line:
                parts = line.split("`")
                if len(parts) >= 2:
                    raw_name = parts[1]
                    func_name = raw_name.split("(")[0].strip()
                    break

        # Check for configured behavior
        if func_name and func_name in self.behaviors:
            behavior = self.behaviors[func_name]
            
            # Apply delay if configured
            if behavior.get("delay", 0) > 0:
                await asyncio.sleep(behavior["delay"])
            
            # Use sequence if available
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

        # Default: return success
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
        payload_dict = {
            "action": "implement_function",
            "reason": "Applying mock fix.",
            "code": self._new_code,
        }
        return json.dumps(payload_dict)

    def reset_messages(self):
        pass

    def set_response_format(self, model):
        pass

    def reset_response_format(self):
        pass


# ────────────────────────────────────────────────────────────────────────────
# Shared Helper Functions
# ────────────────────────────────────────────────────────────────────────────


async def wait_for_state(task, expected_state, timeout=60, poll=0.5):
    """Poll the task's state until it matches expected_state or times out."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if task._state == expected_state:
            return
        await asyncio.sleep(poll)
    tail = "\n".join(task.action_log[-15:]) if hasattr(task, 'action_log') else ""
    raise AssertionError(
        f"Timed out waiting for state {expected_state.name}; "
        f"current state={task._state.name}\n--- Log Tail ---\n{tail}"
    )


async def wait_for_log_entry(task, log_substring: str, timeout=60, poll=0.5):
    """Poll the task's action_log until a specific substring appears or times out."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        log_content = "\n".join(task.action_log)
        if log_substring in log_content:
            return
        await asyncio.sleep(poll)
    tail = "\n".join(task.action_log[-20:])
    raise AssertionError(
        f"Timed out waiting for log entry '{log_substring}'.\n--- Log Tail ---\n{tail}"
    )

