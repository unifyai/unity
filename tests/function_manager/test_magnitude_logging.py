"""
Tests for magnitude agent-service logging integration.

Verifies that:
1. _get_current_lineage() reads TOOL_LOOP_LINEAGE correctly
2. ComputerSession.act() passes lineage in the HTTP payload
3. MagnitudeBackend._log_consumer routes logs through Unity's LOGGER
4. _handle_magnitude_debug_payload persists TEXT/IMG/TRACE payloads
5. _log_consumer routes __MAG_DEBUG__ lines to the debug handler
"""

import asyncio
import json
import logging
import pytest
from unittest import mock

from unity.common._async_tool.loop_config import TOOL_LOOP_LINEAGE
from unity.function_manager.computer_backends import (
    _get_current_lineage,
    _handle_magnitude_debug_payload,
    ComputerSession,
    MagnitudeBackend,
)


class TestGetCurrentLineage:
    """_get_current_lineage reads TOOL_LOOP_LINEAGE and returns a list copy."""

    def test_empty_default(self):
        token = TOOL_LOOP_LINEAGE.set([])
        try:
            assert _get_current_lineage() == []
        finally:
            TOOL_LOOP_LINEAGE.reset(token)

    def test_reads_current_lineage(self):
        lineage = ["CodeActActor.act(ab12)", "execute_code(cd34)"]
        token = TOOL_LOOP_LINEAGE.set(lineage)
        try:
            result = _get_current_lineage()
            assert result == lineage
            assert result is not lineage  # must be a copy
        finally:
            TOOL_LOOP_LINEAGE.reset(token)

    def test_nested_lineage(self):
        lineage = [
            "CodeActActor.act(0001)",
            "execute_code(0002)",
            "TaskScheduler.execute(0003)",
        ]
        token = TOOL_LOOP_LINEAGE.set(lineage)
        try:
            assert _get_current_lineage() == lineage
        finally:
            TOOL_LOOP_LINEAGE.reset(token)


class TestComputerSessionActLineage:
    """ComputerSession.act() includes lineage in the HTTP payload."""

    @pytest.mark.asyncio
    async def test_act_sends_lineage(self):
        captured_payloads: list[dict] = []

        session = ComputerSession(
            session_id="test-session",
            mode="desktop",
            agent_base_url="http://fake:3000",
        )

        async def _mock_request(method, endpoint, payload=None):
            captured_payloads.append(
                {"method": method, "endpoint": endpoint, "payload": payload},
            )
            return {"summary": "ok", "screenshot": ""}

        session._request = _mock_request

        lineage = ["CodeActActor.act(ab12)"]
        token = TOOL_LOOP_LINEAGE.set(lineage)
        try:
            await session.act("Click the button")
        finally:
            TOOL_LOOP_LINEAGE.reset(token)

        assert len(captured_payloads) == 1
        sent = captured_payloads[0]
        assert sent["method"] == "POST"
        assert sent["endpoint"] == "/act"
        assert sent["payload"]["task"] == "Click the button"
        assert sent["payload"]["lineage"] == ["CodeActActor.act(ab12)"]

    @pytest.mark.asyncio
    async def test_act_sends_empty_lineage_when_unset(self):
        captured_payloads: list[dict] = []

        session = ComputerSession(
            session_id="test-session",
            mode="web",
            agent_base_url="http://fake:3000",
        )

        async def _mock_request(method, endpoint, payload=None):
            captured_payloads.append(payload)
            return {"summary": "ok", "screenshot": ""}

        session._request = _mock_request

        token = TOOL_LOOP_LINEAGE.set([])
        try:
            await session.act("Scroll down")
        finally:
            TOOL_LOOP_LINEAGE.reset(token)

        assert captured_payloads[0]["lineage"] == []


class TestLogConsumerUsesUnityLogger:
    """MagnitudeBackend._log_consumer routes messages through Unity's LOGGER."""

    @pytest.mark.asyncio
    async def test_log_consumer_emits_to_unity_logger(self):
        backend = MagnitudeBackend.__new__(MagnitudeBackend)
        backend._network_log_queue = asyncio.Queue()
        backend._current_capture_queue = None
        backend._current_processing_seq = None
        backend._log_buffer = {}

        captured: list[str] = []
        unity_logger = logging.getLogger("unity")
        original_level = unity_logger.level

        class _Capture(logging.Handler):
            def emit(self, record):
                captured.append(record.getMessage())

        handler = _Capture()
        handler.setLevel(logging.INFO)
        unity_logger.addHandler(handler)

        try:
            await backend._network_log_queue.put(
                "[CodeActActor.act(ab12)->desktop.act] 🛠️ Action 1/2: ⊙ click (512, 384)",
            )
            await backend._network_log_queue.put(
                "[CodeActActor.act(ab12)->desktop.act] ✅ Completed mouse:click [200ms]",
            )

            consumer = asyncio.create_task(backend._log_consumer())
            await asyncio.sleep(0.05)
            consumer.cancel()
            try:
                await consumer
            except asyncio.CancelledError:
                pass

            assert len(captured) >= 2
            assert "⊙ click (512, 384)" in captured[0]
            assert "Completed mouse:click" in captured[1]
            assert "[CodeActActor.act(ab12)->desktop.act]" in captured[0]
        finally:
            unity_logger.removeHandler(handler)
            unity_logger.setLevel(original_level)

    @pytest.mark.asyncio
    async def test_log_consumer_buffers_when_processing(self):
        backend = MagnitudeBackend.__new__(MagnitudeBackend)
        backend._network_log_queue = asyncio.Queue()
        backend._current_capture_queue = None
        backend._current_processing_seq = 42
        backend._log_buffer = {42: []}

        await backend._network_log_queue.put("[desktop.act] 🛠️ Action 1/1: ⊙ click")

        consumer = asyncio.create_task(backend._log_consumer())
        await asyncio.sleep(0.05)
        consumer.cancel()
        try:
            await consumer
        except asyncio.CancelledError:
            pass

        assert len(backend._log_buffer[42]) == 1
        assert "⊙ click" in backend._log_buffer[42][0]

    @pytest.mark.asyncio
    async def test_log_consumer_routes_to_capture_queue(self):
        backend = MagnitudeBackend.__new__(MagnitudeBackend)
        backend._network_log_queue = asyncio.Queue()
        backend._current_capture_queue = asyncio.Queue()
        backend._current_processing_seq = None
        backend._log_buffer = {}

        await backend._network_log_queue.put("[desktop.act] 💭 Reasoning: click button")

        consumer = asyncio.create_task(backend._log_consumer())
        await asyncio.sleep(0.05)
        consumer.cancel()
        try:
            await consumer
        except asyncio.CancelledError:
            pass

        assert not backend._current_capture_queue.empty()
        msg = backend._current_capture_queue.get_nowait()
        assert "Reasoning: click button" in msg


class TestHandleMagnitudeDebugPayload:
    """_handle_magnitude_debug_payload handles TEXT payloads on the Unity side.

    IMG and TRACE payloads are saved locally by agent-service (not streamed
    over WebSocket), so the Unity-side handler only processes TEXT.
    """

    def test_text_payload_appends_to_log(self, tmp_path):
        with mock.patch(
            "unity.function_manager.computer_backends._MAGNITUDE_LOG_DIR",
            str(tmp_path),
        ):
            payload = json.dumps({"line": "debug coordinate transform x=100 y=200"})
            _handle_magnitude_debug_payload(f"TEXT {payload}")

            log_file = tmp_path / "magnitude.log"
            assert log_file.exists()
            contents = log_file.read_text()
            assert "debug coordinate transform x=100 y=200" in contents

    def test_text_payload_appends_multiple_lines(self, tmp_path):
        with mock.patch(
            "unity.function_manager.computer_backends._MAGNITUDE_LOG_DIR",
            str(tmp_path),
        ):
            _handle_magnitude_debug_payload(
                f'TEXT {json.dumps({"line": "line one"})}',
            )
            _handle_magnitude_debug_payload(
                f'TEXT {json.dumps({"line": "line two"})}',
            )

            lines = (tmp_path / "magnitude.log").read_text().strip().splitlines()
            assert len(lines) == 2
            assert "line one" in lines[0]
            assert "line two" in lines[1]

    def test_img_payload_ignored_by_unity_consumer(self, tmp_path):
        """IMG payloads are saved locally by agent-service, not handled here."""
        with mock.patch(
            "unity.function_manager.computer_backends._MAGNITUDE_LOG_DIR",
            str(tmp_path),
        ):
            payload = json.dumps(
                {
                    "actId": "act123",
                    "label": "planning_screenshot",
                    "base64": "iVBORw0KGgo=",
                },
            )
            _handle_magnitude_debug_payload(f"IMG {payload}")
            assert not (tmp_path / "acts").exists()

    def test_trace_payload_ignored_by_unity_consumer(self, tmp_path):
        """TRACE payloads are saved locally by agent-service, not handled here."""
        with mock.patch(
            "unity.function_manager.computer_backends._MAGNITUDE_LOG_DIR",
            str(tmp_path),
        ):
            payload = json.dumps({"actId": "act456", "task": "click"})
            _handle_magnitude_debug_payload(f"TRACE {payload}")
            assert not (tmp_path / "acts").exists()

    def test_noop_when_log_dir_unset(self, tmp_path):
        with mock.patch(
            "unity.function_manager.computer_backends._MAGNITUDE_LOG_DIR",
            "",
        ):
            _handle_magnitude_debug_payload(
                f'TEXT {json.dumps({"line": "should be dropped"})}',
            )
            assert not (tmp_path / "magnitude.log").exists()

    def test_malformed_json_ignored(self, tmp_path):
        with mock.patch(
            "unity.function_manager.computer_backends._MAGNITUDE_LOG_DIR",
            str(tmp_path),
        ):
            _handle_magnitude_debug_payload("TEXT {not valid json}")
            assert not (tmp_path / "magnitude.log").exists()


class TestLogConsumerDebugRouting:
    """_log_consumer routes __MAG_DEBUG__ lines to the debug handler."""

    @pytest.mark.asyncio
    async def test_debug_payloads_not_forwarded_to_logger(self, tmp_path):
        backend = MagnitudeBackend.__new__(MagnitudeBackend)
        backend._network_log_queue = asyncio.Queue()
        backend._current_capture_queue = None
        backend._current_processing_seq = None
        backend._log_buffer = {}

        captured_by_logger: list[str] = []
        unity_logger = logging.getLogger("unity")
        original_level = unity_logger.level

        class _Capture(logging.Handler):
            def emit(self, record):
                captured_by_logger.append(record.getMessage())

        handler = _Capture()
        handler.setLevel(logging.INFO)
        unity_logger.addHandler(handler)

        try:
            with mock.patch(
                "unity.function_manager.computer_backends._MAGNITUDE_LOG_DIR",
                str(tmp_path),
            ):
                debug_line = "__MAG_DEBUG__ TEXT " + json.dumps({"line": "debug info"})
                normal_line = "[desktop.act] 🛠️ Action 1/1: ⊙ click (100, 200)"

                await backend._network_log_queue.put(debug_line)
                await backend._network_log_queue.put(normal_line)

                consumer = asyncio.create_task(backend._log_consumer())
                await asyncio.sleep(0.1)
                consumer.cancel()
                try:
                    await consumer
                except asyncio.CancelledError:
                    pass

                assert len(captured_by_logger) == 1
                assert "⊙ click (100, 200)" in captured_by_logger[0]

                log_file = tmp_path / "magnitude.log"
                assert log_file.exists()
                assert "debug info" in log_file.read_text()
        finally:
            unity_logger.removeHandler(handler)
            unity_logger.setLevel(original_level)
