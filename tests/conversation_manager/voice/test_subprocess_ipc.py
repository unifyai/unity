"""
tests/conversation_manager/voice/test_subprocess_ipc.py
=======================================================

REAL integration test for subprocess IPC communication.

This test actually spawns a subprocess and verifies bidirectional IPC works.
It does NOT mock the subprocess or socket - it tests the real code paths.

This would have caught Ved's bug where the IPC socket was unidirectional
(child → parent only) and call guidance events couldn't reach the voice agent.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import pytest_asyncio

from unity.conversation_manager.domains.ipc_socket import (
    CallEventSocketServer,
    CM_EVENT_SOCKET_ENV,
)
from unity.conversation_manager.in_memory_event_broker import (
    create_in_memory_event_broker,
    reset_in_memory_event_broker,
)

# Path to the minimal test subprocess script
TEST_SUBPROCESS_SCRIPT = Path(__file__).parent / "ipc_test_subprocess.py"


@pytest_asyncio.fixture
async def event_broker():
    """Real in-memory event broker."""
    reset_in_memory_event_broker()
    broker = create_in_memory_event_broker()
    yield broker
    await broker.aclose()
    reset_in_memory_event_broker()


class TestRealSubprocessIPC:
    """
    Integration tests that spawn REAL subprocesses to verify IPC works.

    These tests do NOT mock:
    - subprocess.Popen (real process spawning)
    - Unix sockets (real socket communication)
    - IPC protocol (real message passing)

    They only skip LiveKit (which requires audio hardware).
    """

    @pytest.mark.asyncio
    async def test_bidirectional_ipc_roundtrip(self, event_broker):
        """
        REAL integration test: Full bidirectional IPC roundtrip.

        This test spawns a REAL subprocess and verifies:
        1. Child → Parent: Subprocess can send events to parent
        2. Parent → Child: Parent can send events to subprocess

        The test flow:
        1. Parent starts socket server
        2. Subprocess connects and sends "ready" to parent
        3. Parent receives "ready", sends "guidance" to subprocess
        4. Subprocess receives "guidance", sends "ack" with content back
        5. Parent verifies full roundtrip completed

        WHY THIS TEST MATTERS:
        Before Ved's fix (commit c34270dc), the IPC socket was unidirectional.
        Events could only flow child→parent. The parent→child direction
        (step 3-4) was completely broken, which meant:
        - Call guidance from ConversationManager never reached the voice agent
        - Proactive speech prompts were lost
        - The voice agent couldn't receive status updates

        This single test would have caught that bug because step 4 would
        have timed out - the subprocess would never receive the guidance.
        """
        received_from_child = []

        async def on_event(channel: str, event_json: str):
            received_from_child.append((channel, event_json))

        server = CallEventSocketServer(
            event_broker,
            on_event=on_event,
            forward_channels=["test:parent:*"],
        )
        proc = None

        try:
            socket_path = await server.start()

            env = os.environ.copy()
            env[CM_EVENT_SOCKET_ENV] = socket_path

            # Spawn subprocess that does full roundtrip
            proc = subprocess.Popen(
                [sys.executable, str(TEST_SUBPROCESS_SCRIPT), "full_roundtrip"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Wait for subprocess to send "ready"
            ready_received = False
            for _ in range(50):
                await asyncio.sleep(0.1)
                for channel, _ in received_from_child:
                    if channel == "test:subprocess:ready":
                        ready_received = True
                        break
                if ready_received:
                    break

            assert ready_received, "Subprocess never sent 'ready' event"

            # Clear and send guidance
            received_from_child.clear()
            await event_broker.publish(
                "test:parent:guidance",
                json.dumps({"content": "Ask about their day"}),
            )

            # Wait for acknowledgment
            ack_received = False
            for _ in range(50):
                await asyncio.sleep(0.1)
                for channel, event_json in received_from_child:
                    if channel == "test:subprocess:ack":
                        data = json.loads(event_json)
                        assert (
                            data["received_content"] == "Ask about their day"
                        ), f"Subprocess received wrong content: {data}"
                        ack_received = True
                        break
                if ack_received:
                    break

            # Get subprocess output for debugging
            proc.terminate()
            stdout, stderr = proc.communicate(timeout=2)

            assert ack_received, (
                f"Full roundtrip failed - subprocess didn't acknowledge guidance. "
                f"This means parent→child IPC is broken. "
                f"stdout: {stdout.decode()}, stderr: {stderr.decode()}"
            )

        finally:
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
            await server.stop()
