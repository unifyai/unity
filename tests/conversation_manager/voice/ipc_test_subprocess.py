#!/usr/bin/env python3
"""
Minimal test subprocess for IPC integration testing.

This script uses the REAL IPC client code from unity.conversation_manager.domains.ipc_socket
to test bidirectional communication with the parent process.

It mimics what the real voice agent (call.py) does:
1. Connects to parent's IPC socket
2. Signals readiness
3. Receives events from parent (like call guidance)
4. Sends events back to parent (like acknowledgments)

Usage:
    python ipc_test_subprocess.py full_roundtrip
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# Add the project root to path so we can import unity modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from unity.conversation_manager.domains.ipc_socket import (
    CallEventSocketClient,
    CM_EVENT_SOCKET_ENV,
)


async def full_roundtrip():
    """Full bidirectional test: ready → guidance → ack."""
    socket_path = os.environ.get(CM_EVENT_SOCKET_ENV)
    if not socket_path:
        print("ERROR: No socket path in environment", file=sys.stderr)
        sys.exit(1)

    client = CallEventSocketClient(socket_path)
    if not await client.connect():
        print("ERROR: Failed to connect to parent socket", file=sys.stderr)
        sys.exit(1)

    received_guidance = None
    guidance_ready = asyncio.Event()

    async def on_event(channel: str, event_json: str):
        nonlocal received_guidance
        print(f"RECEIVED: {channel} = {event_json}")
        if channel == "app:call:call_guidance":
            data = json.loads(event_json)
            payload = data.get("payload", data)
            received_guidance = {"content": payload.get("content", "")}
            guidance_ready.set()

    # Start receiving
    started = await client.start_receive_loop(on_event)
    if not started:
        print("ERROR: Failed to start receive loop", file=sys.stderr)
        sys.exit(1)

    # Send "ready" to parent (using app:call:* namespace like production)
    await client.send_event(
        "app:call:ready",
        json.dumps({"status": "ready"}),
    )
    print("Sent 'ready' to parent, waiting for guidance...")

    # Wait for guidance
    try:
        await asyncio.wait_for(guidance_ready.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        print("ERROR: Timeout waiting for guidance", file=sys.stderr)
        sys.exit(1)

    # Send ack with the content we received (using app:call:* namespace)
    content = received_guidance.get("content", "")
    await client.send_event(
        "app:call:ack",
        json.dumps({"received_content": content}),
    )
    print(f"SUCCESS: Full roundtrip complete, received content: {content}")

    await client.close()


def main():
    if len(sys.argv) < 2 or sys.argv[1] != "full_roundtrip":
        print("Usage: ipc_test_subprocess.py full_roundtrip", file=sys.stderr)
        sys.exit(1)

    asyncio.run(full_roundtrip())


if __name__ == "__main__":
    main()
