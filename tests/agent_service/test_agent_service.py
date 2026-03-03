"""Integration tests for the real agent-service (Node.js).

Level 1 — Startup smoke tests:
    Verify the agent-service starts, listens, and rejects bad requests.
    No browser binaries required.

Level 2 — Round-trip tests:
    Create a headless web session, take a screenshot, and stop it.
    Requires Patchright/Playwright browser binaries installed in the
    agent-service environment.
"""

import aiohttp
import pytest

pytestmark = pytest.mark.asyncio


# ── Level 1: Startup smoke tests ────────────────────────────────────────


class TestStartup:
    """The agent-service starts without crashing and responds to HTTP."""

    async def test_rejects_invalid_mode(self, agent_service_url, auth_headers):
        async with aiohttp.ClientSession(headers=auth_headers) as s:
            async with s.post(
                f"{agent_service_url}/start",
                json={"mode": "invalid"},
            ) as resp:
                assert resp.status == 400
                body = await resp.json()
                assert body["error"] == "bad_request"

    async def test_rejects_missing_mode(self, agent_service_url, auth_headers):
        async with aiohttp.ClientSession(headers=auth_headers) as s:
            async with s.post(
                f"{agent_service_url}/start",
                json={},
            ) as resp:
                assert resp.status == 400


# ── Level 2: Round-trip tests ────────────────────────────────────────────


class TestRoundtrip:
    """Create a browser session, interact with it, and tear it down."""

    async def test_web_session_lifecycle(self, agent_service_url, auth_headers):
        async with aiohttp.ClientSession(headers=auth_headers) as s:
            async with s.post(
                f"{agent_service_url}/start",
                json={"mode": "web", "headless": True},
            ) as resp:
                assert resp.status == 200, await resp.text()
                data = await resp.json()
                session_id = data["sessionId"]
                assert session_id

            async with s.post(
                f"{agent_service_url}/screenshot",
                json={"sessionId": session_id},
            ) as resp:
                assert resp.status == 200, await resp.text()
                data = await resp.json()
                assert data.get("screenshot"), "Expected base64 screenshot data"
                assert len(data["screenshot"]) > 100

            async with s.post(
                f"{agent_service_url}/stop",
                json={"sessionId": session_id},
            ) as resp:
                assert resp.status == 200

    async def test_stop_invalid_session_returns_error(
        self,
        agent_service_url,
        auth_headers,
    ):
        async with aiohttp.ClientSession(headers=auth_headers) as s:
            async with s.post(
                f"{agent_service_url}/stop",
                json={"sessionId": "nonexistent-id"},
            ) as resp:
                assert resp.status in (400, 404)
