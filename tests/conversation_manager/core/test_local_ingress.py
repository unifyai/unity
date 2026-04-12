from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import aiohttp
import pytest
import pytest_asyncio

from unity.conversation_manager.in_memory_event_broker import (
    create_in_memory_event_broker,
    reset_in_memory_event_broker,
)
from unity.settings import SETTINGS


async def _get_message_on_channel(pubsub, expected_channel: str, timeout: float = 2.0):
    start = asyncio.get_running_loop().time()
    while asyncio.get_running_loop().time() - start < timeout:
        msg = await pubsub.get_message(
            timeout=0.2,
            ignore_subscribe_messages=True,
        )
        if msg is None:
            continue
        if msg["channel"] == expected_channel:
            return msg
    return None


@pytest_asyncio.fixture
async def broker():
    reset_in_memory_event_broker()
    event_broker = create_in_memory_event_broker()
    yield event_broker
    await event_broker.aclose()
    reset_in_memory_event_broker()


@pytest.fixture
def patched_local_session():
    with (
        patch(
            "unity.conversation_manager.comms_manager.SESSION_DETAILS",
        ) as comms_session,
        patch(
            "unity.conversation_manager.local_ingress.SESSION_DETAILS",
        ) as ingress_session,
    ):
        comms_session.assistant.agent_id = 42
        comms_session.assistant.desktop_mode = "ubuntu"
        comms_session.assistant.desktop_url = ""
        comms_session.user.first_name = "Boss"
        comms_session.user.surname = "User"
        comms_session.user.number = "+15555550000"
        comms_session.user.email = "boss@example.com"
        comms_session.user.whatsapp_number = "+15555550000"
        comms_session.user.id = "user-1"

        ingress_session.assistant.agent_id = 42
        ingress_session.user.id = "user-1"
        yield


class TestLocalIngress:
    @pytest.mark.asyncio
    async def test_envelope_endpoint_publishes_sms_event(
        self,
        broker,
        patched_local_session,
    ):
        from unity.conversation_manager.comms_manager import CommsManager
        from unity.conversation_manager.local_ingress import LocalCommsIngress

        with (
            patch.object(SETTINGS.conversation, "LOCAL_COMMS_HOST", "127.0.0.1"),
            patch.object(SETTINGS.conversation, "LOCAL_COMMS_PORT", 0),
            patch(
                "unity.conversation_manager.local_ingress.local_email.is_email_configured",
                return_value=False,
            ),
        ):
            comms_manager = CommsManager(broker)
            comms_manager.loop = asyncio.get_running_loop()
            ingress = LocalCommsIngress(comms_manager)
            await ingress.start()

            try:
                port = ingress._site._server.sockets[0].getsockname()[1]
                async with broker.pubsub() as pubsub:
                    await pubsub.subscribe("app:comms:msg_message")
                    async with aiohttp.ClientSession() as session:
                        response = await session.post(
                            f"http://127.0.0.1:{port}/local/comms/envelope",
                            json={
                                "thread": "msg",
                                "event": {
                                    "body": "Hello from local ingress",
                                    "from_number": "+15551230001",
                                    "contacts": [
                                        {
                                            "contact_id": 7,
                                            "first_name": "Alice",
                                            "surname": "User",
                                            "phone_number": "+15551230001",
                                            "email_address": "alice@example.com",
                                            "whatsapp_number": "+15551230001",
                                        },
                                    ],
                                },
                            },
                        )
                        assert response.status == 200

                    message = await _get_message_on_channel(
                        pubsub,
                        "app:comms:msg_message",
                    )
                    assert message is not None
                    payload = json.loads(message["data"])
                    assert payload["event_name"] == "SMSReceived"
                    assert payload["payload"]["content"] == "Hello from local ingress"
                    assert payload["payload"]["contact"]["contact_id"] == 7
            finally:
                await ingress.stop()

    @pytest.mark.asyncio
    async def test_outbox_endpoint_stores_and_returns_payloads(
        self,
        broker,
        patched_local_session,
    ):
        from unity.conversation_manager.comms_manager import CommsManager
        from unity.conversation_manager.local_ingress import LocalCommsIngress

        with (
            patch.object(SETTINGS.conversation, "LOCAL_COMMS_HOST", "127.0.0.1"),
            patch.object(SETTINGS.conversation, "LOCAL_COMMS_PORT", 0),
            patch(
                "unity.conversation_manager.local_ingress.local_email.is_email_configured",
                return_value=False,
            ),
        ):
            ingress = LocalCommsIngress(CommsManager(broker))
            await ingress.start()

            try:
                port = ingress._site._server.sockets[0].getsockname()[1]
                async with aiohttp.ClientSession() as session:
                    response = await session.post(
                        f"http://127.0.0.1:{port}/local/comms/outbox",
                        json={
                            "thread": "system_error",
                            "event": {"content": "boom"},
                        },
                    )
                    assert response.status == 200

                    outbox = await session.get(
                        f"http://127.0.0.1:{port}/local/comms/outbox",
                    )
                    assert outbox.status == 200
                    body = await outbox.json()
                    assert len(body["items"]) == 1
                    assert body["items"][0]["thread"] == "system_error"
                    assert body["items"][0]["event"] == {"content": "boom"}
                    assert isinstance(body["items"][0]["publish_timestamp"], float)
            finally:
                await ingress.stop()
