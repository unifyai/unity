import asyncio
import json
import time
import os
import aiohttp
import requests

# Configuration
EVENT_SERVER_HOST = "127.0.0.1"
EVENT_SERVER_PORT = 8090
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds

# Global variables to hold the connection
reader: asyncio.StreamReader | None = None
writer: asyncio.StreamWriter | None = None
_connection_established = False
_last_connection_attempt = 0.0
admin_headers = {"Authorization": f"Bearer {os.getenv('ORCHESTRA_ADMIN_KEY')}"}
unity_comms_url = os.getenv("UNITY_COMMS_URL")


async def _ensure_connection():
    """Ensure we have a connection to the event server with retry logic"""
    global reader, writer, _connection_established, _last_connection_attempt

    if _connection_established and writer is not None:
        # Check if connection is still alive
        try:
            # Try to write a ping (empty line) to test connection
            writer.write(
                (
                    json.dumps({"topic": "ping", "to": "past", "event": {}}) + "\n"
                ).encode(),
            )
            await writer.drain()
            return
        except Exception:
            # Connection is dead, reset and reconnect
            _connection_established = False
            reader = None
            writer = None

    # Rate limiting for connection attempts
    current_time = time.time()
    if current_time - _last_connection_attempt < RETRY_DELAY:
        await asyncio.sleep(RETRY_DELAY - (current_time - _last_connection_attempt))

    _last_connection_attempt = current_time

    # Try to establish connection with retries
    for attempt in range(MAX_RETRIES):
        try:
            reader, writer = await asyncio.open_connection(
                EVENT_SERVER_HOST,
                EVENT_SERVER_PORT,
            )
            _connection_established = True
            print(
                f"Connected to event server at {EVENT_SERVER_HOST}:{EVENT_SERVER_PORT}",
            )
            return
        except Exception as e:
            print(f"Connection attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
            else:
                print(
                    f"Failed to connect to event server at {EVENT_SERVER_HOST}:{EVENT_SERVER_PORT} after {MAX_RETRIES} attempts",
                )
                raise


async def publish_event(ev: dict):
    """Publish an event to the event server"""
    global writer

    await _ensure_connection()

    if writer is None:
        raise RuntimeError("No connection to event server")

    try:
        ev_str = json.dumps(ev) + "\n"
        writer.write(ev_str.encode())
        await writer.drain()
    except Exception as e:
        # Connection might be broken, reset and retry once
        print(f"Failed to publish event, connection may be broken: {e}")
        writer = None

        # Try one more time
        await _ensure_connection()
        if writer is None:
            raise RuntimeError("Failed to reconnect to event server")

        ev_str = json.dumps(ev) + "\n"
        writer.write(ev_str.encode())
        await writer.drain()


async def close_connection():
    """Close the connection to the event server"""
    global reader, writer, _connection_established

    if writer is not None:
        writer.close()
        await writer.wait_closed()
        reader = None
        writer = None
        _connection_established = False
        print("Disconnected from event server")


async def get_reader():
    """Get the current reader for event collection"""
    global reader
    await _ensure_connection()
    return reader


def get_server_info():
    """Get the current server configuration"""
    return {
        "host": EVENT_SERVER_HOST,
        "port": EVENT_SERVER_PORT,
        "connected": _connection_established,
        "max_retries": MAX_RETRIES,
        "retry_delay": RETRY_DELAY,
    }


async def test_connection():
    """Test the connection to the event server"""
    try:
        await _ensure_connection()
        print("Connection test successful")
        return True
    except Exception as e:
        print(f"Connection test failed: {e}")
        return False


# comms related utils
async def find_assistant_whatsapp_number() -> str | None:
    assistant_number = os.getenv("ASSISTANT_NUMBER")
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"https://api.unify.ai/v0/admin/assistant?phone={assistant_number}",
            headers=admin_headers,
        ) as response:
            if response.status != 200:
                print(f"Failed to get assistant number. Status: {response.status}")
                return None
            response_json = await response.json()
            found_number = response_json["info"][0]["assistant_whatsapp_number"]
            if not found_number:
                print("No WhatsApp number found for assistant")
                return None
    return found_number


async def find_assistant_phone_number(
    target_phone_number: str,
    assistant_whatsapp_number: str,
) -> str | None:
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"https://api.unify.ai/v0/admin/assistant?user_phone={target_phone_number}&assistant_whatsapp_number={assistant_whatsapp_number}",
            headers=admin_headers,
        ) as response:
            if response.status != 200:
                print(f"Failed to get assistant number. Status: {response.status}")
                return None
            response_json = await response.json()
            found_number = response_json["info"][0]["phone"]
            if not found_number:
                print("No phone number found for assistant")
                return None
    return found_number


async def check_conflict(
    assistant_whatsapp_number: str,
    target_whatsapp_number: str,
) -> str | None:
    # get user id
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"https://api.unify.ai/v0/credits",
            headers={
                "Authorization": f"Bearer {os.environ.get('UNIFY_KEY')}",
            },
        ) as response:
            if response.status != 200:
                print(
                    f"Failed to assign new WhatsApp number. Status: {response.status}",
                )
                return None
            response_json = await response.json()
            user_id = response_json["id"]

    # check conflict
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{unity_comms_url}/whatsapp/conflict",
            headers=admin_headers,
            json={
                "user_id": user_id,
                "assistant_whatsapp_number": assistant_whatsapp_number,
                "target_whatsapp_number": target_whatsapp_number,
            },
        ) as response:
            if response.status != 200:
                print(
                    f"Failed to check WhatsApp conflict. Message not sent. Status: {response.status}",
                )
                return False
            response_json = await response.json()
            conflict = response_json["conflict"]

    return conflict


async def assign_new_assistant_whatsapp_number(
    assistant_phone_number: str,
    assistant_whatsapp_number: str,
    *,
    conflict_number: str = None,
) -> str | None:
    # find user whatsapp number
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"https://api.unify.ai/v0/admin/assistant?phone={assistant_phone_number}&assistant_whatsapp_number={assistant_whatsapp_number}",
            headers=admin_headers,
        ) as response:
            if response.status != 200:
                print(
                    f"Failed to assign new WhatsApp number. Status: {response.status}",
                )
                return None
            response_json = await response.json()
            user_whatsapp_number = response_json["info"][0]["user_whatsapp_number"]
            user_phone_number = response_json["info"][0]["phone"]
            if not user_whatsapp_number:
                print("No WhatsApp number found for user")
                return None

    # assign new whatsapp number
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{unity_comms_url}/whatsapp/assign",
            headers=admin_headers,
            json={
                "user_whatsapp_number": user_whatsapp_number,
                "conflict_whatsapp_number": conflict_number,
            },
        ) as response:
            if response.status != 200:
                print(
                    f"Failed to assign new WhatsApp number. Status: {response.status}",
                )
                return None
            response_json = await response.json()
            new_whatsapp_number = response_json["whatsapp_number"]
            if not new_whatsapp_number:
                print("No WhatsApp number found for user")
                return None

    return new_whatsapp_number, user_phone_number


async def send_sms_notification(
    from_number: str,
    to_number: str,
    new_whatsapp_number: str,
) -> bool:
    try:
        print(f"Sending SMS notification from {from_number} to {to_number}")
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{unity_comms_url}/phone/send-text",
                headers=admin_headers,
                json={
                    "From": from_number,
                    "To": to_number,
                    "Body": f"Your WhatsApp number has been reassigned. Your new WhatsApp number is {new_whatsapp_number}",
                },
            ) as response:
                if response.status != 200:
                    print(f"Failed to send SMS. Status: {response.status}")
                    return False

                response_text = await response.text()
                print(f"Response: {response_text}")
                return True
    except aiohttp.ClientError as e:
        print(f"Network error while sending SMS: {e}")
        return False
    except Exception as e:
        print(f"Error sending SMS: {e}")
        return False


async def admin_update_assistant(
    assistant_phone_number: str,
    assistant_new_whatsapp_number: str,
) -> bool:
    async with aiohttp.ClientSession() as session:
        async with session.patch(
            f"https://api.unify.ai/v0/admin/assistant?phone={assistant_phone_number}&new_assistant_whatsapp_number={assistant_new_whatsapp_number}",
            headers=admin_headers,
        ) as response:
            if response.status != 200:
                print(f"Failed to update assistant. Status: {response.status}")
                return False
            return True


# dispatch agent
def dispatch_agent(agent_name: str):
    response = requests.post(
        f"{unity_comms_url}/phone/dispatch-agent",
        headers=admin_headers,
        json={"agent_name": agent_name},
    )
    if response.status_code != 200:
        print(f"Failed to dispatch agent. Status: {response.status_code}")
        return False
    return True
