import asyncio
from asyncio import StreamWriter
import json
import os
import aiohttp
import requests

# Configuration
EVENT_SERVER_HOST = "127.0.0.1"
EVENT_SERVER_PORT = 8090
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds

# Admin headers and URLs
admin_headers = {"Authorization": f"Bearer {os.getenv('ORCHESTRA_ADMIN_KEY')}"}
unity_comms_url = os.getenv("UNITY_COMMS_URL")


async def create_connection(connection_type: str = "general"):
    """Create a new connection to the event server with type identification"""
    # Try to establish connection with retries
    for attempt in range(MAX_RETRIES):
        try:
            reader, writer = await asyncio.open_connection(
                EVENT_SERVER_HOST,
                EVENT_SERVER_PORT,
            )

            # Send connection type identification
            connection_msg = (
                json.dumps({"topic": "init", "type": connection_type}) + "\n"
            )
            writer.write(connection_msg.encode())
            await writer.drain()

            print(
                f"Connected to event server as {connection_type} at "
                f"{EVENT_SERVER_HOST}:{EVENT_SERVER_PORT}",
            )
            return reader, writer
        except Exception as e:
            print(f"Connection attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
            else:
                print(
                    f"Failed to connect to event server at {EVENT_SERVER_HOST}:"
                    f"{EVENT_SERVER_PORT} after {MAX_RETRIES} attempts",
                )
                raise


async def publish_event(ev: dict, writer: StreamWriter = None):
    """Publish an event to the event server"""
    try:
        ev_str = json.dumps(ev) + "\n"
        writer.write(ev_str.encode())
        await writer.drain()
    except Exception as e:
        # Connection might be broken, reset and retry once
        print(f"Failed to publish event, connection may be broken: {e}")


async def close_connection(writer: StreamWriter = None):
    """Close the connection to the event server"""
    if writer is not None:
        writer.close()
        await writer.wait_closed()
        print("Disconnected from event server")


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


# google meet helpers
async def join_meet_on_browser(meet_browser, meet_id: str):
    await meet_browser.act(
        f"Go to the page: https://meet.google.com/{meet_id}",
    )
    await asyncio.sleep(2)

    # Set agent mic
    await meet_browser.act(
        "Click on microphone default",
    )
    await asyncio.sleep(1)
    await meet_browser.act(
        "Select 'agent_sink.monitor'",
    )

    # Set user speaker
    await meet_browser.act(
        "Click on speaker default",
    )
    await asyncio.sleep(1)
    await meet_browser.act("Select 'meet_sink'")

    # Enter name and join
    await meet_browser.act(
        "Click 'your name' textbox",
    )


async def get_meet_join_state(meet_browser) -> str:
    """Use observe to determine Meet state: 'asking' (pre-join) or 'joined'."""
    try:
        state = await meet_browser.observe(
            (
                "In the current Google Meet UI, are we inside the meeting (joined) or still on the pre-join screen asking to join, or name is not filled and join button is not yet active? "
                "Return exactly one word: 'joined' if inside the call, or 'asking' if on the pre-join screen, or 'filling' if name is not filled and join button is not yet active."
            ),
            str,
        )
        if isinstance(state, str):
            s = state.strip().lower()
            if s in ("joined", "asking", "filling"):
                return s
    except Exception:
        ...
    return "filling"


async def enter_name_with_retry(
    meet_browser,
    assistant_name: str,
    max_attempts: int = 3,
) -> bool:
    """Enter name and verify via observe-only whether we've joined or still asking."""
    if not meet_browser:
        return False

    for _ in range(max_attempts):
        try:
            await meet_browser.act(
                f"Input your name as {assistant_name} and press enter",
            )
            await asyncio.sleep(0.5)

            # Observe-only join state check
            try:
                state = await get_meet_join_state(meet_browser)
                print("STATE:", state)
                if state in ("joined", "asking"):
                    return True
            except Exception:
                ...
        except Exception:
            ...
        await asyncio.sleep(0.8)
    return False


async def _is_captions_enabled(meet_browser) -> bool:
    try:
        status = await meet_browser.observe(
            (
                "In the current Google Meet UI, are live captions turned on? "
                "Return only true or false."
            ),
            bool,
        )
        return bool(status)
    except Exception:
        return False


async def ensure_captions_enabled(meet_browser, max_attempts: int = 5):
    for _ in range(max_attempts):
        if await _is_captions_enabled(meet_browser):
            return True
        try:
            await meet_browser.act("Turn on captions")
        except Exception:
            ...
        await asyncio.sleep(0.6)
    return await _is_captions_enabled(meet_browser)
