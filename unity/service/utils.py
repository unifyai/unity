import asyncio
import json
import time
import unify

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


def get_contact_details(contact_id: int) -> str:
    ctxs = unify.get_active_context()
    read_ctx, write_ctx = ctxs["read"], ctxs["write"]
    assert (
        read_ctx == write_ctx
    ), "read and write contexts must be the same when instantiating a TranscriptManager."
    if read_ctx:
        _ctx = f"{read_ctx}/Contacts"
    else:
        _ctx = "Contacts"

    logs = unify.get_logs(
        context=_ctx,
        filter=f"contact_id == {contact_id}",
        exclude_fields=[
            k for k in unify.get_fields(context=_ctx).keys() if k.endswith("_emb")
        ],
    )
    return logs[0].entries
