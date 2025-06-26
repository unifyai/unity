import asyncio
import json
import os
import time
from typing import Optional

# Configuration
EVENT_SERVER_HOST = "127.0.0.1"
EVENT_SERVER_PORT = 8090
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds

# Global variables to hold the connection
_reader: Optional[asyncio.StreamReader] = None
_writer: Optional[asyncio.StreamWriter] = None
_connection_established = False
_last_connection_attempt = 0.0


async def _ensure_connection():
    """Ensure we have a connection to the event server with retry logic"""
    global _reader, _writer, _connection_established, _last_connection_attempt

    if _connection_established and _writer is not None:
        # Check if connection is still alive
        try:
            # Try to write a ping (empty line) to test connection
            _writer.write(
                (
                    json.dumps({"topic": "ping", "to": "past", "event": {}}) + "\n"
                ).encode()
            )
            await _writer.drain()
            return
        except Exception:
            # Connection is dead, reset and reconnect
            _connection_established = False
            _reader = None
            _writer = None

    # Rate limiting for connection attempts
    current_time = time.time()
    if current_time - _last_connection_attempt < RETRY_DELAY:
        await asyncio.sleep(RETRY_DELAY - (current_time - _last_connection_attempt))

    _last_connection_attempt = current_time

    # Try to establish connection with retries
    for attempt in range(MAX_RETRIES):
        try:
            _reader, _writer = await asyncio.open_connection(
                EVENT_SERVER_HOST, EVENT_SERVER_PORT
            )
            _connection_established = True
            print(
                f"Connected to event server at {EVENT_SERVER_HOST}:{EVENT_SERVER_PORT}"
            )
            return
        except Exception as e:
            print(f"Connection attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
            else:
                print(
                    f"Failed to connect to event server at {EVENT_SERVER_HOST}:{EVENT_SERVER_PORT} after {MAX_RETRIES} attempts"
                )
                raise


async def publish_event(ev: dict):
    """Publish an event to the event server"""
    global _writer

    await _ensure_connection()

    if _writer is None:
        raise RuntimeError("No connection to event server")

    try:
        ev_str = json.dumps(ev) + "\n"
        _writer.write(ev_str.encode())
        await _writer.drain()
    except Exception as e:
        # Connection might be broken, reset and retry once
        print(f"Failed to publish event, connection may be broken: {e}")
        _writer = None

        # Try one more time
        await _ensure_connection()
        if _writer is None:
            raise RuntimeError("Failed to reconnect to event server")

        ev_str = json.dumps(ev) + "\n"
        _writer.write(ev_str.encode())
        await _writer.drain()


async def close_connection():
    """Close the connection to the event server"""
    global _reader, _writer, _connection_established

    if _writer is not None:
        _writer.close()
        await _writer.wait_closed()
        _reader = None
        _writer = None
        _connection_established = False
        print("Disconnected from event server")


def get_reader():
    """Get the current reader for event collection"""
    return _reader


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
