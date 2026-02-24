"""Tests for ComputerPrimitives interject queue registry and remote-control broadcast."""

import asyncio

import pytest

from unity.manager_registry import ManagerRegistry


@pytest.fixture(autouse=True)
def _clear_singleton():
    """Ensure each test gets a fresh ComputerPrimitives singleton."""
    ManagerRegistry.clear()
    yield
    ManagerRegistry.clear()


def _make_cp():
    from unity.function_manager.primitives.runtime import ComputerPrimitives

    return ComputerPrimitives(computer_mode="mock")


# ─── register / deregister ────────────────────────────────────────────────


def test_register_and_deregister():
    """Queue appears in the registry after register, gone after deregister."""
    cp = _make_cp()
    q: asyncio.Queue = asyncio.Queue()

    cp.register_interject_queue(q)
    assert q in cp._interject_queues

    cp.deregister_interject_queue(q)
    assert q not in cp._interject_queues


def test_double_deregister_is_safe():
    """Deregistering a queue that isn't registered does not raise."""
    cp = _make_cp()
    q: asyncio.Queue = asyncio.Queue()
    # Never registered — should be a no-op
    cp.deregister_interject_queue(q)


# ─── broadcast on state change ────────────────────────────────────────────


def test_broadcast_to_all_registered_queues():
    """set_user_remote_control broadcasts to every registered queue."""
    cp = _make_cp()
    q1: asyncio.Queue = asyncio.Queue()
    q2: asyncio.Queue = asyncio.Queue()
    cp.register_interject_queue(q1)
    cp.register_interject_queue(q2)

    cp.set_user_remote_control(True)

    assert not q1.empty()
    assert not q2.empty()
    msg1 = q1.get_nowait()
    msg2 = q2.get_nowait()
    assert "remote control" in msg1["message"].lower()
    assert msg1 == msg2


def test_started_and_stopped_messages_differ():
    """Started and stopped broadcasts carry distinct messages."""
    cp = _make_cp()
    q: asyncio.Queue = asyncio.Queue()
    cp.register_interject_queue(q)

    cp.set_user_remote_control(True)
    started_msg = q.get_nowait()["message"]

    cp.set_user_remote_control(False)
    stopped_msg = q.get_nowait()["message"]

    assert "taken remote control" in started_msg
    assert "released remote control" in stopped_msg
    assert started_msg != stopped_msg


def test_flag_updated_on_state_change():
    """_user_remote_control_active tracks the latest state."""
    cp = _make_cp()
    assert cp._user_remote_control_active is False

    cp.set_user_remote_control(True)
    assert cp._user_remote_control_active is True

    cp.set_user_remote_control(False)
    assert cp._user_remote_control_active is False


# ─── late registration ────────────────────────────────────────────────────


def test_late_registration_gets_immediate_interjection():
    """A queue registered while remote control is active gets the message immediately."""
    cp = _make_cp()
    cp.set_user_remote_control(True)

    q: asyncio.Queue = asyncio.Queue()
    cp.register_interject_queue(q)

    assert not q.empty()
    msg = q.get_nowait()
    assert "taken remote control" in msg["message"]


def test_late_registration_no_interjection_when_inactive():
    """A queue registered while remote control is inactive gets nothing."""
    cp = _make_cp()
    q: asyncio.Queue = asyncio.Queue()
    cp.register_interject_queue(q)

    assert q.empty()


# ─── conversation context ─────────────────────────────────────────────────


def test_conversation_context_included_in_broadcast():
    """When conversation_context is provided, it appears in the broadcast message."""
    cp = _make_cp()
    q: asyncio.Queue = asyncio.Queue()
    cp.register_interject_queue(q)

    cp.set_user_remote_control(True, conversation_context="user: Let me show you")

    msg = q.get_nowait()["message"]
    assert "Recent conversation context:" in msg
    assert "user: Let me show you" in msg


def test_conversation_context_omitted_when_none():
    """When conversation_context is None, the message has no context block."""
    cp = _make_cp()
    q: asyncio.Queue = asyncio.Queue()
    cp.register_interject_queue(q)

    cp.set_user_remote_control(True, conversation_context=None)

    msg = q.get_nowait()["message"]
    assert "Recent conversation context" not in msg


def test_deregistered_queue_does_not_receive_broadcast():
    """After deregistration, a queue no longer receives broadcasts."""
    cp = _make_cp()
    q: asyncio.Queue = asyncio.Queue()
    cp.register_interject_queue(q)
    cp.deregister_interject_queue(q)

    cp.set_user_remote_control(True)

    assert q.empty()
