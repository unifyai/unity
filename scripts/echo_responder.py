#!/usr/bin/env python3
"""
Echo responder for local chat testing.

Subscribes to ALL ``unity-*`` topics on a Pub/Sub emulator and for every
inbound ``unify_message``, publishes back a ``unify_message_outbound``
echo.  This lets Console <-> Communication <-> Pub/Sub round-trip work
end-to-end without LLM keys or a full Unity runtime.

The responder periodically polls the emulator for new topics so that
assistants created after startup (e.g. via seed or hire) are picked up
automatically.

Usage (standalone):
    PUBSUB_EMULATOR_HOST=localhost:8085 \\
    GCP_PROJECT_ID=local-test-project \\
        python scripts/echo_responder.py

The responder is also started automatically by ``scripts/local.sh`` when
LLM keys are not available.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time

import requests
from google.cloud import pubsub_v1


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _echo_reply(body: str) -> str:
    return f"[Echo] {body}" if body else "[Echo] (empty message)"


_subscriber: pubsub_v1.SubscriberClient | None = None
_publisher: pubsub_v1.PublisherClient | None = None
_subscribed_topics: set[str] = set()
_streaming_pulls: list = []
_stop = False


def _get_emulator_base() -> str:
    host = _env("PUBSUB_EMULATOR_HOST", "localhost:8085")
    if not host.startswith("http"):
        host = f"http://{host}"
    return host


def _discover_topics(project_id: str) -> list[str]:
    """List all topics from the emulator REST API, filtering for unity-* topics."""
    try:
        resp = requests.get(
            f"{_get_emulator_base()}/v1/projects/{project_id}/topics",
            timeout=5,
        )
        resp.raise_for_status()
        topics = resp.json().get("topics", [])
        return [
            t["name"].rsplit("/", 1)[-1]
            for t in topics
            if "/topics/unity-" in t["name"] and "startup" not in t["name"]
        ]
    except Exception as exc:
        print(f"[echo-responder] Topic discovery failed: {exc}", file=sys.stderr)
        return []


def _subscribe_to_topic(project_id: str, topic_name: str) -> None:
    """Create an echo subscription for a topic and start pulling messages."""
    global _subscriber, _publisher, _subscribed_topics

    if topic_name in _subscribed_topics:
        return

    sub_id = f"{topic_name}-echo-sub"
    topic_path = _publisher.topic_path(project_id, topic_name)
    sub_path = _subscriber.subscription_path(project_id, sub_id)

    try:
        _subscriber.create_subscription(request={"name": sub_path, "topic": topic_path})
        print(f"[echo-responder] Created subscription: {sub_id}", file=sys.stderr)
    except Exception as exc:
        if "ALREADY_EXISTS" in str(exc) or "409" in str(exc):
            pass
        else:
            print(f"[echo-responder] Warning creating {sub_id}: {exc}", file=sys.stderr)
            return

    def callback(message: pubsub_v1.subscriber.message.Message) -> None:
        try:
            data = json.loads(message.data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            message.ack()
            return

        thread = data.get("thread", "")
        if thread != "unify_message":
            message.ack()
            return

        event = data.get("event", {})
        body = event.get("body", "")
        contact_id = event.get("contact_id", 1)

        print(
            f"[echo-responder] Received on {topic_name}: {body!r} (contact={contact_id})",
            file=sys.stderr,
        )

        reply_data = {
            "thread": "unify_message_outbound",
            "event": {
                "content": _echo_reply(body),
                "role": "assistant",
                "contact_id": int(contact_id) if contact_id else 1,
            },
        }

        try:
            future = _publisher.publish(
                topic_path,
                json.dumps(reply_data).encode("utf-8"),
                thread="unify_message_outbound",
            )
            msg_id = future.result(timeout=10)
            print(
                f"[echo-responder] Reply published on {topic_name} (id={msg_id})",
                file=sys.stderr,
            )
        except Exception as exc:
            print(f"[echo-responder] Failed to publish reply: {exc}", file=sys.stderr)

        message.ack()

    pull = _subscriber.subscribe(sub_path, callback=callback)
    _streaming_pulls.append(pull)
    _subscribed_topics.add(topic_name)
    print(f"[echo-responder] Listening on: {topic_name}", file=sys.stderr)


def _poll_for_new_topics(project_id: str, interval: float = 5.0) -> None:
    """Background thread that periodically checks for new topics."""
    while not _stop:
        topics = _discover_topics(project_id)
        for t in topics:
            if t not in _subscribed_topics:
                print(f"[echo-responder] New topic discovered: {t}", file=sys.stderr)
                _subscribe_to_topic(project_id, t)
        time.sleep(interval)


def main() -> None:
    global _subscriber, _publisher, _stop

    project_id = _env("GCP_PROJECT_ID", "local-test-project")

    emulator = _env("PUBSUB_EMULATOR_HOST")
    if not emulator:
        print(
            "[echo-responder] PUBSUB_EMULATOR_HOST not set — exiting.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[echo-responder] project={project_id}", file=sys.stderr)
    print(f"[echo-responder] Using Pub/Sub emulator at {emulator}", file=sys.stderr)

    _subscriber = pubsub_v1.SubscriberClient()
    _publisher = pubsub_v1.PublisherClient()

    # Initial topic discovery.
    topics = _discover_topics(project_id)
    if topics:
        print(
            f"[echo-responder] Found {len(topics)} topic(s): {', '.join(topics)}",
            file=sys.stderr,
        )
        for t in topics:
            _subscribe_to_topic(project_id, t)
    else:
        print(
            "[echo-responder] No unity-* topics found yet. Will poll for new ones.",
            file=sys.stderr,
        )

    # Start background poller for topics created after startup.
    poller = threading.Thread(
        target=_poll_for_new_topics,
        args=(project_id,),
        daemon=True,
    )
    poller.start()

    print(
        "[echo-responder] Ready. Polling for new topics every 5s. (Ctrl+C to stop)",
        file=sys.stderr,
    )

    def _handle_signal(signum: int, _frame: object) -> None:
        global _stop
        _stop = True
        for pull in _streaming_pulls:
            pull.cancel()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # Block until stopped.
    try:
        while not _stop:
            time.sleep(1)
    except KeyboardInterrupt:
        _stop = True

    for pull in _streaming_pulls:
        try:
            pull.cancel()
            pull.result(timeout=5)
        except Exception:
            pass

    print("[echo-responder] Stopped.", file=sys.stderr)


if __name__ == "__main__":
    main()
