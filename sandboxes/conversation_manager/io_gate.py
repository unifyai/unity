"""
Shared stdin gate for the ConversationManager sandbox.

Why:
Real-comms safety confirmations may prompt from inside CM background tasks.
The REPL also reads stdin. Concurrent `input()` calls from multiple threads
can deadlock or steal each other's lines.

We serialize *all* stdin reads through a single global lock.
"""

from __future__ import annotations

import threading

STDIN_LOCK = threading.Lock()


def gated_input(prompt: str) -> str:
    """Thread-safe stdin read for sandbox components."""
    with STDIN_LOCK:
        return input(prompt)
