"""Persistent LiveKit agent worker.

Started once at pod startup by the ConversationManager.  Maintains a pool
of pre-warmed child processes (STT, VAD, inference runner already
initialised) so that incoming calls skip the ~22 s cold-start overhead.

Each call is dispatched as a LiveKit job via ``CreateAgentDispatchRequest``.
The warm child process runs ``call.entrypoint``, reads per-call config from
``ctx.job.metadata``, and handles the conversation.  When the call ends the
child exits and the pool automatically pre-warms a replacement.
"""

import os
import sys

os.environ["UNITY_TERMINAL_LOG"] = "true"

from dotenv import load_dotenv

load_dotenv()

from livekit import agents

from unity.conversation_manager.medium_scripts.call import (
    entrypoint,
    prewarm as _base_prewarm,
)
from unity.conversation_manager.medium_scripts.common import FastBrainLogger

_log = FastBrainLogger()

WORKER_READY_PATH = "/tmp/unity_worker_ready"


def _prewarm_and_signal(ctx=None):
    """Run the standard prewarm, then touch a ready-file so the parent knows."""
    _base_prewarm(ctx)
    try:
        with open(WORKER_READY_PATH, "w") as f:
            f.write("")
    except OSError:
        pass


def main() -> None:
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} dev <agent_name>")
        sys.exit(1)

    # Extract agent_name before LiveKit's Click CLI parses sys.argv.
    agent_name = sys.argv.pop(2)
    _log.dispatch(f"Starting persistent worker: {agent_name}")

    # Remove stale ready-file so the parent waits for a fresh signal.
    try:
        os.remove(WORKER_READY_PATH)
    except FileNotFoundError:
        pass

    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=agent_name,
            prewarm_fnc=_prewarm_and_signal,
            num_idle_processes=1,
            initialize_process_timeout=60,
        ),
    )


if __name__ == "__main__":
    main()
