"""knowledge_sandbox.py  (optional voice mode, Deepgram SDK v4, sync)
====================================================================
Interactive sandbox for **KnowledgeManager**.

It supports:
• Fixed or LLM‑generated seed data.
• Voice or plain‑text input (same helpers as the other sandboxes).
• Automatic dispatch to `ask`, `update` or `refactor` depending on intent.
• Mid‑conversation interruption (pause / interject / cancel).
"""

from __future__ import annotations

# ─────────────────────────────── stdlib / vendored ──────────────────────────
import os
import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Dict
from datetime import datetime

# Added for graceful shutdown handling
import signal
import threading
from typing import Tuple, Optional, List, Dict

# NOTE: ScenarioBuilder and synthetic seeding removed – the demo now attaches
# directly to the pre-initialised "Intranet" project and forwards every user
# turn to the RAG agent’s `query()` helper.
# Ensure repository root resolves for local execution
# Repo root sits two levels up from this file (…/unity)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Now import RAG agent
from intranet.scripts.utils import initialize_script_environment

if not initialize_script_environment():
    sys.exit(1)

# After ensuring repository root is on sys.path, set up logging and imports
from intranet.core.system_utils import setup_logging

setup_logging()

# Honour LOG_LEVEL env var – if set to OFF/NONE/0 disable logging entirely
if os.getenv("LOG_LEVEL", "INFO").upper() in {"OFF", "NONE", "0"}:
    logging.disable(logging.CRITICAL)

# Optional: silence INFO logs on stdout while retaining them in rag_agent.log
if os.getenv("SILENT_CONSOLE", "false").lower() == "true":
    logging.getLogger("UnifyAsyncLogger.EventBus").setLevel(logging.CRITICAL)
    root_logger = logging.getLogger()
    for _h in root_logger.handlers:
        if isinstance(_h, logging.StreamHandler):
            root_logger.removeHandler(_h)

from intranet.core.rag_agent import IntranetRAGAgent
from sandboxes.utils import (  # shared helpers reused in other sandboxes
    record_until_enter as _record_until_enter,
    transcribe_deepgram as _transcribe_deepgram,
    speak as _speak,
    await_with_interrupt as _await_with_interrupt,
    build_cli_parser,
    activate_project,
    _wait_for_tts_end as _wait_tts_end,
)

# Lightweight intent schema (mirrors regular sandbox)
from pydantic import BaseModel, Field
import unify
from unity.common.llm_helpers import SteerableToolHandle  # type hint only

LG = logging.getLogger("intranet_sandbox")
logging.getLogger("UnifyAsyncLogger").setLevel(logging.INFO)

# ═════════════════════════════════ signal handling ═══════════════════════════

_shutdown_requested = False


def _signal_handler(signum, _frame):
    """Catch SIGINT / SIGTERM and request a clean exit."""

    global _shutdown_requested

    sig_names = {signal.SIGINT: "SIGINT (Ctrl+C)", signal.SIGTERM: "SIGTERM"}
    name = sig_names.get(signum, f"Signal {signum}")
    print(f"\n🛑 Received {name} – shutting down sandbox…")
    _shutdown_requested = True

    # If the loop doesn’t exit within 10 s, force-kill the process.
    def _force_exit():
        if _shutdown_requested:
            LG.warning("⏳ Graceful shutdown timed out – forcing exit.")
            os._exit(1)  # hard exit, avoids async cleanup deadlocks

    _t = threading.Timer(10.0, _force_exit)
    _t.daemon = True
    _t.start()


# Register handlers early
for _sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(_sig, _signal_handler)

# ═════════════════════════════════ demo helpers ═════════════════════════════
# (synthetic scenario generation dropped – we rely on pre-seeded data)


# ═════════════════════════════════ intent dispatcher (ask | update | refactor) ═════════════════════════════
class _Intent(BaseModel):
    action: str = Field(..., pattern="^(ask|update|refactor)$")


_INTENT_SYS_MSG = (
    "Classify the user's message into exactly one of: 'ask' | 'update' | 'refactor'.\n"
    "- ask: read-only retrieval or analysis over existing knowledge.\n"
    "- update: add or modify rows/columns/tables.\n"
    "- refactor: schema normalization or structural changes (rename/split/move columns, joins migration).\n"
    "Return ONLY JSON: {'action': 'ask'|'update'|'refactor'}"
)


async def _dispatch_with_context(
    rag_agent: "IntranetRAGAgent",
    raw: str,
    *,
    show_steps: bool,
    parent_chat_context: List[
        Dict[str, str]
    ],  # kept for parity; forwarded where useful
    clarifications_enabled: bool,  # unused (RAG agent has no clarification loop)
    enable_voice: bool,
) -> Tuple[
    str,
    "SteerableToolHandle" | object,
    Optional[asyncio.Queue[str]],
    Optional[asyncio.Queue[str]],
]:
    """
    Decide whether to call rag_agent.ask / rag_agent.update / rag_agent.refactor.
    Returns (kind, handle_or_result, clar_up_q, clar_down_q).  Clar queues are always None here.
    """
    lowered = raw.lower()

    # Fast-path heuristics (mirror regular sandbox)
    if lowered.startswith(
        (
            "add ",
            "create ",
            "update ",
            "change ",
            "delete ",
            "store ",
            "remember ",
            "note ",
        ),
    ):
        if enable_voice:
            try:
                _speak("Working on it.")
            except Exception:
                pass
        return (
            "update",
            await rag_agent.update(
                update_prompt=raw,
                conversation_context=list(parent_chat_context),
            ),
            None,
            None,
        )

    if lowered.startswith(
        ("refactor ", "restructure ", "normalize ", "normalise ", "schema "),
    ):
        if enable_voice:
            try:
                _speak("Working on it.")
            except Exception:
                pass
        return (
            "refactor",
            await rag_agent.refactor(
                schema_prompt=raw,
                conversation_context=list(parent_chat_context),
            ),
            None,
            None,
        )

    # LLM judge for everything else
    judge = unify.Unify("gpt-5@openai", response_format=_Intent)
    intent = _Intent.model_validate_json(
        judge.set_system_message(_INTENT_SYS_MSG).generate(raw),
    )
    action = intent.action

    if enable_voice:
        try:
            _speak("Working on it.")
        except Exception:
            pass

    if action == "ask":
        result = await rag_agent.ask(
            query_text=raw,
            conversation_context=list(parent_chat_context),
        )
    elif action == "update":
        result = await rag_agent.update(
            update_prompt=raw,
            conversation_context=list(parent_chat_context),
        )
    else:
        result = await rag_agent.refactor(
            schema_prompt=raw,
            conversation_context=list(parent_chat_context),
        )

    return action, result, None, None


# ══════════════════════════════════  CLI  ═══════════════════════════════════


async def _main_async() -> None:
    parser = build_cli_parser("Intranet sandbox")
    args = parser.parse_args()

    os.environ["UNIFY_TRACED"] = "true" if args.traced else "false"

    # Attach to the existing "Intranet" project and instantiate the RAG agent
    os.environ["RAG_SKIP_INIT"] = (
        "true"  # assume intranet/scripts/01_initialize_system.py already ran
    )

    activate_project(args.project_name, args.overwrite)

    rag_agent = IntranetRAGAgent()

    _COMMANDS_HELP = (
        "\nKnowledgeManager sandbox (Intranet) – type commands below (press ↵ with an empty "
        "line to dictate via voice when --voice mode is active – type 'r' to record).  'quit' to exit.\n\n"
        "┌────────────────── accepted commands ─────────────────────┐\n"
        "│ r / free text         – freeform ask / update / refactor │\n"
        "│ save_project | sp     – save project snapshot            │\n"
        "│ help | h              – show this help                   │\n"
        "└──────────────────────────────────────────────────────────┘\n"
    )

    def _explain_commands() -> None:
        print(_COMMANDS_HELP)

    if args.voice:
        _speak(
            "Sandbox ready. You can type commands, or press enter on an empty line to record a voice query.",
        )

    # Running transcript for display and context
    chat_history: List[Dict[str, str]] = []

    # ─────────────────────────── interaction loop ───────────────────────────
    while not _shutdown_requested:
        try:
            print()
            _explain_commands()
            print()

            if args.voice:
                _wait_tts_end()
            if args.voice:
                raw = input("command ('r' to record)> ").strip()
                if raw.lower() == "r":
                    audio = _record_until_enter()
                    raw = _transcribe_deepgram(audio).strip()
                    if not raw:
                        continue
                    print(f"▶️  {raw}")
            else:
                raw = input("command> ").strip()
            if raw.lower() in {"help", "h", "?"}:
                _explain_commands()
                continue

            if raw.lower() in {"quit", "exit"}:
                break
            if not raw:
                continue

            # ─────────────── save project snapshot ────────────────
            import unify

            if raw.lower() in {"save_project", "sp"}:
                commit_hash = unify.commit_project(
                    args.project_name,
                    commit_message=f"Sandbox save {datetime.utcnow().isoformat()}",
                ).get("commit_hash")
                print(f"💾 Project saved at commit {commit_hash}")
                if args.voice:
                    _speak("Project saved")
                continue

            if raw.startswith("/"):
                print(
                    "(no active request) Steering commands are only available while a call is running.",
                )
                continue

            # ════════════════ route & query RAG agent ═════════════════════════════
            if args.voice:
                _speak("Let me think…")

            LG.info(f"🧠 chat_history: {chat_history}")

            from time import perf_counter

            _t0 = perf_counter()

            async def _thinking_anim(task: asyncio.Task):
                phrases = [
                    "🔍 Searching through documents",
                    "📚 Gathering relevant context",
                    "🧠 Processing retrieved information",
                    "📑 Summarising key passages",
                    "🤔 Putting everything together",
                    "⌛ Almost done, finalising answer",
                ]

                cycle_time = 0.7  # seconds per dot update
                for base in phrases:
                    t_phrase_start = perf_counter()
                    while perf_counter() - t_phrase_start < 20:
                        if task.done():
                            print("\r", end="")
                            return
                        elapsed = perf_counter() - t_phrase_start
                        dots = int((elapsed / cycle_time)) % 4  # 0‒3
                        print(f"\r{base}{'.' * dots}   ", end="", flush=True)
                        await asyncio.sleep(cycle_time)
                    print()  # newline after each 20-s phase
                if not task.done():
                    print(
                        "🙏 Sorry, this is taking longer than expected… still working …",
                    )

            # Dispatch to RAG method (intent-based)
            async def _dispatch_async():
                kind, result, _cu, _cd = await _dispatch_with_context(
                    rag_agent,
                    raw,
                    show_steps=args.debug,
                    parent_chat_context=list(chat_history),
                    clarifications_enabled=not getattr(
                        args,
                        "no_clarifications",
                        False,
                    ),
                    enable_voice=bool(args.voice),
                )
                return kind, result

            query_task = asyncio.create_task(_dispatch_async())
            anim_task = asyncio.create_task(_thinking_anim(query_task))

            _kind, rag_response = await query_task
            anim_task.cancel()
            _duration = perf_counter() - _t0

            # ─────────────── structured pretty-print ────────────────
            if _kind == "ask":
                answer = rag_response.get("answer", "(no answer)")
                sources = rag_response.get("sources", [])
                follow_ups = rag_response.get("follow_up_questions", [])
                confidence = rag_response.get("confidence")

                lines: list[str] = []
                lines.append(
                    "\n📄 ==== RAG ANSWER =================================================",
                )
                lines.append(answer.strip())

                if sources:
                    lines.append(f"\n🔗 Sources (top {len(sources)}):")
                    for idx, src in enumerate(sources, 1):
                        title = src.get("title") or src.get(
                            "content_text",
                            "(no source)",
                        )
                        score = src.get("score")
                        score_txt = (
                            f"  (score {score:.2f})" if score is not None else ""
                        )
                        lines.append(f"  {idx}. {title}{score_txt}")
                else:
                    lines.append("\n🔗 Sources: none")

                if follow_ups:
                    lines.append("\n❓ Follow-up questions:")
                    for q in follow_ups:
                        lines.append(f"  • {q}")

                if confidence is not None:
                    lines.append(f"\n🔍 Confidence: {confidence:.2f}")

                lines.append(f"\n⏱️  Response time: {_duration:.2f} s")
                lines.append(
                    "===============================================================\n",
                )

                print("\n".join(lines))

                # Remember dialogue only for ask
                chat_history.append({"role": "user", "content": raw})
                chat_history.append({"role": "assistant", "content": answer})
            else:
                # update/refactor: print raw result as-is (dict or string)
                print(
                    "\n🛠️ ==== RAG OPERATION RESULT ==========================================",
                )
                if isinstance(rag_response, dict):
                    # If our agent returned {status, result}, show result
                    op_res = rag_response.get("result", rag_response)
                    print(op_res)
                    assistant_text = str(op_res)
                else:
                    print(rag_response)
                    assistant_text = str(rag_response)
                print("\n⏱️  Duration: {:.2f} s".format(_duration))
                print(
                    "================================================================\n",
                )

                # Persist dialogue for non-ask modes as well
                chat_history.append({"role": "user", "content": raw})
                chat_history.append({"role": "assistant", "content": assistant_text})

            if args.voice:
                _speak(
                    chat_history[-1]["content"] if _kind == "ask" else assistant_text,
                )
                _speak("Anything else?")
        except (EOFError, KeyboardInterrupt):
            print("Exiting…")
            break

    # Final farewell if shutdown requested via signal
    if _shutdown_requested:
        print("Sandbox terminated.")


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
