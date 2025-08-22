"""knowledge_sandbox.py  (optional voice mode, Deepgram SDK v4, sync)
====================================================================
Interactive sandbox for **KnowledgeManager**.

It supports:
• Fixed or LLM‑generated seed data.
• Voice or plain‑text input (same helpers as the other sandboxes).
• Automatic dispatch to `retrieve`, `store` *or* `refactor` depending on intent.
• Mid‑conversation interruption (pause / interject / cancel).
"""

from __future__ import annotations

# ─────────────────────────────── stdlib / vendored ──────────────────────────
import os
import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

import unify
from pydantic import BaseModel, Field
from sandboxes.scenario_builder import ScenarioBuilder

# Ensure repository root resolves for local execution
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ────────────────────────────────  unity imports  ───────────────────────────
from unity.knowledge_manager.knowledge_manager import KnowledgeManager
from unity.common.llm_helpers import SteerableToolHandle
from sandboxes.utils import (  # shared helpers reused in other sandboxes
    record_until_enter as _record_until_enter,
    transcribe_deepgram as _transcribe_deepgram,
    speak as _speak,
    get_custom_scenario,
    await_with_interrupt as _await_with_interrupt,
    build_cli_parser,
    activate_project,
)

LG = logging.getLogger("knowledge_sandbox")

# ═════════════════════════════════ seed helpers ═════════════════════════════


async def _build_scenario(custom: Optional[str] = None) -> Optional[str]:
    """
    Populate the knowledge store **via the official tools** (`store`/`retrieve`)
    using :class:`ScenarioBuilder`.  Falls back to the fixed seed on error.
    """
    km = KnowledgeManager()

    description = (
        custom.strip()
        if custom
        else (
            "Generate 20 diverse facts about electric-vehicle manufacturers. "
            "Cover launch years, battery capacities, warranty terms and sales "
            "figures in different regions.  Include numbers, dates and named "
            "entities so the schema has to evolve."
        )
    )
    description += (
        "\nTry to batch actions – each `store` can add multiple rows/columns "
        "and `retrieve` can verify to avoid duplication."
    )

    builder = ScenarioBuilder(
        description=description,
        tools={
            "update": km.update,
            "ask": km.ask,
        },
    )

    try:
        await builder.create()
    except Exception as exc:
        raise RuntimeError(f"LLM seeding via ScenarioBuilder failed. {exc}")

    return None


# ═════════════════════════════ intent dispatcher ════════════════════════════


class _Intent(BaseModel):
    action: str = Field(..., pattern="^(retrieve|store|refactor)$")
    cleaned_text: str


_INTENT_SYS_MSG = (
    "Decide whether the user input is a *query* about existing knowledge "
    "(`retrieve`), a *mutation* that adds/updates knowledge (`store`), "
    "or a schema-level restructuring (`refactor`). "
    "Return JSON "
    "{'action':'retrieve'|'store'|'refactor','cleaned_text':<fixed_input>}."
)


async def _dispatch_with_context(
    km: KnowledgeManager,
    raw: str,
    *,
    show_steps: bool,
    parent_chat_context: List[Dict[str, str]],
) -> Tuple[str, SteerableToolHandle]:
    """
    Figure out whether to call `store`, `retrieve` or `refactor`, forwarding
    *parent_chat_context* to the KnowledgeManager methods.
    """

    lowered = raw.lower()

    # ───── quick heuristics (fast-path) ───────────────────────────────
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
        handle = await km.update(
            raw,
            parent_chat_context=parent_chat_context,
            _return_reasoning_steps=show_steps,
        )
        return "update", handle

    if lowered.startswith(
        (
            "refactor ",
            "restructure ",
            "normalize ",
            "normalise ",
            "schema ",
        ),
    ):
        handle = await km.refactor(
            raw,
            parent_chat_context=parent_chat_context,
            _return_reasoning_steps=show_steps,
        )
        return "refactor", handle

    # ───── everything else – ask an LLM judge ────────────────────────
    judge = unify.Unify("gpt-4o@openai", response_format=_Intent)
    intent = _Intent.model_validate_json(
        judge.set_system_message(_INTENT_SYS_MSG).generate(raw),
    )

    fn = (
        km.update
        if intent.action == "update"
        else km.refactor if intent.action == "refactor" else km.ask
    )
    handle = await fn(
        raw,
        parent_chat_context=parent_chat_context,
        _return_reasoning_steps=show_steps,
    )
    return intent.action, handle


# ══════════════════════════════════  CLI  ═══════════════════════════════════


async def _main_async() -> None:
    parser = build_cli_parser("KnowledgeManager sandbox")
    args = parser.parse_args()

    os.environ["UNIFY_TRACED"] = "true" if args.traced else "false"

    activate_project(args.project_name, args.overwrite)

    # ─────────────────── project version handling ────────────────────
    if args.project_version != -1:
        commits = unify.get_project_commits(args.project_name)
        if commits:
            try:
                target = commits[args.project_version]
                unify.rollback_project(args.project_name, target["commit_hash"])
                LG.info("[version] Rolled back to commit %s", target["commit_hash"])
            except IndexError:
                LG.warning(
                    "[version] project_version index %s out of range, ignoring",
                    args.project_version,
                )

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    LG.setLevel(logging.INFO)

    km = KnowledgeManager()
    if args.traced:
        km = unify.traced(km)

    scenario_text: Optional[str] = get_custom_scenario(args)
    LG.info("[seed] building synthetic knowledge base – this can take 20-40 s…")
    if args.voice:
        _speak("Sure thing, building your custom scenario now.")
    await _build_scenario(scenario_text)
    LG.info("[seed] done.")
    if args.voice:
        _speak("All done, your custom scenario is built and ready to go.")

    print("KnowledgeManager sandbox – type or speak. 'quit' to exit.\n")

    _speak(
        "Press enter to record a question or request an update for the knowledge base.",
    )

    # running memory of the dialogue
    chat_history: List[Dict[str, str]] = []

    # interaction loop
    while True:
        try:
            if args.voice:
                audio = _record_until_enter()
                raw = _transcribe_deepgram(audio).strip()
                if not raw:
                    continue
                print(f"▶️  {raw}")
            else:
                raw = input("> ").strip()

            if raw.lower() in {"quit", "exit"}:
                break
            if not raw:
                continue

            # ─────────────── save project snapshot ────────────────
            if raw.lower() in {"save_project", "sp"}:
                commit_hash = unify.commit_project(
                    args.project_name,
                    commit_message=f"Sandbox save {datetime.utcnow().isoformat()}",
                ).get("commit_hash")
                print(f"💾 Project saved at commit {commit_hash}")
                if args.voice:
                    _speak("Project saved")
                continue

            # ──────────────── remember the user's utterance ────────────────
            _kind, _handle = await _dispatch_with_context(
                km,
                raw,
                show_steps=args.debug,
                parent_chat_context=list(chat_history),
            )
            chat_history.append({"role": "user", "content": raw})
            if args.voice:
                _speak("Let me take a look, give me a moment")

            answer = await _await_with_interrupt(_handle)
            if args.voice:
                _speak("Okay that's all done")
            if isinstance(answer, tuple):  # reasoning steps requested
                answer, _steps = answer
            print(f"[{_kind}] → {answer}\n")

            # ──────────────── remember the assistant's reply ───────────────
            chat_history.append({"role": "assistant", "content": answer})
            if args.voice:
                _speak(f"{answer} Anything else I can help with?")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting…")
            break


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
