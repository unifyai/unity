"""
================================================================
Interactive sandbox for **ContactManager**.

It supports:
• Fixed or LLM‑generated seed data.
• Voice or plain‑text input (LiveKit for voice I/O).
• Automatic dispatch to `ask` *or* `update` depending on intent.
• Mid‑conversation interruption (pause / interject / cancel).

Run:
    poetry run python -m sandboxes.user_manager_contact       # text mode
    poetry run python -m sandboxes.user_manager_contact --voice   # with LiveKit voice
"""

import os
import argparse
import asyncio
import logging
import select
import sys
from typing import AsyncIterable, List, Optional, Tuple, Dict
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import (
    AgentSession,
    Agent,
    RoomInputOptions,
    ChatContext,
    ChatMessage,
)
from livekit.plugins import openai, noise_cancellation, deepgram, silero, cartesia
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.agents import ModelSettings, llm, FunctionTool
from unity.contact_manager.contact_manager import ContactManager
import unify
from pydantic import BaseModel, Field
from sandboxes.scenario_builder import ScenarioBuilder

load_dotenv()

# ────────────────────────────────  unity imports  ───────────────────────────
from unity.contact_manager.contact_manager import ContactManager
from unity.common.llm_helpers import SteerableToolHandle
from sandboxes.utils import (  # shared helpers reused in other sandboxes
    record_until_enter as _record_until_enter,
    transcribe_deepgram as _transcribe_deepgram,
    speak as _speak,
    run_in_loop,
    get_custom_scenario,
    activate_project,
)

LG = logging.getLogger("contact_sandbox")

# ═════════════════════════════════ seed helpers ═════════════════════════════


async def _build_scenario(
    custom: Optional[str] = None,
) -> Optional[str]:
    """
    Populate the contact store **through the official tools** using
    :class:`ScenarioBuilder`.  Falls back to the fixed seed on any error.
    """
    cm = ContactManager()
    description = (
        custom.strip()
        if custom
        else (
            "Generate 10 realistic business contacts across EMEA, APAC and AMER. "
            "Each contact needs first_name, surname, email_address and phone_number. "
            "Also create custom columns with varying industries and locations."
        )
    )
    description += (
        "\nTry to get as much done as you can with each `update` and `ask` call. "
        "They can deal with complex multi-step requests just fine."
    )

    builder = ScenarioBuilder(
        description=description,
        tools={  # expose only the public surface
            "update": cm.update,
            "ask": cm.ask,  # allows the LLM to check for duplicates if it wishes
        },
    )

    try:
        await builder.create()
    except Exception as exc:
        raise (f"LLM seeding via ScenarioBuilder failed. {exc}")

    # The new flow doesn't produce a structured "theme"; preserve signature.
    return None


# ═════════════════════════════ intent dispatcher ════════════════════════════


class _Intent(BaseModel):
    action: str = Field(..., pattern="^(ask|update)$")
    cleaned_text: str


_INTENT_SYS_MSG = (
    "Decide whether the user input is a *query* about existing contacts "
    "or a *mutation* (create / update).  "
    "Return JSON {'action':'ask'|'update','cleaned_text':<fixed_input>}."
)


async def _dispatch_with_context(
    cm: ContactManager,
    raw: str,
    *,
    show_steps: bool,
    parent_chat_context: List[Dict[str, str]],
) -> Tuple[str, SteerableToolHandle]:
    """
    Same as :pyfunc:`_dispatch` but forwards *parent_chat_context* to the CM
    methods.  This indirection keeps the diff minimal.
    """

    # quick heuristic – verbs that virtually always imply an update
    if raw.lower().startswith(("add ", "create ", "update ", "change ", "delete ")):
        handle = await cm.update(
            raw,
            parent_chat_context=parent_chat_context,
            _return_reasoning_steps=show_steps,
        )
        return "update", handle

    # ask an LLM if less obvious
    judge = unify.Unify("gpt-4o@openai", response_format=_Intent)
    intent = _Intent.model_validate_json(
        judge.set_system_message(_INTENT_SYS_MSG).generate(raw),
    )
    fn = cm.update if intent.action == "update" else cm.ask
    handle = await fn(
        raw,
        parent_chat_context=parent_chat_context,
        _return_reasoning_steps=show_steps,
    )
    return intent.action, handle


# ═════════════════════════════ interruption helper ══════════════════════════


def _input_now(timeout: float = 0.1) -> Optional[str]:
    """Non‑blocking stdin check (POSIX & Windows)."""
    r, _, _ = select.select([sys.stdin], [], [], timeout)
    return sys.stdin.readline().strip() if r else None


async def _await_with_interrupt(
    handle: SteerableToolHandle,
) -> str:  # returns final answer
    while not handle.done():
        txt = _input_now(0.1)
        if txt:
            if txt.lower() in {"stop", "cancel"}:
                handle.stop()
                break
            run_in_loop(handle.interject(txt))
        await asyncio.sleep(0.05)

    return await handle.result()


class ConversationManager(Agent):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(
            instructions="You are a helpful voice AI assistant.",
            llm=openai.LLM(model="gpt-4o"),
        )
        self.cm = ContactManager()
        self.debug = args.debug
        self.contact_manager_response = None

    async def on_user_turn_completed(
        self,
        turn_ctx: ChatContext,
        new_message: ChatMessage,
    ) -> None:
        # Use contact manager to get additional context/information
        raw = new_message.text_content
        print(f"User: {raw}")

        # Get response from contact manager and store it
        _kind, _handle = await _dispatch_with_context(
            self.cm,
            raw,
            show_steps=self.debug,
            parent_chat_context=[item.model_dump() for item in turn_ctx.items],
        )

        answer = await _await_with_interrupt(_handle)
        if isinstance(answer, tuple):  # reasoning steps requested
            answer, _steps = answer

        # Store the response for use in llm_node
        self.contact_manager_response = str(answer)
        print(f"Contact Manager: {answer}")

        # Add the ContactManager response to the chat context as a system message
        # so the LLM can process it naturally
        system_message = ChatMessage.create(
            text=f"Contact Manager Result: {answer}",
            role="system",
        )
        turn_ctx.messages.append(system_message)

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[FunctionTool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[llm.ChatChunk]:
        print("running custom llm node...")

        # Let the default LLM process the chat context (which now includes ContactManager response)
        # and generate a natural response
        async for chunk in Agent.default.llm_node(
            self,
            chat_ctx,
            tools,
            model_settings,
        ):
            yield chunk


async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    # Parse arguments to get debug setting
    parser = argparse.ArgumentParser(description="ContactManager voice assistant")
    parser.add_argument("--debug", "-d", action="store_true", help="verbose tool logs")
    parser.add_argument("--traced", "-t", action="store_true", help="include tracing")
    parser.add_argument(
        "--project_name",
        "-p",
        default="Sandbox",
        help="Unify project / context name (default: Sandbox)",
    )
    parser.add_argument(
        "--overwrite",
        "-o",
        action="store_true",
        help="overwrite existing data for the chosen project",
    )
    args, unknown = parser.parse_known_args()

    # Setup Unify context
    activate_project(args.project_name, args.overwrite)
    base_ctx = unify.get_active_context().get("write")
    traces_ctx = f"{base_ctx}/Traces" if base_ctx else "Traces"
    unify.set_trace_context(traces_ctx)
    if args.overwrite:
        ctxs = unify.get_contexts()
        if "Contacts" in ctxs:
            unify.delete_context("Contacts")
        if traces_ctx in ctxs:
            unify.delete_context(traces_ctx)
        unify.create_context(traces_ctx)

    # Build scenario and seed data
    scenario_text = "Generate 10 realistic business contacts across EMEA, APAC and AMER. Each contact needs first_name, surname, email_address and phone_number. Also create custom columns with varying industries and locations."
    LG.info("[seed] building synthetic contacts – this can take 20-40 s…")
    await _build_scenario(scenario_text)
    LG.info("[seed] done.")

    voice_id = os.environ.get("VOICE_ID", "")

    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=openai.LLM(model="gpt-4o"),
        tts=cartesia.TTS(voice=voice_id if voice_id != "" else None),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=ConversationManager(args),
        room_input_options=RoomInputOptions(
            # LiveKit Cloud enhanced noise cancellation
            # - If self-hosting, omit this parameter
            # - For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await session.generate_reply(
        instructions="Greet the user and explain that you can help them manage their contacts - search, add, update, or answer questions about their contact database.",
    )


# ══════════════════════════════════  CLI  ═══════════════════════════════════


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
