from __future__ import annotations

import asyncio
import logging
import os
import json
from pathlib import Path
from typing import AsyncIterable

from dotenv import load_dotenv
from jinja2 import Template

from livekit.agents import (
    Agent,
    AgentSession,
    FunctionTool,
    JobContext,
    llm,
    ModelSettings,
    RoomInputOptions,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
)
from livekit import agents

load_dotenv()

# OpenAI Realtime API (voice-to-voice) requires livekit-plugins-openai;
# not available through unify due to different API architecture.
try:
    from livekit.plugins.openai import realtime as openai_realtime
except ImportError:
    openai_realtime = None

from unity.conversation_manager.utils import dispatch_agent
from unity.conversation_manager.events import *

# NEW: shared helpers
from unity.conversation_manager.medium_scripts.common import (
    event_broker,
    create_end_call,
    setup_inactivity_timeout,
    setup_participant_disconnect_handler,
    publish_call_started,
    configure_from_cli,
    should_dispatch_agent,
)

with open(
    Path(__file__).resolve().parent.parent / "prompts" / "realtime_phone_agent.md",
) as f:
    SYSTEM_PROMPT = f.read()

logger = logging.getLogger("gpt-realtime-agent")
logger.setLevel(logging.INFO)


class Assistant(Agent):
    def __init__(
        self,
        contact: dict,
        boss: dict,
        instructions: str = "",
        outbound: bool = False,
    ) -> None:
        self.contact = contact
        self.boss = boss
        self.call_received = not outbound

        super().__init__(instructions=instructions)

    async def on_user_turn_completed(self, turn_ctx, new_message):
        print(turn_ctx)
        print(new_message)

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[FunctionTool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[llm.ChatChunk]:
        print("waiting for call to be received...")
        while not self.call_received:
            await asyncio.sleep(0.1)
        print("call received")

        print("running llm node...")
        async for chunk in super().llm_node(chat_ctx, tools, model_settings):
            yield chunk


async def entrypoint(ctx: JobContext) -> None:
    print("Connecting to room...")
    await ctx.connect()
    print("Connected to room")

    # read static config
    voice_provider = os.environ.get("VOICE_PROVIDER")
    voice_id = os.environ.get("VOICE_ID")
    outbound = os.environ.get("OUTBOUND") == "True"
    channel = os.environ.get("CHANNEL")
    assistant_bio = os.environ.get("ASSISTANT_ABOUT")
    print("voice_provider", voice_provider)
    print("voice_id", voice_id)
    print("outbound", outbound)
    print("channel", channel)

    # contact/boss payloads passed as json env vars
    contact = json.loads(os.getenv("CONTACT", "{}"))
    boss = json.loads(os.getenv("BOSS", "{}"))

    # Configure the OpenAI Realtime model.
    # This requires the livekit-plugins-openai package with OpenAI's Realtime API.
    if openai_realtime is None:
        raise RuntimeError(
            "OpenAI Realtime API is required for voice_mode='sts' (speech-to-speech). "
            "This API is not available through unify. Install livekit-plugins-openai "
            "and configure OPENAI_API_KEY to use realtime voice mode.",
        )

    llm_model = openai_realtime.RealtimeModel(
        model="gpt-realtime",
        voice=voice_id,
        modalities=["audio"],
    )

    session = AgentSession(
        llm=llm_model,
        # If you prefer a separate TTS instead of Realtime audio,
        # set llm.modalities=["text"] above and provide a TTS here.
    )

    user_is_speaking = False
    if channel == "phone":
        user_utterance_event = PhoneUtterance
        assistant_utterance_event = AssistantPhoneUtterance
    else:
        user_utterance_event = UnifyCallUtterance
        assistant_utterance_event = AssistantUnifyCallUtterance

    # NEW: shared end_call + inactivity + participant disconnect handler
    end_call = create_end_call(contact, channel)
    touch_activity = setup_inactivity_timeout(end_call)
    setup_participant_disconnect_handler(ctx.room, end_call)

    @session.on("user_state_changed")
    def _on_user_state_changed(ev: "UserStateChangedEvent"):
        nonlocal user_is_speaking
        user_is_speaking = ev.new_state == "speaking"
        touch_activity()

    @session.on("conversation_item_added")
    def _on_chat_item_added(ev: "UserInputTranscribedEvent"):
        role = ev.item.role  # "user" | "assistant"
        text = ev.item.text_content or ""  # reliably the final text
        if role == "user":
            event = user_utterance_event(contact, content=text)
        else:
            event = assistant_utterance_event(contact, content=text)

        asyncio.create_task(
            event_broker.publish(f"app:comms:{channel}_utterance", event.to_json()),
        )
        print(role, text)
        touch_activity()

    rio = RoomInputOptions()

    # high-level behavior for the assistant.
    system = Template(SYSTEM_PROMPT).render(
        bio=assistant_bio,
        boss_first_name=boss["first_name"],
        boss_surname=boss["surname"],
        boss_email_address=boss["email_address"],
        boss_phone_number=boss["phone_number"],
        contact_first_name=contact["first_name"],
        contact_surname=contact["surname"],
        contact_phone_number=contact["phone_number"],
        contact_email=contact["email_address"],
        is_boss_user=contact["is_boss"],
    )
    print("PRINTING SYSTEM PROMPT")
    print(system)

    agent = Assistant(
        contact=contact,
        boss=boss,
        instructions=system,
        outbound=outbound,
    )

    # publish call started (shared helper)
    await publish_call_started(contact, channel)
    touch_activity()

    async def wait_for_nudges():
        print("waiting")
        rt = agent.realtime_llm_session  # underlying OpenAI RealtimeSession
        async with event_broker.pubsub() as pubsub:
            await pubsub.subscribe("app:call:call_notifs")
            while True:
                msg = await pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=None,
                )
                if msg is not None:
                    print("got notif", msg)
                    msg = json.loads(msg["data"])["payload"]
                    chat_ctx = rt.chat_ctx
                    chat_ctx.add_message(
                        role="user",
                        content=[f"""[notification] {msg["content"]}"""],
                    )
                    await rt.update_chat_ctx(chat_ctx)
                    print(rt.chat_ctx.items)

                    nonlocal user_is_speaking
                    touch_activity()

                    if not user_is_speaking and chat_ctx.items[-1].role != "assistant":
                        await session.generate_reply(allow_interruptions=True)

                await asyncio.sleep(0.1)

    logger.info("starting AgentSession")
    await session.start(room=ctx.room, agent=agent, room_input_options=rio)
    asyncio.create_task(wait_for_nudges())
    await session.generate_reply(allow_interruptions=True)


if __name__ == "__main__":
    # Shared CLI handling
    agent_name, room_name = configure_from_cli(
        extra_env=[
            ("CONTACT", True),
            ("BOSS", True),
            ("ASSISTANT_BIO", False),
        ],
    )

    # dispatch agent
    if should_dispatch_agent():
        print(f"Dispatching agent {agent_name}")
        dispatch_agent(agent_name, room_name)
        print(f"Agent {agent_name} dispatched")

    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=agent_name,
            initialize_process_timeout=60,
        ),
    )
