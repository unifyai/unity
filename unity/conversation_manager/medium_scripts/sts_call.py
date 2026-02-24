from __future__ import annotations

import os
import asyncio
import logging
import json
from typing import AsyncIterable

os.environ["UNITY_TERMINAL_LOG"] = "true"

from dotenv import load_dotenv

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
    from livekit.plugins import openai as openai_plugin
    from livekit.plugins.openai import realtime as openai_realtime
except ImportError:
    openai_plugin = None
    openai_realtime = None

from unity.conversation_manager.utils import dispatch_livekit_agent
from unity.conversation_manager.events import *
from unity.conversation_manager.prompt_builders import build_voice_agent_prompt
from unity.conversation_manager.tracing import content_trace_id
from unity.session_details import SESSION_DETAILS

# Shared helpers
import unillm

from unity.conversation_manager.medium_scripts.common import (
    event_broker,
    create_end_call,
    match_say_meta,
    setup_inactivity_timeout,
    setup_participant_disconnect_handler,
    publish_call_started,
    configure_from_cli,
    should_dispatch_livekit_agent,
    start_event_broker_receive,
    UserTrackCaptureManager,
    ScreenshotHistory,
    capture_assistant_screenshot,
    render_event_for_fast_brain,
    render_participant_comms,
    publish_meet_interaction_from_track,
    FastBrainLogger,
    hydrate_fast_brain_history,
    trim_fast_brain_context,
)
from unity.conversation_manager.types.screenshot import (
    ScreenshotEntry,
    generate_screenshot_path,
    write_screenshot_to_disk,
)

logger = logging.getLogger("gpt-realtime-agent")
logger.setLevel(logging.INFO)

_log = FastBrainLogger(mode="sts")


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

    def set_call_received(self) -> None:
        self.call_received = True

    async def on_user_turn_completed(self, turn_ctx, new_message):
        """
        Hook called when user finishes speaking.

        Note: User utterance publishing is handled by _on_chat_item_added
        to keep all transcript logging in one place alongside assistant utterances.
        """
        _log.user_speech(new_message.text_content or "")

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[FunctionTool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[llm.ChatChunk]:
        _log.info("Waiting for call to be received…")
        while not self.call_received:
            await asyncio.sleep(0.1)
        _log.call_status("call_received")

        await self._capture_screenshots_for_llm(chat_ctx)

        asyncio.create_task(
            event_broker.publish("app:comms:fast_brain_generating", "{}"),
        )

        _log.llm_thinking(reason="llm_node_start")
        async for chunk in super().llm_node(chat_ctx, tools, model_settings):
            yield chunk


async def entrypoint(ctx: JobContext) -> None:
    event_broker.set_logger(_log)

    _log.session_start("Connecting to room…")
    await ctx.connect()
    _log.session_start("Connected to room")

    # User screen share and webcam capture (subscribe to LiveKit room tracks automatically)
    screen_capture = UserTrackCaptureManager(
        ctx.room,
        track_source="screenshare",
        on_track_change=publish_meet_interaction_from_track,
        fb_logger=_log,
    )
    webcam_capture = UserTrackCaptureManager(
        ctx.room,
        track_source="camera",
        on_track_change=publish_meet_interaction_from_track,
        fb_logger=_log,
    )

    # Flag for call_answered that may arrive during initialization
    call_answered_flag = asyncio.Event()

    # Start receiving events from parent (callbacks registered later)
    await start_event_broker_receive()

    # Populate SESSION_DETAILS from environment (set by configure_from_cli)
    SESSION_DETAILS.populate_from_env()

    # Read config from SESSION_DETAILS
    voice_provider = SESSION_DETAILS.voice.provider
    voice_id = SESSION_DETAILS.voice.id
    outbound = SESSION_DETAILS.voice_call.outbound
    channel = SESSION_DETAILS.voice_call.channel
    assistant_bio = SESSION_DETAILS.assistant.about
    _log.config(
        f"voice_provider={voice_provider} voice_id={voice_id} outbound={outbound} channel={channel}",
    )

    # Contact/boss payloads from SESSION_DETAILS
    contact = json.loads(SESSION_DETAILS.voice_call.contact_json or "{}")
    boss = json.loads(SESSION_DETAILS.voice_call.boss_json or "{}")

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
        # OpenAI TTS with the same voice as the Realtime model, used exclusively
        # for session.say() when the guidance articulator provides pre-generated
        # speech. Normal conversation still uses the Realtime model's native audio.
        tts=openai_plugin.TTS(voice=voice_id) if voice_id else openai_plugin.TTS(),
    )

    user_is_speaking = False
    _queued_speech: list[tuple[str, str, str]] = []  # (text, guidance_id, source)
    _last_say_meta: dict | None = None
    generation_seq = 0
    user_state_seq = 0

    def _log_reply_task(task: asyncio.Task) -> None:
        try:
            task.result()
            _log.llm_completed()
        except asyncio.CancelledError:
            _log.llm_cancelled()
        except Exception as exc:  # noqa: BLE001
            _log.llm_error(str(exc))

    def trigger_generate_reply(
        reason: str,
        source_id: str,
        *,
        allow_interruptions: bool = True,
        wait_for_completion: bool = False,
    ):
        nonlocal generation_seq
        generation_seq += 1
        generation_id = f"gen-{generation_seq:06d}"
        _log.llm_thinking(
            reason=reason,
            generation_id=generation_id,
            source_id=source_id,
            queued_speech=len(_queued_speech),
        )
        maybe_result = session.generate_reply(
            allow_interruptions=allow_interruptions,
        )
        if isinstance(maybe_result, asyncio.Task):
            maybe_result.add_done_callback(_log_reply_task)
        if wait_for_completion:
            return maybe_result
        return maybe_result

    if channel == "phone":
        user_utterance_event = InboundPhoneUtterance
        assistant_utterance_event = OutboundPhoneUtterance
    else:
        user_utterance_event = InboundUnifyMeetUtterance
        assistant_utterance_event = OutboundUnifyMeetUtterance

    # Shared end_call + inactivity + participant disconnect handler
    end_call = create_end_call(contact, channel, room_name=ctx.room.name)
    touch_activity = setup_inactivity_timeout(end_call)
    setup_participant_disconnect_handler(ctx.room, end_call)

    @session.on("user_state_changed")
    def _on_user_state_changed(ev: "UserStateChangedEvent"):
        nonlocal user_is_speaking, user_state_seq
        user_state_seq += 1
        state_id = f"usrstate-{user_state_seq:06d}"
        user_is_speaking = ev.new_state == "speaking"
        _log.user_state(ev.new_state, state_id=state_id)
        touch_activity()

    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev):
        """Try queued speech only after the agent settles into a quiescent state.

        We intentionally do NOT trigger from user_state_changed because there is
        a gap between VAD silence detection and the turn detector confirming the
        turn. During that gap, agent_state is still "listening" and current_speech
        is None — firing then would race ahead of the fast brain's reply.

        Triggering here guarantees the full thinking → speaking → listening cycle
        has completed before queued guidance speech plays.
        """
        if ev.new_state in ("listening", "idle"):
            maybe_speak_queued()

    # -- Screenshot state --
    screenshot_history = ScreenshotHistory()
    assistant_screen_share_active = False

    def _inject_visual_context_sts() -> None:
        """Fire-and-forget: rebuild and push visual context to the Realtime API."""
        if not rt_ref:
            return
        content = screenshot_history.build_visual_context_content()
        if not content:
            return
        current_rt = rt_ref[0]
        current_rt.chat_ctx.add_message(role="user", content=content)
        asyncio.create_task(_trimmed_update_chat_ctx())

    def _publish_screenshot(entry: "ScreenshotEntry", filepath: str) -> None:
        """Fire-and-forget: write to disk and publish to slow brain via IPC."""

        async def _background():
            await asyncio.to_thread(write_screenshot_to_disk, entry, filepath)
            await event_broker.publish(
                "app:comms:screenshot",
                json.dumps(
                    {
                        "b64": entry.b64,
                        "utterance": entry.utterance,
                        "timestamp": entry.timestamp.isoformat(),
                        "source": entry.source,
                        "filepath": filepath,
                    },
                ),
            )

        asyncio.create_task(_background())

    def _handle_screenshot(entry: "ScreenshotEntry") -> None:
        """Process a captured screenshot: history, visual context, disk, IPC."""
        filepath = generate_screenshot_path(entry)
        screenshot_history.add(entry, filepath)
        _inject_visual_context_sts()
        if entry.source != "assistant":
            _publish_screenshot(entry, filepath)

    @session.on("conversation_item_added")
    def _on_chat_item_added(ev: "UserInputTranscribedEvent"):
        """Publish both user and assistant utterances from a single location."""
        nonlocal _last_say_meta
        role = ev.item.role  # "user" | "assistant"
        text = ev.item.text_content or ""  # reliably the final text
        utterance_id = content_trace_id("utt", f"{role}:{text}")
        say_meta = match_say_meta(_last_say_meta, text) if role == "assistant" else None
        if say_meta:
            _last_say_meta = None
        if role == "user":
            _log.user_speech(text)
        else:
            _log.assistant_speech(
                text,
                source=(say_meta or {}).get("source", "generate_reply"),
                guidance_id=(say_meta or {}).get("guidance_id", ""),
            )
        if role == "user":
            event = user_utterance_event(contact, content=text)
            from datetime import datetime, timezone

            b64 = screen_capture.capture_screenshot()
            if b64:
                _handle_screenshot(
                    ScreenshotEntry(
                        b64=b64,
                        utterance=text,
                        timestamp=datetime.now(timezone.utc),
                        source="user",
                    ),
                )
            webcam_b64 = webcam_capture.capture_screenshot()
            if webcam_b64:
                _handle_screenshot(
                    ScreenshotEntry(
                        b64=webcam_b64,
                        utterance=text,
                        timestamp=datetime.now(timezone.utc),
                        source="webcam",
                    ),
                )
        else:
            event = assistant_utterance_event(contact, content=text)

        asyncio.create_task(
            event_broker.publish(f"app:comms:{channel}_utterance", event.to_json()),
        )
        touch_activity()

    rio = RoomInputOptions()

    # high-level behavior for the assistant.
    from unity.settings import SETTINGS

    assistant_name = SESSION_DETAILS.assistant.name
    system = build_voice_agent_prompt(
        bio=assistant_bio,
        assistant_name=assistant_name or None,
        boss_first_name=boss.get("first_name", ""),
        boss_surname=boss.get("surname", ""),
        boss_email_address=boss.get("email_address", ""),
        boss_phone_number=boss.get("phone_number", ""),
        boss_bio=boss.get("bio") or None,
        contact_first_name=contact.get("first_name", ""),
        contact_surname=contact.get("surname", ""),
        contact_phone_number=contact.get("phone_number", ""),
        contact_email=contact.get("email_address", ""),
        contact_bio=contact.get("bio") or None,
        is_boss_user=contact.get("contact_id") == 1,
        contact_rolling_summary=contact.get("rolling_summary", ""),
        demo_mode=SETTINGS.DEMO_MODE,
    ).flatten()
    _log.config(f"System prompt ({len(system)} chars)")

    agent = Assistant(
        contact=contact,
        boss=boss,
        instructions=system,
        outbound=outbound,
    )

    async def _capture_screenshots_for_llm(chat_ctx) -> None:
        """Capture fresh screenshots and inject into the LLM's chat_ctx copy.

        The LiveKit pipeline passes a copy of the chat context to llm_node.
        _handle_screenshot updates the live session context (for IPC and
        subsequent turns), but the copy is stale.  After capturing we inject
        directly into the chat_ctx parameter so the current LLM call sees it.
        """
        try:
            from datetime import datetime, timezone

            captured_any = False
            utterance = ""
            is_user_turn = False
            if chat_ctx.items:
                last = chat_ctx.items[-1]
                utterance = getattr(last, "text_content", None) or ""
                is_user_turn = getattr(last, "role", None) == "user"

            if screen_capture._latest_frame_data is not None:
                b64 = screen_capture.capture_screenshot()
                if b64:
                    _handle_screenshot(
                        ScreenshotEntry(
                            b64=b64,
                            utterance=utterance,
                            timestamp=datetime.now(timezone.utc),
                            source="user",
                        ),
                    )
                    captured_any = True
            if webcam_capture._latest_frame_data is not None:
                b64 = webcam_capture.capture_screenshot()
                if b64:
                    _handle_screenshot(
                        ScreenshotEntry(
                            b64=b64,
                            utterance=utterance,
                            timestamp=datetime.now(timezone.utc),
                            source="webcam",
                        ),
                    )
                    captured_any = True
            if assistant_screen_share_active and is_user_turn:
                entry = await capture_assistant_screenshot(utterance)
                if entry:
                    _handle_screenshot(entry)
                    captured_any = True

            if captured_any:
                content = screenshot_history.build_visual_context_content()
                if content:
                    msg = chat_ctx.add_message(role="user", content=content)
                    chat_ctx.items.pop()
                    chat_ctx.items.insert(-1, msg)
        except Exception as e:
            print(f"[llm_node] screenshot capture error (non-fatal): {e}")

    agent._capture_screenshots_for_llm = _capture_screenshots_for_llm

    # publish call started (shared helper)
    await publish_call_started(contact, channel)
    touch_activity()

    pending_guidance: list[tuple[str, str, bool, str, str]] = (
        []
    )  # (content, response_text, should_speak, guidance_id, guidance_source)
    session_ready = False

    def on_status(data: dict) -> None:
        """Handle status events (call_answered, stop)."""
        event_type = data.get("type", "")
        _log.call_status(event_type)
        touch_activity()

        if event_type == "call_answered":
            call_answered_flag.set()
            agent.set_call_received()
        elif event_type == "stop":
            asyncio.create_task(end_call())

    def apply_guidance(
        content: str,
        response_text: str = "",
        should_speak: bool = False,
        *,
        guidance_id: str = "",
        source: str = "",
        guidance_source: str = "",
    ) -> None:
        _log.guidance_applied(guidance_id, source=guidance_source or source)

        if should_speak and response_text:
            _queued_speech.append(
                (response_text, guidance_id, guidance_source, content),
            )
            maybe_speak_queued()
        else:
            rt.chat_ctx.add_message(
                role="system",
                content=[f"[notification] {content}"],
            )
            asyncio.create_task(_trimmed_update_chat_ctx())
            trigger_generate_reply(
                reason="notification",
                source_id=guidance_id or "guidance_notify",
            )

    def maybe_speak_queued() -> None:
        """Speak the next queued response when user is silent and assistant is idle.

        Gates on agent_state to avoid racing with the fast brain's reply pipeline.
        After the user stops speaking, the agent transitions through thinking →
        speaking → listening. We only speak queued text once the agent has settled
        back to a quiescent state, guaranteeing the fast brain's reply comes first.
        """
        nonlocal _last_say_meta
        if not _queued_speech or user_is_speaking:
            return
        if session.agent_state not in ("listening", "idle"):
            return
        current = session.current_speech
        if current is not None and not current.done:
            return
        text, guidance_id, guidance_source, notification_content = _queued_speech.pop(0)
        _last_say_meta = {
            "guidance_id": guidance_id,
            "source": guidance_source,
            "text": text,
        }

        rt.chat_ctx.add_message(
            role="system",
            content=[f"[notification] {notification_content}"],
        )
        asyncio.create_task(_trimmed_update_chat_ctx())

        _log.guidance_say(guidance_id, text, guidance_source=guidance_source)
        session.say(text, allow_interruptions=True, add_to_chat_ctx=True)

    def on_guidance(data: dict) -> None:
        """Handle guidance from conversation manager."""
        nonlocal assistant_screen_share_active
        payload = data.get("payload") or data
        content = payload.get("content", "")
        # Track screen share state from meet interaction guidance.
        if payload.get("source") == "meet_interaction":
            low = content.lower()
            if "screen sharing is now on" in low:
                assistant_screen_share_active = True
            elif "screen sharing is now off" in low:
                assistant_screen_share_active = False
                screenshot_history.clear(source="assistant")
            if "stopped sharing" in low or "screen sharing is now off" in low:
                source = "user" if "user" in low else "assistant"
                screenshot_history.clear(source=source)
        response_text = payload.get("response_text", "")
        should_speak = payload.get("should_speak", False)
        guidance_source = payload.get("source", "")
        guidance_id = content_trace_id("guid", content)
        _log.guidance_received(
            guidance_source,
            should_speak,
            content,
            guidance_id=guidance_id,
        )
        touch_activity()

        if content:
            if not session_ready:
                pending_guidance.append(
                    (
                        content,
                        response_text,
                        should_speak,
                        guidance_id,
                        guidance_source,
                    ),
                )
                _log.guidance_buffered(guidance_id, len(pending_guidance))
            else:
                apply_guidance(
                    content,
                    response_text,
                    should_speak,
                    guidance_id=guidance_id,
                    guidance_source=guidance_source,
                    source="socket_callback",
                )

    event_broker.register_callback("app:call:status", on_status)
    event_broker.register_callback("app:call:call_guidance", on_guidance)

    # --- Tier 1: Comms from call participants (all calls) ---
    is_boss_user = contact.get("contact_id") == 1
    rt_ref: list = []  # mutable container so the closure captures the live value
    participant_ids: set[int] = set()
    if contact.get("contact_id") is not None:
        participant_ids.add(contact["contact_id"])

    def _sts_inject_and_reply(msg: str, reason: str) -> None:
        """Inject a system message into the STS chat context and trigger a reply."""
        if not session_ready or not rt_ref:
            return
        rt_ref[0].chat_ctx.add_message(role="system", content=[msg])
        asyncio.create_task(_trimmed_update_chat_ctx())
        trigger_generate_reply(reason=reason, source_id="system_event")

    def on_participant_comms(data: dict) -> None:
        raw = data.get("event") if "event" in data else json.dumps(data)
        text = render_participant_comms(
            raw if isinstance(raw, str) else json.dumps(raw),
            participant_ids,
        )
        if not text:
            return
        _log.participant_comms(text)
        touch_activity()
        _sts_inject_and_reply(text, reason="participant_comms")

    event_broker.register_callback("app:comms:*", on_participant_comms)

    # --- Tier 2: All other system events (boss calls only) ---
    if is_boss_user:

        def on_system_event(data: dict) -> None:
            raw = data.get("event") if "event" in data else json.dumps(data)
            text = render_event_for_fast_brain(
                raw if isinstance(raw, str) else json.dumps(raw),
            )
            if not text:
                return
            _log.boss_event(text)
            touch_activity()
            _sts_inject_and_reply(f"[notification] {text}", reason="boss_event")

        event_broker.register_callback("app:actor:*", on_system_event)
        event_broker.register_callback("app:managers:output", on_system_event)
        event_broker.register_callback("app:logging:message_logged", on_system_event)

    # Handle call_answered that arrived during initialization
    if call_answered_flag.is_set():
        _log.call_status("call_answered (arrived during init)")
        agent.set_call_received()

    _log.session_start("Starting AgentSession")
    await session.start(room=ctx.room, agent=agent, room_input_options=rio)

    # Get realtime session (only available after session.start())
    rt = agent.realtime_llm_session
    rt_ref.append(rt)

    # Trim-aware wrapper for syncing context to the Realtime API server.
    async def _trimmed_update_chat_ctx() -> None:
        window = SETTINGS.conversation.FAST_BRAIN_CONTEXT_WINDOW
        trimmed = trim_fast_brain_context(rt.chat_ctx.items, window)
        if len(trimmed) < len(rt.chat_ctx.items):
            rt.chat_ctx._items[:] = trimmed
        await rt.update_chat_ctx(rt.chat_ctx)

    # Hydrate historical context from EventBus into the fast brain.
    history_lines = await hydrate_fast_brain_history(
        participant_ids=participant_ids,
        is_boss_user=is_boss_user,
        assistant_name=assistant_name or "Assistant",
        limit=SETTINGS.conversation.FAST_BRAIN_CONTEXT_WINDOW,
    )
    if history_lines:
        history_block = (
            "--- Recent conversation history ---\n"
            + "\n".join(history_lines)
            + "\n--- Current call ---"
        )
        rt.chat_ctx.add_message(role="system", content=[history_block])
        await _trimmed_update_chat_ctx()
        _log.info(f"Hydrated {len(history_lines)} historical events into context")

    # Log real usage per turn via unillm (replaces duration-based heuristic)
    @rt.on("metrics_collected")
    def _on_metrics(metrics):
        usage = {
            "input_tokens": metrics.input_tokens,
            "output_tokens": metrics.output_tokens,
            "total_tokens": metrics.total_tokens,
            "input_token_details": {
                "audio_tokens": metrics.input_token_details.audio_tokens,
                "text_tokens": metrics.input_token_details.text_tokens,
                "cached_tokens": metrics.input_token_details.cached_tokens,
            },
            "output_token_details": {
                "audio_tokens": metrics.output_token_details.audio_tokens,
                "text_tokens": metrics.output_token_details.text_tokens,
            },
        }
        transcript = [
            {"role": item.role, "content": item.text_content or ""}
            for item in rt.chat_ctx.items
            if item.text_content
        ]
        unillm.log_usage(
            metrics.model,
            usage,
            transcript=transcript,
            label=metrics.model,
        )

    # For outbound calls, wait for call_answered before speaking.
    # Unlike call.py, the Realtime API bypasses llm_node, so we must wait here.
    # Note: await on an already-set Event returns immediately.
    if outbound:
        _log.info("Outbound call: waiting for call_answered before speaking…")
        await call_answered_flag.wait()
        _log.call_status("call_answered (outbound)")

    # Mark session ready and process any buffered guidance BEFORE first utterance.
    # After this, the on_guidance callback will apply guidance immediately.
    session_ready = True
    if pending_guidance:
        _log.session_ready(
            f"Applying {len(pending_guidance)} buffered guidance message(s)",
        )
        for (
            content,
            response_text,
            should_speak,
            guidance_id,
            guidance_source,
        ) in pending_guidance:
            apply_guidance(
                content,
                response_text,
                should_speak,
                guidance_id=guidance_id,
                source="pending_buffer_flush",
                guidance_source=guidance_source,
            )
        pending_guidance.clear()

    await trigger_generate_reply(
        reason="session_start",
        source_id="startup",
        allow_interruptions=True,
        wait_for_completion=True,
    )


if __name__ == "__main__":
    # Shared CLI handling
    room_name = configure_from_cli(
        extra_env=[
            ("CONTACT", True),
            ("BOSS", True),
            ("ASSISTANT_BIO", False),
            ("ASSISTANT_ID", False),
            ("USER_ID", False),
        ],
    )

    if should_dispatch_livekit_agent():
        _log.dispatch(f"Dispatching LiveKit agent {room_name}")
        dispatch_livekit_agent(
            room_name,
            record=True,
            assistant_id=SESSION_DETAILS.assistant.id,
            user_id=SESSION_DETAILS.user.id,
        )
        _log.dispatch(f"LiveKit agent {room_name} dispatched")

    # Run the agent using the standard CLI - this is the natural way to run LiveKit agents.
    # The process will be terminated via SIGTERM when cleanup_call_proc() is called.
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=room_name,
            initialize_process_timeout=60,
        ),
    )
