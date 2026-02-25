import os
import sys
import json
import asyncio

os.environ["UNITY_TERMINAL_LOG"] = "true"

from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    cartesia,
    deepgram,
    elevenlabs,
    silero,
)

from unity.conversation_manager.livekit_unify_adapter import UnifyLLM

if sys.platform == "darwin":
    from livekit.plugins import noise_cancellation

from livekit.plugins.turn_detector.english import EnglishModel
from livekit.agents import ChatContext, ChatMessage
from livekit.agents import ModelSettings, llm, FunctionTool

from typing import AsyncIterable

load_dotenv()

from unity.conversation_manager.events import *
from unity.conversation_manager.utils import dispatch_livekit_agent
from unity.conversation_manager.prompt_builders import build_voice_agent_prompt
from unity.conversation_manager.tracing import (
    content_trace_id,
    monotonic_ms,
    now_utc_iso,
)
from unity.session_details import SESSION_DETAILS

# Shared helpers
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

# Globals initialized lazily or via prewarm to avoid duplicate heavy init
STT = None
VAD = None


# Module-level logger created early for prewarm (before entrypoint runs).
_log = FastBrainLogger()


def prewarm(_ctx=None):
    global STT, VAD
    try:
        _log.info("Prewarm: initializing STT, VAD and turn detector…")
        STT = deepgram.STT(model="nova-3", language="en-GB")
        VAD = silero.VAD.load(min_speech_duration=0.15)
        _log.info("Prewarm complete")
    except Exception as e:  # noqa: BLE001
        _log.error(f"Prewarm failed: {e}")
        STT = None
        VAD = None


class Assistant(Agent):
    """
    TTS Fast Brain - handles real-time conversation independently.

    Uses a lightweight LLM (gpt-5-mini via UnifyLLM adapter) for fast
    conversational responses. Routes through unillm.AsyncUnify for local
    caching (CI) and usage tracking.
    Communicates with the Main CM Brain (slow brain) via Unix domain socket IPC.
    """

    def __init__(
        self,
        contact: dict,
        boss: dict,
        channel: str,
        instructions: str,
        outbound: bool = False,
    ) -> None:
        self.contact = contact
        self.boss = boss
        self.channel = channel
        self.utterance_event = (
            InboundPhoneUtterance if channel == "phone" else InboundUnifyMeetUtterance
        )
        self.assistant_utterance_event = (
            OutboundPhoneUtterance if channel == "phone" else OutboundUnifyMeetUtterance
        )
        self.call_received = not outbound

        super().__init__(instructions=instructions)

    def set_call_received(self):
        self.call_received = True

    async def on_user_turn_completed(
        self,
        turn_ctx: ChatContext,
        new_message: ChatMessage,
    ) -> None:
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
        """Wait for call connection then delegate to parent LLM."""
        _log.info("Waiting for call to be received…")
        while not self.call_received:
            await asyncio.sleep(0.1)
        _log.call_status("call_received")

        await self._capture_screenshots_for_llm(chat_ctx)

        asyncio.create_task(
            event_broker.publish("app:comms:fast_brain_generating", "{}"),
        )

        from unity.settings import SETTINGS

        window = SETTINGS.conversation.FAST_BRAIN_CONTEXT_WINDOW
        trimmed_items = trim_fast_brain_context(chat_ctx.items, window)
        if len(trimmed_items) < len(chat_ctx.items):
            trimmed_ctx = llm.ChatContext()
            for item in trimmed_items:
                trimmed_ctx.items.append(item)
        else:
            trimmed_ctx = chat_ctx

        _log.info("LLM thinking… reason=llm_node_start")
        async for chunk in super().llm_node(trimmed_ctx, tools, model_settings):
            yield chunk


async def entrypoint(ctx: agents.JobContext):
    global STT, VAD

    # Wire the module-level logger into the shared event broker.
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

    # Fallback for whenever pre-loading fails
    if STT is None:
        STT = deepgram.STT(model="nova-3", language="en-GB")
        VAD = silero.VAD.load(min_speech_duration=0.15)

    from unity.settings import SETTINGS

    # Fast brain LLM - lightweight model for responsive conversation
    # Uses UnifyLLM adapter for local caching (CI) and usage tracking
    llm_model = UnifyLLM(
        model=SETTINGS.conversation.FAST_BRAIN_MODEL,
        reasoning_effort="low",
    )

    assistant_name = SESSION_DETAILS.assistant.name
    system_prompt = build_voice_agent_prompt(
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
    _log.config(f"System prompt ({len(system_prompt)} chars)")

    session = AgentSession(
        llm=llm_model,
        stt=STT,
        tts=(
            elevenlabs.TTS(
                voice_id=voice_id if voice_id != "" else elevenlabs.DEFAULT_VOICE_ID,
                model="eleven_multilingual_v2",
            )
            if voice_provider == "elevenlabs"
            else cartesia.TTS(
                voice=voice_id if voice_id != "" else cartesia.tts.TTSDefaultVoiceId,
            )
        ),
        vad=VAD,
        turn_detection=EnglishModel(),
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
        last_role = (
            assistant._chat_ctx.items[-1].role if assistant._chat_ctx.items else "none"
        )
        trigger = {
            "generation_id": generation_id,
            "reason": reason,
            "source_id": source_id,
            "user_is_speaking": user_is_speaking,
            "last_chat_role": last_role,
            "ts_utc": now_utc_iso(),
            "monotonic_ms": monotonic_ms(),
        }
        enqueue_trace_context = getattr(llm_model, "enqueue_trace_context", None)
        if callable(enqueue_trace_context):
            enqueue_trace_context(trigger)
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
    def _on_user_state_changed(ev):
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
    _agent_service_url: str | None = None
    _visual_ctx_msg_id: str | None = None
    import aiohttp as _aiohttp

    _screenshot_http_session = _aiohttp.ClientSession()

    def _clear_visual_context(source: str | None = None) -> None:
        """Remove visual context from chat contexts and clear screenshot history."""
        nonlocal _visual_ctx_msg_id
        screenshot_history.clear(source=source)
        if not screenshot_history.build_visual_context_content():
            for ctx in (assistant._chat_ctx, session._chat_ctx):
                if _visual_ctx_msg_id is not None:
                    idx = ctx.index_by_id(_visual_ctx_msg_id)
                    if idx is not None:
                        ctx.items.pop(idx)
            _visual_ctx_msg_id = None

    def _inject_visual_context() -> None:
        """Replace the visual context system message in the chat context."""
        nonlocal _visual_ctx_msg_id
        content = screenshot_history.build_visual_context_content()
        if not content:
            return
        # Remove the previous visual context message if present.
        for ctx in (assistant._chat_ctx, session._chat_ctx):
            if _visual_ctx_msg_id is not None:
                idx = ctx.index_by_id(_visual_ctx_msg_id)
                if idx is not None:
                    ctx.items.pop(idx)
        msg = assistant._chat_ctx.add_message(role="user", content=content)
        session._chat_ctx.add_message(
            role="user",
            content=content,
            id=msg.id,
        )
        _visual_ctx_msg_id = msg.id

    def _publish_screenshot(entry: ScreenshotEntry, filepath: str) -> None:
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

    def _handle_screenshot(entry: ScreenshotEntry) -> None:
        """Process a captured screenshot: history, visual context, disk, IPC."""
        filepath = generate_screenshot_path(entry)
        screenshot_history.add(entry, filepath)
        _inject_visual_context()
        if entry.source != "assistant":
            _publish_screenshot(entry, filepath)

    @session.on("conversation_item_added")
    def _on_chat_item_added(ev):
        """Publish both user and assistant utterances from a single location."""
        nonlocal _last_say_meta
        role = ev.item.role  # "user" | "assistant"
        text = ev.item.text_content or ""
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

    assistant = Assistant(
        contact=contact,
        boss=boss,
        channel=channel,
        instructions=system_prompt,
        outbound=outbound,
    )

    async def _capture_screenshots_for_llm(chat_ctx) -> None:
        """Capture fresh screenshots and inject them into the LLM's chat_ctx.

        The LiveKit pipeline passes a **copy** of the chat context to
        ``llm_node``.  ``_handle_screenshot`` updates the live
        ``session._chat_ctx`` (for subsequent turns and IPC), but that copy
        is stale.  After capturing, we rebuild the visual context content and
        inject it directly into the ``chat_ctx`` parameter so the current
        LLM call sees the screenshot.

        Sync captures (user screen / webcam) are negligible (~1 ms).
        The async assistant capture (~500 ms HTTP) only runs on user turns.
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

            copy_visual_id = _visual_ctx_msg_id

            if is_user_turn and screen_capture._latest_frame_data is not None:
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
            if is_user_turn and webcam_capture._latest_frame_data is not None:
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
                entry = await capture_assistant_screenshot(
                    utterance,
                    fb_logger=_log,
                    agent_service_url=_agent_service_url,
                    http_session=_screenshot_http_session,
                )
                if entry:
                    if not assistant_screen_share_active:
                        if copy_visual_id is not None:
                            idx = chat_ctx.index_by_id(copy_visual_id)
                            if idx is not None:
                                chat_ctx.items.pop(idx)
                    else:
                        _handle_screenshot(entry)
                        captured_any = True

            if captured_any:
                content = screenshot_history.build_visual_context_content()
                if content:
                    if copy_visual_id is not None:
                        idx = chat_ctx.index_by_id(copy_visual_id)
                        if idx is not None:
                            chat_ctx.items.pop(idx)
                    msg = chat_ctx.add_message(role="user", content=content)
                    chat_ctx.items.pop()
                    chat_ctx.items.insert(-1, msg)
        except Exception as e:
            print(f"[llm_node] screenshot capture error (non-fatal): {e}")

    assistant._capture_screenshots_for_llm = _capture_screenshots_for_llm

    rio = RoomInputOptions(
        noise_cancellation=(
            noise_cancellation.BVC() if sys.platform == "darwin" else None
        ),
    )

    # Publish call started (shared helper)
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
            assistant.set_call_received()
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
            guidance_message = f"[notification] {content}"
            assistant._chat_ctx.add_message(
                role="system",
                content=[guidance_message],
            )
            session._chat_ctx.add_message(
                role="system",
                content=[guidance_message],
            )
            if guidance_source != "meet_interaction":
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

        guidance_message = f"[notification] {notification_content}"
        assistant._chat_ctx.add_message(
            role="system",
            content=[guidance_message],
        )
        session._chat_ctx.add_message(
            role="system",
            content=[guidance_message],
        )

        _log.guidance_say(guidance_id, text, guidance_source=guidance_source)
        session.say(text, allow_interruptions=True, add_to_chat_ctx=True)

    def on_guidance(data: dict) -> None:
        """Handle guidance from conversation manager."""
        nonlocal assistant_screen_share_active, _agent_service_url
        payload = data.get("payload") or data
        content = payload.get("content", "")
        # Track screen share state from meet interaction guidance.
        if payload.get("source") == "meet_interaction":
            low = content.lower()
            if "screen sharing is now on" in low:
                assistant_screen_share_active = True
                if payload.get("agent_service_url"):
                    _agent_service_url = payload["agent_service_url"]
            elif "screen sharing is now off" in low:
                assistant_screen_share_active = False
                _clear_visual_context(source="assistant")
            elif "stopped sharing" in low:
                source = "user" if "user" in low else "assistant"
                _clear_visual_context(source=source)
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
                    source="socket_callback",
                    guidance_source=guidance_source,
                )

    event_broker.register_callback("app:call:status", on_status)
    event_broker.register_callback("app:call:call_guidance", on_guidance)

    # --- Tier 1: Comms from call participants (all calls) ---
    # Build the set of contact_ids on this call.
    is_boss_user = contact.get("contact_id") == 1
    participant_ids: set[int] = set()
    if contact.get("contact_id") is not None:
        participant_ids.add(contact["contact_id"])

    def _inject_and_reply(msg: str, reason: str) -> None:
        """Inject a system message into chat context and trigger a reply."""
        assistant._chat_ctx.add_message(role="system", content=[msg])
        session._chat_ctx.add_message(role="system", content=[msg])
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
        if not session_ready:
            return
        _inject_and_reply(text, reason="participant_comms")

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
            if not session_ready:
                return
            _inject_and_reply(f"[notification] {text}", reason="boss_event")

        event_broker.register_callback("app:actor:*", on_system_event)
        event_broker.register_callback("app:managers:output", on_system_event)
        event_broker.register_callback("app:logging:message_logged", on_system_event)

    # Handle call_answered that arrived during initialization
    if call_answered_flag.is_set():
        _log.call_status("call_answered (arrived during init)")
        assistant.set_call_received()

    _log.session_start("Starting AgentSession")
    await session.start(room=ctx.room, agent=assistant, room_input_options=rio)

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
        assistant._chat_ctx.add_message(role="system", content=[history_block])
        session._chat_ctx.add_message(role="system", content=[history_block])
        _log.info(f"Hydrated {len(history_lines)} historical events into context")

    # Mark session ready and process any buffered guidance BEFORE first utterance.
    # After this, the on_guidance callback will apply guidance immediately.
    # Note: For outbound calls, llm_node will wait for call_received (set by on_status).
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
    # CLI handling
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
            assistant_id=SESSION_DETAILS.assistant.agent_id,
            user_id=SESSION_DETAILS.user.id,
        )
        _log.dispatch(f"LiveKit agent {room_name} dispatched")

    # Run the agent using the standard CLI - this is the natural way to run LiveKit agents.
    # The process will be terminated via SIGTERM when cleanup_call_proc() is called.
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=room_name,
            prewarm_fnc=prewarm,
            initialize_process_timeout=60,
        ),
    )
