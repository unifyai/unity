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
    create_end_call,  # kept for test monkeypatch compatibility
    match_say_meta,
    setup_inactivity_timeout,
    setup_participant_disconnect_handler,  # kept for test monkeypatch compatibility
    publish_call_started,
    publish_call_ended,
    delete_livekit_room,
    configure_from_cli,
    should_dispatch_livekit_agent,
    start_event_broker_receive,
    UserTrackCaptureManager,
    ScreenshotHistory,
    capture_assistant_screenshot,
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
        VAD = silero.VAD.load(min_speech_duration=0.15, min_silence_duration=1.0)
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
        if channel == "phone_call":
            self.utterance_event = InboundPhoneUtterance
            self.assistant_utterance_event = OutboundPhoneUtterance
        elif channel == "whatsapp_call":
            self.utterance_event = InboundWhatsAppCallUtterance
            self.assistant_utterance_event = OutboundWhatsAppCallUtterance
        elif channel == "google_meet":
            self.utterance_event = InboundGoogleMeetUtterance
            self.assistant_utterance_event = OutboundGoogleMeetUtterance
        else:
            self.utterance_event = InboundUnifyMeetUtterance
            self.assistant_utterance_event = OutboundUnifyMeetUtterance
        self.call_received = not outbound
        self._user_speech_logged = False
        self.user_turn_generating = False

        super().__init__(instructions=instructions)

    def set_call_received(self):
        self.call_received = True

    async def on_user_turn_completed(
        self,
        turn_ctx: ChatContext,
        new_message: ChatMessage,
    ) -> None:
        """Hook called when user finishes speaking — before LLM generation starts."""
        text = new_message.text_content or ""
        if text:
            _log.user_speech(text)
            self._user_speech_logged = True

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[FunctionTool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[llm.ChatChunk]:
        """Wait for call connection then delegate to parent LLM."""
        self.user_turn_generating = True
        try:
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

            _log.info("LLM thinking… (llm_node_start)")
            async for chunk in super().llm_node(
                trimmed_ctx,
                tools,
                model_settings,
            ):
                yield chunk
        finally:
            self.user_turn_generating = False


def _load_config_from_metadata(ctx: agents.JobContext) -> dict | None:
    """Parse call config from job dispatch metadata (persistent worker path).

    Returns the parsed dict, or None when no metadata is present (legacy
    subprocess path).
    """
    raw = getattr(ctx.job, "metadata", None)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


def _configure_child_logging() -> None:
    """Ensure Unity's LOGGER works in LiveKit's pre-warmed child processes.

    LiveKit agents (v1.2.x) uses ``forkserver`` on Linux.  Child processes
    are forked from a lean server process, not from the worker, so the
    ``unity`` logger's handlers may point to stale file descriptors.

    The framework routes child logs through a ``LogQueueHandler`` on the
    **root** logger, which serialises records back to the worker process.
    We enable propagation so Unity records flow through that channel, and
    remove any direct handlers that could double-emit or silently fail.
    """
    import logging as _logging

    from unity.logger import LOGGER as _L

    _L.propagate = True
    for h in list(_L.handlers):
        _L.removeHandler(h)

    for name in ("livekit", "livekit.agents", "livekit.plugins"):
        lg = _logging.getLogger(name)
        lg.propagate = True
        for h in list(lg.handlers):
            lg.removeHandler(h)


async def entrypoint(ctx: agents.JobContext):
    global STT, VAD

    _configure_child_logging()

    # Wire the module-level logger into the shared event broker.
    event_broker.set_logger(_log)

    # --- Config: persistent worker (job metadata) or legacy subprocess (env) ---
    meta = _load_config_from_metadata(ctx)
    _log.info(f"Entrypoint started (has_metadata={meta is not None})")
    if meta:
        from unity.conversation_manager.domains.ipc_socket import init_socket_for_job

        ipc_path = meta.get("ipc_socket_path", "")
        if ipc_path:
            init_socket_for_job(ipc_path)
            event_broker.reinit_socket()
            _log.info(f"IPC socket initialised: {ipc_path}")
        else:
            _log.warning("No ipc_socket_path in job metadata — IPC disabled")

        voice_provider = meta.get("voice_provider", "cartesia")
        voice_id = meta.get("voice_id", "")
        outbound = meta.get("outbound", False)
        channel = meta.get("channel", "phone")
        assistant_bio = meta.get("assistant_bio", "")
        contact = meta.get("contact", {})
        boss = meta.get("boss", {})
        SESSION_DETAILS.assistant.about = assistant_bio
        if meta.get("assistant_id"):
            try:
                SESSION_DETAILS.assistant.agent_id = int(meta["assistant_id"])
            except (ValueError, TypeError):
                pass
        if meta.get("user_id"):
            SESSION_DETAILS.user.id = meta["user_id"]
        if meta.get("assistant_name"):
            parts = meta["assistant_name"].split(None, 1)
            SESSION_DETAILS.assistant.first_name = parts[0] if parts else ""
            SESSION_DETAILS.assistant.surname = parts[1] if len(parts) > 1 else ""
    else:
        _log.warning(
            "No job metadata — falling back to env-based config (IPC disabled)",
        )
        SESSION_DETAILS.populate_from_env()
        voice_provider = SESSION_DETAILS.voice.provider
        voice_id = SESSION_DETAILS.voice.id
        outbound = SESSION_DETAILS.voice_call.outbound
        channel = SESSION_DETAILS.voice_call.channel
        assistant_bio = SESSION_DETAILS.assistant.about
        contact = json.loads(SESSION_DETAILS.voice_call.contact_json or "{}")
        boss = json.loads(SESSION_DETAILS.voice_call.boss_json or "{}")

    _log.config(
        f"voice_provider={voice_provider} voice_id={voice_id} outbound={outbound} channel={channel}",
    )

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

    # Fallback for whenever pre-loading fails
    if STT is None:
        STT = deepgram.STT(model="nova-3", language="en-GB")
        VAD = silero.VAD.load(min_speech_duration=0.15, min_silence_duration=1.0)

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
        is_boss_user=bool(contact.get("is_system", False)),
        contact_rolling_summary=contact.get("rolling_summary", ""),
        demo_mode=SETTINGS.DEMO_MODE,
        channel=channel,
        user_desktop_control=SETTINGS.conversation.USER_DESKTOP_CONTROL_ENABLED,
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
        min_endpointing_delay=0.75,
    )

    user_is_speaking = False
    _queued_speech: list[tuple[str, str, str, str, str]] = []
    _say_meta_queue: list[dict] = []
    _dedup_in_flight = False
    generation_seq = 0
    user_state_seq = 0
    _was_quiescent = True
    _pending_reply_timer: asyncio.TimerHandle | None = None
    _NOTIFY_COALESCE_S = 0.05

    def _log_reply_task(task: asyncio.Task) -> None:
        try:
            task.result()
            _log.llm_completed()
        except asyncio.CancelledError:
            _log.llm_cancelled()
        except Exception as exc:  # noqa: BLE001
            _log.llm_error(str(exc))

    def _fire_generate_reply(
        reason: str,
        source_id: str,
        allow_interruptions: bool = True,
        user_input: str | None = None,
    ):
        nonlocal generation_seq, _pending_reply_timer
        _pending_reply_timer = None
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
            queued_speech=len(_queued_speech),
        )
        reply_kwargs = {"allow_interruptions": allow_interruptions}
        if user_input is not None:
            reply_kwargs["user_input"] = user_input
        maybe_result = session.generate_reply(**reply_kwargs)
        if isinstance(maybe_result, asyncio.Task):
            maybe_result.add_done_callback(_log_reply_task)
        return maybe_result

    def trigger_generate_reply(
        reason: str,
        source_id: str,
        *,
        allow_interruptions: bool = True,
        wait_for_completion: bool = False,
        user_input: str | None = None,
    ):
        nonlocal _pending_reply_timer
        if _pending_reply_timer is not None:
            _pending_reply_timer.cancel()
            _pending_reply_timer = None

        if wait_for_completion:
            return _fire_generate_reply(
                reason,
                source_id,
                allow_interruptions,
                user_input,
            )

        loop = asyncio.get_event_loop()
        _pending_reply_timer = loop.call_later(
            _NOTIFY_COALESCE_S,
            _fire_generate_reply,
            reason,
            source_id,
            allow_interruptions,
            user_input,
        )

    def _invalidate_current_generation(reason: str, source_id: str) -> None:
        """Cancel in-flight FastBrain generation and re-trigger with updated context.

        Called when a significant IPC event (slow brain notification, outbound
        message confirmation) arrives while the FastBrain LLM is mid-generation.
        The 50 ms coalescence in ``trigger_generate_reply`` naturally collapses
        bursts (e.g. notification + message_sent arriving ~100 ms apart) into a
        single regeneration.
        """
        if not assistant.user_turn_generating:
            return
        _log.info(f"Invalidating in-flight generation: {reason}")
        session.interrupt()
        trigger_generate_reply(reason=reason, source_id=source_id)

    if channel == "phone_call":
        user_utterance_event = InboundPhoneUtterance
        assistant_utterance_event = OutboundPhoneUtterance
    elif channel == "whatsapp_call":
        user_utterance_event = InboundWhatsAppCallUtterance
        assistant_utterance_event = OutboundWhatsAppCallUtterance
    elif channel == "google_meet":
        user_utterance_event = InboundGoogleMeetUtterance
        assistant_utterance_event = OutboundGoogleMeetUtterance
    else:
        user_utterance_event = InboundUnifyMeetUtterance
        assistant_utterance_event = OutboundUnifyMeetUtterance

    # Register cleanup as a LiveKit shutdown callback so it runs on any
    # exit path: participant disconnect, inactivity, or explicit stop.
    async def _on_job_shutdown():
        await delete_livekit_room(ctx.room.name)
        await publish_call_ended(contact, channel)

    ctx.add_shutdown_callback(_on_job_shutdown)

    # Bridge AgentSession close → job shutdown.  close_on_disconnect
    # (RoomInputOptions, default True) closes the AgentSession when the
    # linked participant leaves, but does NOT resolve the JobContext's
    # shutdown future — so our shutdown callbacks never fire.  Listening
    # for the session "close" event completes the chain.
    @session.on("close")
    def _on_session_close(ev):
        from livekit.agents.voice.events import CloseReason

        if ev.reason == CloseReason.PARTICIPANT_DISCONNECTED:
            ctx.shutdown(reason="participant_disconnected")

    async def _shutdown_inactivity():
        ctx.shutdown(reason="inactivity")

    touch_activity = setup_inactivity_timeout(_shutdown_inactivity)

    def _check_quiescence_transition() -> None:
        nonlocal _was_quiescent
        now_quiescent = _is_pipeline_quiescent()
        if now_quiescent != _was_quiescent:
            _was_quiescent = now_quiescent
            import json as _json

            asyncio.create_task(
                event_broker.publish(
                    "app:comms:pipeline_quiescent",
                    _json.dumps({"quiescent": now_quiescent}),
                ),
            )

    @session.on("user_state_changed")
    def _on_user_state_changed(ev):
        nonlocal user_is_speaking, user_state_seq
        user_state_seq += 1
        state_id = f"usrstate-{user_state_seq:06d}"
        user_is_speaking = ev.new_state == "speaking"
        _log.user_state(ev.new_state, state_id=state_id)
        touch_activity()
        _check_quiescence_transition()

    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev):
        """Try queued speech only after the agent settles into a quiescent state.

        We intentionally do NOT trigger from user_state_changed because there is
        a gap between VAD silence detection and the turn detector confirming the
        turn. During that gap, agent_state is still "listening" and current_speech
        is None — firing then would race ahead of the fast brain's reply.

        Triggering here guarantees the full thinking → speaking → listening cycle
        has completed before queued notification speech plays.
        """
        if ev.new_state in ("listening", "idle"):
            maybe_speak_queued()
        _check_quiescence_transition()

    # -- Screenshot state --
    screenshot_history = ScreenshotHistory()
    assistant_screen_share_active = False
    user_remote_control_active = False
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

    async def _refresh_screenshots() -> None:
        """Capture fresh screenshots from all active sources and update visual context.

        Called before any module that needs the latest visual state (e.g., the
        notification reply evaluator, the fast-brain LLM).  Sync captures
        (user screen, webcam) are ~1 ms.  The assistant capture reads from the
        agent-service screenshot cache (~0 ms) unless the user has remote
        control, in which case a live capture (~500 ms) is used.
        """
        from datetime import datetime, timezone

        if screen_capture._latest_frame_data is not None:
            b64 = screen_capture.capture_screenshot()
            if b64:
                _handle_screenshot(
                    ScreenshotEntry(
                        b64=b64,
                        utterance="",
                        timestamp=datetime.now(timezone.utc),
                        source="user",
                    ),
                )

        if webcam_capture._latest_frame_data is not None:
            b64 = webcam_capture.capture_screenshot()
            if b64:
                _handle_screenshot(
                    ScreenshotEntry(
                        b64=b64,
                        utterance="",
                        timestamp=datetime.now(timezone.utc),
                        source="webcam",
                    ),
                )

        if assistant_screen_share_active:
            entry = await capture_assistant_screenshot(
                utterance="",
                cached=not user_remote_control_active,
                fb_logger=_log,
                agent_service_url=_agent_service_url,
                http_session=_screenshot_http_session,
            )
            if entry and assistant_screen_share_active:
                _handle_screenshot(entry)

    @session.on("conversation_item_added")
    def _on_chat_item_added(ev):
        """Publish both user and assistant utterances from a single location."""
        role = ev.item.role  # "user" | "assistant"
        text = ev.item.text_content or ""
        utterance_id = content_trace_id("utt", f"{role}:{text}")
        say_meta: dict | None = None
        if role == "assistant" and _say_meta_queue:
            for i, candidate in enumerate(_say_meta_queue):
                if match_say_meta(candidate, text):
                    say_meta = _say_meta_queue.pop(i)
                    break
        if role == "user":
            if not assistant._user_speech_logged:
                _log.user_speech(text)
            assistant._user_speech_logged = False
        else:
            source = (say_meta or {}).get("source", "reply")
            if say_meta and say_meta.get("llm_log_path"):
                log_path = say_meta["llm_log_path"]
            elif source == "reply":
                log_path = getattr(llm_model, "last_log_path", "")
            else:
                log_path = ""
            _log.assistant_speech(text, source=source, llm_log_path=log_path)
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
        ``llm_node``.  ``_refresh_screenshots`` updates the live
        ``session._chat_ctx`` (for subsequent turns and IPC), but that copy
        is stale.  After refreshing, we rebuild the visual context content
        and inject it directly into the ``chat_ctx`` parameter so the
        current LLM call sees the screenshot.
        """
        try:
            copy_visual_id = _visual_ctx_msg_id

            await _refresh_screenshots()

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

    pending_notifications: list[tuple[str, str, bool, str, str, str]] = (
        []
    )  # (content, response_text, should_speak, notification_id, notification_source, llm_log_path)
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
            ctx.shutdown(reason="stopped")

    def _is_pipeline_quiescent() -> bool:
        """True when the voice pipeline is completely idle (no speech in flight)."""
        if user_is_speaking:
            return False
        if session.agent_state not in ("listening", "idle"):
            return False
        current = session.current_speech
        if current is not None and not current.done:
            return False
        return True

    def _speak_now(
        text: str,
        notification_id: str,
        notification_source: str,
        notification_content: str,
        llm_log_path: str,
    ) -> None:
        _say_meta_queue.append(
            {
                "notification_id": notification_id,
                "source": notification_source,
                "text": text,
                "llm_log_path": llm_log_path,
            },
        )
        # Context injection is handled by apply_notification unconditionally
        # for non-proactive notifications. No injection here to avoid doubles.
        _log.notification_say(text, notification_source=notification_source)
        session.say(text, allow_interruptions=True, add_to_chat_ctx=True)

    def _extract_chat_messages(
        ctx,
        *,
        strip_images: bool = False,
        tail: int | None = None,
    ) -> list[dict]:
        """Convert a LiveKit ChatContext into a list of message dicts for direct LLM calls.

        Parameters
        ----------
        strip_images : bool
            When True, image content parts are dropped and only the text portions
            of multi-modal messages are kept.  Messages that become empty after
            stripping are omitted entirely.
        tail : int | None
            When set, only the last *tail* messages are returned (after any
            image stripping).  Useful for keeping the context compact.
        """
        from livekit.agents.llm import ImageContent

        messages: list[dict] = []
        for item in ctx.items:
            role = getattr(item, "role", None)
            if role is None:
                continue
            raw_content = getattr(item, "content", None)
            if not raw_content:
                continue
            has_images = isinstance(raw_content, list) and any(
                isinstance(c, ImageContent) for c in raw_content
            )
            if has_images:
                if strip_images:
                    text_parts = [c for c in raw_content if isinstance(c, str)]
                    text = " ".join(text_parts).strip()
                    if text:
                        messages.append({"role": role, "content": text})
                else:
                    parts: list[dict] = []
                    for c in raw_content:
                        if isinstance(c, str):
                            parts.append({"type": "text", "text": c})
                        elif isinstance(c, ImageContent) and isinstance(c.image, str):
                            parts.append(
                                {"type": "image_url", "image_url": {"url": c.image}},
                            )
                    if parts:
                        messages.append({"role": role, "content": parts})
            else:
                text = getattr(item, "text_content", None)
                if not text:
                    continue
                messages.append({"role": role, "content": text})
        if tail is not None and len(messages) > tail:
            messages = messages[-tail:]
        return messages

    def apply_notification(
        content: str,
        response_text: str = "",
        should_speak: bool = False,
        *,
        notification_id: str = "",
        source: str = "",
        notification_source: str = "",
        llm_log_path: str = "",
    ) -> None:
        # Inject into chat context unconditionally so the fast brain always
        # sees the latest slow brain understanding.  Proactive speech is
        # fire-and-forget filler — it never updates context.
        if notification_source != "proactive_speech":
            notification_message = f"[notification] {content}"
            assistant._chat_ctx.add_message(
                role="system",
                content=[notification_message],
            )
            session._chat_ctx.add_message(
                role="system",
                content=[notification_message],
            )

        if should_speak and response_text:
            if notification_source == "proactive_speech":
                # Proactive speech exists purely to fill silence — never queue it.
                # Play immediately if the pipeline is fully quiescent and nothing
                # else is waiting; otherwise discard silently.
                if not _is_pipeline_quiescent() or _queued_speech:
                    return
                _speak_now(
                    response_text,
                    notification_id,
                    notification_source,
                    content,
                    llm_log_path,
                )
            else:
                # Latest slow brain guidance supersedes older queued speech.
                _queued_speech.clear()
                _queued_speech.append(
                    (
                        response_text,
                        notification_id,
                        notification_source,
                        content,
                        llm_log_path,
                    ),
                )
                maybe_speak_queued()

    def _get_recent_assistant_utterances(n: int = 10) -> list[str]:
        """Return the last *n* assistant utterances from the fast brain's chat context.

        Walks ``assistant._chat_ctx.items`` in reverse, collecting up to *n*
        text strings from items with ``role == "assistant"``.  Returns them in
        chronological order (oldest first).
        """
        results: list[str] = []
        for item in reversed(assistant._chat_ctx.items):
            if getattr(item, "role", None) != "assistant":
                continue
            raw = getattr(item, "content", None)
            if isinstance(raw, str) and raw:
                results.append(raw)
            elif isinstance(raw, list):
                text = " ".join(c for c in raw if isinstance(c, str)).strip()
                if text:
                    results.append(text)
            if len(results) >= n:
                break
        results.reverse()
        return results

    def _get_recent_notifications(n: int = 5) -> list[str]:
        """Return the last *n* ``[notification]`` system messages from the chat context."""
        results: list[str] = []
        prefix = "[notification] "
        for item in reversed(assistant._chat_ctx.items):
            if getattr(item, "role", None) != "system":
                continue
            raw = getattr(item, "content", None)
            if isinstance(raw, list):
                raw = " ".join(c for c in raw if isinstance(c, str)).strip()
            if isinstance(raw, str) and raw.startswith(prefix):
                results.append(raw[len(prefix) :])
            if len(results) >= n:
                break
        results.reverse()
        return results

    async def _dedup_and_speak(
        text: str,
        notification_id: str,
        notification_source: str,
        notification_content: str,
        llm_log_path: str,
    ) -> None:
        nonlocal _dedup_in_flight
        _dedup_in_flight = True
        try:
            recent = _get_recent_assistant_utterances()
            notifications = _get_recent_notifications() if recent else []
            if recent and SETTINGS.conversation.SPEECH_DEDUP_ENABLED:
                from unity.conversation_manager.domains.speech_dedup import (
                    SpeechDeduplicationChecker,
                )

                dedup = await SpeechDeduplicationChecker().evaluate(
                    proposed_speech=text,
                    recent_utterances=recent,
                    recent_notifications=notifications,
                )
                if dedup.should_suppress:
                    _log.dedup_suppressed(text, dedup.reasoning)
                    return
            _speak_now(
                text,
                notification_id,
                notification_source,
                notification_content,
                llm_log_path,
            )
        finally:
            _dedup_in_flight = False
            maybe_speak_queued()

    def maybe_speak_queued() -> None:
        """Speak the next queued response when user is silent and assistant is idle.

        Gates on agent_state to avoid racing with the fast brain's reply pipeline.
        After the user stops speaking, the agent transitions through thinking →
        speaking → listening. We only speak queued text once the agent has settled
        back to a quiescent state, guaranteeing the fast brain's reply comes first.

        When dedup is enabled, the actual speak is async (LLM call to check for
        redundancy).  The ``_dedup_in_flight`` guard prevents a second item from
        being dispatched while the first is still being checked.
        """
        if _dedup_in_flight or not _queued_speech or not _is_pipeline_quiescent():
            return
        (
            text,
            notification_id,
            notification_source,
            notification_content,
            llm_log_path,
        ) = _queued_speech.pop(0)
        asyncio.ensure_future(
            _dedup_and_speak(
                text,
                notification_id,
                notification_source,
                notification_content,
                llm_log_path,
            ),
        )

    def on_notification(data: dict) -> None:
        """Handle notifications from conversation manager."""
        nonlocal assistant_screen_share_active, _agent_service_url
        payload = data.get("payload") or data
        content = payload.get("content", "")
        # Track screen share state from meet interaction notifications.
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
            elif "took remote control" in low:
                user_remote_control_active = True
            elif "released remote control" in low:
                user_remote_control_active = False

                async def _update_cache_after_remote_control():
                    entry = await capture_assistant_screenshot(
                        utterance="",
                        cached=False,
                        fb_logger=_log,
                        agent_service_url=_agent_service_url,
                        http_session=_screenshot_http_session,
                    )
                    if entry and assistant_screen_share_active:
                        _handle_screenshot(entry)

                if assistant_screen_share_active:
                    asyncio.create_task(_update_cache_after_remote_control())
        response_text = payload.get("response_text", "")
        should_speak = payload.get("should_speak", False)
        notification_source = payload.get("source", "")
        llm_log_path = payload.get("llm_log_path", "")
        notification_id = content_trace_id("guid", content)
        triggers_turn = notification_source not in (
            "meet_interaction",
            "proactive_speech",
        )
        _log.notification(
            notification_source,
            content,
            speak=should_speak,
            turn=triggers_turn,
        )
        touch_activity()

        if content:
            if not session_ready:
                pending_notifications.append(
                    (
                        content,
                        response_text,
                        should_speak,
                        notification_id,
                        notification_source,
                        llm_log_path,
                    ),
                )
                _log.notification_buffered(len(pending_notifications))
            else:
                apply_notification(
                    content,
                    response_text,
                    should_speak,
                    notification_id=notification_id,
                    source="socket_callback",
                    notification_source=notification_source,
                    llm_log_path=llm_log_path,
                )
                if triggers_turn:
                    _invalidate_current_generation(
                        "notification_during_generation",
                        notification_id,
                    )

    event_broker.register_callback("app:call:status", on_status)
    event_broker.register_callback("app:call:notification", on_notification)

    # --- Tier 1: Comms from call participants (all calls) ---
    is_boss_user = bool(contact.get("is_system", False))
    participant_ids: set[int] = set()
    if contact.get("contact_id") is not None:
        participant_ids.add(contact["contact_id"])

    def _inject_silent_context(msg: str) -> None:
        """Inject a system message into chat context as silent background."""
        assistant._chat_ctx.add_message(role="system", content=[msg])
        session._chat_ctx.add_message(role="system", content=[msg])

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
        _inject_silent_context(text)
        if text.startswith("[You "):
            if assistant.user_turn_generating:
                _invalidate_current_generation(
                    "outbound_action_during_generation",
                    "participant_comms",
                )
            else:
                trigger_generate_reply(
                    reason="outbound_message_acknowledgment",
                    source_id="participant_comms",
                )

    event_broker.register_callback("app:comms:*", on_participant_comms)

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

    # Mark session ready and process any buffered notifications BEFORE first utterance.
    # After this, the on_notification callback will apply notifications immediately.
    # Note: For outbound calls, llm_node will wait for call_received (set by on_status).
    session_ready = True
    if pending_notifications:
        _log.session_ready(
            f"Applying {len(pending_notifications)} buffered notification(s)",
        )
        for (
            content,
            response_text,
            should_speak,
            notification_id,
            notification_source,
            llm_log_path,
        ) in pending_notifications:
            apply_notification(
                content,
                response_text,
                should_speak,
                notification_id=notification_id,
                source="pending_buffer_flush",
                notification_source=notification_source,
                llm_log_path=llm_log_path,
            )
        pending_notifications.clear()

    # Pre-generate the opening greeting via a direct sidecar LLM call so that
    # the full LLM latency is absorbed before audio playback begins.
    # - Meet: hides the delay behind the "waiting for assistant" spinner, then
    #   signals "ready_to_speak" so the avatar appears right before speech.
    # - Phone (inbound): eliminates dead air after the call connects.
    # - Phone (outbound): waits for the callee to answer first, then generates.
    if outbound:
        _log.info("Outbound call — waiting for callee to answer…")
        await call_answered_flag.wait()
        _log.call_status("call_answered — generating greeting")

    from unity.common.llm_client import new_llm_client

    greeting_client = new_llm_client(
        model=SETTINGS.conversation.FAST_BRAIN_MODEL,
        origin="fast_brain_greeting",
        reasoning_effort="low",
    )
    greeting_messages = [
        {"role": "system", "content": system_prompt},
        *_extract_chat_messages(session._chat_ctx),
    ]
    greeting_text = await greeting_client.generate(messages=greeting_messages)

    if channel != "phone":
        await ctx.room.local_participant.publish_data(
            json.dumps({"type": "ready_to_speak"}).encode(),
            topic="agent_status",
            reliable=True,
        )

    session.say(
        greeting_text,
        allow_interruptions=True,
        add_to_chat_ctx=True,
    )

    # Inject the initializing-state system message *after* the greeting has
    # been generated and spoken.  Placing it before the greeting caused the
    # LLM to proactively mention "still setting up" in the opening line,
    # which sounds odd when no action has been requested yet.  The note only
    # matters for subsequent turns where the user might ask for something
    # that requires initialized managers.
    if not os.environ.get("UNITY_CM_INITIALIZED"):
        _init_note = (
            "[system] You have just started up and your systems are still "
            "syncing — loading files, pulling up previous conversations, "
            "and connecting to your tools. This takes a few moments. "
            "If the user asks you to do something that requires looking "
            "things up or taking action, let them know naturally that "
            "you are still getting set up (e.g. 'I'm just pulling up our "
            "previous sessions — give me a moment and I'll get right on "
            "that'). Do NOT say 'I can't do that' — frame it as a brief "
            "delay, not a limitation. You will receive a notification "
            "when everything is ready."
        )
        assistant._chat_ctx.add_message(role="system", content=[_init_note])
        session._chat_ctx.add_message(role="system", content=[_init_note])
        _log.info("Injected initializing-state system message (CM not yet initialized)")


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
