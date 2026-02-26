#!/usr/bin/env python3
"""Post-hoc call transcript builder for ConversationManager sandbox runs.

Parses .logs_voice_agent.txt (and optionally .logs_conversation_sandbox.txt)
to produce a deterministic, source-traced call transcript with anomaly detection.

Every assistant utterance is traced to its source path:
  - Path 1: fast_brain (generate_reply)
  - Path 2: proactive_speech
  - Path 3: slow_brain / actor_notification

Usage:
    .venv/bin/python sandboxes/conversation_manager/call_transcript.py .logs_voice_agent.txt
    .venv/bin/python sandboxes/conversation_manager/call_transcript.py .logs_voice_agent.txt --cm-log .logs_conversation_sandbox.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Utterance:
    """A spoken or received utterance in the call (from conversation_item_added)."""

    monotonic_ms: int
    ts_utc: str
    role: str
    text: str
    utterance_id: str
    speech_source: str
    guidance_id: str


@dataclass
class GuidanceReceived:
    """A guidance event received by the voice agent."""

    monotonic_ms: int
    ts_utc: str
    guidance_id: str
    source: str
    should_speak: bool
    user_is_speaking: bool
    content: str


@dataclass
class SessionSay:
    """A session.say() call — queued speech from guidance."""

    monotonic_ms: int
    ts_utc: str
    guidance_id: str
    source: str
    text: str


@dataclass
class FastBrainRequest:
    """A fast brain LLM request-response cycle."""

    request_id: str
    start_ms: int
    start_utc: str
    end_ms: int | None = None
    end_utc: str | None = None
    trigger_json: str = ""
    generation_id: str = ""
    chunk_count: int | None = None


@dataclass
class GenerateReplyTrigger:
    """A generate_reply trigger event."""

    monotonic_ms: int
    ts_utc: str
    generation_id: str
    reason: str
    source_id: str
    queued_speech_count: int


@dataclass
class UserStateChange:
    """A user voice state transition."""

    monotonic_ms: int
    ts_utc: str
    state_id: str
    new_state: str


@dataclass
class Anomaly:
    """A detected issue in the call timeline."""

    anomaly_type: str
    severity: str  # "error", "warning", "info"
    description: str
    related_ids: list[str] = field(default_factory=list)
    utterance_index: int | None = None


@dataclass
class SlowBrainRun:
    """A slow-brain LLM run with its reasoning and decision."""

    ts_utc: str
    run_id: str
    request_id: str
    origin_event: str
    origin_event_id: str
    thoughts: str
    action: str
    dispatch_utc: str = ""


@dataclass
class ActorNotification:
    """An actor event delivered to the fast brain (Action started/update/completed)."""

    ts_utc: str
    notification_type: str  # "started", "update", "completed"
    content: str
    generation_id: str = ""


@dataclass
class VoiceLogData:
    """All parsed data from the voice agent log."""

    utterances: list[Utterance] = field(default_factory=list)
    guidance_events: list[GuidanceReceived] = field(default_factory=list)
    session_says: list[SessionSay] = field(default_factory=list)
    fb_requests: list[FastBrainRequest] = field(default_factory=list)
    fb_triggers: list[GenerateReplyTrigger] = field(default_factory=list)
    user_states: list[UserStateChange] = field(default_factory=list)
    actor_notifications: list[ActorNotification] = field(default_factory=list)


@dataclass
class CMGuidanceDecision:
    """A guidance articulation decision from the CM log."""

    guidance_id: str
    run_id: str
    send: bool
    speak: bool


@dataclass
class SteeringContent:
    """Content of a steering action (interject/ask) sent to the actor."""

    ts_utc: str
    kind: str  # "interject", "ask"
    content: str


@dataclass
class CMLogData:
    """Parsed data from the CM log (optional cross-reference)."""

    decisions: list[CMGuidanceDecision] = field(default_factory=list)
    blocked_ids: list[str] = field(default_factory=list)
    slow_brain_runs: list[SlowBrainRun] = field(default_factory=list)
    steering_contents: list[SteeringContent] = field(default_factory=list)


@dataclass
class TimelineEntry:
    """A single entry in the final timeline (utterance + metadata)."""

    index: int
    utterance: Utterance
    gap_before_s: float | None = None
    guidance_chain: GuidanceReceived | None = None
    session_say: SessionSay | None = None
    fb_trigger: GenerateReplyTrigger | None = None
    fb_request: FastBrainRequest | None = None
    anomalies: list[Anomaly] = field(default_factory=list)


@dataclass
class Timeline:
    """The complete call timeline."""

    entries: list[TimelineEntry]
    slow_brain_runs: list[SlowBrainRun]
    steering_contents: list[SteeringContent]
    actor_notifications: list[ActorNotification]
    silent_guidance: list[GuidanceReceived]
    anomalies: list[Anomaly]
    call_start_ms: int
    call_end_ms: int
    call_start_utc: str
    call_end_utc: str


# ─────────────────────────────────────────────────────────────────────────────
# Trace Line Parsing
# ─────────────────────────────────────────────────────────────────────────────


import re
import hashlib

_FB_RE = re.compile(
    r"\[FastBrain[^\]]*\]\s+(.*)",
)

_USER_SPEECH_ICON = "\U0001f9d1\u200d\U0001f4bb"  # 🧑‍💻
_ASSISTANT_SPEECH_ICON = "\U0001f50a"  # 🔊

_TS_MILLIS_RE = re.compile(r"^(\d{2}:\d{2}:\d{2}\.\d{3})\s")
_TS_FULL_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})[,.](\d{3})")


def _extract_log_date(lines: list[str]) -> str:
    """Extract YYYY-MM-DD from the first standard-format log line."""
    for raw in lines[:100]:
        m = _TS_FULL_RE.match(raw)
        if m:
            return m.group(1)
    return "2026-01-01"


def _parse_fb_line(
    line: str,
    log_date: str,
    line_idx: int,
) -> tuple[str, int, str] | None:
    """Parse a FastBrainLogger line into (ts_utc, monotonic_proxy, body).

    Returns None if the line is not a FastBrainLogger line.
    """
    m = _FB_RE.search(line)
    if not m:
        return None
    body = m.group(1)

    ts_utc = ""
    tm = _TS_MILLIS_RE.match(line)
    if tm:
        ts_utc = f"{log_date}T{tm.group(1)}+00:00"
    else:
        tfm = _TS_FULL_RE.match(line)
        if tfm:
            ts_utc = f"{tfm.group(1)}T{tfm.group(2)}.{tfm.group(3)}+00:00"

    return ts_utc, line_idx * 10, body


def _content_hash(prefix: str, content: str) -> str:
    digest = hashlib.sha1((content or "").encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:12]}"


def _extract_field(line: str, key: str, next_key: str | None = None) -> str:
    """Extract value for `key=...` bounded by ` next_key=` or end of line."""
    marker = f"{key}="
    idx = line.find(marker)
    if idx == -1:
        return ""
    start = idx + len(marker)
    if next_key:
        end_marker = f" {next_key}="
        end = line.find(end_marker, start)
        if end != -1:
            return line[start:end]
    return line[start:].rstrip()


def _utc_to_ms(ts: str) -> int:
    """Convert a UTC timestamp like '2026-01-01T11:24:34.275+00:00' to ms since midnight."""
    if "T" not in ts:
        return 0
    time_part = ts.split("T")[1].split("+")[0].split("-")[0]
    parts = time_part.split(":")
    h, m = int(parts[0]), int(parts[1])
    s_parts = parts[2].split(".")
    s = int(s_parts[0])
    ms = int(s_parts[1][:3]) if len(s_parts) > 1 else 0
    return ((h * 3600) + (m * 60) + s) * 1000 + ms


def _extract_bool(line: str, key: str, next_key: str | None = None) -> bool:
    return _extract_field(line, key, next_key).lower() == "true"


def _extract_int(line: str, key: str, next_key: str | None = None) -> int:
    try:
        return int(_extract_field(line, key, next_key))
    except (ValueError, TypeError):
        return 0


def _find_next_generation_id(
    lines: list[str],
    start_idx: int,
    log_date: str,
) -> str:
    """Find the generation_id from the LLM thinking line that follows an actor notification."""
    for j in range(start_idx + 1, min(start_idx + 3, len(lines))):
        if "LLM thinking" in lines[j] and "generation_id=" in lines[j]:
            return _extract_field(lines[j], "generation_id", "source_id")
    return ""


def _collect_full_text(lines: list[str], trace_line_idx: int, role: str) -> str:
    """Collect full utterance text from content lines following a trace event.

    After a conversation_item_added trace, the next line starts with
    '{role} <text>' and may continue across multiple lines until a line
    starting with '[' or a timestamp log line.
    """
    prefix = f"{role} "
    text_lines: list[str] = []
    collecting = False

    for i in range(trace_line_idx + 1, min(trace_line_idx + 50, len(lines))):
        raw = lines[i]
        if raw.startswith("[") or (raw[:4].isdigit() and " - " in raw[:30]):
            if collecting:
                break
            continue
        if not collecting and raw.startswith(prefix):
            text_lines.append(raw[len(prefix) :])
            collecting = True
        elif collecting:
            text_lines.append(raw)

    return "\n".join(text_lines).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Voice Log Parser
# ─────────────────────────────────────────────────────────────────────────────


def parse_voice_log(path: Path) -> VoiceLogData:
    """Parse all relevant trace events from the voice agent log.

    Supports both the legacy ``[TRACE::FAST_BRAIN_CALL]`` format and the
    current ``FastBrainLogger`` emoji format introduced in c9e3c8f4a.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    data = VoiceLogData()
    log_date = _extract_log_date(lines)

    for i, line in enumerate(lines):

        # ══════════════════════════════════════════════════════════════════
        # Legacy [TRACE::...] format (pre-c9e3c8f4a)
        # ══════════════════════════════════════════════════════════════════

        # ── conversation_item_added ──
        if "event=conversation_item_added" in line and "[TRACE::" in line:
            role = _extract_field(line, "role", "utterance_id")
            utterance_id = _extract_field(line, "utterance_id", "text_preview")
            speech_source = _extract_field(line, "speech_source", "guidance_id")
            guidance_id = _extract_field(line, "guidance_id")
            full_text = _collect_full_text(lines, i, role)
            if not full_text:
                full_text = _extract_field(
                    line,
                    "text_preview",
                    "speech_source",
                ).replace("\\n", "\n")

            data.utterances.append(
                Utterance(
                    monotonic_ms=_extract_int(line, "monotonic_ms", "role"),
                    ts_utc=_extract_field(line, "ts_utc", "monotonic_ms"),
                    role=role,
                    text=full_text,
                    utterance_id=utterance_id,
                    speech_source=speech_source,
                    guidance_id=guidance_id,
                ),
            )

        # ── guidance_received ──
        elif "event=guidance_received" in line and "[TRACE::" in line:
            data.guidance_events.append(
                GuidanceReceived(
                    monotonic_ms=_extract_int(line, "monotonic_ms", "guidance_id"),
                    ts_utc=_extract_field(line, "ts_utc", "monotonic_ms"),
                    guidance_id=_extract_field(line, "guidance_id", "guidance_source"),
                    source=_extract_field(line, "guidance_source", "session_ready"),
                    should_speak=_extract_bool(
                        line,
                        "should_speak",
                        "user_is_speaking",
                    ),
                    user_is_speaking=_extract_bool(
                        line,
                        "user_is_speaking",
                        "content_preview",
                    ),
                    content=_extract_field(line, "content_preview"),
                ),
            )

        # ── session_say ──
        elif "event=session_say" in line and "[TRACE::" in line:
            data.session_says.append(
                SessionSay(
                    monotonic_ms=_extract_int(line, "monotonic_ms", "guidance_id"),
                    ts_utc=_extract_field(line, "ts_utc", "monotonic_ms"),
                    guidance_id=_extract_field(line, "guidance_id", "guidance_source"),
                    source=_extract_field(line, "guidance_source", "text_preview"),
                    text=_extract_field(line, "text_preview"),
                ),
            )

        # ── generate_reply_trigger ──
        elif "event=generate_reply_trigger" in line and "[TRACE::" in line:
            data.fb_triggers.append(
                GenerateReplyTrigger(
                    monotonic_ms=_extract_int(line, "monotonic_ms", "generation_id"),
                    ts_utc=_extract_field(line, "ts_utc", "monotonic_ms"),
                    generation_id=_extract_field(line, "generation_id", "reason"),
                    reason=_extract_field(line, "reason", "source_id"),
                    source_id=_extract_field(line, "source_id", "queued_speech_count"),
                    queued_speech_count=_extract_int(
                        line,
                        "queued_speech_count",
                        "user_is_speaking",
                    ),
                ),
            )

        # ── FAST_BRAIN_REQUEST_START ──
        elif "[TRACE::FAST_BRAIN_REQUEST_START]" in line:
            data.fb_requests.append(
                FastBrainRequest(
                    request_id=_extract_field(line, "request_id", "model"),
                    start_ms=_extract_int(line, "monotonic_ms"),
                    start_utc=_extract_field(line, "ts_utc", "monotonic_ms"),
                    trigger_json=_extract_field(line, "trigger", "ts_utc"),
                ),
            )

        # ── FAST_BRAIN_REQUEST_END ──
        elif "[TRACE::FAST_BRAIN_REQUEST_END]" in line:
            req_id = _extract_field(line, "request_id", "chunk_count")
            end_ms = _extract_int(line, "monotonic_ms")
            end_utc = _extract_field(line, "ts_utc", "monotonic_ms")
            chunk_count = _extract_int(line, "chunk_count", "trigger_id")
            for req in reversed(data.fb_requests):
                if req.request_id == req_id:
                    req.end_ms = end_ms
                    req.end_utc = end_utc
                    req.chunk_count = chunk_count
                    break

        # ── user_state_changed ──
        elif "event=user_state_changed" in line and "[TRACE::" in line:
            data.user_states.append(
                UserStateChange(
                    monotonic_ms=_extract_int(line, "monotonic_ms", "state_id"),
                    ts_utc=_extract_field(line, "ts_utc", "monotonic_ms"),
                    state_id=_extract_field(line, "state_id", "new_state"),
                    new_state=_extract_field(line, "new_state", "user_is_speaking"),
                ),
            )

        # ══════════════════════════════════════════════════════════════════
        # FastBrainLogger emoji format (post-c9e3c8f4a)
        # ══════════════════════════════════════════════════════════════════

        elif "[FastBrain" in line:
            fb = _parse_fb_line(line, log_date, i)
            if fb is None:
                continue
            ts_utc, mono, body = fb

            if body.startswith("User state:"):
                new_state = body.split(":", 1)[1].strip().split()[0]
                data.user_states.append(
                    UserStateChange(
                        monotonic_ms=mono,
                        ts_utc=ts_utc,
                        state_id=_extract_field(line, "state_id"),
                        new_state=new_state,
                    ),
                )

            elif body.startswith("Guidance from "):
                source = body.split("Guidance from ", 1)[1].split(":")[0]
                should_speak = "speak=True" in body
                content = (
                    body.split("speak=True ", 1)[-1]
                    if should_speak
                    else body.split("speak=False ", 1)[-1]
                )
                gid = _extract_field(line, "guidance_id")
                if " guidance_id=" in content:
                    content = content[: content.index(" guidance_id=")]
                data.guidance_events.append(
                    GuidanceReceived(
                        monotonic_ms=mono,
                        ts_utc=ts_utc,
                        guidance_id=gid or _content_hash("guid", content),
                        source=source,
                        should_speak=should_speak,
                        user_is_speaking=False,
                        content=content.strip(),
                    ),
                )

            elif body.startswith("Speaking guidance "):
                parts = body.split("Speaking guidance ", 1)[1]
                gid = parts.split(":")[0].strip()
                text = parts.split(":", 1)[1].strip() if ":" in parts else ""
                gsource = _extract_field(line, "guidance_source")
                if " guidance_source=" in text:
                    text = text[: text.index(" guidance_source=")]
                data.session_says.append(
                    SessionSay(
                        monotonic_ms=mono,
                        ts_utc=ts_utc,
                        guidance_id=gid,
                        source=gsource,
                        text=text,
                    ),
                )

            elif body.startswith("LLM thinking"):
                data.fb_triggers.append(
                    GenerateReplyTrigger(
                        monotonic_ms=mono,
                        ts_utc=ts_utc,
                        generation_id=_extract_field(
                            line,
                            "generation_id",
                            "source_id",
                        ),
                        reason=_extract_field(line, "reason", "generation_id"),
                        source_id=_extract_field(
                            line,
                            "source_id",
                            "queued_speech",
                        ),
                        queued_speech_count=_extract_int(
                            line,
                            "queued_speech",
                        ),
                    ),
                )

            elif body.startswith("Action started:"):
                content = body[len("Action started:") :].strip()
                gen_id = _find_next_generation_id(lines, i, log_date)
                data.actor_notifications.append(
                    ActorNotification(
                        ts_utc=ts_utc,
                        notification_type="started",
                        content=content,
                        generation_id=gen_id,
                    ),
                )

            elif body.startswith("Action update:"):
                content = body[len("Action update:") :].strip()
                gen_id = _find_next_generation_id(lines, i, log_date)
                data.actor_notifications.append(
                    ActorNotification(
                        ts_utc=ts_utc,
                        notification_type="update",
                        content=content,
                        generation_id=gen_id,
                    ),
                )

            elif body.startswith("Action completed"):
                content = body.split(":", 1)[1].strip() if ":" in body else body
                gen_id = _find_next_generation_id(lines, i, log_date)
                data.actor_notifications.append(
                    ActorNotification(
                        ts_utc=ts_utc,
                        notification_type="completed",
                        content=content,
                        generation_id=gen_id,
                    ),
                )

            elif _ASSISTANT_SPEECH_ICON in line:
                source = "generate_reply"
                gid = ""
                if " source=" in body:
                    source = _extract_field(line, "source", "guidance_id")
                    gid = _extract_field(line, "guidance_id")
                    text = body.split(" source=")[0].strip()
                else:
                    text_parts = [body.strip()]
                    for j in range(i + 1, min(i + 10, len(lines))):
                        nxt = lines[j]
                        if " source=" in nxt:
                            pre = nxt[: nxt.index(" source=")].strip()
                            if pre:
                                text_parts.append(pre)
                            raw_src = nxt[nxt.index("source=") + 7 :]
                            source = raw_src.split()[0]
                            if "guidance_id=" in nxt:
                                gid = _extract_field(nxt, "guidance_id")
                            break
                        if (
                            nxt.strip()
                            and not _TS_MILLIS_RE.match(nxt)
                            and not _TS_FULL_RE.match(nxt)
                        ):
                            text_parts.append(nxt.strip())
                    text = "\n".join(text_parts)
                if text.endswith("\u2026"):
                    text = text[:-1]
                if text:
                    data.utterances.append(
                        Utterance(
                            monotonic_ms=mono,
                            ts_utc=ts_utc,
                            role="assistant",
                            text=text,
                            utterance_id=_content_hash(
                                "utt",
                                f"assistant:{text}",
                            ),
                            speech_source=source,
                            guidance_id=gid,
                        ),
                    )

            elif _USER_SPEECH_ICON in line:
                text = body.strip()
                if text.endswith("\u2026"):
                    text = text[:-1]
                if text:
                    data.utterances.append(
                        Utterance(
                            monotonic_ms=mono,
                            ts_utc=ts_utc,
                            role="user",
                            text=text,
                            utterance_id=_content_hash("utt", f"user:{text}"),
                            speech_source="generate_reply",
                            guidance_id="",
                        ),
                    )

    data.utterances.sort(key=lambda u: u.monotonic_ms)
    data.guidance_events.sort(key=lambda g: g.monotonic_ms)
    data.session_says.sort(key=lambda s: s.monotonic_ms)
    data.fb_requests.sort(key=lambda r: r.start_ms)
    data.fb_triggers.sort(key=lambda t: t.monotonic_ms)
    data.user_states.sort(key=lambda u: u.monotonic_ms)
    data.actor_notifications.sort(key=lambda a: a.ts_utc)

    seen_utt: set[str] = set()
    deduped: list[Utterance] = []
    for u in data.utterances:
        if u.utterance_id not in seen_utt:
            seen_utt.add(u.utterance_id)
            deduped.append(u)
    data.utterances = deduped

    return data


# ─────────────────────────────────────────────────────────────────────────────
# CM Log Parser (optional cross-referencing)
# ─────────────────────────────────────────────────────────────────────────────


_CM_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})\s")


def _parse_cm_ts(line: str) -> str:
    """Extract UTC timestamp from a CM log line like '2026-02-26 11:24:34 [DEBUG]...'"""
    m = _TS_FULL_RE.match(line)
    if m:
        return f"{m.group(1)}T{m.group(2)}.{m.group(3)}+00:00"
    m2 = _CM_TS_RE.match(line)
    if m2:
        return f"{m2.group(1)}T{m2.group(2)}.000+00:00"
    return ""


def parse_cm_log(path: Path) -> CMLogData:
    """Parse slow-brain decisions, guidance decisions, and actor events from the CM log."""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    data = CMLogData()

    run_meta: dict[str, dict[str, str]] = {}

    for line in lines:
        # Slow-brain dispatch: captures origin metadata per run_id
        if "Dispatching slow-brain" in line and "run_id=" in line:
            run_id = _extract_field(line, "run_id", "request_id")
            run_meta[run_id] = {
                "request_id": _extract_field(line, "request_id", "origin_event_id"),
                "origin_event_id": _extract_field(
                    line,
                    "origin_event_id",
                    "origin_event",
                ),
                "origin_event": _extract_field(
                    line,
                    "origin_event",
                    "dropped_requests",
                ),
                "dispatch_utc": _parse_cm_ts(line),
            }

        # Slow-brain thought + action decision
        # Format: run_id=llmrun-NNNNN thoughts: <text> | action: <action>
        # Can't use _extract_field for run_id because "thoughts:" uses colon, not equals.
        elif "thoughts:" in line and "| action:" in line and "run_id=" in line:
            run_id_raw = line.split("run_id=", 1)[1]
            run_id = run_id_raw.split()[0]
            raw = line.split("thoughts:", 1)[1]
            parts = raw.rsplit("| action:", 1)
            thoughts = parts[0].strip()
            action = parts[1].strip() if len(parts) > 1 else ""
            ts = _parse_cm_ts(line)
            meta = run_meta.get(run_id, {})
            data.slow_brain_runs.append(
                SlowBrainRun(
                    ts_utc=ts,
                    run_id=run_id,
                    request_id=meta.get("request_id", ""),
                    origin_event=meta.get("origin_event", ""),
                    origin_event_id=meta.get("origin_event_id", ""),
                    thoughts=thoughts,
                    action=action,
                    dispatch_utc=meta.get("dispatch_utc", ""),
                ),
            )

        # Steering action content (interject/ask sent to actor)
        elif "Interject requested:" in line and "CodeActActor" in line:
            content = line.split("Interject requested:", 1)[1].strip()
            data.steering_contents.append(
                SteeringContent(
                    ts_utc=_parse_cm_ts(line),
                    kind="interject",
                    content=content,
                ),
            )

        elif "Ask requested:" in line and "CodeActActor" in line:
            content = line.split("Ask requested:", 1)[1].strip()
            data.steering_contents.append(
                SteeringContent(
                    ts_utc=_parse_cm_ts(line),
                    kind="ask",
                    content=content,
                ),
            )

        # Guidance decisions (existing)
        elif "Decision " in line and "guidance_id=" in line:
            gid = _extract_field(line, "guidance_id", "send")
            rid = _extract_field(line, "run_id", "guidance_id")
            send = _extract_bool(line, "send", "speak")
            speak = _extract_bool(line, "speak")
            if gid:
                data.decisions.append(
                    CMGuidanceDecision(
                        guidance_id=gid,
                        run_id=rid,
                        send=send,
                        speak=speak,
                    ),
                )

        elif "Blocked guidance" in line and "guidance_id=" in line:
            gid = _extract_field(line, "guidance_id")
            if gid:
                data.blocked_ids.append(gid)

    return data


# ─────────────────────────────────────────────────────────────────────────────
# Index Builders
# ─────────────────────────────────────────────────────────────────────────────


def _build_guidance_index(
    data: VoiceLogData,
) -> dict[str, GuidanceReceived]:
    return {g.guidance_id: g for g in data.guidance_events}


def _build_session_say_index(data: VoiceLogData) -> dict[str, SessionSay]:
    idx: dict[str, SessionSay] = {}
    for s in data.session_says:
        idx[s.guidance_id] = s
    return idx


def _find_fb_request_for_utterance(
    utt: Utterance,
    requests: list[FastBrainRequest],
) -> FastBrainRequest | None:
    """Find the fast brain request whose response window contains this utterance.

    The utterance's monotonic_ms should fall between the request's end_ms
    and end_ms + reasonable TTS playout window (~30s).
    """
    best: FastBrainRequest | None = None
    best_gap = float("inf")
    for req in requests:
        if req.end_ms is None:
            continue
        gap = utt.monotonic_ms - req.end_ms
        if 0 <= gap < 30_000 and gap < best_gap:
            best = req
            best_gap = gap
    return best


def _find_fb_trigger_for_request(
    req: FastBrainRequest,
    triggers: list[GenerateReplyTrigger],
) -> GenerateReplyTrigger | None:
    best: GenerateReplyTrigger | None = None
    best_gap = float("inf")
    for t in triggers:
        gap = req.start_ms - t.monotonic_ms
        if 0 <= gap < 5_000 and gap < best_gap:
            best = t
            best_gap = gap
    return best


def _find_fb_trigger_for_utterance(
    utt: Utterance,
    triggers: list[GenerateReplyTrigger],
    *,
    exclude: set[str] | None = None,
) -> GenerateReplyTrigger | None:
    """Direct trigger-to-utterance linking via UTC timestamps.

    Used as a fallback when no FastBrainRequest intermediary exists (emoji log format).
    Finds the closest preceding trigger within 30s whose generation_id hasn't been consumed.
    """
    utt_ms = _utc_to_ms(utt.ts_utc)
    best: GenerateReplyTrigger | None = None
    best_gap = float("inf")
    for t in triggers:
        if exclude and t.generation_id in exclude:
            continue
        t_ms = _utc_to_ms(t.ts_utc)
        gap = utt_ms - t_ms
        if 0 < gap < 45_000 and gap < best_gap:
            best = t
            best_gap = gap
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Timeline Builder
# ─────────────────────────────────────────────────────────────────────────────


def build_timeline(
    data: VoiceLogData,
    cm_data: CMLogData | None = None,
) -> Timeline:
    """Assemble the ordered timeline with gap computation and source linking."""
    guidance_idx = _build_guidance_index(data)
    say_idx = _build_session_say_index(data)

    entries: list[TimelineEntry] = []
    prev_utc_ms: int | None = None
    used_trigger_ids: set[str] = set()

    for seq, utt in enumerate(data.utterances, start=1):
        gap_s: float | None = None
        utt_utc_ms = _utc_to_ms(utt.ts_utc)
        if prev_utc_ms is not None and utt_utc_ms > 0 and prev_utc_ms > 0:
            gap_s = (utt_utc_ms - prev_utc_ms) / 1000.0

        guidance_chain = guidance_idx.get(utt.guidance_id) if utt.guidance_id else None
        session_say = say_idx.get(utt.guidance_id) if utt.guidance_id else None

        fb_request: FastBrainRequest | None = None
        fb_trigger: GenerateReplyTrigger | None = None
        if utt.speech_source == "generate_reply" and utt.role == "assistant":
            fb_request = _find_fb_request_for_utterance(utt, data.fb_requests)
            if fb_request:
                fb_trigger = _find_fb_trigger_for_request(fb_request, data.fb_triggers)
            if not fb_trigger:
                fb_trigger = _find_fb_trigger_for_utterance(
                    utt,
                    data.fb_triggers,
                    exclude=used_trigger_ids,
                )
        if fb_trigger and fb_trigger.generation_id:
            used_trigger_ids.add(fb_trigger.generation_id)

        entries.append(
            TimelineEntry(
                index=seq,
                utterance=utt,
                gap_before_s=gap_s,
                guidance_chain=guidance_chain,
                session_say=session_say,
                fb_trigger=fb_trigger,
                fb_request=fb_request,
            ),
        )
        if utt_utc_ms > 0:
            prev_utc_ms = utt_utc_ms

    spoken_gids = {u.guidance_id for u in data.utterances if u.guidance_id}
    silent_guidance = [
        g
        for g in data.guidance_events
        if g.guidance_id not in spoken_gids and not g.should_speak
    ]

    call_start_utc = data.utterances[0].ts_utc if data.utterances else ""
    call_end_utc = data.utterances[-1].ts_utc if data.utterances else ""
    call_start_ms = _utc_to_ms(call_start_utc)
    call_end_ms = _utc_to_ms(call_end_utc)

    anomalies = detect_anomalies(entries, data)

    slow_brain_runs = cm_data.slow_brain_runs if cm_data else []
    steering_contents = cm_data.steering_contents if cm_data else []
    actor_notifications = list(data.actor_notifications)

    return Timeline(
        entries=entries,
        slow_brain_runs=slow_brain_runs,
        steering_contents=steering_contents,
        actor_notifications=actor_notifications,
        silent_guidance=silent_guidance,
        anomalies=anomalies,
        call_start_ms=call_start_ms,
        call_end_ms=call_end_ms,
        call_start_utc=call_start_utc,
        call_end_utc=call_end_utc,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Anomaly Detection
# ─────────────────────────────────────────────────────────────────────────────


def _has_user_turn_between(
    entries: list[TimelineEntry],
    idx_a: int,
    idx_b: int,
) -> bool:
    """Check if any user utterance exists in the full timeline between two indices."""
    for e in entries:
        if e.index > idx_a and e.index < idx_b and e.utterance.role == "user":
            return True
    return False


def detect_anomalies(
    entries: list[TimelineEntry],
    data: VoiceLogData,
) -> list[Anomaly]:
    anomalies: list[Anomaly] = []
    guidance_idx = _build_guidance_index(data)
    say_idx = _build_session_say_index(data)

    assistant_entries = [e for e in entries if e.utterance.role == "assistant"]

    for i, entry in enumerate(assistant_entries):
        utt = entry.utterance

        # ── Stale proactive speech ──
        # Proactive speech that plays AFTER a slow_brain result on the SAME topic.
        # Excluded when user messages intervened (indicating a new topic).
        if utt.speech_source == "proactive_speech" and i > 0:
            prev_asst = assistant_entries[i - 1]
            if prev_asst.utterance.speech_source in (
                "slow_brain",
                "actor_notification",
            ):
                if not _has_user_turn_between(
                    entries,
                    prev_asst.index,
                    entry.index,
                ):
                    a = Anomaly(
                        anomaly_type="STALE_PROACTIVE",
                        severity="warning",
                        description=(
                            f"Proactive speech #{entry.index} played at "
                            f"{_short_utc(utt.ts_utc)} AFTER "
                            f"{prev_asst.utterance.speech_source} result "
                            f"#{prev_asst.index} at "
                            f"{_short_utc(prev_asst.utterance.ts_utc)} with no "
                            f"user turn in between. Proactive text may "
                            f"contradict the delivered result."
                        ),
                        related_ids=[utt.guidance_id, prev_asst.utterance.utterance_id],
                        utterance_index=entry.index,
                    )
                    anomalies.append(a)
                    entry.anomalies.append(a)

        # ── Tag mismatch ──
        # session_say text differs from conversation_item_added text for the
        # same guidance_id — indicates the guidance was interrupted or the
        # _last_say_meta poisoned the tag on a different utterance.
        if utt.guidance_id:
            say = say_idx.get(utt.guidance_id)
            if say and say.text and utt.text:
                say_norm = say.text[:80].strip()
                utt_norm = utt.text[:80].strip()
                if say_norm != utt_norm:
                    a = Anomaly(
                        anomaly_type="TAG_MISMATCH",
                        severity="warning",
                        description=(
                            f"Utterance #{entry.index} has guidance_id="
                            f"{utt.guidance_id} but session_say text "
                            f'"{say.text[:60]}" differs from spoken text '
                            f'"{utt.text[:60]}". Possible _last_say_meta '
                            f"collision or TTS interruption."
                        ),
                        related_ids=[utt.utterance_id, utt.guidance_id],
                        utterance_index=entry.index,
                    )
                    anomalies.append(a)
                    entry.anomalies.append(a)

    # ── Excessive consecutive filler ──
    consecutive_proactive = 0
    streak_start_idx: int | None = None
    for entry in entries:
        if (
            entry.utterance.role == "assistant"
            and entry.utterance.speech_source == "proactive_speech"
        ):
            consecutive_proactive += 1
            if streak_start_idx is None:
                streak_start_idx = entry.index
        else:
            if consecutive_proactive >= 3:
                anomalies.append(
                    Anomaly(
                        anomaly_type="EXCESSIVE_FILLER",
                        severity="info",
                        description=(
                            f"{consecutive_proactive} consecutive proactive speeches "
                            f"starting at #{streak_start_idx} without substantive content."
                        ),
                    ),
                )
            consecutive_proactive = 0
            streak_start_idx = None

    if consecutive_proactive >= 3:
        anomalies.append(
            Anomaly(
                anomaly_type="EXCESSIVE_FILLER",
                severity="info",
                description=(
                    f"{consecutive_proactive} consecutive proactive speeches "
                    f"starting at #{streak_start_idx} (end of call)."
                ),
            ),
        )

    # ── Guidance received while user speaking (queued) ──
    for g in data.guidance_events:
        if g.should_speak and g.user_is_speaking:
            anomalies.append(
                Anomaly(
                    anomaly_type="GUIDANCE_WHILE_SPEAKING",
                    severity="info",
                    description=(
                        f"Guidance {g.guidance_id} ({g.source}) received while "
                        f"user was speaking. Speech was queued."
                    ),
                    related_ids=[g.guidance_id],
                ),
            )

    return anomalies


# ─────────────────────────────────────────────────────────────────────────────
# Output Formatting
# ─────────────────────────────────────────────────────────────────────────────

_SEP = "\u2500" * 72  # ────────
_DSEP = "\u2550" * 72  # ════════

SOURCE_LABELS = {
    "generate_reply": "fast_brain",
    "proactive_speech": "proactive_speech",
    "slow_brain": "slow_brain",
    "actor_notification": "actor_notification",
    "initial_call": "initial_call",
    "meet_interaction": "meet_interaction",
}


def _time_sort_key(ts_utc: str) -> str:
    """Extract HH:MM:SS.mmm from a UTC timestamp for chronological sorting."""
    if "T" in ts_utc:
        time_part = ts_utc.split("T")[1]
        return time_part[:12]
    return ts_utc[:12]


_ACTION_LABELS = {
    "wait": "wait",
    "act": "act",
}


def _format_slow_brain(
    sb: SlowBrainRun,
    *,
    verbose: bool = False,
    notification_index: dict[str, ActorNotification] | None = None,
    steering_index: dict[str, SteeringContent] | None = None,
) -> list[str]:
    """Format a slow-brain run as timeline lines."""
    is_wait = sb.action == "wait"
    action_label = _ACTION_LABELS.get(sb.action, sb.action)
    parts: list[str] = []

    # Cross-reference: show the notification content that triggered this run
    origin_content = ""
    if notification_index and sb.origin_event in (
        "ActorSessionResponse",
        "ActorHandleStarted",
        "ActorHandleResponse",
        "ActorResult",
    ):
        an = notification_index.get(sb.origin_event_id)
        if an:
            origin_content = an.content

    if is_wait:
        thought_preview = ""
        if sb.thoughts:
            preview = sb.thoughts[:100]
            if len(sb.thoughts) > 100:
                preview += "..."
            thought_preview = f"  \u2014 {preview}"
        parts.append(
            f"  \u00b7 {_short_utc(sb.ts_utc)} "
            f"SLOW BRAIN [{sb.run_id}] "
            f"\u2190 {sb.origin_event} "
            f"\u2192 wait{thought_preview}",
        )
    else:
        parts.append(
            f"  \u25c6 {_short_utc(sb.ts_utc)} "
            f"SLOW BRAIN [{sb.run_id}] "
            f"\u2190 {sb.origin_event}",
        )
        if sb.thoughts:
            text = sb.thoughts
            if not verbose and len(text) > 200:
                text = text[:197] + "..."
            parts.append(f"    \U0001f4ad {text}")
        parts.append(f"    \u2b95 {action_label}")

        # Show the steering message sent to the actor
        if steering_index:
            sc = steering_index.get(sb.run_id)
            if sc:
                preview = sc.content
                if not verbose and len(preview) > 200:
                    preview = preview[:197] + "..."
                label = "\U0001f4ac" if sc.kind == "interject" else "\u2753"
                parts.append(f"    {label} {sc.kind}: {preview}")

    if origin_content:
        preview = origin_content
        if not verbose and len(preview) > 120:
            preview = preview[:117] + "..."
        parts.append(f"    \U0001f4e8 notification: {preview}")

    return parts


def _format_actor_notification(
    an: ActorNotification,
    *,
    verbose: bool = False,
) -> list[str]:
    """Format an actor notification as timeline lines."""
    type_label = {
        "started": "\U0001f4e3 ACTOR STARTED",
        "update": "\U0001f4e3 ACTOR UPDATE",
        "completed": "\U0001f4e3 ACTOR COMPLETED",
    }.get(an.notification_type, f"\U0001f4e3 ACTOR {an.notification_type.upper()}")

    parts: list[str] = []
    parts.append(f"  \u25c6 {_short_utc(an.ts_utc)} {type_label}")
    content = an.content
    if not verbose and len(content) > 200:
        content = content[:197] + "..."
    parts.append(f"    {content}")
    if an.generation_id:
        parts.append(f"    \u2192 triggered fast brain {an.generation_id}")
    return parts


def _format_duration(ms: int) -> str:
    total_s = ms / 1000.0
    h, remainder = divmod(int(total_s), 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def _format_gap(seconds: float | None) -> str:
    if seconds is None or seconds < 2.0:
        return ""
    if seconds >= 60:
        m, s = divmod(int(seconds), 60)
        return f"  >> {m}m {s}s gap"
    return f"  >> {seconds:.1f}s gap"


def _short_utc(ts: str) -> str:
    """Extract HH:MM:SS from a UTC timestamp like 2026-02-19T06:28:59.559+00:00."""
    if "T" in ts:
        time_part = ts.split("T")[1]
        return time_part[:8]
    return ts[:8]


def _source_label(speech_source: str) -> str:
    return SOURCE_LABELS.get(speech_source, speech_source)


def format_timeline(timeline: Timeline, *, verbose: bool = False) -> str:
    """Format the complete timeline as a human-readable string.

    When slow_brain_runs and actor_notifications are present (from CM log),
    they are interleaved chronologically with utterances to show the full
    causal chain.
    """
    parts: list[str] = []
    has_system_events = bool(
        timeline.slow_brain_runs or timeline.actor_notifications,
    )
    has_cm_data = bool(timeline.slow_brain_runs)

    # ── Header ──
    duration_ms = timeline.call_end_ms - timeline.call_start_ms
    total = len(timeline.entries)
    user_count = sum(1 for e in timeline.entries if e.utterance.role == "user")
    asst_count = total - user_count

    source_counts: dict[str, int] = {}
    for e in timeline.entries:
        if e.utterance.role == "assistant":
            src = _source_label(e.utterance.speech_source)
            source_counts[src] = source_counts.get(src, 0) + 1

    parts.append(_DSEP)
    title = "CALL TRANSCRIPT TIMELINE"
    if has_cm_data:
        title += " (FULL TRACE)"
    parts.append(title)
    parts.append(_DSEP)
    parts.append("")
    parts.append(
        f"Duration:   {_format_duration(duration_ms)} "
        f"({_short_utc(timeline.call_start_utc)} -> "
        f"{_short_utc(timeline.call_end_utc)} UTC)",
    )
    parts.append(
        f"Utterances: {total} total ({user_count} user, {asst_count} assistant)",
    )

    if has_system_events:
        sb_total = len(timeline.slow_brain_runs)
        sb_action = sum(1 for s in timeline.slow_brain_runs if s.action != "wait")
        sb_wait = sb_total - sb_action
        parts.append(
            f"Slow brain: {sb_total} runs " f"({sb_action} action, {sb_wait} wait)",
        )
        parts.append(
            f"Actor notifications: {len(timeline.actor_notifications)}",
        )

    parts.append("")
    parts.append("Source breakdown (assistant only):")
    for src in ["fast_brain", "proactive_speech", "slow_brain", "actor_notification"]:
        cnt = source_counts.get(src, 0)
        if cnt:
            parts.append(f"  {src:24s} {cnt}")
    parts.append("")
    parts.append(
        f"Silent guidance (should_speak=False): {len(timeline.silent_guidance)}",
    )

    # ── Transcript ──
    parts.append("")
    parts.append(_DSEP)
    parts.append("TRANSCRIPT")
    parts.append(_DSEP)

    if has_system_events:
        _format_interleaved_transcript(timeline, parts, verbose=verbose)
    else:
        _format_utterance_only_transcript(timeline, parts, verbose=verbose)

    # ── Silent Guidance ──
    if timeline.silent_guidance:
        parts.append(_DSEP)
        parts.append("SILENT GUIDANCE (should_speak=False, injected as notification)")
        parts.append(_DSEP)
        parts.append("")
        for g in timeline.silent_guidance:
            parts.append(
                f"  {g.guidance_id} | {_short_utc(g.ts_utc)} UTC | {g.source}",
            )
            parts.append(f"  {g.content}")
            parts.append("")

    # ── JSON summary (for machine consumption) ──
    parts.append(_DSEP)
    parts.append("MACHINE-READABLE SUMMARY (JSON)")
    parts.append(_DSEP)

    summary: dict = {
        "call_duration_ms": timeline.call_end_ms - timeline.call_start_ms,
        "call_start_utc": timeline.call_start_utc,
        "call_end_utc": timeline.call_end_utc,
        "utterance_count": len(timeline.entries),
        "user_utterances": sum(
            1 for e in timeline.entries if e.utterance.role == "user"
        ),
        "assistant_utterances": sum(
            1 for e in timeline.entries if e.utterance.role == "assistant"
        ),
        "source_breakdown": source_counts,
        "silent_guidance_count": len(timeline.silent_guidance),
    }

    if has_system_events:
        summary["slow_brain_runs"] = len(timeline.slow_brain_runs)
        summary["slow_brain_actions"] = sum(
            1 for s in timeline.slow_brain_runs if s.action != "wait"
        )
        summary["actor_notifications"] = len(timeline.actor_notifications)

    summary["transcript"] = [
        {
            "index": e.index,
            "ts_utc": e.utterance.ts_utc,
            "role": e.utterance.role,
            "text": e.utterance.text,
            "speech_source": e.utterance.speech_source,
            "guidance_id": e.utterance.guidance_id or None,
            "utterance_id": e.utterance.utterance_id,
            "gap_before_s": round(e.gap_before_s, 2) if e.gap_before_s else None,
        }
        for e in timeline.entries
    ]
    parts.append(json.dumps(summary, indent=2))

    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Transcript Formatters
# ─────────────────────────────────────────────────────────────────────────────

# Sort order: system events before utterances at the same timestamp,
# so causal events (actor notification) appear before the response they trigger.
_KIND_ORDER = {"slow_brain": 0, "actor_notification": 1, "utterance": 2}


def _format_utterance_entry(
    entry: TimelineEntry,
    parts: list[str],
    *,
    verbose: bool = False,
) -> None:
    """Format a single utterance entry (used by both transcript modes)."""
    utt = entry.utterance
    gap_str = _format_gap(entry.gap_before_s)
    if gap_str:
        parts.append(gap_str)
        parts.append("")

    role_tag = "USER" if utt.role == "user" else "ASSISTANT"
    source_tag = ""
    if utt.role == "assistant":
        source_tag = f" | {_source_label(utt.speech_source)}"
        if entry.fb_trigger and entry.fb_trigger.reason:
            source_tag += f" ({entry.fb_trigger.reason})"

    parts.append(
        f"#{entry.index:<3d} | {_short_utc(utt.ts_utc)} UTC | {role_tag}{source_tag}",
    )
    parts.append(_SEP)

    text = utt.text
    if not verbose and len(text) > 300:
        text = text[:297] + "..."
    parts.append(text)
    parts.append("")

    ids = [utt.utterance_id]
    if utt.guidance_id:
        ids.append(utt.guidance_id)
    if entry.fb_request:
        ids.append(f"req:{entry.fb_request.request_id[:8]}")
    if entry.fb_trigger:
        ids.append(entry.fb_trigger.generation_id)

    parts.append(f"  [{' | '.join(ids)}]")

    for a in entry.anomalies:
        parts.append(f"  !! {a.anomaly_type}: {a.description}")

    parts.append("")


def _format_utterance_only_transcript(
    timeline: Timeline,
    parts: list[str],
    *,
    verbose: bool = False,
) -> None:
    """Original voice-only transcript format (no CM log)."""
    for entry in timeline.entries:
        _format_utterance_entry(entry, parts, verbose=verbose)


def _build_notification_index(
    slow_brain_runs: list[SlowBrainRun],
    actor_notifications: list[ActorNotification],
) -> dict[str, ActorNotification]:
    """Map slow-brain origin_event_ids to their triggering ActorNotification.

    Since the CM log doesn't share IDs with the voice log, matches by
    finding the nearest notification within 2 seconds of each actor-triggered
    slow-brain run.
    """
    actor_origin_types = {
        "ActorSessionResponse",
        "ActorHandleStarted",
        "ActorHandleResponse",
        "ActorResult",
    }
    idx: dict[str, ActorNotification] = {}
    used: set[int] = set()

    for sb in slow_brain_runs:
        if sb.origin_event not in actor_origin_types or not sb.origin_event_id:
            continue
        dispatch_ms = _utc_to_ms(sb.dispatch_utc) if sb.dispatch_utc else 0
        if dispatch_ms == 0:
            continue
        best_j: int | None = None
        best_gap = float("inf")
        for j, an in enumerate(actor_notifications):
            if j in used:
                continue
            gap = abs(dispatch_ms - _utc_to_ms(an.ts_utc))
            if gap < 3_000 and gap < best_gap:
                best_j = j
                best_gap = gap
        if best_j is not None:
            idx[sb.origin_event_id] = actor_notifications[best_j]
            used.add(best_j)

    return idx


def _build_steering_index(
    slow_brain_runs: list[SlowBrainRun],
    steering_contents: list[SteeringContent],
) -> dict[str, SteeringContent]:
    """Map slow-brain run_ids (for interject/ask actions) to their steering content."""
    action_runs = [
        sb
        for sb in slow_brain_runs
        if sb.action.startswith("interject_") or sb.action.startswith("ask_")
    ]
    idx: dict[str, SteeringContent] = {}
    used: set[int] = set()

    for sb in action_runs:
        sb_ms = _utc_to_ms(sb.ts_utc)
        best_j: int | None = None
        best_gap = float("inf")
        expected_kind = "interject" if sb.action.startswith("interject_") else "ask"
        for j, sc in enumerate(steering_contents):
            if j in used or sc.kind != expected_kind:
                continue
            gap = abs(sb_ms - _utc_to_ms(sc.ts_utc))
            if gap < 5_000 and gap < best_gap:
                best_j = j
                best_gap = gap
        if best_j is not None:
            idx[sb.run_id] = steering_contents[best_j]
            used.add(best_j)

    return idx


def _format_interleaved_transcript(
    timeline: Timeline,
    parts: list[str],
    *,
    verbose: bool = False,
) -> None:
    """Interleaved transcript with slow-brain decisions and actor notifications."""
    notif_idx = _build_notification_index(
        timeline.slow_brain_runs,
        timeline.actor_notifications,
    )
    steer_idx = _build_steering_index(
        timeline.slow_brain_runs,
        timeline.steering_contents,
    )

    merged: list[tuple[str, str, object]] = []

    for entry in timeline.entries:
        sk = _time_sort_key(entry.utterance.ts_utc)
        merged.append((sk, "utterance", entry))

    for sb in timeline.slow_brain_runs:
        sk = _time_sort_key(sb.ts_utc)
        merged.append((sk, "slow_brain", sb))

    for an in timeline.actor_notifications:
        sk = _time_sort_key(an.ts_utc)
        merged.append((sk, "actor_notification", an))

    merged.sort(key=lambda x: (x[0], _KIND_ORDER.get(x[1], 9)))

    for _, kind, event in merged:
        if kind == "utterance":
            _format_utterance_entry(event, parts, verbose=verbose)  # type: ignore[arg-type]
        elif kind == "slow_brain":
            sb_lines = _format_slow_brain(
                event,  # type: ignore[arg-type]
                verbose=verbose,
                notification_index=notif_idx,
                steering_index=steer_idx,
            )
            parts.extend(sb_lines)
            parts.append("")
        elif kind == "actor_notification":
            an_lines = _format_actor_notification(event, verbose=verbose)  # type: ignore[arg-type]
            parts.extend(an_lines)
            parts.append("")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a source-traced call transcript from sandbox voice logs.",
    )
    parser.add_argument(
        "voice_log",
        type=Path,
        help="Path to .logs_voice_agent.txt",
    )
    parser.add_argument(
        "--cm-log",
        type=Path,
        default=None,
        help="Path to .logs_conversation_sandbox.txt (optional, for CM cross-reference)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show full text for long utterances (default truncates at 300 chars)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_only",
        help="Output only the JSON summary (machine-readable)",
    )
    args = parser.parse_args()

    if not args.voice_log.exists():
        print(f"Error: voice log not found: {args.voice_log}", file=sys.stderr)
        sys.exit(1)

    data = parse_voice_log(args.voice_log)

    if not data.utterances:
        print("No utterances found in voice log.", file=sys.stderr)
        sys.exit(1)

    cm_data: CMLogData | None = None
    if args.cm_log and args.cm_log.exists():
        cm_data = parse_cm_log(args.cm_log)

    timeline = build_timeline(data, cm_data)

    if args.json_only:
        source_counts: dict[str, int] = {}
        for e in timeline.entries:
            if e.utterance.role == "assistant":
                src = _source_label(e.utterance.speech_source)
                source_counts[src] = source_counts.get(src, 0) + 1

        summary = {
            "call_duration_ms": timeline.call_end_ms - timeline.call_start_ms,
            "call_start_utc": timeline.call_start_utc,
            "call_end_utc": timeline.call_end_utc,
            "utterance_count": len(timeline.entries),
            "source_breakdown": source_counts,
            "anomaly_count": len(timeline.anomalies),
            "transcript": [
                {
                    "index": e.index,
                    "ts_utc": e.utterance.ts_utc,
                    "role": e.utterance.role,
                    "text": e.utterance.text,
                    "speech_source": e.utterance.speech_source,
                    "guidance_id": e.utterance.guidance_id or None,
                    "utterance_id": e.utterance.utterance_id,
                    "gap_before_s": (
                        round(e.gap_before_s, 2) if e.gap_before_s else None
                    ),
                }
                for e in timeline.entries
            ],
        }
        print(json.dumps(summary, indent=2))
    else:
        print(format_timeline(timeline, verbose=args.verbose))


if __name__ == "__main__":
    main()
