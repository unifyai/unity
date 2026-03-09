#!/usr/bin/env python3
"""Post-hoc call transcript builder for ConversationManager sandbox runs.

Parses .logs_voice_agent.txt (and optionally .logs_conversation_sandbox.txt)
to produce a deterministic, source-traced call transcript with anomaly detection.

Every assistant utterance is traced to its source path:
  - Path 1: fast_brain (reply)
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
    llm_log_path: str = ""


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
    """A reply trigger event from the fast brain."""

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
    llm_log_path: str = ""
    was_queued: bool = False
    preempted: bool = False


@dataclass
class ProactiveSpeechDecision:
    """A proactive speech LLM decision from the CM side."""

    ts_utc: str
    should_speak: bool
    delay_s: int
    content: str  # empty when should_speak=False


@dataclass
class DirectActionEntry:
    """A single low-level action executed via /execute-actions (click, type, etc.)."""

    rendered: str  # e.g. "⊙ click (460, 412)"
    execution_ms: int


@dataclass
class DirectActionGroup:
    """A group of direct actions from a single execute_code block."""

    actions: list[DirectActionEntry] = field(default_factory=list)

    @property
    def total_ms(self) -> int:
        return sum(a.execution_ms for a in self.actions)


@dataclass
class MagnitudeTrace:
    """A parsed magnitude act_trace.json for a single desktop/web act() call."""

    act_id: str
    task: str
    total_ms: int
    iterations: int
    reasoning: str
    action_traces: list[dict]
    lineage: list[str] = field(default_factory=list)


@dataclass
class ActorToolCall:
    """A tool call within the CodeActActor execution loop."""

    ts_utc: str
    actor_id: str  # e.g. "accb"
    event_type: str  # "request", "llm_thinking", "tool_scheduled", "tool_completed", "execute_code", "persist_wait"
    tool_name: str
    duration_s: float | None = None
    llm_log_path: str = ""
    detail: str = ""
    code: str = ""
    thought: str = ""
    result_summary: str = ""
    magnitude_trace: MagnitudeTrace | None = None
    direct_actions: DirectActionGroup | None = None


@dataclass
class ActorNotification:
    """An actor event delivered to the fast brain (Action started/update/completed)."""

    ts_utc: str
    notification_type: str  # "started", "update", "completed"
    content: str
    generation_id: str = ""
    magnitude_traces: list[MagnitudeTrace] = field(default_factory=list)


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
    proactive_decisions: list[ProactiveSpeechDecision] = field(default_factory=list)
    actor_tool_calls: list[ActorToolCall] = field(default_factory=list)


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
    proactive_decisions: list[ProactiveSpeechDecision]
    actor_tool_calls: list[ActorToolCall]
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
    """Find a generation link from the LLM thinking line that follows an actor notification.

    Since generation_id was removed from log output, this now returns a
    content-derived hash when a thinking line follows within 3 lines.
    """
    for j in range(start_idx + 1, min(start_idx + 3, len(lines))):
        if "LLM thinking" in lines[j]:
            return _content_hash("gen", lines[j])
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Voice Log Parser
# ─────────────────────────────────────────────────────────────────────────────


def parse_voice_log(path: Path) -> VoiceLogData:
    """Parse all relevant trace events from the voice agent log.

    Parses the ``FastBrainLogger`` emoji format: ``[FastBrain...] <body>``.
    Actor notifications are extracted from the unified notification channel
    (embedded inside ``Notification (...)`` lines).
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    data = VoiceLogData()
    log_date = _extract_log_date(lines)

    for i, line in enumerate(lines):
        if "[FastBrain" not in line:
            continue

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

        elif body.startswith("Notification ("):
            should_speak = "speak=True" in body
            src_match = re.search(r"src=([^,)]+)", body)
            source = src_match.group(1).strip() if src_match else ""
            content_match = re.search(r"\):\s*(.*)", body)
            content = content_match.group(1).strip() if content_match else ""
            nid = _content_hash("guid", content)

            data.guidance_events.append(
                GuidanceReceived(
                    monotonic_ms=mono,
                    ts_utc=ts_utc,
                    guidance_id=nid,
                    source=source,
                    should_speak=should_speak,
                    user_is_speaking=False,
                    content=content,
                ),
            )

            if content.startswith("Action started:"):
                data.actor_notifications.append(
                    ActorNotification(
                        ts_utc=ts_utc,
                        notification_type="started",
                        content=content[len("Action started:") :].strip(),
                        generation_id=_find_next_generation_id(lines, i, log_date),
                    ),
                )
            elif content.startswith("Action progress:"):
                data.actor_notifications.append(
                    ActorNotification(
                        ts_utc=ts_utc,
                        notification_type="update",
                        content=content[len("Action progress:") :].strip(),
                        generation_id=_find_next_generation_id(lines, i, log_date),
                    ),
                )
            elif content.startswith("Action update:"):
                data.actor_notifications.append(
                    ActorNotification(
                        ts_utc=ts_utc,
                        notification_type="update",
                        content=content[len("Action update:") :].strip(),
                        generation_id=_find_next_generation_id(lines, i, log_date),
                    ),
                )
            elif content.startswith(("Action completed", "Action failed")):
                detail = content.split(":", 1)[1].strip() if ":" in content else content
                data.actor_notifications.append(
                    ActorNotification(
                        ts_utc=ts_utc,
                        notification_type="completed",
                        content=detail,
                        generation_id=_find_next_generation_id(lines, i, log_date),
                    ),
                )
            elif content.startswith("Desktop action completed:"):
                data.actor_notifications.append(
                    ActorNotification(
                        ts_utc=ts_utc,
                        notification_type="completed",
                        content=content[len("Desktop action completed:") :].strip(),
                        generation_id=_find_next_generation_id(lines, i, log_date),
                    ),
                )

        elif body.startswith("Speaking notification"):
            text = body.split(":", 1)[1].strip() if ":" in body else ""
            nsource = _extract_field(line, "notification_source")
            if " notification_source=" in text:
                text = text[: text.index(" notification_source=")]
            nid = _content_hash("guid", text)
            data.session_says.append(
                SessionSay(
                    monotonic_ms=mono,
                    ts_utc=ts_utc,
                    guidance_id=nid,
                    source=nsource,
                    text=text,
                ),
            )

        elif body.startswith("LLM thinking"):
            reason = _extract_field(line, "reason", "queued_speech")
            if not reason:
                reason = _extract_field(line, "reason")
            data.fb_triggers.append(
                GenerateReplyTrigger(
                    monotonic_ms=mono,
                    ts_utc=ts_utc,
                    generation_id=_content_hash("gen", f"{ts_utc}:{reason}"),
                    reason=reason,
                    source_id="",
                    queued_speech_count=_extract_int(
                        line,
                        "queued_speech",
                    ),
                ),
            )

        elif _ASSISTANT_SPEECH_ICON in line:
            source = "reply"
            gid = ""
            llm_log_path = ""
            raw_body = body
            arrow_idx = raw_body.rfind(" \u2192 /")
            if arrow_idx != -1:
                llm_log_path = raw_body[arrow_idx + 3 :].strip()
                raw_body = raw_body[:arrow_idx].strip()
            m = re.match(r"^([\w ]+):\s+(.+)$", raw_body)
            if m:
                source = m.group(1).strip().lower().replace(" ", "_")
                text = m.group(2).strip()
            else:
                text = raw_body.strip()
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
                        llm_log_path=llm_log_path,
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
                        speech_source="reply",
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


def parse_magnitude_traces(magnitude_dir: Path) -> dict[str, MagnitudeTrace]:
    """Load all act_trace.json files and index by execute_code exec_id from lineage.

    Returns a dict mapping exec_id (e.g. "6801") to the corresponding trace.
    """
    traces: dict[str, MagnitudeTrace] = {}
    acts_dir = magnitude_dir / "acts"
    if not acts_dir.is_dir():
        return traces
    for trace_file in acts_dir.glob("*/act_trace.json"):
        try:
            raw = json.loads(trace_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        lineage = raw.get("lineage", [])
        exec_id = ""
        for segment in lineage:
            m = re.search(r"execute_code\((\w+)\)", segment)
            if m:
                exec_id = m.group(1)
                break
        trace = MagnitudeTrace(
            act_id=raw.get("actId", ""),
            task=raw.get("task", ""),
            total_ms=raw.get("totalMs", 0),
            iterations=raw.get("iterations", 0),
            reasoning=raw.get("reasoning", ""),
            action_traces=raw.get("actionTraces", []),
            lineage=lineage,
        )
        if exec_id:
            traces[exec_id] = trace
        else:
            traces[trace.act_id] = trace
    return traces


_EXEC_ACTIONS_START_RE = re.compile(
    r"\[execute-actions\] Executing (\d+) direct action\(s\) for session (\S+)",
)
_EXEC_ACTIONS_DONE_RE = re.compile(
    r"\[execute-actions\] (\d+) action\(s\) executed \[(\d+)ms\]",
)


def parse_agent_service_log(path: Path) -> list[DirectActionGroup]:
    """Parse [execute-actions] blocks from the agent-service supervisor log.

    Returns a list of DirectActionGroup in chronological order.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    groups: list[DirectActionGroup] = []
    current_group: DirectActionGroup | None = None
    pending_rendered: str = ""

    for line in lines:
        m_start = _EXEC_ACTIONS_START_RE.search(line)
        if m_start:
            if current_group is None:
                current_group = DirectActionGroup()
            pending_rendered = ""
            continue

        m_done = _EXEC_ACTIONS_DONE_RE.search(line)
        if m_done and current_group is not None:
            ms = int(m_done.group(2))
            current_group.actions.append(
                DirectActionEntry(rendered=pending_rendered.strip(), execution_ms=ms),
            )
            pending_rendered = ""
            # Peek ahead: if the next non-blank line isn't another [execute-actions],
            # close the group.
            continue

        # Between start and done, the line is the rendered action (e.g. "  ⊙ click ...")
        if current_group is not None and not line.startswith("["):
            stripped = line.strip()
            if stripped:
                pending_rendered = stripped
            continue

        # Any non-execute-actions line after a group means the group is done
        if current_group is not None and current_group.actions:
            groups.append(current_group)
            current_group = None

    if current_group is not None and current_group.actions:
        groups.append(current_group)

    return groups


def _collect_json_block(lines: list[str], start_idx: int) -> tuple[str, int]:
    """Collect a multi-line JSON block starting from lines[start_idx].

    The opening ``{`` is expected on lines[start_idx] (possibly after a log prefix).
    Returns (json_string, end_idx) where end_idx is the line AFTER the closing ``}``.
    """
    first_line = lines[start_idx]
    brace_pos = first_line.find("{")
    if brace_pos < 0:
        return "", start_idx + 1
    collected = [first_line[brace_pos:]]
    depth = first_line[brace_pos:].count("{") - first_line[brace_pos:].count("}")
    idx = start_idx + 1
    while idx < len(lines) and depth > 0:
        raw = lines[idx]
        collected.append(raw)
        depth += raw.count("{") - raw.count("}")
        idx += 1
    return "\n".join(collected), idx


def _extract_code_from_assistant_msg(json_str: str) -> dict[str, tuple[str, str]]:
    """Extract execute_code arguments from an assistant tool-call JSON.

    Returns ``{call_id: (code, thought)}`` for each execute_code tool call.
    """
    try:
        msg = json.loads(json_str, strict=False)
    except json.JSONDecodeError:
        return {}
    results: dict[str, tuple[str, str]] = {}
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        if fn.get("name") != "execute_code":
            continue
        args = fn.get("arguments", {})
        code = args.get("code", "")
        if isinstance(code, str):
            code = re.sub(r"^\s*```\w*\n?", "", code)
            code = re.sub(r"\n?\s*```\s*$", "", code)
            import textwrap

            code = textwrap.dedent(code).strip()
        thought = args.get("thought", "")
        call_id = tc.get("id", "")
        results[call_id] = (code, thought)
    return results


def _extract_result_from_tool_msg(json_str: str) -> str:
    """Extract a concise result summary from a ToolCall Completed JSON block."""
    try:
        msg = json.loads(json_str, strict=False)
    except json.JSONDecodeError:
        return ""
    content = msg.get("content")
    if not isinstance(content, list):
        return ""

    error = ""
    result_val = ""
    stdout_parts: list[str] = []
    has_image = False

    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "image_url":
            has_image = True
            continue
        if part.get("type") != "text":
            continue
        inner_text = part.get("text", "")
        try:
            inner = json.loads(inner_text)
            if inner.get("error"):
                err = str(inner["error"])
                last_line = err.strip().splitlines()[-1] if err.strip() else err
                error = f"ERROR: {last_line[:200]}"
            if inner.get("result"):
                rv = str(inner["result"])
                if "base64" in rv[:100] or (len(rv) > 200 and rv[:20].isalnum()):
                    result_val = "(screenshot captured)"
                else:
                    result_val = rv[:300]
            if inner.get("stdout"):
                stdout_parts.append(inner["stdout"].strip())
        except json.JSONDecodeError:
            text = inner_text.strip()
            if text.startswith("--- stdout ---"):
                text = text[len("--- stdout ---") :].strip()
            if text:
                stdout_parts.append(text[:200])

    if error:
        return error
    if result_val:
        return result_val
    stdout_combined = "\n".join(stdout_parts).strip()
    if stdout_combined:
        if "iVBOR" in stdout_combined:
            idx = stdout_combined.index("iVBOR")
            return stdout_combined[:idx] + "(image data)..."
        if "data:image/" in stdout_combined:
            idx = stdout_combined.index("data:image/")
            return stdout_combined[:idx] + "(image data)..."
        return stdout_combined[:300]
    if has_image:
        return "(screenshot captured)"
    return "(ok)"


def parse_cm_log(path: Path) -> CMLogData:
    """Parse slow-brain decisions, guidance decisions, and actor events from the CM log."""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    data = CMLogData()

    run_meta: dict[str, dict[str, str]] = {}
    sb_llm_paths: dict[str, str] = {}

    pending_code: dict[str, tuple[str, str]] = {}

    i = 0
    while i < len(lines):
        line = lines[i]
        # Slow-brain LLM log path: [ConversationManager] LLM thinking… (...) → /path
        if "[ConversationManager] LLM thinking" in line and "\u2192 " in line:
            path_part = line.split("\u2192 ", 1)[1].strip()
            origin_match = re.search(r"\(([^)]+)\)", line.split("LLM thinking", 1)[1])
            if origin_match:
                origin_key = _parse_cm_ts(line)
                sb_llm_paths[origin_key] = path_part

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
                "cancel_running": "cancel_running=True" in line,
            }

        # Slow-brain run started: captures was_queued per run_id
        if "Slow-brain run started" in line and "run_id=" in line:
            run_id = _extract_field(line, "run_id", "request_id")
            if run_id in run_meta:
                run_meta[run_id]["was_queued"] = "was_queued=True" in line

        # Slow-brain thought + action decision
        # Format: run_id=llmrun-NNNNN thoughts: <text> | actions: ['wait']
        elif "thoughts:" in line and "| actions:" in line and "run_id=" in line:
            run_id_raw = line.split("run_id=", 1)[1]
            run_id = run_id_raw.split()[0]
            raw = line.split("thoughts:", 1)[1]
            parts = raw.rsplit("| actions:", 1)
            thoughts = parts[0].strip()
            actions_raw = parts[1].strip() if len(parts) > 1 else "[]"
            actions_list = re.findall(r"'([^']*)'", actions_raw)
            action = actions_list[0] if actions_list else ""
            ts = _parse_cm_ts(line)
            meta = run_meta.get(run_id, {})
            llm_path = sb_llm_paths.get(ts, "")
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
                    llm_log_path=llm_path,
                    was_queued=bool(meta.get("was_queued")),
                    preempted=bool(meta.get("cancel_running")),
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

        # Proactive speech decisions: 🗣️ [ProactiveSpeech] should_speak=True, delay=3s: text
        elif "[ProactiveSpeech] should_speak=" in line:
            should_speak = "should_speak=True" in line
            delay_match = re.search(r"delay=(\d+)s", line)
            delay_s = int(delay_match.group(1)) if delay_match else 0
            content = ""
            if should_speak and ": " in line.split("delay=")[1]:
                content = (
                    line.split("delay=")[1].split(": ", 1)[1].strip()
                    if ": " in line.split("delay=")[1]
                    else ""
                )
            data.proactive_decisions.append(
                ProactiveSpeechDecision(
                    ts_utc=_parse_cm_ts(line),
                    should_speak=should_speak,
                    delay_s=delay_s,
                    content=content,
                ),
            )

        # Actor request: ➡️ [CodeActActor.act(XXXX)] Request: ...
        elif "CodeActActor.act(" in line and "] Request:" in line:
            aid_match = re.search(r"CodeActActor\.act\((\w+)\)", line)
            actor_id = aid_match.group(1) if aid_match else ""
            detail = line.split("] Request:", 1)[1].strip()
            data.actor_tool_calls.append(
                ActorToolCall(
                    ts_utc=_parse_cm_ts(line),
                    actor_id=actor_id,
                    event_type="request",
                    tool_name="",
                    detail=detail,
                ),
            )

        # Actor LLM thinking: 🧠 [CodeActActor.act(XXXX)] LLM thinking… → /path
        elif "CodeActActor.act(" in line and "LLM thinking" in line:
            aid_match = re.search(r"CodeActActor\.act\((\w+)\)", line)
            actor_id = aid_match.group(1) if aid_match else ""
            llm_path = ""
            if "\u2192 " in line:
                llm_path = line.split("\u2192 ", 1)[1].strip()
            data.actor_tool_calls.append(
                ActorToolCall(
                    ts_utc=_parse_cm_ts(line),
                    actor_id=actor_id,
                    event_type="llm_thinking",
                    tool_name="",
                    llm_log_path=llm_path,
                ),
            )

        # Actor tool scheduled: 🛠️  ToolCall Scheduled [CodeActActor.act(XXXX)] tool_name - id
        elif "ToolCall Scheduled" in line and "CodeActActor.act(" in line:
            aid_match = re.search(r"CodeActActor\.act\((\w+)\)", line)
            actor_id = aid_match.group(1) if aid_match else ""
            parts = line.split("]", 2)
            tool_part = parts[2].strip() if len(parts) > 2 else ""
            tool_name = (
                tool_part.split(" - ")[0].strip() if " - " in tool_part else tool_part
            )
            data.actor_tool_calls.append(
                ActorToolCall(
                    ts_utc=_parse_cm_ts(line),
                    actor_id=actor_id,
                    event_type="tool_scheduled",
                    tool_name=tool_name,
                ),
            )

        # Actor assistant message with code: 🤖 [CodeActActor.act(XXXX)] {
        elif "\U0001f916 [CodeActActor.act(" in line and "{" in line:
            json_str, end_idx = _collect_json_block(lines, i)
            code_by_call_id = _extract_code_from_assistant_msg(json_str)
            pending_code.update(code_by_call_id)
            i = end_idx
            continue

        # Actor execute_code: 🛠️ [CodeActActor.act(XXXX)->execute_code(YYYY)] Executing code...
        elif "->execute_code(" in line and "Executing code" in line:
            aid_match = re.search(r"CodeActActor\.act\((\w+)\)", line)
            actor_id = aid_match.group(1) if aid_match else ""
            exec_match = re.search(r"execute_code\((\w+)\)", line)
            exec_id = exec_match.group(1) if exec_match else ""
            sched_match = re.search(
                r"ToolCall Scheduled.*?execute_code - (\S+)",
                lines[i - 1] if i > 0 else "",
            )
            call_id = sched_match.group(1) if sched_match else ""
            code, thought = pending_code.pop(call_id, ("", ""))
            data.actor_tool_calls.append(
                ActorToolCall(
                    ts_utc=_parse_cm_ts(line),
                    actor_id=actor_id,
                    event_type="execute_code",
                    tool_name=f"execute_code({exec_id})",
                    code=code,
                    thought=thought,
                ),
            )

        # Actor tool completed: ✅  ToolCall Completed [Ns] [CodeActActor.act(XXXX)] {
        elif "ToolCall Completed" in line and "CodeActActor.act(" in line:
            aid_match = re.search(r"CodeActActor\.act\((\w+)\)", line)
            actor_id = aid_match.group(1) if aid_match else ""
            dur_match = re.search(r"ToolCall Completed \[(\d+\.?\d*)s\]", line)
            duration = float(dur_match.group(1)) if dur_match else None
            result_summary = ""
            if "{" in line:
                json_str, end_idx = _collect_json_block(lines, i)
                result_summary = _extract_result_from_tool_msg(json_str)
                i = end_idx
            else:
                i += 1
            data.actor_tool_calls.append(
                ActorToolCall(
                    ts_utc=_parse_cm_ts(line),
                    actor_id=actor_id,
                    event_type="tool_completed",
                    tool_name="",
                    duration_s=duration,
                    result_summary=result_summary,
                ),
            )
            continue

        # Actor persist wait: ⏸️ [CodeActActor.act(XXXX)] Persist mode: waiting...
        elif "Persist mode:" in line and "CodeActActor.act(" in line:
            aid_match = re.search(r"CodeActActor\.act\((\w+)\)", line)
            actor_id = aid_match.group(1) if aid_match else ""
            data.actor_tool_calls.append(
                ActorToolCall(
                    ts_utc=_parse_cm_ts(line),
                    actor_id=actor_id,
                    event_type="persist_wait",
                    tool_name="",
                ),
            )

        # StorageCheck: ➡️ [StorageCheck(...)(XXXX)] Request: ...
        elif "StorageCheck(" in line and "] Request:" in line:
            aid_match = re.search(r"StorageCheck\([^)]*\)\((\w+)\)", line)
            actor_id = aid_match.group(1) if aid_match else ""
            detail = line.split("] Request:", 1)[1].strip()
            data.actor_tool_calls.append(
                ActorToolCall(
                    ts_utc=_parse_cm_ts(line),
                    actor_id=actor_id,
                    event_type="storage_check",
                    tool_name="StorageCheck",
                    detail=detail,
                ),
            )

        i += 1

    # Merge tool_completed results back into their preceding execute_code entries.
    last_exec: ActorToolCall | None = None
    merged_completed_indices: set[int] = set()
    for idx_atc, atc in enumerate(data.actor_tool_calls):
        if atc.event_type == "execute_code":
            last_exec = atc
        elif atc.event_type == "tool_completed" and last_exec is not None:
            last_exec.duration_s = atc.duration_s
            last_exec.result_summary = atc.result_summary
            merged_completed_indices.add(idx_atc)
            last_exec = None
        elif atc.event_type not in ("llm_thinking",):
            last_exec = None
    if merged_completed_indices:
        data.actor_tool_calls = [
            atc
            for idx_atc, atc in enumerate(data.actor_tool_calls)
            if idx_atc not in merged_completed_indices
        ]

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
        if utt.speech_source == "reply" and utt.role == "assistant":
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
    proactive_decisions = cm_data.proactive_decisions if cm_data else []
    actor_tool_calls = cm_data.actor_tool_calls if cm_data else []

    return Timeline(
        entries=entries,
        slow_brain_runs=slow_brain_runs,
        steering_contents=steering_contents,
        actor_notifications=actor_notifications,
        silent_guidance=silent_guidance,
        proactive_decisions=proactive_decisions,
        actor_tool_calls=actor_tool_calls,
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
    "reply": "fast_brain",
    "notification_reply": "notification_reply",
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

    queue_tag = ""
    if sb.preempted and sb.was_queued:
        queue_tag = " [preempted previous, queued]"
    elif sb.preempted:
        queue_tag = " [preempted previous]"
    elif sb.was_queued:
        queue_tag = " [queued]"

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
            f"\u2192 wait{queue_tag}{thought_preview}",
        )
    else:
        parts.append(
            f"  \u25c6 {_short_utc(sb.ts_utc)} "
            f"SLOW BRAIN [{sb.run_id}] "
            f"\u2190 {sb.origin_event}{queue_tag}",
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

    if sb.llm_log_path:
        parts.append(f"    \U0001f4dd {sb.llm_log_path}")

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
    for mt in an.magnitude_traces:
        parts.extend(_format_magnitude_trace(mt, verbose=verbose))
    return parts


def _format_proactive_decision(
    pd: ProactiveSpeechDecision,
    *,
    verbose: bool = False,
) -> list[str]:
    """Format a proactive speech decision as a timeline line."""
    if pd.should_speak:
        preview = pd.content
        if not verbose and len(preview) > 120:
            preview = preview[:117] + "..."
        return [
            f"  \U0001f5e3\ufe0f {_short_utc(pd.ts_utc)} "
            f"PROACTIVE SPEECH \u2192 speak (delay={pd.delay_s}s): {preview}",
        ]
    return [
        f"  \U0001f5e3\ufe0f {_short_utc(pd.ts_utc)} "
        f"PROACTIVE SPEECH \u2192 silent (delay={pd.delay_s}s)",
    ]


def _format_direct_actions(dag: DirectActionGroup) -> list[str]:
    """Render an inline direct-actions block under an execute_code entry."""
    total_s = dag.total_ms / 1000.0
    parts = [
        f"      \u250c\u2500 DIRECT ACTIONS: {len(dag.actions)} actions [{total_s:.1f}s]",
    ]
    for a in dag.actions:
        parts.append(f"      \u2502   {a.rendered:40s} [{a.execution_ms}ms]")
    parts.append(f"      \u2514\u2500 done")
    return parts


def _format_magnitude_trace(mt: MagnitudeTrace, *, verbose: bool = False) -> list[str]:
    """Render an inline magnitude trace block under an execute_code entry."""
    task_preview = mt.task if verbose or len(mt.task) <= 80 else mt.task[:77] + "..."
    total_s = mt.total_ms / 1000.0
    header = (
        f'      \u250c\u2500 MAGNITUDE: "{task_preview}" '
        f"[{total_s:.1f}s, {mt.iterations} iter, {len(mt.action_traces)} actions]"
    )
    parts = [header]

    reasoning_steps = [s.strip() for s in mt.reasoning.split("\n---\n") if s.strip()]
    reasoning_by_iter: dict[int, str] = {}
    for i, step in enumerate(reasoning_steps):
        reasoning_by_iter[i] = step

    last_iter = -1
    for a in mt.action_traces:
        iteration = a.get("iteration", 0)
        if iteration != last_iter and iteration in reasoning_by_iter:
            thought = reasoning_by_iter[iteration]
            if not verbose and len(thought) > 150:
                thought = thought[:147] + "..."
            parts.append(f"      \u2502 \U0001f4ad {thought}")
            last_iter = iteration
        rendered = a.get("rendered", a.get("variant", "?"))
        ms = a.get("executionMs", 0)
        parts.append(f"      \u2502   {rendered:40s} [{ms}ms]")

    parts.append(f"      \u2514\u2500 done")
    return parts


def _format_actor_tool_call(
    atc: ActorToolCall,
    *,
    verbose: bool = False,
) -> list[str]:
    """Format an actor tool call as a timeline line."""
    ts = _short_utc(atc.ts_utc)
    if atc.event_type == "request":
        preview = atc.detail
        if not verbose and len(preview) > 150:
            preview = preview[:147] + "..."
        return [f"  \u2699\ufe0f {ts} ACTOR [{atc.actor_id}] Request: {preview}"]
    elif atc.event_type == "llm_thinking":
        suffix = f" \u2192 {atc.llm_log_path}" if atc.llm_log_path else ""
        return [f"  \U0001f9e0 {ts} ACTOR [{atc.actor_id}] LLM thinking{suffix}"]
    elif atc.event_type == "tool_scheduled":
        if atc.tool_name.startswith("execute_code"):
            return []
        return [
            f"  \U0001f6e0\ufe0f  {ts} ACTOR [{atc.actor_id}] Tool: {atc.tool_name}",
        ]
    elif atc.event_type == "execute_code":
        dur = f" [{atc.duration_s:.1f}s]" if atc.duration_s is not None else ""
        result_tag = ""
        if atc.result_summary:
            if atc.result_summary.startswith("ERROR:"):
                result_tag = f"  \u274c {atc.result_summary}"
            else:
                result_tag = (
                    f"  \u2192 {atc.result_summary}"
                    if atc.result_summary != "(ok)"
                    else ""
                )
        parts = [
            f"  \U0001f4bb {ts} ACTOR [{atc.actor_id}] {atc.tool_name}{dur}{result_tag}",
        ]
        if atc.thought:
            thought_preview = atc.thought
            if not verbose and len(thought_preview) > 200:
                thought_preview = thought_preview[:197] + "..."
            parts.append(f"      \U0001f4ad {thought_preview}")
        if atc.code:
            code_preview = atc.code
            if not verbose:
                code_lines = code_preview.splitlines()
                if len(code_lines) > 5:
                    code_preview = "\n".join(code_lines[:5]) + "\n..."
            for cl in code_preview.splitlines():
                parts.append(
                    (
                        f"      code: {cl}"
                        if parts[-1].endswith(result_tag or dur or atc.tool_name)
                        else f"            {cl}"
                    ),
                )
        if atc.magnitude_trace:
            parts.extend(_format_magnitude_trace(atc.magnitude_trace, verbose=verbose))
        if atc.direct_actions and atc.direct_actions.actions:
            parts.extend(_format_direct_actions(atc.direct_actions))
        return parts
    elif atc.event_type == "tool_completed":
        dur = f" [{atc.duration_s:.1f}s]" if atc.duration_s is not None else ""
        line = f"  \u2705 {ts} ACTOR [{atc.actor_id}] Tool completed{dur}"
        if atc.result_summary:
            preview = atc.result_summary
            if not verbose and len(preview) > 120:
                preview = preview[:117] + "..."
            line += f"  \u2192 {preview}"
        return [line]
    elif atc.event_type == "persist_wait":
        return [
            f"  \u23f8\ufe0f  {ts} ACTOR [{atc.actor_id}] Persist mode \u2014 waiting for interjection",
        ]
    elif atc.event_type == "storage_check":
        return [f"  \U0001f4be {ts} STORAGE CHECK [{atc.actor_id}] {atc.detail}"]
    return [
        f"  \u2699\ufe0f {ts} ACTOR [{atc.actor_id}] {atc.event_type}: {atc.tool_name}",
    ]


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
        if timeline.actor_tool_calls:
            exec_count = sum(
                1 for a in timeline.actor_tool_calls if a.event_type == "execute_code"
            )
            llm_count = sum(
                1 for a in timeline.actor_tool_calls if a.event_type == "llm_thinking"
            )
            mag_count = sum(
                1
                for a in timeline.actor_tool_calls
                if a.event_type == "execute_code" and a.magnitude_trace
            )
            mag_suffix = f", {mag_count} magnitude acts" if mag_count else ""
            parts.append(
                f"Actor internals: {llm_count} LLM calls, "
                f"{exec_count} code executions{mag_suffix}",
            )
        if timeline.proactive_decisions:
            speak = sum(1 for p in timeline.proactive_decisions if p.should_speak)
            silent = len(timeline.proactive_decisions) - speak
            parts.append(
                f"Proactive decisions: {len(timeline.proactive_decisions)} "
                f"({speak} speak, {silent} silent)",
            )

    parts.append("")
    parts.append("Source breakdown (assistant only):")
    for src in [
        "fast_brain",
        "notification_reply",
        "proactive_speech",
        "slow_brain",
        "actor_notification",
    ]:
        cnt = source_counts.get(src, 0)
        if cnt:
            parts.append(f"  {src:24s} {cnt}")
    parts.append("")
    parts.append(
        f"Silent notifications (should_speak=False): {len(timeline.silent_guidance)}",
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
        parts.append(
            "SILENT NOTIFICATIONS (should_speak=False, injected as [notification])",
        )
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
_KIND_ORDER = {
    "proactive_decision": 0,
    "actor_tool_call": 1,
    "slow_brain": 2,
    "actor_notification": 3,
    "utterance": 4,
}


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

    for pd in timeline.proactive_decisions:
        sk = _time_sort_key(pd.ts_utc)
        merged.append((sk, "proactive_decision", pd))

    for atc in timeline.actor_tool_calls:
        sk = _time_sort_key(atc.ts_utc)
        merged.append((sk, "actor_tool_call", atc))

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
        elif kind == "proactive_decision":
            pd_lines = _format_proactive_decision(event, verbose=verbose)  # type: ignore[arg-type]
            parts.extend(pd_lines)
            parts.append("")
        elif kind == "actor_tool_call":
            atc_lines = _format_actor_tool_call(event, verbose=verbose)  # type: ignore[arg-type]
            parts.extend(atc_lines)
            parts.append("")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def collect_magnitude_traces_from_docker(
    container_name: str = "unity-desktop-sandbox",
    remote_dir: str = "/var/log/magnitude",
) -> tuple[dict[str, MagnitudeTrace], Path | None, Path | None]:
    """Copy magnitude traces and agent-service log from a running Docker container.

    Returns (traces_dict, local_dir, agent_service_log_path).
    """
    import subprocess
    import tempfile

    try:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", container_name],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0 or "true" not in result.stdout.lower():
            return {}, None, None
    except Exception:
        return {}, None, None

    local_dir = Path(tempfile.mkdtemp(prefix="mag_traces_"))
    try:
        subprocess.run(
            [
                "docker",
                "cp",
                f"{container_name}:{remote_dir}/acts",
                str(local_dir / "acts"),
            ],
            capture_output=True,
            timeout=30,
        )
    except Exception:
        pass

    agent_log_path = local_dir / "agent-service.log"
    try:
        subprocess.run(
            [
                "docker",
                "cp",
                f"{container_name}:/var/log/supervisor/agent-service.log",
                str(agent_log_path),
            ],
            capture_output=True,
            timeout=30,
        )
    except Exception:
        agent_log_path = None

    traces: dict[str, MagnitudeTrace] = {}
    if (local_dir / "acts").is_dir():
        traces = parse_magnitude_traces(local_dir)

    if agent_log_path and not agent_log_path.exists():
        agent_log_path = None

    return traces, local_dir, agent_log_path


def _attach_unmatched_magnitude_traces(
    actor_notifications: list[ActorNotification],
    unmatched: list[MagnitudeTrace],
) -> None:
    """Attach fast-path magnitude traces to web_act/desktop_act ACTOR STARTED notifications by task match."""
    started_notifs = [
        n
        for n in actor_notifications
        if n.notification_type == "started"
        and ("web_act" in n.content or "desktop_act" in n.content)
    ]
    for trace in unmatched:
        task_lower = trace.task.lower()
        for notif in started_notifs:
            if (
                task_lower in notif.content.lower()
                or notif.content.lower().find(task_lower[:30]) >= 0
            ):
                notif.magnitude_traces.append(trace)
                break


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
        "--magnitude-dir",
        type=Path,
        default=None,
        help="Path to magnitude log directory (contains acts/ with act_trace.json files)",
    )
    parser.add_argument(
        "--agent-service-log",
        type=Path,
        default=None,
        help="Path to agent-service supervisor log (for direct-action traces)",
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

    mag_traces: dict[str, MagnitudeTrace] = {}
    if args.magnitude_dir:
        mag_traces = parse_magnitude_traces(args.magnitude_dir)
        if cm_data and mag_traces:
            matched_ids: set[str] = set()
            for atc in cm_data.actor_tool_calls:
                if atc.event_type != "execute_code":
                    continue
                exec_match = re.search(r"execute_code\((\w+)\)", atc.tool_name)
                if exec_match:
                    exec_id = exec_match.group(1)
                    if exec_id in mag_traces:
                        atc.magnitude_trace = mag_traces[exec_id]
                        matched_ids.add(exec_id)
            unmatched = [
                t
                for k, t in mag_traces.items()
                if k not in matched_ids and not t.lineage
            ]
            if unmatched:
                _attach_unmatched_magnitude_traces(
                    data.actor_notifications,
                    unmatched,
                )

    if args.agent_service_log and args.agent_service_log.exists() and cm_data:
        direct_groups = parse_agent_service_log(args.agent_service_log)
        if direct_groups:
            exec_codes = [
                atc
                for atc in cm_data.actor_tool_calls
                if atc.event_type == "execute_code"
                and not atc.magnitude_trace
                and atc.result_summary not in ("",)
                and not (atc.result_summary or "").startswith("ERROR:")
            ]
            group_idx = 0
            for atc in exec_codes:
                if group_idx >= len(direct_groups):
                    break
                if atc.code and (
                    "session.click" in atc.code
                    or "session.type_text" in atc.code
                    or "session.scroll" in atc.code
                    or "session.drag" in atc.code
                ):
                    atc.direct_actions = direct_groups[group_idx]
                    group_idx += 1

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
