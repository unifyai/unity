"""
Log parsers for Unity session data.

Handles four log formats:
  - unillm:  LLM request/response text files
  - unify:   Orchestra API call JSON files
  - unity:   Framework debug log (unity.log)
  - cloud:   Cloud Logging plain text
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# unillm log parsing (LLM calls)
# ---------------------------------------------------------------------------

_HEADER_RE = re.compile(r"\[([^\]]+)\] LLM request")
_CACHE_RE = re.compile(r"LLM response.*\[cache: (\w+)\]")
_USAGE_RE = re.compile(r"LLM response.*\[usage\]")
_LEADING_SPACES_RE = re.compile(r"^ {8,}", re.MULTILINE)


def _reescape_json_newlines(text: str) -> str:
    """Re-escape literal newlines inside JSON string values."""
    out: list[str] = []
    in_str = False
    esc = False
    for ch in text:
        if esc:
            out.append(ch)
            esc = False
            continue
        if ch == "\\" and in_str:
            out.append(ch)
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
        if in_str and ch == "\n":
            out.append("\\n")
            continue
        if in_str and ch == "\r":
            out.append("\\r")
            continue
        if in_str and ch == "\t":
            out.append("\\t")
            continue
        out.append(ch)
    return "".join(out)


def _extract_json_blocks(content: str) -> list[dict]:
    """Extract top-level JSON objects from mixed text by brace-counting."""
    blocks: list[dict] = []
    i = 0
    while i < len(content):
        if content[i] == "{" and (i == 0 or content[i - 1] == "\n"):
            depth, in_str, esc, j = 0, False, False, i
            while j < len(content):
                ch = content[j]
                if esc:
                    esc = False
                    j += 1
                    continue
                if ch == "\\" and in_str:
                    esc = True
                    j += 1
                    continue
                if ch == '"':
                    in_str = not in_str
                if not in_str:
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            raw = content[i : j + 1]
                            try:
                                blocks.append(json.loads(_reescape_json_newlines(raw)))
                            except json.JSONDecodeError:
                                pass
                            i = j
                            break
                j += 1
        i += 1
    return blocks


def _clean_content(text: str) -> str:
    """Strip excessive leading whitespace from content lines."""
    return _LEADING_SPACES_RE.sub("    ", text)


def cache_status_from_filename(name: str) -> str:
    if ".cache_hit." in name:
        return "hit"
    if ".cache_miss." in name:
        return "miss"
    if ".cache_pending." in name or ".pending." in name:
        return "pending"
    if "_usage" in name:
        return "usage"
    return "none"


def origin_from_filename(name: str) -> str:
    base = name
    for ext in (
        ".cache_hit.txt",
        ".cache_miss.txt",
        ".cache_pending.txt",
        ".pending.txt",
        ".txt",
    ):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    parts = base.split("_", 2)
    return parts[2] if len(parts) >= 3 else ""


def time_from_filename(name: str) -> str:
    parts = name.split("_", 2)
    t = parts[0] if parts else ""
    if len(t) >= 6:
        return f"{t[:2]}:{t[2:4]}:{t[4:6]}"
    return t


def list_llm_calls(unillm_dir: Path) -> list[dict]:
    """Return lightweight metadata for all LLM call log files."""
    if not unillm_dir.is_dir():
        return []
    files = sorted(unillm_dir.glob("*.txt"), key=lambda p: p.name)
    result = []
    for f in files:
        entry: dict[str, Any] = {
            "filename": f.name,
            "origin": origin_from_filename(f.name),
            "time": time_from_filename(f.name),
            "cache_status": cache_status_from_filename(f.name),
        }
        try:
            head = f.read_text(encoding="utf-8", errors="replace")[:2000]
            hm = _HEADER_RE.search(head)
            if hm:
                hdr = hm.group(1)
                if " :: " in hdr:
                    entry["origin_full"], entry["model"] = hdr.split(" :: ", 1)
                else:
                    entry["model"] = hdr
        except Exception:
            pass
        result.append(entry)
    return result


def parse_llm_call(file_path: Path) -> dict[str, Any]:
    """Fully parse a single unillm log file."""
    content = file_path.read_text(encoding="utf-8", errors="replace")
    result: dict[str, Any] = {
        "filename": file_path.name,
        "origin": origin_from_filename(file_path.name),
        "time": time_from_filename(file_path.name),
        "cache_status": cache_status_from_filename(file_path.name),
        "model": None,
        "request": None,
        "response": None,
        "messages": [],
        "usage": None,
    }

    hm = _HEADER_RE.search(content)
    if hm:
        hdr = hm.group(1)
        if " :: " in hdr:
            result["origin"], result["model"] = hdr.split(" :: ", 1)
        else:
            result["model"] = hdr

    cm = _CACHE_RE.search(content)
    if cm:
        result["cache_status"] = cm.group(1)
    elif _USAGE_RE.search(content):
        result["cache_status"] = "usage"

    blocks = _extract_json_blocks(content)
    if len(blocks) >= 1:
        result["request"] = blocks[0]
    if len(blocks) >= 2:
        result["response"] = blocks[1]

    messages: list[dict] = []
    if result["request"]:
        for msg in result["request"].get("messages", []):
            messages.append(_simplify_message(msg, is_response=False))

    if result["response"]:
        choices = result["response"].get("choices", [])
        if choices:
            resp_msg = choices[0].get("message")
            if resp_msg:
                messages.append(_simplify_message(resp_msg, is_response=True))
        result["usage"] = result["response"].get("usage")

    result["messages"] = messages
    # Drop raw request/response to keep payloads manageable
    result.pop("request", None)
    result.pop("response", None)
    return result


def _simplify_message(msg: dict, *, is_response: bool) -> dict:
    """Flatten a message for the frontend."""
    role = msg.get("role", "unknown")
    content = msg.get("content", "")

    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif part.get("type") == "image_url":
                    text_parts.append("[image]")
        content = "\n".join(text_parts)

    if isinstance(content, str):
        content = _clean_content(content)

    tool_calls = msg.get("tool_calls")
    tool_call_id = msg.get("tool_call_id")

    reasoning = msg.get("reasoning_content") or ""
    thinking = (
        msg.get("thinking_blocks")
        or (msg.get("provider_specific_fields") or {}).get("thinking_blocks")
        or []
    )

    return {
        "role": role,
        "content": content if isinstance(content, str) else str(content),
        "is_response": is_response,
        "tool_calls": tool_calls,
        "tool_call_id": tool_call_id,
        "reasoning": reasoning,
        "thinking": thinking,
        "content_length": len(content) if isinstance(content, str) else 0,
    }


# ---------------------------------------------------------------------------
# unify API call parsing
# ---------------------------------------------------------------------------


def list_api_calls(unify_dir: Path) -> list[dict]:
    """Parse unify API call JSON log files."""
    if not unify_dir.is_dir():
        return []
    files = sorted(unify_dir.glob("*.json"), key=lambda p: p.name)
    result = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8", errors="replace"))
            result.append(
                {
                    "filename": f.name,
                    "timestamp": data.get("timestamp", ""),
                    "method": data.get("method", ""),
                    "route": data.get("route", ""),
                    "url": data.get("url", ""),
                    "status_code": (data.get("response") or {}).get("status_code"),
                    "duration_ms": data.get("duration_ms"),
                    "status": data.get("status", ""),
                },
            )
        except Exception:
            continue
    return result


def parse_api_call(file_path: Path) -> dict[str, Any]:
    """Parse a single unify API call JSON file with full request/response."""
    data = json.loads(file_path.read_text(encoding="utf-8", errors="replace"))
    req = data.get("request") or {}
    resp = data.get("response") or {}

    body_preview = resp.get("body_preview") or resp.get("body") or ""
    if isinstance(body_preview, (dict, list)):
        body_preview = json.dumps(body_preview, indent=2)
    if len(str(body_preview)) > 5000:
        body_preview = str(body_preview)[:5000] + "..."

    req_body = req.get("json") or req.get("data") or ""
    if isinstance(req_body, (dict, list)):
        req_body = json.dumps(req_body, indent=2)
    if len(str(req_body)) > 5000:
        req_body = str(req_body)[:5000] + "..."

    return {
        "filename": file_path.name,
        "timestamp": data.get("timestamp", ""),
        "method": data.get("method", ""),
        "route": data.get("route", ""),
        "url": data.get("url", ""),
        "status_code": resp.get("status_code"),
        "duration_ms": data.get("duration_ms"),
        "status": data.get("status", ""),
        "request_params": req.get("params") or {},
        "request_body": req_body,
        "response_body": body_preview,
    }


# ---------------------------------------------------------------------------
# Cloud logging parsing
# ---------------------------------------------------------------------------

_CLOUD_EVENT_RE = re.compile(
    r"^(\d{2}:\d{2}:\d{2}\.\d{3})\s+" r"[^\[]*" r"\[([^\]]+)\]\s+" r"(.*)",
    re.MULTILINE,
)
_USER_MSG_RE = re.compile(r"Message from (.+?):\s+(.+)")


def parse_cloud_log(cloud_log_path: Path) -> dict[str, Any]:
    """Parse cloud_logging.txt into structured events."""
    if not cloud_log_path.exists():
        return {"events": [], "user_messages": [], "errors": [], "raw_lines": 0}

    text = cloud_log_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    events: list[dict] = []
    user_messages: list[dict] = []
    errors: list[dict] = []

    for m in _CLOUD_EVENT_RE.finditer(text):
        ts, component, message = m.group(1), m.group(2), m.group(3).strip()
        level = "info"
        if "Error" in message or "error" in message or "FAILED" in message:
            level = "error"
        elif "WARN" in message or "warn" in message.lower():
            level = "warning"

        ev = {"time": ts, "component": component, "message": message, "level": level}
        events.append(ev)

        if level == "error":
            errors.append(ev)

        um = _USER_MSG_RE.search(message)
        if um:
            user_messages.append(
                {
                    "time": ts,
                    "from": um.group(1),
                    "message": um.group(2),
                },
            )

    return {
        "events": events,
        "user_messages": user_messages,
        "errors": errors,
        "raw_lines": len(lines),
    }


# ---------------------------------------------------------------------------
# Framework log parsing (unity.log)
# ---------------------------------------------------------------------------

_UNITY_LOG_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\s+" r"(\w+)\s+" r"(.*?)$",
    re.MULTILINE,
)
_MEMORY_RE = re.compile(r"\[(\d+/\d+ MiB \(\d+%\))\]")


def parse_framework_log(
    unity_log_path: Path,
    *,
    level: str | None = None,
    component: str | None = None,
    offset: int = 0,
    limit: int = 200,
) -> dict[str, Any]:
    """Parse unity.log with optional filtering and pagination."""
    if not unity_log_path.exists():
        return {"entries": [], "total": 0, "offset": offset, "limit": limit}

    text = unity_log_path.read_text(encoding="utf-8", errors="replace")
    all_entries: list[dict] = []

    for m in _UNITY_LOG_RE.finditer(text):
        ts, lvl, msg = m.group(1), m.group(2), m.group(3).strip()
        comp = ""
        comp_match = re.search(r"\[(\w+)\]", msg)
        if comp_match:
            comp = comp_match.group(1)

        mem = ""
        mem_match = _MEMORY_RE.search(msg)
        if mem_match:
            mem = mem_match.group(1)

        if level and lvl.upper() != level.upper():
            continue
        if component and comp.lower() != component.lower():
            continue

        all_entries.append(
            {
                "timestamp": ts,
                "level": lvl,
                "component": comp,
                "message": msg,
                "memory": mem,
            },
        )

    total = len(all_entries)
    page = all_entries[offset : offset + limit]
    return {"entries": page, "total": total, "offset": offset, "limit": limit}
