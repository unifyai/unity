from typing import Union, Dict, List
from .loop_config import LoopConfig


# ── small helper: publish to the EventBus (if configured) ──────────────
async def to_event_bus(
    messages: Union[Dict, List[Dict]],
    loop_cfg: LoopConfig,
    origin: str | None = None,
) -> None:
    """
    Emit *messages* to the shared EventBus (if configured).

    Every `ToolLoop` event now carries **both** the raw chat *message*
    and the *public method* that spawned the loop so downstream
    subscribers can easily group / filter events.

    Uses the typed ToolLoopPayload model for schema consistency.
    """
    # Import lazily to avoid import-time cycles with `unity.events.event_bus`.
    from ...events.event_bus import (
        Event,
        EVENT_BUS,
    )  # local import to break circular dependency
    from ...events.types.tool_loop import ToolLoopPayload

    if not EVENT_BUS:
        return
    if isinstance(messages, dict):
        messages = [messages]
    _lookup = loop_cfg.tool_alias_lookup
    for message in messages:
        _aliases = None
        if _lookup and isinstance(message, dict):
            _tool_calls = message.get("tool_calls") or []
            _sparse = {}
            for tc in _tool_calls:
                name = (tc.get("function") or {}).get("name", "")
                if name in _lookup:
                    _sparse[name] = _lookup[name]
            _aliases = _sparse or None

        payload = ToolLoopPayload(
            message=message,
            method=loop_cfg.loop_id,
            hierarchy=list(loop_cfg.lineage),
            hierarchy_label=loop_cfg.label,
            origin=origin,
            tool_aliases=_aliases,
        )
        await EVENT_BUS.publish(
            Event(type="ToolLoop", payload=payload),
        )
