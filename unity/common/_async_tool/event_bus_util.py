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
    for message in messages:
        payload = ToolLoopPayload(
            message=message,
            method=loop_cfg.loop_id,
            hierarchy=list(loop_cfg.lineage),
            hierarchy_label=loop_cfg.label,
            origin=origin,
        )
        await EVENT_BUS.publish(
            Event(type="ToolLoop", payload=payload),
        )
