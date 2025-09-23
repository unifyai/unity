from typing import Union, Dict, List
from .loop_config import LoopConfig
from ...events.event_bus import Event, EVENT_BUS


# ── small helper: publish to the EventBus (if configured) ──────────────
async def to_event_bus(
    messages: Union[Dict, List[Dict]],
    loop_cfg: LoopConfig,
) -> None:
    """
    Emit *messages* to the shared EventBus (if configured).

    Every `ToolLoop` event now carries **both** the raw chat *message*
    and the *public method* that spawned the loop so downstream
    subscribers can easily group / filter events.
    """
    if not EVENT_BUS:
        return
    if isinstance(messages, dict):
        messages = [messages]
    for message in messages:
        await EVENT_BUS.publish(
            Event(
                type="ToolLoop",
                payload={
                    "message": message,
                    "method": loop_cfg.loop_id,
                    "hierarchy": list(loop_cfg.lineage),
                    "hierarchy_label": loop_cfg.label,
                },
            ),
        )
