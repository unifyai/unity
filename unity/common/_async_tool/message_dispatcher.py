from .loop_config import LoopConfig
from .timeout_timer import TimeoutTimer
from .event_bus_util import to_event_bus
import unillm


class LoopMessageDispatcher:
    """
    A helper class for appending messages to an async tool loop client
    while also publishing them to the event bus and ensuring that the timer is reset.
    """

    def __init__(
        self,
        client: unillm.AsyncUnify,
        cfg: LoopConfig,
        timer: TimeoutTimer,
    ):
        self._client = client
        self._cfg = cfg
        self._timer = timer

    async def append_msgs(
        self,
        msgs: list[dict],
        origin: str | None = None,
        *,
        skip_event_bus: bool = False,
    ) -> None:
        self._client.append_messages(msgs)
        if not skip_event_bus:
            await to_event_bus(msgs, self._cfg, origin=origin)
        self._timer.reset()

    async def publish_to_event_bus(
        self,
        msgs: list[dict],
        origin: str | None = None,
    ) -> None:
        """Publish messages to EventBus without appending to client transcript.

        Used when a message already exists in the transcript (e.g., placeholder
        that was updated in-place) and just needs to be published to EventBus.
        """
        await to_event_bus(msgs, self._cfg, origin=origin)
