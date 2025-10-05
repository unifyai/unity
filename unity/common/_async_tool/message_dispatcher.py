from .loop_config import LoopConfig
from .timeout_timer import TimeoutTimer
from .event_bus_util import to_event_bus
import unify
from ...constants import LOGGER


class LoopMessageDispatcher:
    """
    A helper class for appending messages to an async tool loop client
    while also publishing them to the event bus and ensuring that the timer is reset.
    """

    def __init__(
        self,
        client: unify.AsyncUnify,
        cfg: LoopConfig,
        timer: TimeoutTimer,
    ):
        self._client = client
        self._cfg = cfg
        self._timer = timer

    async def append_msgs(
        self,
        msgs: list[dict],
    ) -> None:
        self._client.append_messages(msgs)

        # Hard-coded debug logs for investigation – summarize appended messages and transcript
        try:

            def _summarize(m: dict) -> str:
                role = m.get("role")
                if role == "assistant":
                    tcs = m.get("tool_calls")
                    if tcs:
                        names = []
                        try:
                            for tc in tcs:
                                fn = (tc.get("function", {}) or {}).get("name")
                                if isinstance(fn, str):
                                    names.append(fn)
                        except Exception:
                            pass
                        return f"assistant tool_calls={names}"
                    return "assistant (no tool_calls)"
                if role == "tool":
                    return f"tool name={m.get('name')} id={m.get('tool_call_id')}"
                if role == "system":
                    c = m.get("content") or ""
                    return f"system content={str(c)[:80]}"
                if role == "user":
                    c = m.get("content") or ""
                    return f"user content={str(c)[:80]}"
                return str({k: m.get(k) for k in ("role", "name")})

            appended = ", ".join(_summarize(m) for m in msgs)
            LOGGER.info(f"dispatcher: append count={len(msgs)} -> [{appended}]")

            snapshot = "; ".join(
                f"#{i} " + _summarize(m) for i, m in enumerate(self._client.messages)
            )
            LOGGER.info(f"dispatcher: transcript snapshot: {snapshot}")
        except Exception:
            pass

        await to_event_bus(msgs, self._cfg)
        self._timer.reset()
