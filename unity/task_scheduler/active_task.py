import asyncio
import functools
from typing import Optional

from ..common.llm_helpers import SteerableToolHandle
from ..planner.base import BasePlanner
from ..planner.simulated import SimulatedPlanner


class ActiveTask(SteerableToolHandle):

    def __init__(
        self,
        description: str,
        planner: Optional[BasePlanner] = SimulatedPlanner(1),
    ) -> None:
        self._plan = planner.plan(description)

    @functools.wraps(SteerableToolHandle.interject, updated=())
    async def interject(self, message: str) -> None:
        # The underlying plan’s `interject` is synchronous, so dispatch it to a
        # worker thread to avoid blocking the event loop.
        await asyncio.to_thread(self._plan.interject, message)

    @functools.wraps(SteerableToolHandle.stop, updated=())
    def stop(self) -> None:
        # `stop` may return a string for some plans, but the steering contract
        # expects `None`, so we discard any return value.
        self._plan.stop()

    @functools.wraps(SteerableToolHandle.pause, updated=())
    def pause(self) -> None:
        self._plan.pause()

    @functools.wraps(SteerableToolHandle.resume, updated=())
    def resume(self) -> None:
        self._plan.resume()

    @functools.wraps(SteerableToolHandle.done, updated=())
    def done(self) -> bool:
        return self._plan.done()

    @functools.wraps(SteerableToolHandle.result, updated=())
    async def result(self) -> str:
        return await self._plan.result()
