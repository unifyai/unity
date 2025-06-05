import asyncio
import functools
from typing import Optional, Dict, Callable

from ..planner.base import BasePlanner, BasePlan
from ..planner.simulated import SimulatedPlanner


class ActiveTask(BasePlan):

    def __init__(
        self,
        description: str,
        planner: Optional[BasePlanner] = SimulatedPlanner(1),
    ) -> None:
        self._plan = planner.plan(description)

    @functools.wraps(BasePlan.ask, updated=())
    async def ask(self, message: str) -> str:
        await asyncio.to_thread(self._plan.ask, message)

    @functools.wraps(BasePlan.interject, updated=())
    async def interject(self, message: str) -> str:
        await asyncio.to_thread(self._plan.interject, message)

    @functools.wraps(BasePlan.stop, updated=())
    def stop(self) -> Optional[str]:
        return self._plan.stop()

    @functools.wraps(BasePlan.pause, updated=())
    def pause(self) -> Optional[str]:
        return self._plan.pause()

    @functools.wraps(BasePlan.resume, updated=())
    def resume(self) -> Optional[str]:
        return self._plan.resume()

    @functools.wraps(BasePlan.done, updated=())
    def done(self) -> bool:
        return self._plan.done()

    @functools.wraps(BasePlan.result, updated=())
    async def result(self) -> str:
        return await self._plan.result()

    @property
    @functools.wraps(BasePlan.valid_tools, updated=())
    def valid_tools(self) -> Dict[str, Callable]:
        tools = {
            self.interject.__name__: self.interject,
            self.stop.__name__: self.stop,
        }
        if self._paused:
            tools[self.resume.__name__] = self.resume
        else:
            tools[self.pause.__name__] = self.pause
        return tools
