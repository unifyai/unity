from dataclasses import dataclass

@dataclass
class State:
    _task_running: bool = False
    _task_paused: bool = False
    _last_task_result: str = ""
    _last_step_results: list[str] = []

    def set_task_running(self, running: bool):
        self._task_running = running

    def set_task_paused(self, paused: bool):
        self._task_paused = paused

    def set_last_task_result(self, result: str):
        self._last_task_result = result

    def add_last_step_results(self, results: list[str]):
        self._last_step_results.append(results)
    
    def reset_last_step_results(self):
        self._last_step_results = []

    @property
    def task_running(self) -> bool:
        return self._task_running

    @property
    def task_paused(self) -> bool:
        return self._task_paused

    @property
    def last_task_result(self) -> str:
        return self._last_task_result

    @property
    def last_step_results(self) -> list[str]:
        return self._last_step_results
