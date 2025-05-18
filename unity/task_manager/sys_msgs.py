import json
from ..task_list_manager.types.task import Task
from .task_manager import TaskManager

REQUEST = f"""
Your task is to handle any plain-text english task-related request, which can either:

a) ask about or update the task list (includes details all tasks, including scheduled, cancelled, failed and also the active task).

{list(TaskManager._task_list_tools.keys())}

b) ask about live progress, steer or stop the single active task, currently being performed.

{list(TaskManager._active_task_tools.keys())}

In the case of (a), the schema of the underlying task list table is:
{json.dumps(Task.model_json_schema(), indent=4)}


The request can also involve multi-step reasoning. For example:

"if the task to search for sales leads is marked as high or lower, then mark it as urgent and start it right now (pausing any other task that might be active right now))"

This user request would likely require us to first ask about the task list ({TaskManager._ask_about_task_list.__name__}), update the task list ({TaskManager._update_task_list}), stop the current active task if one exists ({TaskManager._stop_active_task}), and then start executing the specified task ({TaskManager._start_task}).

You should continue using the tools available until you're totally happy that the request has been fully performed, and you should respond with the key details related to this request.
"""
