from task_manager.task_manager import TaskManager

REQUEST = f"""
Your task is to handle any plain-text english task-related request, which can refer to inactive (cancelled, scheduled, failed) or active tasks. It can involve updates to those tasks and/or simply answering questions about them. It can also involve multi-step reasoning.

For example:

"if the task to search for sales leads is marked as high or lower, then mark it as urgent and start it right now (pausing any other task that might be active right now))"

This user request would likely require us to first ask about the task list ({TaskManager.request.__name__}), update the task list ({TaskManager.request.__name__}), and then execute that task ({TaskManager.request.__name__}).
"""
