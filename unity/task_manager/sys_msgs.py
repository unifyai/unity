import json
from ..task_scheduler.types.task import Task

ASK = f"""
Your task is to handle any plain-text english task-related question, which can either ask about:

a) the task list (includes details all tasks, including scheduled, cancelled, failed and also the active task).

b) stored contact details of anyone we've previously corresponded with, or we need to correspond with

c) transcripts from any of the previous conversations with anyone

d) live progress on the single active task, currently being performed.

In the case of (a), the schema of the underlying task list table is:
{json.dumps(Task.model_json_schema(), indent=4)}


The question can also involve multi-step reasoning. For example:

"If the task to search for sales leads is marked as high or urgent, then give me the task description, otherwise give me the description of the highest priority task."

This user question would likely require us to make multiple queries of the task list, for example.

You should continue using the tools available until you're totally happy that the question has been fully answered, and you should respond with the key details related to this question.
"""

REQUEST = f"""
Your task is to handle any plain-text english task-related request, which can either:

a) update the task list (includes details all tasks, including scheduled, cancelled, failed and also the active task).

b) update the stored contact details of anyone we've previously corresponded with, or we need to correspond with

c) summarize any of the exchanges you've had, making it easier to then search for information via vector similarity search etc.

d) steer or stop the single active task, currently being performed.

In the case of (a), the schema of the underlying task list table is:
{json.dumps(Task.model_json_schema(), indent=4)}


The request can also involve multi-step reasoning. For example:

"if the task to search for sales leads is marked as high or lower, then mark it as urgent and start it right now (pausing any other task that might be active right now))"

This user request would likely require us to first ask about the task list, update the task list, stop the current active task if one exists, and then start executing the specified task.

You should continue using the tools available until you're totally happy that the request has been fully performed, and you should respond with the key details related to this request.
"""
