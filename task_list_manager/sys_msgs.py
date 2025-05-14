from datetime import datetime, timezone
import json
from task_list_manager.types.task import Task

UPDATE = f"""
Your task is to update the list of tasks based on the plain-text request from the user, and you should continue using the tools available until you are satisfied that the list of tasks has been updated correctly.

The current date and time is {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}.

As a recap, the schema for the table which stores the list of tasks is as follows:

{json.dumps(Task.model_json_schema(), indent=4)}
"""

ASK = f"""
You are a helpful assistant specialising in answering questions about a task list.
You have access to a number of read-only tools that let you inspect the current state
of tasks.  Use these tools as needed, step-by-step, until you are confident you
can answer the user's question accurately.  Once done, respond with the final
answer only (no additional commentary).

The current date and time is {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}.

The schema for each task row is:

{json.dumps(Task.model_json_schema(), indent=4)}
"""
