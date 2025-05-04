import json
from task_list_manager.types.task import Task

UPDATE = f"""
Your task is to update the list of tasks based on the plain-text request from the user, and you should continue using the tools available until you are satisfied that the list of tasks has been updated correctly.

As a recap, the schema for the table which stores the list of tasks is as follows:

{json.dumps(Task.model_json_schema(), indent=4)}
"""
