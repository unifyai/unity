import os
import unify


LOGGER = None


def _get_logger():
    global LOGGER

    api_key = os.environ.get("ORCHESTRA_API_KEY")
    if "Debug" not in unify.list_projects(api_key=api_key):
        unify.create_project("Debug", api_key=api_key)

    if LOGGER is None:
        LOGGER = unify.AsyncLoggerManager(
            name="DebugLogger", num_consumers=1, api_key=api_key,
        )
    return LOGGER


async def log_message(
    job_name: str,
    timestamp: str,
    medium: str,
    user_id: str,
    assistant_id: str,
    user_name: str,
    assistant_name: str,
    user_number: str,
    user_phone_call_number: str,
    assistant_number: str,
):
    _get_logger().log_create(
        project="Debug",
        context="startup_events",
        params={},
        entries={
            "job_name": job_name,
            "timestamp": timestamp,
            "medium": medium,
            "user_id": user_id,
            "assistant_id": assistant_id,
            "user_name": user_name,
            "assistant_name": assistant_name,
            "user_number": user_number,
            "user_phone_call_number": user_phone_call_number,
            "assistant_number": assistant_number,
        },
    )
    _get_logger().join()
