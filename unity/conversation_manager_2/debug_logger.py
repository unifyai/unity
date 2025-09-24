from dotenv import load_dotenv

load_dotenv()
import os
import traceback
import unify


api_key = os.environ.get("SHARED_UNIFY_KEY")
if "Debug" not in unify.list_projects(api_key=api_key):
    unify.create_project("Debug", api_key=api_key)


def log_job_startup(
    job_name: str,
    timestamp: str,
    medium: str,
    user_id: str,
    assistant_id: str,
    user_name: str,
    assistant_name: str,
    user_number: str,
    user_whatsapp_number: str,
    assistant_number: str,
    user_email: str,
    assistant_email: str,
):
    try:
        unify.create_logs(
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
                "user_whatsapp_number": user_whatsapp_number,
                "assistant_number": assistant_number,
                "user_email": user_email,
                "assistant_email": assistant_email,
                "running": True,
            },
            api_key=api_key,
        )
        print("Logged Startup Event", job_name)
    except Exception as e:
        print(f"Error creating logs: {e}")
        traceback.print_exc()


def mark_job_done(job_name: str):
    try:
        job_log = unify.get_logs(
            project="Debug",
            context="startup_events",
            filter=f"job_name == '{job_name}'",
            api_key=api_key,
        )[0]
        job_log.update_entries(running=False)
        print("Job marked done", job_name)
    except Exception as e:
        print(f"Error finding job: {e}")
        traceback.print_exc()
