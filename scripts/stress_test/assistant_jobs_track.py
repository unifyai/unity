from dotenv import load_dotenv
import os

load_dotenv()
import unify

unify.activate("AssistantJobs", api_key=os.getenv("SHARED_UNIFY_KEY"))
jobs = unify.get_logs(
    context="startup_events",
    api_key=os.getenv("SHARED_UNIFY_KEY"),
    filter="running == 'true'",
    limit=100,
)
for idx, job in enumerate(jobs):
    print("--------------------------------")
    job_name = ""
    if "job_name" in job.entries:
        job_name = job.entries.get("job_name")
    assistant_id = job.entries.get("assistant_id")
    print(f"{idx+1}. {job_name} --> {assistant_id}")
