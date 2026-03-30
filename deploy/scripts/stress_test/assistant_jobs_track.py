import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dev"))

from job_utils import fetch_running_jobs

namespace = os.getenv("UNITY_NAMESPACE", "staging")
jobs = fetch_running_jobs(namespace)

print(f"Found {len(jobs)} running job(s)\n")

for idx, job in enumerate(jobs):
    job_name = job.get("job_name", "?")
    assistant_id = job.get("assistant_id", "?")
    print("--------------------------------")
    print(f"{idx+1}. {job_name} --> {assistant_id}")
