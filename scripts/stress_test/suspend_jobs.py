from dotenv import load_dotenv
import os

load_dotenv()
import requests
import unify

COMMS_URL = os.getenv("UNITY_COMMS_URL", "").rstrip("/")
ADMIN_KEY = os.getenv("ORCHESTRA_ADMIN_KEY", "")
NAMESPACE = os.getenv("UNITY_NAMESPACE", "staging")

api_key = os.getenv("SHARED_UNIFY_KEY")
unify.activate("AssistantJobs", api_key=api_key)
jobs = unify.get_logs(
    context="startup_events",
    api_key=api_key,
    filter="running == 'true'",
    limit=10,
)

print(f"Found {len(jobs)} running job(s)\n")

if not COMMS_URL or not ADMIN_KEY:
    print("Error: UNITY_COMMS_URL and ORCHESTRA_ADMIN_KEY must be set")
    exit(1)

for idx, job in enumerate(jobs):
    print("--------------------------------")
    job_name = job.entries.get("job_name", "")
    assistant_id = job.entries.get("assistant_id", "")
    print(f"{idx+1}. {job_name} --> {assistant_id}")

    if not job_name:
        print("     ⚠ no job_name — skipping")
        continue

    try:
        resp = requests.post(
            f"{COMMS_URL}/infra/job/stop",
            data={"job_name": job_name, "namespace": NAMESPACE},
            headers={"Authorization": f"Bearer {ADMIN_KEY}"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        print(f"     ✅ suspended: {data.get('message', data)}")
    except requests.exceptions.HTTPError as e:
        print(f"     ❌ failed ({e.response.status_code}): {e.response.text}")
    except requests.RequestException as e:
        print(f"     ❌ unreachable: {e}")
