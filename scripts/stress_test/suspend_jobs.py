import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dev"))

from job_utils import _admin_key, _comms_url, fetch_running_jobs

namespace = os.getenv("UNITY_NAMESPACE", "staging")
comms_url = _comms_url(namespace)
admin_key = _admin_key()

if not comms_url or not admin_key:
    print("Error: UNITY_COMMS_URL and ORCHESTRA_ADMIN_KEY must be set")
    exit(1)

jobs = fetch_running_jobs(namespace)
print(f"Found {len(jobs)} running job(s)\n")

for idx, job in enumerate(jobs):
    job_name = job.get("job_name", "")
    assistant_id = job.get("assistant_id", "")
    print("--------------------------------")
    print(f"{idx+1}. {job_name} --> {assistant_id}")

    if not job_name:
        print("     ⚠ no job_name — skipping")
        continue

    try:
        resp = requests.post(
            f"{comms_url}/infra/job/stop",
            data={"job_name": job_name, "namespace": namespace},
            headers={"Authorization": f"Bearer {admin_key}"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        print(f"     ✅ suspended: {data.get('message', data)}")
    except requests.exceptions.HTTPError as e:
        print(f"     ❌ failed ({e.response.status_code}): {e.response.text}")
    except requests.RequestException as e:
        print(f"     ❌ unreachable: {e}")
