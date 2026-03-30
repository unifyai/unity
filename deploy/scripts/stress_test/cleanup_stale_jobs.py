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
print(f"Found {len(jobs)} running job(s) to clean up\n")

for idx, job in enumerate(jobs):
    job_name = job.get("job_name", "")
    assistant_id = job.get("assistant_id", "")
    print("--------------------------------")
    print(f"{idx+1}. {job_name} --> {assistant_id}")

    # Suspend the job via comms
    try:
        resp = requests.post(
            f"{comms_url}/infra/job/stop",
            data={"job_name": job_name, "namespace": namespace},
            headers={"Authorization": f"Bearer {admin_key}"},
            timeout=30,
        )
        resp.raise_for_status()
        print(f"   ✅ Job suspended")
    except requests.RequestException as e:
        print(f"   ❌ Job suspend failed: {e}")

    # Release VM
    if assistant_id:
        resp = requests.post(
            f"{comms_url}/infra/vm/pool/release",
            json={"assistant_id": assistant_id},
            headers={"Authorization": f"Bearer {admin_key}"},
            timeout=60,
        )
        if resp.ok and resp.json().get("released"):
            print(f"   VM released for {assistant_id}")
        else:
            print(
                f"   VM release failed for {assistant_id}: {resp.status_code} {resp.text}",
            )
            print(f"   Detaching disk for {assistant_id}, just in case...")
            detach_resp = requests.post(
                f"{comms_url}/infra/vm/pool/disk/detach/{assistant_id}",
                headers={"Authorization": f"Bearer {admin_key}"},
                timeout=60,
            )
            if detach_resp.ok:
                print(f"   Disk detached for {assistant_id}")
            else:
                print(
                    f"   Disk detach failed for {assistant_id}: {detach_resp.status_code} {detach_resp.text}",
                )
    else:
        print("   Skipping VM release (no assistant_id)")
