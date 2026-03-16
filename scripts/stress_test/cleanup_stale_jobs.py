from dotenv import load_dotenv
import os

load_dotenv()
import requests
import unify

ASSISTANT_IDS = []
# ASSISTANT_IDS = list(range(656, 666))

COMMS_URL = os.getenv("UNITY_COMMS_URL", "").rstrip("/")
ADMIN_KEY = os.getenv("ORCHESTRA_ADMIN_KEY", "")

unify.activate("AssistantJobs", api_key=os.getenv("SHARED_UNIFY_KEY"))

# cleanup stale jobs in general
if not ASSISTANT_IDS:
    jobs = unify.get_logs(
        context="startup_events",
        api_key=os.getenv("SHARED_UNIFY_KEY"),
        filter="running == 'true'",
        limit=100,
    )
    print(f"Cleaning up {len(jobs)} stale jobs in general")
# cleanup for specific assistants
else:
    filterString = " or ".join([f"assistant_id == '{a}'" for a in ASSISTANT_IDS])
    jobs = unify.get_logs(
        context="startup_events",
        api_key=os.getenv("SHARED_UNIFY_KEY"),
        filter=filterString,
        limit=10,
    )
    print(f"Cleaning up {len(jobs)} stale jobs for assistants {ASSISTANT_IDS}")

for idx, job in enumerate(jobs):
    print("--------------------------------")
    job_name = ""
    if "job_name" in job.entries:
        job_name = job.entries.get("job_name")
    assistant_id = job.entries.get("assistant_id")
    print(f"{idx+1}. {job_name} --> {assistant_id}")

    # mark job as done
    job.update_entries(running=False)

    # release vm
    if COMMS_URL and ADMIN_KEY and assistant_id:
        resp = requests.post(
            f"{COMMS_URL}/infra/vm/pool/release",
            json={"assistant_id": assistant_id},
            headers={"Authorization": f"Bearer {ADMIN_KEY}"},
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
                f"{COMMS_URL}/infra/vm/pool/disk/detach/{assistant_id}",
                headers={"Authorization": f"Bearer {ADMIN_KEY}"},
                timeout=60,
            )
            if detach_resp.ok:
                print(f"   Disk detached for {assistant_id}")
            else:
                print(
                    f"   Disk detach failed for {assistant_id}: {detach_resp.status_code} {detach_resp.text}",
                )
    else:
        print("   Skipping VM release (UNITY_COMMS_URL or ORCHESTRA_ADMIN_KEY not set)")
