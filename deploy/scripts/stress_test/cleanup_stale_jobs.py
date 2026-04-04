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
BINDING_ID_LABEL = "assistantsession.unify.ai/binding-id"


def _suspend_job(job_name: str) -> None:
    """Suspend the Kubernetes Job via Comms."""

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


def _release_binding_runtime(
    *,
    job_name: str,
    assistant_id: str,
    binding_id: str,
) -> None:
    """Request binding-scoped runtime release via Comms."""

    try:
        resp = requests.post(
            f"{comms_url}/infra/vm/pool/release",
            json={
                "assistant_id": assistant_id,
                "binding_id": binding_id,
                "job_name": job_name,
            },
            headers={"Authorization": f"Bearer {admin_key}"},
            timeout=60,
        )
        if not resp.ok:
            print(
                f"   ❌ VM release request failed for {assistant_id}: {resp.status_code} {resp.text}",
            )
            return

        payload = resp.json()
        if payload.get("released"):
            print(f"   ✅ VM release requested for {assistant_id} ({binding_id})")
        else:
            print(
                f"   ⚠️ VM release skipped for {assistant_id} ({binding_id}): "
                f"{payload.get('message') or payload}",
            )
    except requests.RequestException as e:
        print(f"   ❌ VM release request failed for {assistant_id}: {e}")


def cleanup_jobs(jobs: list[dict]) -> None:
    """Suspend stale jobs and request binding-scoped runtime release when safe."""

    print(f"Found {len(jobs)} running job(s) to clean up\n")

    for idx, job in enumerate(jobs):
        job_name = str(job.get("job_name", "") or "")
        assistant_id = str(job.get("assistant_id", "") or "")
        labels = job.get("labels", {}) or {}
        binding_id = str(labels.get(BINDING_ID_LABEL, "") or "")
        print("--------------------------------")
        print(f"{idx+1}. {job_name} --> {assistant_id}")

        _suspend_job(job_name)

        if assistant_id and binding_id:
            _release_binding_runtime(
                job_name=job_name,
                assistant_id=assistant_id,
                binding_id=binding_id,
            )
        elif assistant_id:
            print("   Skipping VM release (missing binding-id label)")
        else:
            print("   Skipping VM release (no assistant_id)")


def main() -> int:
    """Run the stale-job cleanup flow."""

    if not comms_url or not admin_key:
        print("Error: UNITY_COMMS_URL and ORCHESTRA_ADMIN_KEY must be set")
        return 1

    cleanup_jobs(fetch_running_jobs(namespace))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
