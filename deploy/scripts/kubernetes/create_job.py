#!/usr/bin/env python3
"""
Script to create Kubernetes Jobs for Unity assistants.

Creates a job with assistant-specific environment variables.
All other config comes from cluster-wide ConfigMaps and Secrets.

Usage:
    python create_job.py --assistant-id ID --user-first-name NAME --user-number NUMBER
"""

import argparse
import sys
from kubernetes import client, config
from kubernetes.client.rest import ApiException


def setup_kubernetes_client():
    """Initialize Kubernetes client using Google Cloud SDK"""
    try:
        # Use Google Cloud SDK to get cluster credentials
        import subprocess
        import json
        import tempfile
        import os

        # Get cluster credentials using gcloud
        project_id = "responsive-city-458413-a2"
        region = "us-central1"  # Use region instead of zone
        cluster_name = "unity"

        print(f"🔗 Connecting to GKE cluster: {cluster_name}")

        # Run gcloud command to get cluster credentials
        result = subprocess.run(
            [
                "gcloud",
                "container",
                "clusters",
                "get-credentials",
                cluster_name,
                "--region",
                region,
                "--project",
                project_id,
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"❌ Failed to get cluster credentials: {result.stderr}")
            print("💡 Make sure you have:")
            print("   1. gcloud CLI installed")
            print("   2. Access to the GKE cluster")
            print("   3. GOOGLE_APPLICATION_CREDENTIALS set correctly")
            return None, None

        print("✅ Got cluster credentials")

        # Get the cluster endpoint and token directly
        cluster_info = subprocess.run(
            [
                "gcloud",
                "container",
                "clusters",
                "describe",
                cluster_name,
                "--region",
                region,
                "--project",
                project_id,
                "--format",
                "json",
            ],
            capture_output=True,
            text=True,
        )

        if cluster_info.returncode != 0:
            print(f"❌ Failed to get cluster info: {cluster_info.stderr}")
            return None, None

        cluster_data = json.loads(cluster_info.stdout)
        cluster_endpoint = cluster_data["endpoint"]
        cluster_ca_cert = cluster_data["masterAuth"]["clusterCaCertificate"]

        # Get access token using service account
        token_result = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            capture_output=True,
            text=True,
        )

        if token_result.returncode != 0:
            print(f"❌ Failed to get access token: {token_result.stderr}")
            return None, None

        access_token = token_result.stdout.strip()

        # Create a temporary kubeconfig
        kubeconfig = {
            "apiVersion": "v1",
            "kind": "Config",
            "clusters": [
                {
                    "name": "unity-cluster",
                    "cluster": {
                        "server": f"https://{cluster_endpoint}",
                        "certificate-authority-data": cluster_ca_cert,
                    },
                },
            ],
            "users": [{"name": "unity-user", "user": {"token": access_token}}],
            "contexts": [
                {
                    "name": "unity-context",
                    "context": {"cluster": "unity-cluster", "user": "unity-user"},
                },
            ],
            "current-context": "unity-context",
        }

        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            json.dump(kubeconfig, f)
            temp_config = f.name

        # Load the temporary config
        config.load_kube_config(config_file=temp_config)

        # Clean up
        os.unlink(temp_config)

        return client.BatchV1Api(), client.CoreV1Api()

    except Exception as e:
        print(f"❌ Error setting up Kubernetes client: {e}")
        return None, None


def check_job_exists(batch_api, assistant_id: str, namespace: str = "default"):
    """Check if a job for this assistant already exists and is running"""
    try:
        job_name = f"unity-{assistant_id}"
        job = batch_api.read_namespaced_job(name=job_name, namespace=namespace)

        # Check if job is active (has running pods)
        if job.status.active and job.status.active > 0:
            return True, "running"
        elif job.status.succeeded and job.status.succeeded > 0:
            return True, "completed"
        elif job.status.failed and job.status.failed > 0:
            return True, "failed"
        else:
            return True, "unknown"

    except ApiException as e:
        if e.status == 404:
            return False, None
        else:
            raise e


def create_unity_job(
    batch_api,
    assistant_id: str,
    user_first_name: str,
    user_surname: str = "",
    user_number: str = "",
    assistant_number: str = "",
    assistant_email: str = "",
    user_phone_number: str = "",
    user_email: str = "",
    namespace: str = "default",
    image: str = "us-central1-docker.pkg.dev/responsive-city-458413-a2/unity/unity:latest",
):
    """
    Create a Kubernetes Job for a Unity assistant.

    Args:
        batch_api: Kubernetes Batch API client
        assistant_id: Unique assistant identifier
        user_first_name: User's first name
        user_surname: User's surname
        user_number: User's phone number
        assistant_number: Assistant's phone number (optional)
        assistant_email: Assistant's email address (optional)
        user_phone_number: User's phone for calls (defaults to user_number)
        user_email: User's email address
        namespace: Kubernetes namespace
        image: Docker image to use
    """
    try:
        # Check if job already exists and is running
        exists, status = check_job_exists(batch_api, assistant_id, namespace)

        if exists and status == "running":
            print(f"✅ Assistant {assistant_id} is already running")
            return None
        elif exists and status in ["completed", "failed"]:
            print(f"🗑️  Cleaning up old job for {assistant_id} (status: {status})")
            delete_job(batch_api, assistant_id, namespace)

        # Create the job name with unity- prefix
        job_name = f"unity-{assistant_id}"

        # Define the job manifest
        job_manifest = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": job_name,
                "namespace": namespace,
                "labels": {
                    "app": "unity",
                    "assistant-id": assistant_id,
                    "created-by": "create_job_script",
                },
            },
            "spec": {
                "backoffLimit": 1,  # Allow 1 retry for resource issues
                "activeDeadlineSeconds": 7200,  # 2 hours max runtime
                "ttlSecondsAfterFinished": 0,  # Auto-delete job and pods after specified delay
                "template": {
                    "metadata": {
                        "labels": {"app": "unity", "assistant-id": assistant_id},
                    },
                    "spec": {
                        "restartPolicy": "Never",
                        "serviceAccountName": "comm-sa",
                        "terminationGracePeriodSeconds": 30,  # Faster termination
                        "containers": [
                            {
                                "name": "unity-assistant",
                                "image": image,
                                "imagePullPolicy": "IfNotPresent",  # Use cached images for faster startup
                                "ports": [
                                    {"containerPort": 8000},
                                    {"containerPort": 6379},
                                ],
                                "envFrom": [
                                    {"configMapRef": {"name": "unity-config"}},
                                    {"secretRef": {"name": "unity-secrets"}},
                                ],
                                "env": [
                                    # Assistant-specific environment variables
                                    {"name": "ASSISTANT_ID", "value": assistant_id},
                                    {
                                        "name": "USER_FIRST_NAME",
                                        "value": user_first_name,
                                    },
                                    {"name": "USER_SURNAME", "value": user_surname},
                                    {"name": "USER_EMAIL", "value": user_email},
                                    {
                                        "name": "ASSISTANT_EMAIL",
                                        "value": assistant_email,
                                    },
                                    {
                                        "name": "ASSISTANT_NUMBER",
                                        "value": assistant_number,
                                    },
                                    {"name": "USER_NUMBER", "value": user_number},
                                    {
                                        "name": "USER_PHONE_NUMBER",
                                        "value": user_phone_number or user_number,
                                    },
                                    {
                                        "name": "GCP_PROJECT_ID",
                                        "value": project_id,
                                    },
                                    {
                                        "name": "GOOGLE_APPLICATION_CREDENTIALS",
                                        "value": "/secrets/key.json",
                                    },
                                    # Startup optimizations
                                    {"name": "PYTHONUNBUFFERED", "value": "1"},
                                    {
                                        "name": "TOKENIZERS_PARALLELISM",
                                        "value": "false",
                                    },
                                    {"name": "OMP_NUM_THREADS", "value": "2"},
                                    {"name": "MKL_NUM_THREADS", "value": "2"},
                                ],
                                "resources": {
                                    "requests": {
                                        "cpu": "2",
                                        "memory": "8Gi",
                                        "ephemeral-storage": "100Gi",
                                    },
                                    "limits": {
                                        "cpu": "4",
                                        "memory": "16Gi",
                                        "ephemeral-storage": "100Gi",
                                    },
                                },
                                "volumeMounts": [
                                    {
                                        "name": "sa-key",
                                        "mountPath": "/secrets",
                                        "readOnly": True,
                                    },
                                ],
                            },
                        ],
                        "volumes": [
                            {"name": "sa-key", "secret": {"secretName": "comm-sa-key"}},
                        ],
                    },
                },
            },
        }

        # Create the job
        try:
            api_response = batch_api.create_namespaced_job(
                namespace=namespace,
                body=job_manifest,
            )

            print(f"✅ Job created successfully!")
            print(f"   Job name: {api_response.metadata.name}")
            print(f"   Job UID: {api_response.metadata.uid}")
            print(f"   Assistant ID: {assistant_id}")
            print(f"   Namespace: {namespace}")
            print(f"   Image: {image}")

            return api_response

        except ApiException as e:
            if e.status == 409:  # Conflict - job already exists
                print(f"⚠️  Job already exists: {job_name}")
                return None
            else:
                raise e

    except Exception as e:
        print(f"❌ Error creating job: {e}")
        return None


def get_job_status(batch_api, core_api, assistant_id: str, namespace: str = "default"):
    """Get the status of a Unity job and its pods"""
    try:
        job_name = f"unity-{assistant_id}"

        api_response = batch_api.read_namespaced_job(name=job_name, namespace=namespace)

        status = {
            "job_name": api_response.metadata.name,
            "assistant_id": assistant_id,
            "namespace": namespace,
            "creation_timestamp": (
                api_response.metadata.creation_timestamp.isoformat()
                if api_response.metadata.creation_timestamp
                else None
            ),
            "active": api_response.status.active or 0,
            "succeeded": api_response.status.succeeded or 0,
            "failed": api_response.status.failed or 0,
            "conditions": [],
        }

        # Add conditions if they exist
        if api_response.status.conditions:
            for condition in api_response.status.conditions:
                status["conditions"].append(
                    {
                        "type": condition.type,
                        "status": condition.status,
                        "reason": condition.reason,
                        "message": condition.message,
                        "last_transition_time": (
                            condition.last_transition_time.isoformat()
                            if condition.last_transition_time
                            else None
                        ),
                    },
                )

        # Get pod information
        pods = core_api.list_namespaced_pod(
            namespace=namespace,
            label_selector=f"job-name={job_name}",
        )

        status["pods"] = []
        for pod in pods.items:
            pod_status = {
                "name": pod.metadata.name,
                "phase": pod.status.phase,
                "ready": pod.status.ready,
                "scheduled": pod.status.conditions is not None
                and any(
                    cond.type == "PodScheduled" and cond.status == "True"
                    for cond in pod.status.conditions
                ),
            }

            # Add scheduling issues
            if pod.status.conditions:
                for condition in pod.status.conditions:
                    if condition.type == "PodScheduled" and condition.status == "False":
                        pod_status["scheduling_issue"] = {
                            "reason": condition.reason,
                            "message": condition.message,
                        }

            status["pods"].append(pod_status)

        return status

    except ApiException as e:
        if e.status == 404:
            print(f"❌ Job not found: unity-{assistant_id}")
            return None
        else:
            print(f"❌ Error getting job status: {e}")
            return None


def delete_job(batch_api, assistant_id: str, namespace: str = "default"):
    """Delete a Unity job"""
    try:
        job_name = f"unity-{assistant_id}"

        api_response = batch_api.delete_namespaced_job(
            name=job_name,
            namespace=namespace,
            propagation_policy="Background",  # Delete pods as well
        )

        print(f"✅ Job deleted successfully: {job_name}")
        return True

    except ApiException as e:
        if e.status == 404:
            print(f"⚠️  Job not found (already deleted): unity-{assistant_id}")
            return True
        else:
            print(f"❌ Error deleting job: {e}")
            return False


def cleanup_completed_jobs(
    batch_api,
    namespace: str = "default",
    max_age_hours: int = 24,
    force_all: bool = False,
):
    """Clean up completed jobs older than max_age_hours"""
    try:
        import datetime

        jobs = batch_api.list_namespaced_job(
            namespace=namespace,
            label_selector="app=unity",
        )

        cutoff_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
            hours=max_age_hours,
        )
        cleaned_count = 0

        for job in jobs.items:
            assistant_id = job.metadata.labels.get("assistant-id", "unknown")

            # Check if job is completed or failed
            if (
                job.status.succeeded or job.status.failed
            ) and job.metadata.creation_timestamp:
                # Clean up if force_all or job is older than cutoff
                if force_all or job.metadata.creation_timestamp < cutoff_time:
                    try:
                        delete_job(batch_api, assistant_id, namespace)
                        cleaned_count += 1
                    except Exception as e:
                        print(f"   ❌ Failed to delete {job.metadata.name}: {e}")

        if cleaned_count > 0:
            print(f"✅ Cleaned up {cleaned_count} old jobs")
        else:
            print("✅ No old jobs to clean up")

    except Exception as e:
        print(f"❌ Error cleaning up jobs: {e}")


def list_jobs(batch_api, namespace: str = "default"):
    """List all Unity jobs in the namespace"""
    try:
        jobs = batch_api.list_namespaced_job(
            namespace=namespace,
            label_selector="app=unity",
        )

        print(f"📋 Unity Jobs in namespace '{namespace}':")
        print()

        for job in jobs.items:
            assistant_id = job.metadata.labels.get("assistant-id", "unknown")
            status = "Unknown"

            if job.status.active:
                status = "Running"
            elif job.status.succeeded:
                status = "Completed"
            elif job.status.failed:
                status = "Failed"

            print(f"   - {job.metadata.name}")
            print(f"     Assistant ID: {assistant_id}")
            print(f"     Status: {status}")
            print(f"     Created: {job.metadata.creation_timestamp}")
            print()

        if not jobs.items:
            print("   (no jobs found)")

    except Exception as e:
        print(f"❌ Error listing jobs: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Create Kubernetes Jobs for Unity assistants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_job.py --assistant-id test --user-first-name "John" --user-surname "Doe" --user-number "+1234567890"
  python create_job.py --assistant-id test --cleanup-delay 600          # 10 minute cleanup delay
  python create_job.py --assistant-id test --status                     # Check job status
  python create_job.py --assistant-id test --delete                     # Delete job
  python create_job.py --list                                          # List all jobs
  python create_job.py --cleanup                                       # Clean up old jobs
  python create_job.py --cleanup-all                                   # Clean up ALL completed jobs
  python pubsub_trigger.py --assistant-id test --user-first-name "John" --user-number "+1234567890"
        """,
    )

    parser.add_argument(
        "--assistant-id",
        required=False,
        help="Assistant ID (required for create/status/delete)",
    )

    parser.add_argument(
        "--user-first-name",
        help="User's first name",
    )
    parser.add_argument(
        "--user-surname",
        default="",
        help="User's surname",
    )

    parser.add_argument(
        "--user-number",
        # required=True,
        help="User's phone number",
    )

    parser.add_argument(
        "--assistant-number",
        default="",
        help="Assistant's phone number (optional)",
    )

    parser.add_argument(
        "--user-phone-number",
        default="",
        help="User's phone number for calls (defaults to user-number)",
    )

    parser.add_argument(
        "--namespace",
        default="default",
        help="Kubernetes namespace (default: default)",
    )

    parser.add_argument(
        "--image",
        default="us-central1-docker.pkg.dev/responsive-city-458413-a2/unity/unity:latest",
        help="Docker image to use",
    )

    parser.add_argument(
        "--cleanup-delay",
        type=int,
        default=300,
        help="Seconds to wait after job completion before auto-cleanup (default: 300 = 5 minutes)",
    )

    parser.add_argument("--status", action="store_true", help="Get job status")

    parser.add_argument("--delete", action="store_true", help="Delete job")

    parser.add_argument("--list", action="store_true", help="List all Unity jobs")

    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up completed jobs older than 24 hours",
    )

    parser.add_argument(
        "--cleanup-all",
        action="store_true",
        help="Clean up ALL completed jobs immediately",
    )

    parser.add_argument(
        "--max-age-hours",
        type=int,
        default=24,
        help="Maximum age in hours for cleanup (default: 24)",
    )

    args = parser.parse_args()

    print(f"🔧 Unity Kubernetes Job Management")
    print(f"   Namespace: {args.namespace}")
    print()

    # Initialize Kubernetes client
    batch_api, core_api = setup_kubernetes_client()
    if not batch_api or not core_api:
        print("❌ Failed to connect to Kubernetes cluster")
        sys.exit(1)
    print("✅ Connected to Kubernetes cluster")

    # Handle different operations
    if args.list:
        list_jobs(batch_api, args.namespace)
        return

    if args.cleanup:
        cleanup_completed_jobs(batch_api, args.namespace, args.max_age_hours)
        return

    if args.cleanup_all:
        cleanup_completed_jobs(batch_api, args.namespace, force_all=True)
        return

    if args.status:
        if not args.assistant_id:
            print("❌ --assistant-id is required for --status")
            sys.exit(1)

        status = get_job_status(batch_api, core_api, args.assistant_id, args.namespace)
        if status:
            print(f"📊 Job Status:")
            print(f"   Job Name: {status['job_name']}")
            print(f"   Assistant ID: {status['assistant_id']}")
            print(f"   Active: {status['active']}")
            print(f"   Succeeded: {status['succeeded']}")
            print(f"   Failed: {status['failed']}")
            print(f"   Created: {status['creation_timestamp']}")

            if status["conditions"]:
                print(f"   Conditions:")
                for condition in status["conditions"]:
                    print(f"     - {condition['type']}: {condition['status']}")
                    if condition["reason"]:
                        print(f"       Reason: {condition['reason']}")
                    if condition["message"]:
                        print(f"       Message: {condition['message']}")

            if status["pods"]:
                print(f"   Pods:")
                for pod in status["pods"]:
                    print(f"     - {pod['name']}: {pod['phase']}")
                    if "scheduling_issue" in pod:
                        issue = pod["scheduling_issue"]
                        print(
                            f"       ❌ Scheduling issue: {issue['reason']} - {issue['message']}",
                        )
            else:
                print(f"   Pods: (none created yet)")
        return

    if args.delete:
        if not args.assistant_id:
            print("❌ --assistant-id is required for --delete")
            sys.exit(1)

        success = delete_job(batch_api, args.assistant_id, args.namespace)
        if not success:
            sys.exit(1)
        return

    # Create job (default operation)
    if not args.assistant_id:
        print("❌ --assistant-id is required")
        sys.exit(1)

    print(f"🎯 Creating job for assistant: {args.assistant_id}")

    job = create_unity_job(
        batch_api=batch_api,
        assistant_id=args.assistant_id,
        user_first_name=args.user_first_name,
        user_surname=args.user_surname,
        user_number=args.user_number,
        assistant_number=args.assistant_number,
        assistant_email=args.assistant_email,
        user_phone_number=args.user_phone_number,
        user_email=args.user_email,
        namespace=args.namespace,
        image=args.image,
        cleanup_delay_seconds=args.cleanup_delay,
    )

    if job:
        print("\n✅ Job created successfully!")
        print("\n💡 Next steps:")
        print(
            f"   1. Check job status: python create_job.py --assistant-id {args.assistant_id} --status",
        )
        print(
            f"   2. View pod logs: kubectl logs -n {args.namespace} -l job-name={job.metadata.name}",
        )
        print(
            f"   3. Delete job: python create_job.py --assistant-id {args.assistant_id} --delete",
        )
    else:
        print("❌ Failed to create job")
        sys.exit(1)


if __name__ == "__main__":
    main()
