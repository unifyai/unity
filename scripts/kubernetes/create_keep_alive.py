#!/usr/bin/env python3
"""
Script to create a keep-alive deployment for GKE Autopilot.

This deployment ensures at least one node is always available to reduce
pod scheduling delays when new jobs are created.

Usage:
    python create_keep_alive.py --create
    python create_keep_alive.py --delete
"""

import argparse
import sys
from kubernetes import client, config
from kubernetes.client.rest import ApiException


def setup_kubernetes_client():
    """Initialize Kubernetes client using Google Cloud SDK"""
    try:
        import subprocess

        # Get cluster credentials using gcloud
        project_id = "responsive-city-458413-a2"
        region = "us-central1"
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
            return None

        print("✅ Got cluster credentials")
        config.load_kube_config()
        return client.AppsV1Api()

    except Exception as e:
        print(f"❌ Error setting up Kubernetes client: {e}")
        return None


def create_keep_alive_deployment(api_client, namespace="default"):
    """Create a minimal keep-alive deployment to maintain at least one node"""
    try:
        deployment_manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "unity-keep-alive",
                "namespace": namespace,
                "labels": {"app": "unity", "component": "keep-alive"},
            },
            "spec": {
                "replicas": 1,
                "selector": {
                    "matchLabels": {"app": "unity", "component": "keep-alive"},
                },
                "template": {
                    "metadata": {"labels": {"app": "unity", "component": "keep-alive"}},
                    "spec": {
                        "priorityClassName": "unity-low",  # Use low priority for keep-alive
                        "containers": [
                            {
                                "name": "keep-alive",
                                "image": "busybox:1.35",
                                "command": ["/bin/sh"],
                                "args": ["-c", "while true; do sleep 30; done"],
                                "resources": {
                                    "requests": {"cpu": "2", "memory": "8Gi"},
                                    "limits": {"cpu": "2", "memory": "8Gi"},
                                },
                            },
                        ],
                    },
                },
            },
        }

        # Check if deployment exists
        try:
            api_client.read_namespaced_deployment(
                name="unity-keep-alive",
                namespace=namespace,
            )
            print("✅ Keep-alive deployment already exists")
            return True
        except ApiException as e:
            if e.status == 404:
                # Create the deployment
                api_client.create_namespaced_deployment(
                    namespace=namespace,
                    body=deployment_manifest,
                )
                print("✅ Created keep-alive deployment")
                return True
            else:
                raise e

    except Exception as e:
        print(f"❌ Error creating keep-alive deployment: {e}")
        return False


def delete_keep_alive_deployment(api_client, namespace="default"):
    """Delete the keep-alive deployment"""
    try:
        api_client.delete_namespaced_deployment(
            name="unity-keep-alive",
            namespace=namespace,
        )
        print("✅ Deleted keep-alive deployment")
        return True

    except ApiException as e:
        if e.status == 404:
            print("⚠️  Keep-alive deployment not found (already deleted)")
            return True
        else:
            print(f"❌ Error deleting keep-alive deployment: {e}")
            return False


def check_keep_alive_status(api_client, namespace="default"):
    """Check the status of the keep-alive deployment"""
    try:
        deployment = api_client.read_namespaced_deployment(
            name="unity-keep-alive",
            namespace=namespace,
        )

        print(f"📊 Keep-alive deployment status:")
        print(f"   Name: {deployment.metadata.name}")
        print(f"   Replicas: {deployment.spec.replicas}")
        print(f"   Available: {deployment.status.available_replicas or 0}")
        print(f"   Ready: {deployment.status.ready_replicas or 0}")
        print(f"   Updated: {deployment.status.updated_replicas or 0}")

        return True

    except ApiException as e:
        if e.status == 404:
            print("❌ Keep-alive deployment not found")
            return False
        else:
            print(f"❌ Error checking keep-alive status: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Manage keep-alive deployment for GKE Autopilot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_keep_alive.py --create          # Create keep-alive deployment
  python create_keep_alive.py --delete          # Delete keep-alive deployment
  python create_keep_alive.py --status          # Check deployment status

Purpose:
  This deployment maintains at least one node in the cluster to reduce
  scheduling delays when new jobs are created.
        """,
    )

    parser.add_argument(
        "--create",
        action="store_true",
        help="Create keep-alive deployment",
    )

    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete keep-alive deployment",
    )

    parser.add_argument("--status", action="store_true", help="Check deployment status")

    parser.add_argument(
        "--namespace",
        default="default",
        help="Kubernetes namespace (default: default)",
    )

    args = parser.parse_args()

    print(f"🔧 Unity Keep-Alive Deployment Management")
    print(f"   Namespace: {args.namespace}")
    print()

    # Initialize Kubernetes client
    api_client = setup_kubernetes_client()
    if not api_client:
        print("❌ Failed to connect to Kubernetes cluster")
        sys.exit(1)

    # Handle different operations
    if args.create:
        if create_keep_alive_deployment(api_client, args.namespace):
            print("\n✅ Keep-alive deployment created successfully!")
            print(
                "\n💡 This deployment will maintain at least one node to reduce scheduling delays.",
            )
            print(
                "   Monitor with: kubectl get pods -n default -l app=unity,component=keep-alive",
            )
        else:
            sys.exit(1)
        return

    if args.delete:
        if delete_keep_alive_deployment(api_client, args.namespace):
            print("\n✅ Keep-alive deployment deleted successfully!")
        else:
            sys.exit(1)
        return

    if args.status:
        check_keep_alive_status(api_client, args.namespace)
        return

    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    main()
