#!/usr/bin/env python3
"""
Script to set up Kubernetes PriorityClasses for Unity jobs.

This is a one-time setup that creates priority classes for different job priorities.
PriorityClasses are cluster-wide resources that determine scheduling order.

Usage:
    python setup_priority_classes.py --create
    python setup_priority_classes.py --list
    python setup_priority_classes.py --delete
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
        return client.SchedulingV1Api()

    except Exception as e:
        print(f"❌ Error setting up Kubernetes client: {e}")
        return None


def create_priority_classes(api_client):
    """Create priority classes for Unity jobs"""

    priority_classes = [
        {
            "name": "unity-critical",
            "value": 1000000,
            "description": "Critical priority for urgent Unity assistant jobs",
        },
        {
            "name": "unity-high",
            "value": 500000,
            "description": "High priority for important Unity assistant jobs",
        },
        {
            "name": "unity-normal",
            "value": 100000,
            "description": "Normal priority for standard Unity assistant jobs",
        },
        {
            "name": "unity-low",
            "value": 50000,
            "description": "Low priority for background Unity jobs",
        },
    ]

    created_count = 0

    for pc in priority_classes:
        try:
            priority_class_manifest = {
                "apiVersion": "scheduling.k8s.io/v1",
                "kind": "PriorityClass",
                "metadata": {
                    "name": pc["name"],
                    "labels": {"app": "unity", "priority": pc["name"].split("-")[-1]},
                },
                "value": pc["value"],
                "globalDefault": False,
                "description": pc["description"],
            }

            # Check if PriorityClass exists
            try:
                api_client.read_priority_class(name=pc["name"])
                print(f"✅ PriorityClass '{pc['name']}' already exists")
            except ApiException as e:
                if e.status == 404:
                    # Create the PriorityClass
                    api_client.create_priority_class(body=priority_class_manifest)
                    print(
                        f"✅ Created PriorityClass '{pc['name']}' (value: {pc['value']})",
                    )
                    created_count += 1
                else:
                    print(f"❌ Error checking PriorityClass '{pc['name']}': {e}")
                    continue

        except Exception as e:
            print(f"❌ Error creating PriorityClass '{pc['name']}': {e}")
            continue

    return created_count


def list_priority_classes(api_client):
    """List all priority classes in the cluster"""
    try:
        priority_classes = api_client.list_priority_class()

        print("📋 PriorityClasses in cluster:")
        print()

        if not priority_classes.items:
            print("   No PriorityClasses found")
            return

        for pc in priority_classes.items:
            print(f"   Name: {pc.metadata.name}")
            print(f"   Value: {pc.value}")
            print(f"   Global Default: {pc.global_default}")
            print(f"   Description: {pc.description}")
            print(f"   Labels: {pc.metadata.labels}")
            print()

    except Exception as e:
        print(f"❌ Error listing PriorityClasses: {e}")


def delete_priority_classes(api_client):
    """Delete Unity priority classes"""
    unity_priority_classes = [
        "unity-critical",
        "unity-high",
        "unity-normal",
        "unity-low",
    ]
    deleted_count = 0

    for pc_name in unity_priority_classes:
        try:
            api_client.delete_priority_class(name=pc_name)
            print(f"✅ Deleted PriorityClass '{pc_name}'")
            deleted_count += 1
        except ApiException as e:
            if e.status == 404:
                print(f"⚠️  PriorityClass '{pc_name}' not found (already deleted)")
            else:
                print(f"❌ Error deleting PriorityClass '{pc_name}': {e}")

    return deleted_count


def main():
    parser = argparse.ArgumentParser(
        description="Set up Kubernetes PriorityClasses for Unity jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_priority_classes.py --create          # Create all priority classes
  python setup_priority_classes.py --list            # List all priority classes
  python setup_priority_classes.py --delete          # Delete Unity priority classes

Priority Levels:
  unity-critical (1000000) - For urgent jobs that need immediate scheduling
  unity-high     (500000)  - For important jobs with high priority
  unity-normal   (100000)  - For standard jobs (default)
  unity-low      (50000)   - For background jobs that can wait

Usage in Jobs:
  spec:
    priorityClassName: unity-high  # Reference by name
        """,
    )

    parser.add_argument(
        "--create",
        action="store_true",
        help="Create Unity priority classes",
    )

    parser.add_argument("--list", action="store_true", help="List all priority classes")

    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete Unity priority classes",
    )

    args = parser.parse_args()

    print(f"🔧 Unity PriorityClass Management")
    print()

    # Initialize Kubernetes client
    api_client = setup_kubernetes_client()
    if not api_client:
        print("❌ Failed to connect to Kubernetes cluster")
        sys.exit(1)

    # Handle different operations
    if args.create:
        print("🚀 Creating Unity PriorityClasses...")
        created_count = create_priority_classes(api_client)
        print(f"\n✅ Created {created_count} new PriorityClasses")
        print("\n💡 Next steps:")
        print("   1. Use in jobs: priorityClassName: unity-high")
        print("   2. Check status: python setup_priority_classes.py --list")
        return

    if args.list:
        list_priority_classes(api_client)
        return

    if args.delete:
        print("🗑️  Deleting Unity PriorityClasses...")
        deleted_count = delete_priority_classes(api_client)
        print(f"\n✅ Deleted {deleted_count} PriorityClasses")
        return

    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    main()
