"""Smoke test: push a single metric to GCP Cloud Monitoring.

Can run locally (hardcodes GKE resource attributes) or inside a GKE pod.

    GOOGLE_APPLICATION_CREDENTIALS=gcp_sa_key.json uv run scripts/test_metrics_push.py
"""

import logging
import random
import socket
import time

from opentelemetry import metrics
from opentelemetry.exporter.cloud_monitoring import CloudMonitoringMetricsExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource

logging.basicConfig(level=logging.DEBUG)

# Hardcode the resource attributes that GKE metadata server + our manual
# attributes would produce.  This lets us test the full write path locally.
resource = Resource.create(
    {
        "cloud.provider": "gcp",
        "cloud.account.id": "responsive-city-458413-a2",
        "cloud.platform": "gcp_kubernetes_engine",
        "cloud.region": "us-central1",
        "k8s.cluster.name": "unity",
        "k8s.namespace.name": "staging",
        "k8s.pod.name": f"test-metrics-{socket.gethostname()}",
        "k8s.container.name": "unity-assistant",
    },
)
print(f"Resource: {resource.attributes}")

# Set up exporter
exporter = CloudMonitoringMetricsExporter()
reader = PeriodicExportingMetricReader(exporter, export_interval_millis=10_000)
provider = MeterProvider(resource=resource, metric_readers=[reader])
metrics.set_meter_provider(provider)
print("MeterProvider configured")

# Record a test metric
meter = metrics.get_meter("test", version="0.1.0")
test_histogram = meter.create_histogram(
    name="unity_test_metric",
    description="Smoke test metric — safe to delete.",
    unit="s",
)

value = round(random.uniform(1.0, 20.0), 2)
test_histogram.record(value)
print(f"Recorded test value: {value}s")

# Wait for one export cycle
print("Waiting 15s for export cycle...")
time.sleep(15)

print("Shutting down...")
provider.shutdown()
print(
    "Done. Check Metrics Explorer for 'unity_test_metric' under Kubernetes Container.",
)
