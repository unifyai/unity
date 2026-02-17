"""Metrics push lifecycle: init, periodic export, and shutdown flush.

Call ``init_metrics()`` early in the process to activate the GCP
Monitoring exporter. Background export happens automatically every 5 s
via the ``PeriodicExportingMetricReader``.  Call ``flush_metrics()``
before exit to push any remaining data.

In test mode or when GCP credentials are absent the exporter is *not*
created — metric instruments still exist but record into a no-op provider,
adding zero overhead.
"""

from __future__ import annotations

import logging
import os

from opentelemetry import metrics
from opentelemetry.exporter.cloud_monitoring import CloudMonitoringMetricsExporter
from opentelemetry.resourcedetector.gcp_resource_detector import (
    GoogleCloudResourceDetector,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

_provider: MeterProvider | None = None


def init_metrics() -> None:
    """Initialise the OTel MeterProvider with the GCP Monitoring exporter.

    Skipped automatically when:
    - ``TEST`` env var is set (unit-test runs)
    - ``GOOGLE_APPLICATION_CREDENTIALS`` is not set (local dev without GCP)
    """
    global _provider

    if os.getenv("TEST"):
        print("[metrics] Metrics export disabled (test mode)")
        return

    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        print("[metrics] Metrics export disabled (no GCP credentials)")
        return

    # Surface export errors that would otherwise be swallowed by the
    # background PeriodicExportingMetricReader thread.
    logging.getLogger("opentelemetry.exporter.cloud_monitoring").setLevel(logging.DEBUG)
    logging.getLogger("opentelemetry.sdk.metrics").setLevel(logging.DEBUG)

    # Detect GKE resource attributes (cluster, namespace, pod, container)
    # so the exporter maps data points to the correct Cloud Monitoring
    # monitored resource type.  Without this, writes are silently rejected.
    resource = GoogleCloudResourceDetector().detect()
    print(f"[metrics] Detected GCP resource: {resource.attributes}")

    exporter = CloudMonitoringMetricsExporter()
    reader = PeriodicExportingMetricReader(
        exporter,
        export_interval_millis=5_000,  # 5 seconds — short for ephemeral GKE jobs
    )
    _provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(_provider)
    print("[metrics] GMP metrics export initialised (5 s interval)")


def flush_metrics() -> None:
    """Force-flush all pending metrics (call before exit)."""
    if _provider is not None:
        try:
            _provider.force_flush(timeout_millis=5_000)
            print("[metrics] Final metrics flushed")
        except Exception as exc:
            print(f"[metrics] Flush failed: {exc}")


def shutdown_metrics() -> None:
    """Shut down the metrics provider (flushes + releases resources)."""
    if _provider is not None:
        try:
            _provider.shutdown()
            print("[metrics] Metrics provider shut down")
        except Exception as exc:
            print(f"[metrics] Shutdown failed: {exc}")
