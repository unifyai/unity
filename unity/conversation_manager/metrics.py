"""Unity telemetry metric definitions (OpenTelemetry SDK).

All metric instruments are defined at module level using the global
MeterProvider. They become active once ``init_metrics()`` is called
from ``metrics_push.py``. Until then they are harmless no-ops (the
default NoOpMeterProvider handles them).

Metrics land in GCP Managed Prometheus as ``prometheus.googleapis.com/*``
and are queryable with PromQL via the Google Cloud Monitoring data source
in Grafana — the same path used by communication/adapters metrics.
"""

from __future__ import annotations

from opentelemetry import metrics

meter = metrics.get_meter("unity", version="0.1.0")

# ---------------------------------------------------------------------------
# U1  Container spin-up time
# ---------------------------------------------------------------------------
container_spinup = meter.create_histogram(
    name="unity_container_spinup_seconds",
    description="Time from container start (entrypoint.sh) to first idle ping.",
    unit="s",
)

# ---------------------------------------------------------------------------
# U2  Manager init total
# ---------------------------------------------------------------------------
manager_init_total = meter.create_histogram(
    name="unity_manager_init_seconds",
    description="Total duration of init_conv_manager().",
    unit="s",
)

# ---------------------------------------------------------------------------
# U3  Per-manager init
# ---------------------------------------------------------------------------
per_manager_init = meter.create_histogram(
    name="unity_per_manager_init_seconds",
    description="Init time per manager step.",
    unit="s",
)

# ---------------------------------------------------------------------------
# U9  Session duration
# ---------------------------------------------------------------------------
session_duration = meter.create_histogram(
    name="unity_session_duration_seconds",
    description="Total assistant session duration (startup to shutdown).",
    unit="s",
)

# ---------------------------------------------------------------------------
# X1  Running job count (cluster-wide snapshot)
# ---------------------------------------------------------------------------
running_job_count = meter.create_gauge(
    name="unity_running_job_count",
    description=(
        "Number of assistant jobs with running==True at the moment a "
        "StartupEvent is received or a session shuts down."
    ),
)
