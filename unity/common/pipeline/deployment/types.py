from __future__ import annotations

from typing import Any, Callable, Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, Field

from .._utils import utc_now_iso

__all__ = [
    "DeploymentArtifactKind",
    "DeploymentBundle",
    "DeploymentBundleArtifact",
    "DeploymentBundleRef",
    "DeploymentBundleRuntimeConfig",
    "DeploymentBundleStore",
    "DeploymentExecutionTarget",
    "DeploymentIdentity",
    "DeploymentIngestionExecutor",
    "DeploymentIngestionJob",
    "DeploymentIngestionOutcome",
    "DeploymentIngestionStageStatus",
    "DeploymentJobState",
    "DeploymentJobStore",
    "DeploymentObservabilityRefs",
    "DeploymentQueuePayload",
    "DeploymentRunMode",
    "DispatchManifest",
    "StageReporter",
]

DeploymentArtifactKind = Literal[
    "actor_config",
    "seed_data",
    "integration_asset",
    "data_pipeline_config",
    "source_data",
    "deployment_spec",
    "function_dir",
    "data_dir",
]
DeploymentStoreBackend = Literal["local", "gcs"]
DeploymentExecutorBackend = Literal["local", "cloud_tasks"]
DeploymentQueueBackend = Literal["in_memory", "pubsub", "cloud_tasks"]
DeploymentRunMode = Literal["file_manager", "data_manager", "hybrid"]
DeploymentExecutionTarget = Literal["local", "local_with_gcp", "staging", "production"]
DeploymentJobState = Literal["queued", "running", "success", "error", "cancelled"]


class DeploymentBundleArtifact(BaseModel):
    """Typed manifest entry for one artifact referenced by a deployment bundle."""

    artifact_id: str = Field(default_factory=lambda: uuid4().hex)
    kind: DeploymentArtifactKind
    logical_name: str
    source_path: str
    artifact_uri: str | None = None
    is_directory: bool = False
    media_type: str | None = None
    size_bytes: int | None = None
    checksum: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DeploymentIdentity(BaseModel):
    """Stable identity for a deployment bundle submission."""

    client: str
    deployment: str
    project: str | None = None
    environment: str | None = None
    tenant_id: str | None = None


class DeploymentBundle(BaseModel):
    """Typed manifest describing an offline deployment data-dump ingest."""

    bundle_id: str = Field(default_factory=lambda: uuid4().hex)
    deployment_identity: DeploymentIdentity
    actor_config: DeploymentBundleArtifact | None = None
    seed_data: list[DeploymentBundleArtifact] = Field(default_factory=list)
    integration_assets: list[DeploymentBundleArtifact] = Field(default_factory=list)
    data_pipeline_config: DeploymentBundleArtifact | None = None
    source_artifact_manifest: list[DeploymentBundleArtifact] = Field(
        default_factory=list,
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)


class DeploymentBundleRef(BaseModel):
    """Reference to a persisted bundle manifest."""

    bundle_id: str
    manifest_path: str


class DeploymentObservabilityRefs(BaseModel):
    """Operator-facing pointers to logs and ledgers for one job run."""

    progress_file: str | None = None
    log_file: str | None = None
    run_ledger_path: str | None = None
    cost_ledger_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DeploymentIngestionStageStatus(BaseModel):
    """Status for one named stage inside a deployment ingestion job."""

    stage_name: str
    status: DeploymentJobState
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DeploymentIngestionOutcome(BaseModel):
    """Typed result returned by a deployment ingestion executor callback."""

    cost_ledger_path: str | None = None
    observability_refs: DeploymentObservabilityRefs = Field(
        default_factory=DeploymentObservabilityRefs,
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class DeploymentIngestionJob(BaseModel):
    """Tracked execution record for one deployment bundle ingest."""

    job_id: str = Field(default_factory=lambda: uuid4().hex)
    dispatch_id: str = ""
    bundle_ref: DeploymentBundleRef
    run_mode: DeploymentRunMode
    execution_target: DeploymentExecutionTarget = "local"
    status: DeploymentJobState = "queued"
    stage_status: dict[str, DeploymentIngestionStageStatus] = Field(
        default_factory=dict,
    )
    created_at: str = Field(default_factory=utc_now_iso)
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    cancelled_at: str | None = None
    cancel_reason: str | None = None
    cost_ledger_path: str | None = None
    observability_refs: DeploymentObservabilityRefs = Field(
        default_factory=DeploymentObservabilityRefs,
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class DispatchManifest(BaseModel):
    """Batch-level index linking all jobs spawned by a single dispatch invocation.

    Written to ``dispatches/<dispatch_id>/manifest.json`` in GCS by
    both ``dispatch_pipeline.py`` and ``pipeline_control submit``.
    The ``list`` CLI scans these manifests to enumerate recent dispatches.
    """

    dispatch_id: str
    created_at: str = Field(default_factory=utc_now_iso)
    source: Literal["dispatch_pipeline", "pipeline_control"] = "dispatch_pipeline"
    mode: str = ""
    config_path: str = ""
    job_ids: list[str] = Field(default_factory=list)
    total_files: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class DeploymentQueuePayload(BaseModel):
    """Typed queue payload for one queued deployment ingestion job."""

    job_id: str
    bundle_id: str
    run_mode: DeploymentRunMode
    execution_target: DeploymentExecutionTarget = "local"


def _default_deployment_root() -> str:
    from pathlib import Path

    return str(Path.cwd() / ".deployments")


class DeploymentBundleRuntimeConfig(BaseModel):
    """Typed adapter/runtime selection for deployment bundle execution."""

    bundle_store_backend: DeploymentStoreBackend = "local"
    job_store_backend: DeploymentStoreBackend = "local"
    executor_backend: DeploymentExecutorBackend = "local"
    queue_backend: DeploymentQueueBackend = "in_memory"
    local_root_dir: str = Field(default_factory=_default_deployment_root)
    max_workers: int = Field(default=1, ge=1)


class DeploymentBundleStore(Protocol):
    """Port for persisting and retrieving deployment bundle manifests."""

    def write_bundle(self, bundle: DeploymentBundle) -> DeploymentBundleRef: ...

    def read_bundle(self, bundle_id: str) -> DeploymentBundle: ...


class DeploymentJobStore(Protocol):
    """Port for persisting and retrieving deployment ingestion jobs."""

    def upsert_job(self, job: DeploymentIngestionJob) -> str: ...

    def read_job(self, job_id: str) -> DeploymentIngestionJob: ...


StageReporter = Callable[
    [str, DeploymentJobState, str | None, dict[str, Any] | None],
    None,
]


class DeploymentIngestionExecutor(Protocol):
    """Callable used by the runner to execute one deployment bundle job."""

    def __call__(
        self,
        bundle: DeploymentBundle,
        report_stage: StageReporter,
    ) -> DeploymentIngestionOutcome | None: ...
