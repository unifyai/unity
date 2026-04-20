from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["LocalDeploymentRuntime", "build_local_deployment_runtime"]

from .local import (
    LocalDeploymentBundleStore,
    LocalDeploymentIngestionRunner,
    LocalDeploymentJobStore,
    LocalQueuedDeploymentCoordinator,
)
from .types import DeploymentBundleRuntimeConfig
from ..work_queue import InMemoryWorkQueue, WorkQueue


@dataclass(frozen=True)
class LocalDeploymentRuntime:
    """Resolved local deployment-ingestion runtime objects."""

    config: DeploymentBundleRuntimeConfig
    bundle_store: LocalDeploymentBundleStore
    job_store: LocalDeploymentJobStore
    work_queue: WorkQueue
    runner: LocalDeploymentIngestionRunner
    queued_coordinator: LocalQueuedDeploymentCoordinator


def build_local_deployment_runtime(
    config: DeploymentBundleRuntimeConfig,
) -> LocalDeploymentRuntime:
    """Build a local-first deployment runtime from typed adapter config."""

    if config.bundle_store_backend != "local":
        raise NotImplementedError(
            "Only local deployment bundle stores are implemented in this repo.",
        )
    if config.job_store_backend != "local":
        raise NotImplementedError(
            "Only local deployment job stores are implemented in this repo.",
        )
    if config.executor_backend != "local":
        raise NotImplementedError(
            "Only local deployment executors are implemented in this repo.",
        )
    if config.queue_backend != "in_memory":
        raise NotImplementedError(
            "Only in-memory deployment queues are implemented in this repo.",
        )

    root_dir = Path(config.local_root_dir).expanduser().resolve()
    bundle_store = LocalDeploymentBundleStore(root_dir)
    job_store = LocalDeploymentJobStore(root_dir)
    work_queue = InMemoryWorkQueue()
    runner = LocalDeploymentIngestionRunner(
        bundle_store=bundle_store,
        job_store=job_store,
        max_workers=config.max_workers,
    )
    queued_coordinator = LocalQueuedDeploymentCoordinator(
        queue=work_queue,
        runner=runner,
    )
    return LocalDeploymentRuntime(
        config=config,
        bundle_store=bundle_store,
        job_store=job_store,
        work_queue=work_queue,
        runner=runner,
        queued_coordinator=queued_coordinator,
    )
