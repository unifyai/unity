from __future__ import annotations

import asyncio
import shutil
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

__all__ = [
    "LocalDeploymentBundleStore",
    "LocalDeploymentIngestionRunner",
    "LocalDeploymentJobStore",
    "LocalQueuedDeploymentCoordinator",
]

from .types import (
    DeploymentBundle,
    DeploymentBundleArtifact,
    DeploymentBundleRef,
    DeploymentBundleStore,
    DeploymentExecutionTarget,
    DeploymentIngestionExecutor,
    DeploymentIngestionJob,
    DeploymentIngestionOutcome,
    DeploymentIngestionStageStatus,
    DeploymentJobState,
    DeploymentJobStore,
    DeploymentQueuePayload,
    DeploymentRunMode,
)
from .._utils import utc_now_iso
from ..work_queue import LocalQueueWorker, WorkQueue


class LocalDeploymentBundleStore:
    """Filesystem-backed store for typed deployment bundle manifests."""

    def __init__(self, root_dir: str | Path):
        self._root = Path(root_dir).expanduser().resolve()
        self._bundles_dir = self._root / "deployment-bundles"
        self._artifacts_dir = self._root / "deployment-bundle-artifacts"
        self._bundles_dir.mkdir(parents=True, exist_ok=True)
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)

    def bundle_path(self, bundle_id: str) -> Path:
        return self._bundles_dir / f"{bundle_id}.json"

    def artifact_dir(self, bundle_id: str) -> Path:
        return self._artifacts_dir / bundle_id

    def prepare_bundle(self, bundle: DeploymentBundle) -> DeploymentBundle:
        prepared = bundle.model_copy(deep=True)
        self.artifact_dir(bundle.bundle_id).mkdir(parents=True, exist_ok=True)

        artifact_groups = [
            [prepared.actor_config] if prepared.actor_config is not None else [],
            prepared.seed_data,
            prepared.integration_assets,
            (
                [prepared.data_pipeline_config]
                if prepared.data_pipeline_config is not None
                else []
            ),
            prepared.source_artifact_manifest,
        ]
        for group in artifact_groups:
            for artifact in group:
                self._stage_artifact(prepared.bundle_id, artifact)
        return prepared

    def write_bundle(self, bundle: DeploymentBundle) -> DeploymentBundleRef:
        path = self.bundle_path(bundle.bundle_id)
        path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
        return DeploymentBundleRef(bundle_id=bundle.bundle_id, manifest_path=str(path))

    def read_bundle(self, bundle_id: str) -> DeploymentBundle:
        path = self.bundle_path(bundle_id)
        return DeploymentBundle.model_validate_json(path.read_text(encoding="utf-8"))

    def _stage_artifact(
        self,
        bundle_id: str,
        artifact: DeploymentBundleArtifact,
    ) -> None:
        source = Path(artifact.source_path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Bundle artifact not found: {source}")

        target = self.artifact_dir(bundle_id) / artifact.logical_name
        if source.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

        artifact.artifact_uri = target.as_uri()
        artifact.is_directory = target.is_dir()
        if not artifact.size_bytes and target.is_file():
            artifact.size_bytes = target.stat().st_size


class LocalDeploymentJobStore:
    """Filesystem-backed store for deployment ingestion job snapshots."""

    def __init__(self, root_dir: str | Path):
        self._root = Path(root_dir).expanduser().resolve()
        self._jobs_dir = self._root / "deployment-jobs"
        self._jobs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def job_path(self, job_id: str) -> Path:
        return self._jobs_dir / f"{job_id}.json"

    def upsert_job(self, job: DeploymentIngestionJob) -> str:
        path = self.job_path(job.job_id)
        with self._lock:
            path.write_text(job.model_dump_json(indent=2), encoding="utf-8")
        return str(path)

    def read_job(self, job_id: str) -> DeploymentIngestionJob:
        path = self.job_path(job_id)
        with self._lock:
            return DeploymentIngestionJob.model_validate_json(
                path.read_text(encoding="utf-8"),
            )

    def update_job(
        self,
        job_id: str,
        mutate: "Callable[[DeploymentIngestionJob], None]",
    ) -> DeploymentIngestionJob:
        """Atomic read-modify-write under a single lock acquisition."""
        with self._lock:
            path = self.job_path(job_id)
            job = DeploymentIngestionJob.model_validate_json(
                path.read_text(encoding="utf-8"),
            )
            mutate(job)
            path.write_text(job.model_dump_json(indent=2), encoding="utf-8")
            return job


class LocalDeploymentIngestionRunner:
    """Run typed deployment bundle ingests in local background worker threads."""

    def __init__(
        self,
        *,
        bundle_store: DeploymentBundleStore,
        job_store: DeploymentJobStore,
        max_workers: int = 1,
    ):
        self._bundle_store = bundle_store
        self._job_store = job_store
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="deployment-ingest",
        )
        self._lock = threading.Lock()
        self._futures: dict[str, Future[None]] = {}

    def submit(
        self,
        bundle: DeploymentBundle,
        *,
        run_mode: DeploymentRunMode,
        execute: DeploymentIngestionExecutor,
        execution_target: DeploymentExecutionTarget = "local",
    ) -> DeploymentIngestionJob:
        job = self.create_job(
            bundle,
            run_mode=run_mode,
            execution_target=execution_target,
        )

        future = self._executor.submit(
            self.run_existing_job,
            job_id=job.job_id,
            bundle_id=bundle.bundle_id,
            execute=execute,
        )
        with self._lock:
            self._futures[job.job_id] = future

        def _cleanup(done_future: Future[None], *, job_id: str = job.job_id) -> None:
            with self._lock:
                if self._futures.get(job_id) is done_future:
                    self._futures.pop(job_id, None)

        future.add_done_callback(_cleanup)
        return self.read_job(job.job_id)

    def create_job(
        self,
        bundle: DeploymentBundle,
        *,
        run_mode: DeploymentRunMode,
        execution_target: DeploymentExecutionTarget = "local",
    ) -> DeploymentIngestionJob:
        bundle_ref = self._bundle_store.write_bundle(bundle)
        job = DeploymentIngestionJob(
            bundle_ref=bundle_ref,
            run_mode=run_mode,
            execution_target=execution_target,
        )
        self._job_store.upsert_job(job)
        return self.read_job(job.job_id)

    def read_job(self, job_id: str) -> DeploymentIngestionJob:
        return self._job_store.read_job(job_id)

    def wait(self, job_id: str, timeout: float | None = None) -> DeploymentIngestionJob:
        future = None
        with self._lock:
            future = self._futures.get(job_id)
        if future is not None:
            future.result(timeout=timeout)
        return self.read_job(job_id)

    def run_existing_job(
        self,
        *,
        job_id: str,
        bundle_id: str,
        execute: DeploymentIngestionExecutor,
    ) -> None:
        self._run_job(job_id, bundle_id, execute)

    def _run_job(
        self,
        job_id: str,
        bundle_id: str,
        execute: DeploymentIngestionExecutor,
    ) -> None:
        bundle = self._bundle_store.read_bundle(bundle_id)

        def _mark_running(j: DeploymentIngestionJob) -> None:
            j.status = "running"
            j.started_at = j.started_at or utc_now_iso()

        self._job_store.update_job(job_id, _mark_running)

        def _report_stage(
            stage_name: str,
            status: DeploymentJobState,
            error: str | None = None,
            metadata: dict[str, Any] | None = None,
        ) -> None:
            def _mutate(current: DeploymentIngestionJob) -> None:
                stage = current.stage_status.get(
                    stage_name,
                    DeploymentIngestionStageStatus(
                        stage_name=stage_name,
                        status=status,
                    ),
                )
                now = utc_now_iso()
                stage.status = status
                if status == "running" and stage.started_at is None:
                    stage.started_at = now
                if status in {"success", "error"}:
                    stage.finished_at = now
                    if stage.started_at is None:
                        stage.started_at = now
                if error is not None:
                    stage.error = error
                if metadata:
                    stage.metadata.update(metadata)
                current.stage_status[stage_name] = stage

            self._job_store.update_job(job_id, _mutate)

        try:
            outcome = execute(bundle, _report_stage) or DeploymentIngestionOutcome()

            def _mark_success(j: DeploymentIngestionJob) -> None:
                j.status = "success"
                j.finished_at = utc_now_iso()
                j.error = None
                j.cost_ledger_path = outcome.cost_ledger_path
                j.observability_refs = outcome.observability_refs
                j.metadata.update(outcome.metadata)

            self._job_store.update_job(job_id, _mark_success)
        except Exception as exc:
            error_msg = str(exc) or "deployment ingestion failed"

            def _mark_error(j: DeploymentIngestionJob) -> None:
                running_stages = [
                    s for s in j.stage_status.values() if s.status == "running"
                ]
                if running_stages:
                    last = running_stages[-1]
                    last.status = "error"
                    last.finished_at = utc_now_iso()
                    last.error = error_msg
                    j.stage_status[last.stage_name] = last
                j.status = "error"
                j.finished_at = utc_now_iso()
                j.error = error_msg

            self._job_store.update_job(job_id, _mark_error)


class LocalQueuedDeploymentCoordinator:
    """Queue-backed local submission and worker flow for deployment bundles."""

    topic_name = "deployment.ingestion"

    def __init__(
        self,
        *,
        queue: WorkQueue,
        runner: LocalDeploymentIngestionRunner,
    ):
        self._queue = queue
        self._runner = runner

    async def submit(
        self,
        bundle: DeploymentBundle,
        *,
        run_mode: DeploymentRunMode,
        execution_target: DeploymentExecutionTarget = "local",
    ) -> DeploymentIngestionJob:
        job = self._runner.create_job(
            bundle,
            run_mode=run_mode,
            execution_target=execution_target,
        )
        payload = DeploymentQueuePayload(
            job_id=job.job_id,
            bundle_id=bundle.bundle_id,
            run_mode=run_mode,
            execution_target=execution_target,
        )
        await self._queue.publish(
            topic=self.topic_name,
            payload=payload.model_dump(mode="json"),
        )
        return self._runner.read_job(job.job_id)

    async def drain_once(
        self,
        *,
        execute: DeploymentIngestionExecutor,
        max_messages: int = 1,
    ) -> int:
        async def _handle(item) -> None:
            payload = DeploymentQueuePayload.model_validate(item.payload)
            await asyncio.to_thread(
                self._runner.run_existing_job,
                job_id=payload.job_id,
                bundle_id=payload.bundle_id,
                execute=execute,
            )

        worker = LocalQueueWorker(queue=self._queue, handler=_handle)
        return await worker.run_once(
            max_messages=max_messages,
            topics=[self.topic_name],
        )
