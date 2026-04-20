from __future__ import annotations

from pathlib import Path

import pytest

from unity.common.pipeline import (
    DeploymentBundle,
    DeploymentBundleArtifact,
    DeploymentIdentity,
    DeploymentObservabilityRefs,
    DeploymentBundleRuntimeConfig,
    DeploymentIngestionOutcome,
    InMemoryWorkQueue,
    LocalDeploymentBundleStore,
    LocalDeploymentIngestionRunner,
    LocalDeploymentJobStore,
    LocalQueuedDeploymentCoordinator,
    build_local_deployment_runtime,
)


def _sample_bundle() -> DeploymentBundle:
    return DeploymentBundle(
        deployment_identity=DeploymentIdentity(
            client="midland_heart",
            deployment="v2",
            project="MidlandHeart",
            environment="local",
        ),
        data_pipeline_config=DeploymentBundleArtifact(
            kind="data_pipeline_config",
            logical_name="pipeline_config.json",
            source_path="/tmp/pipeline_config.json",
        ),
        source_artifact_manifest=[
            DeploymentBundleArtifact(
                kind="source_data",
                logical_name="repairs.csv",
                source_path="/tmp/repairs.csv",
            ),
        ],
        integration_assets=[
            DeploymentBundleArtifact(
                kind="integration_asset",
                logical_name="functions",
                source_path="/tmp/functions",
                is_directory=True,
            ),
        ],
    )


def test_local_deployment_bundle_store_round_trips_manifest(tmp_path):
    store = LocalDeploymentBundleStore(tmp_path)
    bundle = _sample_bundle()

    bundle_ref = store.write_bundle(bundle)
    loaded = store.read_bundle(bundle.bundle_id)

    assert bundle_ref.bundle_id == bundle.bundle_id
    assert bundle_ref.manifest_path.endswith(f"{bundle.bundle_id}.json")
    assert loaded.deployment_identity.client == "midland_heart"
    assert loaded.data_pipeline_config.logical_name == "pipeline_config.json"
    assert loaded.source_artifact_manifest[0].logical_name == "repairs.csv"


def test_local_deployment_bundle_store_can_stage_local_artifacts(tmp_path):
    store = LocalDeploymentBundleStore(tmp_path)
    data_file = tmp_path / "source.csv"
    config_file = tmp_path / "pipeline_config.json"
    function_dir = tmp_path / "functions"
    data_file.write_text("id,value\n1,alpha\n", encoding="utf-8")
    config_file.write_text('{"source_files": []}', encoding="utf-8")
    function_dir.mkdir()
    (function_dir / "logic.py").write_text(
        "def run():\n    return 'ok'\n",
        encoding="utf-8",
    )

    bundle = DeploymentBundle(
        deployment_identity=DeploymentIdentity(
            client="midland_heart",
            deployment="v2",
        ),
        data_pipeline_config=DeploymentBundleArtifact(
            kind="data_pipeline_config",
            logical_name="pipeline_config.json",
            source_path=str(config_file),
        ),
        integration_assets=[
            DeploymentBundleArtifact(
                kind="function_dir",
                logical_name="functions",
                source_path=str(function_dir),
                is_directory=True,
            ),
        ],
        source_artifact_manifest=[
            DeploymentBundleArtifact(
                kind="source_data",
                logical_name="source.csv",
                source_path=str(data_file),
            ),
        ],
    )

    prepared = store.prepare_bundle(bundle)

    staged_data_path = Path(
        prepared.source_artifact_manifest[0].artifact_uri.removeprefix("file://"),
    )
    staged_config_path = Path(
        prepared.data_pipeline_config.artifact_uri.removeprefix("file://"),
    )
    staged_functions_path = Path(
        prepared.integration_assets[0].artifact_uri.removeprefix("file://"),
    )

    assert staged_data_path.exists()
    assert staged_data_path.read_text(encoding="utf-8") == "id,value\n1,alpha\n"
    assert staged_config_path.exists()
    assert staged_functions_path.is_dir()
    assert (staged_functions_path / "logic.py").exists()


def test_local_deployment_runner_records_successful_job(tmp_path):
    bundle_store = LocalDeploymentBundleStore(tmp_path)
    job_store = LocalDeploymentJobStore(tmp_path)
    runner = LocalDeploymentIngestionRunner(
        bundle_store=bundle_store,
        job_store=job_store,
        max_workers=1,
    )
    bundle = _sample_bundle()

    def _execute(bundle, report_stage):
        report_stage("prepare_bundle", "running")
        report_stage("prepare_bundle", "success")
        report_stage("ingest_data", "running", metadata={"file_count": 1})
        report_stage("ingest_data", "success")
        return DeploymentIngestionOutcome(
            cost_ledger_path="/tmp/cost-ledger.jsonl",
            observability_refs=DeploymentObservabilityRefs(
                progress_file="/tmp/progress.jsonl",
                log_file="/tmp/run.log",
            ),
            metadata={"rows_ingested": 25},
        )

    job = runner.submit(bundle, run_mode="hybrid", execute=_execute)
    final_job = runner.wait(job.job_id, timeout=5.0)

    assert final_job.status == "success"
    assert final_job.stage_status["prepare_bundle"].status == "success"
    assert final_job.stage_status["ingest_data"].status == "success"
    assert final_job.stage_status["ingest_data"].metadata["file_count"] == 1
    assert final_job.cost_ledger_path == "/tmp/cost-ledger.jsonl"
    assert final_job.observability_refs.progress_file == "/tmp/progress.jsonl"
    assert final_job.metadata["rows_ingested"] == 25
    assert job_store.job_path(job.job_id).exists()


def test_local_deployment_runner_records_failed_job(tmp_path):
    bundle_store = LocalDeploymentBundleStore(tmp_path)
    job_store = LocalDeploymentJobStore(tmp_path)
    runner = LocalDeploymentIngestionRunner(
        bundle_store=bundle_store,
        job_store=job_store,
        max_workers=1,
    )
    bundle = _sample_bundle()

    def _execute(bundle, report_stage):
        report_stage("prepare_bundle", "running")
        raise RuntimeError("bundle validation failed")

    job = runner.submit(bundle, run_mode="file_manager", execute=_execute)
    final_job = runner.wait(job.job_id, timeout=5.0)

    assert final_job.status == "error"
    assert final_job.error == "bundle validation failed"
    assert final_job.stage_status["prepare_bundle"].status == "error"
    assert final_job.stage_status["prepare_bundle"].error == "bundle validation failed"


def test_build_local_deployment_runtime_defaults_to_local_backends(tmp_path):
    runtime = build_local_deployment_runtime(
        DeploymentBundleRuntimeConfig(local_root_dir=str(tmp_path), max_workers=2),
    )

    assert isinstance(runtime.bundle_store, LocalDeploymentBundleStore)
    assert isinstance(runtime.job_store, LocalDeploymentJobStore)
    assert isinstance(runtime.work_queue, InMemoryWorkQueue)
    assert isinstance(runtime.runner, LocalDeploymentIngestionRunner)
    assert isinstance(runtime.queued_coordinator, LocalQueuedDeploymentCoordinator)
    assert runtime.config.max_workers == 2


def test_build_local_deployment_runtime_rejects_non_local_backends(tmp_path):
    config = DeploymentBundleRuntimeConfig(
        local_root_dir=str(tmp_path),
        bundle_store_backend="gcs",
    )

    try:
        build_local_deployment_runtime(config)
    except NotImplementedError as exc:
        assert "Only local deployment bundle stores" in str(exc)
    else:
        raise AssertionError("Expected explicit failure for non-local backends")


def test_build_local_deployment_runtime_rejects_non_local_queue_backends(tmp_path):
    config = DeploymentBundleRuntimeConfig(
        local_root_dir=str(tmp_path),
        queue_backend="pubsub",
    )

    try:
        build_local_deployment_runtime(config)
    except NotImplementedError as exc:
        assert "Only in-memory deployment queues" in str(exc)
    else:
        raise AssertionError("Expected explicit failure for non-local queue backends")


@pytest.mark.asyncio
async def test_local_queued_deployment_coordinator_executes_submitted_job(tmp_path):
    runtime = build_local_deployment_runtime(
        DeploymentBundleRuntimeConfig(local_root_dir=str(tmp_path)),
    )
    bundle = _sample_bundle()

    def _execute(bundle, report_stage):
        report_stage("prepare_bundle", "running")
        report_stage("prepare_bundle", "success")
        return DeploymentIngestionOutcome(metadata={"mode": "queued"})

    job = await runtime.queued_coordinator.submit(bundle, run_mode="data_manager")
    queued_job = runtime.job_store.read_job(job.job_id)
    assert queued_job.status == "queued"

    processed = await runtime.queued_coordinator.drain_once(
        execute=_execute,
        max_messages=1,
    )
    final_job = runtime.job_store.read_job(job.job_id)

    assert processed == 1
    assert final_job.status == "success"
    assert final_job.stage_status["prepare_bundle"].status == "success"
    assert final_job.metadata["mode"] == "queued"
