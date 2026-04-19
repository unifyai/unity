"""Tests for :mod:`unity.common.pipeline.dispatch`.

Covers the symbolic contract enforced by :func:`publish_parse_request`:

* Exactly one of ``source_local_path`` / ``source_bytes`` /
  ``source_gs_uri`` must be supplied.
* ``ingestion_mode`` and the matching binding are propagated into the
  published :class:`ParseRequested` envelope verbatim.
* ``file_paths`` on the published envelope always has length 1
  (``one file per ParseRequested``).
* ``gs://`` URIs short-circuit the upload; local paths and in-memory
  bytes flow through ``storage.Client`` (here a fake).

No real GCP clients are touched -- a pair of in-memory fakes capture
every upload and publish.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from unity.common.pipeline import (
    DispatchTarget,
    publish_parse_request,
)
from unity.common.pipeline.types import DmBinding, FmBinding

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeBlob:
    def __init__(self, parent: "_FakeBucket", name: str) -> None:
        self._parent = parent
        self.name = name
        self.data: bytes | None = None
        self.from_filename: str | None = None

    def upload_from_string(self, data: bytes) -> None:
        self.data = data

    def upload_from_filename(self, path: str) -> None:
        self.from_filename = path
        with open(path, "rb") as f:
            self.data = f.read()


class _FakeBucket:
    def __init__(self, name: str) -> None:
        self.name = name
        self.blobs: dict[str, _FakeBlob] = {}

    def blob(self, key: str) -> _FakeBlob:
        b = _FakeBlob(self, key)
        self.blobs[key] = b
        return b


class _FakeStorageClient:
    def __init__(self) -> None:
        self.buckets: dict[str, _FakeBucket] = {}

    def bucket(self, name: str) -> _FakeBucket:
        if name not in self.buckets:
            self.buckets[name] = _FakeBucket(name)
        return self.buckets[name]


class _FakeFuture:
    def __init__(self, message_id: str) -> None:
        self._message_id = message_id

    def result(self, timeout: float | None = None) -> str:
        return self._message_id


class _FakePublisherClient:
    def __init__(self) -> None:
        self.published: list[dict[str, Any]] = []

    def topic_path(self, project: str, topic: str) -> str:
        return f"projects/{project}/topics/{topic}"

    def publish(self, topic: str, data: bytes, **attrs: Any) -> _FakeFuture:
        self.published.append(
            {"topic": topic, "data": data, "attrs": dict(attrs)},
        )
        return _FakeFuture(f"msg-{len(self.published)}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def target() -> DispatchTarget:
    return DispatchTarget(
        project_id="unit-test-project",
        bucket_name="unit-test-bucket",
        env_suffix="-staging",
    )


@pytest.fixture
def storage() -> _FakeStorageClient:
    return _FakeStorageClient()


@pytest.fixture
def pubsub() -> _FakePublisherClient:
    return _FakePublisherClient()


# ---------------------------------------------------------------------------
# Success paths
# ---------------------------------------------------------------------------


class TestFmModeBytes:
    def test_uploads_bytes_and_publishes_fm_envelope(
        self,
        target: DispatchTarget,
        storage: _FakeStorageClient,
        pubsub: _FakePublisherClient,
    ) -> None:
        fm_binding = FmBinding(
            user_id="alice",
            assistant_id="42",
            fm_alias="Local",
            logical_path="reports/q1.csv",
        )

        result = publish_parse_request(
            target=target,
            logical_path="reports/q1.csv",
            ingestion_mode="fm",
            fm_binding=fm_binding,
            source_bytes=b"a,b\n1,2\n",
            storage_client=storage,
            publisher_client=pubsub,
        )

        bucket = storage.buckets["unit-test-bucket"]
        assert len(bucket.blobs) == 1
        blob_key, blob = next(iter(bucket.blobs.items()))
        assert blob_key.endswith("/q1.csv")
        assert blob_key.startswith("dispatch/")
        assert blob.data == b"a,b\n1,2\n"
        assert result.gs_uri == f"gs://unit-test-bucket/{blob_key}"

        assert len(pubsub.published) == 1
        published = pubsub.published[0]
        assert (
            published["topic"]
            == "projects/unit-test-project/topics/unity-parse-staging"
        )
        payload = json.loads(published["data"].decode("utf-8"))
        assert payload["ingestion_mode"] == "fm"
        assert payload["fm_binding"]["user_id"] == "alice"
        assert payload["fm_binding"]["assistant_id"] == "42"
        assert payload["fm_binding"]["fm_alias"] == "Local"
        assert payload["fm_binding"]["logical_path"] == "reports/q1.csv"
        assert payload["dm_binding"] is None
        assert payload["file_paths"] == [result.gs_uri]
        assert payload["job_id"] == result.job_id


class TestFmModeLocalPath:
    def test_uploads_local_path(
        self,
        target: DispatchTarget,
        storage: _FakeStorageClient,
        pubsub: _FakePublisherClient,
        tmp_path,
    ) -> None:
        src = tmp_path / "payload.csv"
        src.write_text("x,y\n1,2\n")

        fm_binding = FmBinding(
            user_id="alice",
            assistant_id="42",
            fm_alias="Local",
            logical_path=str(src),
        )
        result = publish_parse_request(
            target=target,
            logical_path=str(src),
            ingestion_mode="fm",
            fm_binding=fm_binding,
            source_local_path=str(src),
            storage_client=storage,
            publisher_client=pubsub,
        )

        bucket = storage.buckets["unit-test-bucket"]
        _, blob = next(iter(bucket.blobs.items()))
        assert blob.from_filename == str(src)
        assert blob.data == b"x,y\n1,2\n"
        assert result.gs_uri.startswith("gs://unit-test-bucket/")


class TestDmModeGsUri:
    def test_gs_uri_short_circuits_upload(
        self,
        target: DispatchTarget,
        storage: _FakeStorageClient,
        pubsub: _FakePublisherClient,
    ) -> None:
        dm_binding = DmBinding(target_context="alice/42/Orders")

        result = publish_parse_request(
            target=target,
            logical_path="gs://bucket/foo.csv",
            ingestion_mode="dm",
            dm_binding=dm_binding,
            source_gs_uri="gs://bucket/foo.csv",
            storage_client=storage,
            publisher_client=pubsub,
        )

        assert storage.buckets == {}
        assert result.gs_uri == "gs://bucket/foo.csv"

        payload = json.loads(pubsub.published[0]["data"].decode("utf-8"))
        assert payload["ingestion_mode"] == "dm"
        assert payload["dm_binding"]["target_context"] == "alice/42/Orders"
        assert payload["fm_binding"] is None
        assert payload["file_paths"] == ["gs://bucket/foo.csv"]


class TestOneFilePerMessage:
    def test_file_paths_length_is_always_one(
        self,
        target: DispatchTarget,
        storage: _FakeStorageClient,
        pubsub: _FakePublisherClient,
    ) -> None:
        """The publisher always constructs ``file_paths=[gs_uri]`` (length 1)."""
        fm_binding = FmBinding(
            user_id="alice",
            assistant_id="42",
            fm_alias="Local",
            logical_path="x.csv",
        )
        publish_parse_request(
            target=target,
            logical_path="x.csv",
            ingestion_mode="fm",
            fm_binding=fm_binding,
            source_bytes=b"x\n",
            storage_client=storage,
            publisher_client=pubsub,
        )
        payload = json.loads(pubsub.published[0]["data"].decode("utf-8"))
        assert len(payload["file_paths"]) == 1


class TestDispatchPropagatesJobId:
    def test_explicit_job_id_is_preserved(
        self,
        target: DispatchTarget,
        storage: _FakeStorageClient,
        pubsub: _FakePublisherClient,
    ) -> None:
        fm_binding = FmBinding(
            user_id="alice",
            assistant_id="42",
            fm_alias="Local",
            logical_path="x.csv",
        )
        result = publish_parse_request(
            target=target,
            logical_path="x.csv",
            ingestion_mode="fm",
            fm_binding=fm_binding,
            source_bytes=b"x\n",
            job_id="my-stable-id",
            storage_client=storage,
            publisher_client=pubsub,
        )
        assert result.job_id == "my-stable-id"
        payload = json.loads(pubsub.published[0]["data"].decode("utf-8"))
        assert payload["job_id"] == "my-stable-id"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_fm_mode_requires_fm_binding(self, target, storage, pubsub):
        with pytest.raises(ValueError, match="fm_binding"):
            publish_parse_request(
                target=target,
                logical_path="x.csv",
                ingestion_mode="fm",
                source_bytes=b"x\n",
                storage_client=storage,
                publisher_client=pubsub,
            )

    def test_dm_mode_requires_dm_binding(self, target, storage, pubsub):
        with pytest.raises(ValueError, match="dm_binding"):
            publish_parse_request(
                target=target,
                logical_path="x.csv",
                ingestion_mode="dm",
                source_bytes=b"x\n",
                storage_client=storage,
                publisher_client=pubsub,
            )

    def test_mode_binding_mismatch_fm(self, target, storage, pubsub):
        with pytest.raises(ValueError, match="must not supply dm_binding"):
            publish_parse_request(
                target=target,
                logical_path="x.csv",
                ingestion_mode="fm",
                fm_binding=FmBinding(
                    user_id="a",
                    assistant_id="0",
                    fm_alias="Local",
                    logical_path="x.csv",
                ),
                dm_binding=DmBinding(target_context="ctx"),
                source_bytes=b"x\n",
                storage_client=storage,
                publisher_client=pubsub,
            )

    def test_mode_binding_mismatch_dm(self, target, storage, pubsub):
        with pytest.raises(ValueError, match="must not supply fm_binding"):
            publish_parse_request(
                target=target,
                logical_path="x.csv",
                ingestion_mode="dm",
                fm_binding=FmBinding(
                    user_id="a",
                    assistant_id="0",
                    fm_alias="Local",
                    logical_path="x.csv",
                ),
                dm_binding=DmBinding(target_context="ctx"),
                source_bytes=b"x\n",
                storage_client=storage,
                publisher_client=pubsub,
            )

    def test_requires_exactly_one_source(self, target, storage, pubsub):
        fm_binding = FmBinding(
            user_id="a",
            assistant_id="0",
            fm_alias="Local",
            logical_path="x.csv",
        )
        with pytest.raises(ValueError, match="Exactly one of"):
            publish_parse_request(
                target=target,
                logical_path="x.csv",
                ingestion_mode="fm",
                fm_binding=fm_binding,
                storage_client=storage,
                publisher_client=pubsub,
            )
        with pytest.raises(ValueError, match="Exactly one of"):
            publish_parse_request(
                target=target,
                logical_path="x.csv",
                ingestion_mode="fm",
                fm_binding=fm_binding,
                source_bytes=b"x",
                source_gs_uri="gs://b/x",
                storage_client=storage,
                publisher_client=pubsub,
            )

    def test_invalid_gs_uri(self, target, storage, pubsub):
        fm_binding = FmBinding(
            user_id="a",
            assistant_id="0",
            fm_alias="Local",
            logical_path="x.csv",
        )
        with pytest.raises(ValueError, match="must start with 'gs://'"):
            publish_parse_request(
                target=target,
                logical_path="x.csv",
                ingestion_mode="fm",
                fm_binding=fm_binding,
                source_gs_uri="s3://bucket/key",
                storage_client=storage,
                publisher_client=pubsub,
            )

    def test_invalid_ingestion_mode(self, target, storage, pubsub):
        with pytest.raises(ValueError, match="ingestion_mode must be"):
            publish_parse_request(
                target=target,
                logical_path="x.csv",
                ingestion_mode="bogus",  # type: ignore[arg-type]
                source_bytes=b"x",
                storage_client=storage,
                publisher_client=pubsub,
            )

    def test_requires_logical_path(self, target, storage, pubsub):
        fm_binding = FmBinding(
            user_id="a",
            assistant_id="0",
            fm_alias="Local",
            logical_path="x.csv",
        )
        with pytest.raises(ValueError, match="logical_path"):
            publish_parse_request(
                target=target,
                logical_path="",
                ingestion_mode="fm",
                fm_binding=fm_binding,
                source_bytes=b"x",
                storage_client=storage,
                publisher_client=pubsub,
            )
