"""Port for durable artifact storage, worker leases, and ingest checkpoints.

The three concerns sit on one protocol because they share a failure model: all
three have to survive the process that wrote them. An ingestion that can be
resumed needs its rows materialised somewhere durable, a fence that stops two
attempts from corrupting each other, and a monotonic record of how far it got.
Splitting them across ports would let an implementation satisfy one and not the
others, which is precisely the state that makes a run unrecoverable.

That is why leases and checkpoints are part of the *port* rather than of one
backend. A store that cannot fence and cannot checkpoint can still ingest, but
its runs are not resumable -- and an executor has no way to tell, so it would
quietly offer a guarantee it cannot keep. Requiring both here means every
binding is resumable, and the difference between running in-process and running
on a worker fleet is only which adapter is bound.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Protocol, Sequence

from .types import (
    IngestCheckpoint,
    InlineRowsHandle,
    ObjectStoreArtifactHandle,
    TableInputHandle,
)

#: Conventional ``table_id`` used when materialising lowered content rows
#: (``/Content/`` rows derived from the document graph) as an
#: ``ObjectStoreArtifactHandle``.  Keeping this constant here means callers
#: never have to coin a synthetic id for the content artifact.
CONTENT_ROWS_TABLE_ID: str = "__content__"


class ArtifactNotFound(FileNotFoundError):
    """Raised when a key has no object behind it.

    Subclasses ``FileNotFoundError`` so filesystem-backed stores can raise it
    where they would already have raised, while object-store backends translate
    their own not-found error into it. Callers that distinguish "absent" from
    "unreachable" -- reading a checkpoint that may not exist yet is the common
    one -- can then catch a single type whatever the binding.
    """


@dataclass(frozen=True)
class LeaseRecord:
    """Ownership of one unit of work by one attempt, with a fencing token.

    ``generation`` is the fencing token: it changes on every write, so a writer
    that has been away can tell it lost ownership rather than overwriting a
    newer attempt's progress. A lease without one is only advisory.
    """

    key: str
    owner_id: str
    attempt_id: str
    stage: str
    acquired_at: str
    heartbeat_at: str
    expires_at: str
    generation: Optional[int]
    takeover_count: int = 0
    previous_owner_id: str = ""


class LeaseNotAcquired(RuntimeError):
    """Another live attempt holds the lease.

    Not an error condition in itself -- the usual cause is an ordinary
    duplicate delivery, and the right response is to let the holder finish.
    """

    def __init__(self, message: str, *, lease: Optional[LeaseRecord] = None):
        super().__init__(message)
        self.lease = lease


class StaleLeaseError(RuntimeError):
    """The caller no longer owns the lease it is writing under.

    Raised on the write path rather than the read path on purpose: this is the
    check that stops a stalled attempt, resumed after its lease was taken over,
    from rolling a checkpoint backwards over the successor's progress.
    """


class ArtifactStore(Protocol):
    """Port for durable artifact storage, leases, and ingest checkpoints.

    Implementations must support four concerns:

    1. **Table materialisation** -- serialise a ``TableInputHandle`` into a
       durable artifact (JSONL today, Parquet/Arrow later).
    2. **Manifest CRUD** -- store and retrieve arbitrary JSON documents
       (run manifests, plans, bundle descriptors) keyed by a logical path.
    3. **Leases** -- fence duplicate attempts at one unit of work, with a
       generation token and takeover of expired holders.
    4. **Checkpoints** -- record committed progress monotonically, so a
       resumed attempt skips what already landed instead of duplicating it.

    ``LocalArtifactStore`` implements all four on the filesystem; the hosted
    deployment binds an object-store adapter with identical semantics.
    """

    def materialize_table_input(
        self,
        handle: TableInputHandle,
        *,
        logical_path: str,
        table_id: str,
        artifact_format: str,
        job_id: str = "",
    ) -> ObjectStoreArtifactHandle:
        """Materialise *handle* as a durable artifact.

        ``job_id`` is optional so callers that operate outside the
        worker pipeline (e.g. a local developer running the
        ``LocalArtifactStore`` with a hash-based on-disk layout) can
        leave it empty. Object-store-backed implementations (e.g.
        ``GcsArtifactStore``) use it to scope every artifact under a
        single ``jobs/<job_id>/artifacts/...`` root so all of a job's
        outputs live in one place.
        """
        ...

    def materialize_content_rows(
        self,
        rows: Iterable[Any],
        *,
        logical_path: str,
        artifact_format: str = "jsonl",
        job_id: str = "",
    ) -> ObjectStoreArtifactHandle:
        """Serialise lowered content rows into a JSONL artifact handle.

        ``rows`` may be Pydantic models (e.g. ``FileContentRow``) or plain
        ``dict`` payloads; each is normalised to a JSON object before being
        written.  The resulting handle uses the conventional
        ``CONTENT_ROWS_TABLE_ID`` so manifests/handles for derived content
        stay consistent across implementations. See
        :meth:`materialize_table_input` for the meaning of ``job_id``.
        """
        ...

    def put_json(
        self,
        key: str,
        data: Any,
        *,
        if_generation_match: Optional[int] = None,
    ) -> str:
        """Serialise *data* as JSON and persist under *key*.

        ``if_generation_match`` makes the write conditional: ``0`` means "only
        if absent", and any other value means "only if the object is still at
        that generation". Omitting it overwrites unconditionally. Implementations
        raise if the condition fails, which is what turns an ordinary write into
        a compare-and-swap.

        Returns the storage URI of the written object.
        """
        ...

    def get_json(self, key: str) -> Any:
        """Read and deserialise a JSON object previously stored at *key*.

        Raises :class:`ArtifactNotFound` when *key* holds nothing.
        """
        ...

    def put_bytes(self, key: str, data: bytes) -> str:
        """Persist raw bytes under *key*, returning its storage URI.

        Distinct from :meth:`put_json` because a source file is opaque: parsing
        decides what it is, and round-tripping it through JSON would corrupt
        anything that is not text. Used where bytes must reach the store through
        a caller that cannot write to it directly.
        """
        ...

    def exists(self, key: str) -> bool:
        """Return ``True`` if an object exists at *key*."""
        ...

    def list_keys(self, prefix: str) -> list[str]:
        """Every key under *prefix*, sorted.

        On the port rather than on one backend because callers legitimately need
        to ask what a job left behind -- which tables it checkpointed, what it
        parked -- and cannot know the artifact ids in advance. A binding that
        could not enumerate would force those callers into backend-specific
        code, and the local binding would silently lose the ability to answer.
        """
        ...

    def delete(self, key: str) -> None:
        """Remove the object at *key* (no-op if absent)."""
        ...

    def download_to_local(self, source: str, dest: Path | str) -> Path:
        """Stage the object at *source* onto the local filesystem at *dest*.

        ``source`` is either a plain key or a fully-qualified storage URI of the
        kind this store emits. Exists so row streaming never needs a
        backend-aware reader: whatever the store, the consumer reads a file.
        """
        ...

    # -- leases ---------------------------------------------------------------

    def acquire_lease(
        self,
        key: str,
        *,
        owner_id: str,
        attempt_id: str,
        stage: str,
        ttl_seconds: int = 900,
        steal_expired_after_seconds: int = 30,
    ) -> LeaseRecord:
        """Take ownership of *key*, or take it over from an expired holder.

        The lease is what makes at-most-one-writer true across processes, and
        with it the checkpoint below is trustworthy. Re-acquiring with the same
        ``owner_id`` and ``attempt_id`` refreshes rather than conflicts, so an
        ordinary redelivery to the same attempt is not an error.

        Takeover is deliberately time-based and requires the holder to be past
        its expiry by ``steal_expired_after_seconds``: a worker that dies
        without releasing must not strand the work forever, and the grace period
        keeps a merely-slow heartbeat from losing its lease to a racing peer.

        Raises :class:`LeaseNotAcquired` when a live holder owns it.
        """
        ...

    def refresh_lease(
        self,
        key: str,
        *,
        owner_id: str,
        attempt_id: str,
        generation: Optional[int],
        ttl_seconds: int = 900,
    ) -> LeaseRecord:
        """Extend a held lease and return its new generation.

        Called as work progresses so a long unit does not expire mid-flight.
        Raises :class:`StaleLeaseError` if ownership or generation moved on,
        which is the signal that this attempt has been superseded and must stop
        rather than continue writing.
        """
        ...

    def verify_lease(
        self,
        key: str,
        *,
        owner_id: str,
        attempt_id: str,
    ) -> LeaseRecord:
        """Confirm this attempt still owns *key*, raising if it does not."""
        ...

    def release_lease(
        self,
        key: str,
        *,
        owner_id: str,
        attempt_id: str,
        generation: Optional[int],
    ) -> None:
        """Give up a held lease so a successor need not wait out its TTL."""
        ...

    # -- checkpoints ----------------------------------------------------------

    def write_checkpoint(
        self,
        job_id: str,
        artifact_id: str,
        checkpoint: IngestCheckpoint,
        *,
        attempt_id: str = "",
        lease_generation: Optional[int] = None,
    ) -> None:
        """Record committed progress for one artifact, monotonically.

        Written after each committed chunk, which is what bounds re-work on
        resume to at most one chunk. Implementations must **refuse a checkpoint
        that moves backwards**: a regression means a superseded attempt is still
        writing, and accepting it would make a resumed run re-insert rows that
        already landed while reporting a total that says otherwise.
        """
        ...

    def read_checkpoint(
        self,
        job_id: str,
        artifact_id: str,
    ) -> Optional[IngestCheckpoint]:
        """Return committed progress for one artifact, or ``None`` if untouched.

        ``None`` means "start from the beginning", not an error -- the first
        attempt at any artifact reads nothing.
        """
        ...

    def delete_checkpoints(
        self,
        job_id: str,
        *,
        artifact_ids: Optional[Sequence[str]] = None,
    ) -> None:
        """Discard recorded checkpoints for one job (no-op if absent).

        ``artifact_ids`` narrows the discard to those tables. Re-attempting one
        file in a batch must not clear the marks belonging to the other
        fourteen: those describe rows that were committed correctly, and losing
        them turns a one-file retry into a re-ingest of everything.

        The one legitimate caller is a full re-run: re-attempting everything
        means the committed marks no longer describe the work, and leaving any
        of them would make the re-run skip rows it was explicitly asked to
        rewrite. Job-scoped rather than per-artifact because the caller cannot
        know which artifacts a previous attempt got as far as checkpointing.
        Never call this while an attempt may be live -- the lease serialises
        writers, not this.
        """
        ...


class LocalArtifactStore:
    """Filesystem-backed artifact store: full port, including resumability.

    Used by self-host and by in-process execution, so it has to offer the same
    guarantees as the hosted backend rather than a subset -- an executor cannot
    know which adapter it was handed, and a store that silently could not fence
    or checkpoint would turn every crash into lost or duplicated rows.

    Compare-and-swap is built from two filesystem primitives. An exclusive
    ``flock`` makes each read-modify-write a critical section, and the kernel
    drops it when the descriptor closes -- including on process death, which a
    lock *file* would not, leaving keys wedged after a crash. The generation
    token lives in a sidecar so payloads stay byte-identical in shape to the
    hosted backend's, and both are replaced under the same lock, so a reader
    never sees a payload and a generation that disagree.
    """

    def __init__(self, *, root_dir: str | Path):
        self.root_dir = Path(root_dir).expanduser().resolve()

    def materialize_table_input(
        self,
        handle: TableInputHandle,
        *,
        logical_path: str,
        table_id: str,
        artifact_format: str,
        job_id: str = "",
    ) -> ObjectStoreArtifactHandle:
        # ``job_id`` is intentionally ignored here: LocalArtifactStore's
        # on-disk layout is hash-based and optimised for developer
        # ergonomics, not for the per-job roll-ups that the GCS store
        # uses in production. Keeping the kwarg keeps the protocol
        # uniform across backends.
        del job_id
        if isinstance(handle, ObjectStoreArtifactHandle):
            return handle
        if artifact_format != "jsonl":
            raise ValueError(
                f"Unsupported artifact format for LocalArtifactStore: {artifact_format!r}",
            )

        target_path = self._artifact_path(
            logical_path=logical_path,
            table_id=table_id,
            artifact_format=artifact_format,
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)

        columns = list(getattr(handle, "columns", []) or [])
        actual_row_count = 0
        from unify.common.pipeline.row_streaming import iter_table_input_rows

        with target_path.open("w", encoding="utf-8", newline="\n") as fh:
            for row in iter_table_input_rows(handle):
                payload = {str(key): value for key, value in dict(row).items()}
                if not columns:
                    columns = [str(key) for key in payload.keys()]
                fh.write(json.dumps(payload, ensure_ascii=False))
                fh.write("\n")
                actual_row_count += 1

        return ObjectStoreArtifactHandle(
            storage_uri=target_path.resolve().as_uri(),
            logical_path=str(logical_path or ""),
            artifact_format="jsonl",
            columns=columns,
            row_count=actual_row_count,
        )

    def materialize_content_rows(
        self,
        rows: Iterable[Any],
        *,
        logical_path: str,
        artifact_format: str = "jsonl",
        job_id: str = "",
    ) -> ObjectStoreArtifactHandle:
        """Serialise content rows as JSONL via ``materialize_table_input``.

        Mirrors the table materialisation flow but fixes the ``table_id`` to
        :data:`CONTENT_ROWS_TABLE_ID`.  Rows may be Pydantic models or
        dicts; non-dict inputs are coerced via ``model_dump(mode="json")``
        when available and otherwise wrapped into a single-field dict as a
        last resort.
        """
        serialised: list[dict[str, Any]] = []
        columns: list[str] = []
        for row in rows:
            payload: dict[str, Any]
            dump = getattr(row, "model_dump", None)
            if callable(dump):
                payload = dict(dump(mode="json", exclude_none=True))
            elif isinstance(row, dict):
                payload = {str(k): v for k, v in row.items()}
            else:
                payload = {"value": row}
            serialised.append(payload)
            if not columns:
                columns = [str(k) for k in payload.keys()]

        inline = InlineRowsHandle(
            rows=serialised,
            columns=columns,
            row_count=len(serialised),
        )
        return self.materialize_table_input(
            inline,
            logical_path=logical_path,
            table_id=CONTENT_ROWS_TABLE_ID,
            artifact_format=artifact_format,
            job_id=job_id,
        )

    # -- manifest CRUD -----------------------------------------------------

    def put_json(
        self,
        key: str,
        data: Any,
        *,
        if_generation_match: Optional[int] = None,
    ) -> str:
        target = self._key_path(key)
        with _exclusive(target):
            self._put_json_locked(
                target,
                data,
                if_generation_match=if_generation_match,
            )
        return target.resolve().as_uri()

    def get_json(self, key: str) -> Any:
        target = self._key_path(key)
        if not target.exists():
            raise ArtifactNotFound(f"Artifact not found: {key}")
        return json.loads(target.read_text(encoding="utf-8"))

    def put_bytes(self, key: str, data: bytes) -> str:
        target = self._key_path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Replaced by rename so a reader never observes a partial file: a parse
        # worker watching this directory must not open a half-written source.
        tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, target)
        return target.resolve().as_uri()

    def exists(self, key: str) -> bool:
        return self._key_path(key).exists()

    def list_keys(self, prefix: str) -> list[str]:
        root = self._key_path(prefix)
        base = root if root.is_dir() else root.parent
        if not base.exists():
            return []
        keys: list[str] = []
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            # Lock files and generation sidecars are this backend's own
            # bookkeeping, not objects a caller stored, so they are not keys.
            if path.name.startswith(".") and (
                path.name.endswith(".lock") or path.name.endswith(".gen")
            ):
                continue
            key = str(path.relative_to(self.root_dir))
            if key.startswith(prefix.lstrip("/")):
                keys.append(key)
        return sorted(keys)

    def delete(self, key: str) -> None:
        target = self._key_path(key)
        with _exclusive(target):
            target.unlink(missing_ok=True)
            _generation_path(target).unlink(missing_ok=True)

    def download_to_local(self, source: str, dest: Path | str) -> Path:
        """Copy a stored object to *dest*.

        The local store's objects are already files, so this is a copy rather
        than a fetch -- but it stays on the port because callers must not have
        to know that. A ``file://`` URI emitted by this store and a ``gs://``
        URI emitted by the hosted one are both accepted by their own store.
        """
        dest_path = Path(dest).expanduser().resolve()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        source_path = (
            Path(_path_from_file_uri(source))
            if source.startswith("file://")
            else self._key_path(source)
        )
        if not source_path.exists():
            raise ArtifactNotFound(f"Artifact not found: {source}")
        if source_path.resolve() != dest_path:
            dest_path.write_bytes(source_path.read_bytes())
        return dest_path

    # -- leases ---------------------------------------------------------------

    def acquire_lease(
        self,
        key: str,
        *,
        owner_id: str,
        attempt_id: str,
        stage: str,
        ttl_seconds: int = 900,
        steal_expired_after_seconds: int = 30,
    ) -> LeaseRecord:
        target = self._key_path(key)
        now = _utc_now()
        with _exclusive(target):
            existing, generation = self._read_lease_locked(target)

            if existing is None:
                record = _lease_payload(
                    key=key,
                    owner_id=owner_id,
                    attempt_id=attempt_id,
                    stage=stage,
                    now=now,
                    ttl_seconds=ttl_seconds,
                )
                written = self._put_json_locked(
                    target,
                    record,
                    if_generation_match=0,
                )
                return LeaseRecord(**record, generation=written)

            held = LeaseRecord(**existing, generation=generation)

            # The same attempt asking again is a redelivery, not a conflict.
            if held.owner_id == owner_id and held.attempt_id == attempt_id:
                return self._refresh_locked(
                    target,
                    key=key,
                    owner_id=owner_id,
                    attempt_id=attempt_id,
                    generation=generation,
                    ttl_seconds=ttl_seconds,
                )

            if not _lease_is_expired(
                held,
                grace_seconds=steal_expired_after_seconds,
            ):
                raise LeaseNotAcquired(
                    f"Lease {key!r} is owned by {held.owner_id!r} "
                    f"until {held.expires_at}",
                    lease=held,
                )

            takeover = _lease_payload(
                key=key,
                owner_id=owner_id,
                attempt_id=attempt_id,
                stage=stage,
                now=now,
                ttl_seconds=ttl_seconds,
                takeover_count=held.takeover_count + 1,
                previous_owner_id=held.owner_id,
            )
            written = self._put_json_locked(
                target,
                takeover,
                if_generation_match=generation,
            )
            return LeaseRecord(**takeover, generation=written)

    def refresh_lease(
        self,
        key: str,
        *,
        owner_id: str,
        attempt_id: str,
        generation: Optional[int],
        ttl_seconds: int = 900,
    ) -> LeaseRecord:
        target = self._key_path(key)
        with _exclusive(target):
            return self._refresh_locked(
                target,
                key=key,
                owner_id=owner_id,
                attempt_id=attempt_id,
                generation=generation,
                ttl_seconds=ttl_seconds,
            )

    def verify_lease(
        self,
        key: str,
        *,
        owner_id: str,
        attempt_id: str,
    ) -> LeaseRecord:
        target = self._key_path(key)
        with _exclusive(target):
            payload, generation = self._read_lease_locked(target)
        if payload is None:
            raise LeaseNotAcquired(f"Lease {key!r} does not exist")
        record = LeaseRecord(**payload, generation=generation)
        if record.owner_id != owner_id or record.attempt_id != attempt_id:
            raise StaleLeaseError(
                f"Lease {key!r} owner changed to {record.owner_id!r}/"
                f"{record.attempt_id!r}",
            )
        return record

    def release_lease(
        self,
        key: str,
        *,
        owner_id: str,
        attempt_id: str,
        generation: Optional[int],
    ) -> None:
        # Verified before deleting so a superseded attempt cannot release the
        # lease its successor now holds and hand the work to a third writer.
        self.verify_lease(key, owner_id=owner_id, attempt_id=attempt_id)
        self.delete(key)

    # -- checkpoints ----------------------------------------------------------

    def write_checkpoint(
        self,
        job_id: str,
        artifact_id: str,
        checkpoint: IngestCheckpoint,
        *,
        attempt_id: str = "",
        lease_generation: Optional[int] = None,
    ) -> None:
        target = self._key_path(_checkpoint_key(job_id, artifact_id))
        with _exclusive(target):
            existing, generation = self._read_json_locked(target)
            if existing is not None:
                prior = IngestCheckpoint.model_validate(existing)
                _reject_regression(
                    checkpoint,
                    prior,
                    job_id=job_id,
                    artifact_id=artifact_id,
                )
            payload = checkpoint.model_copy(
                update={
                    "attempt_id": attempt_id or checkpoint.attempt_id,
                    "lease_generation": (
                        lease_generation
                        if lease_generation is not None
                        else checkpoint.lease_generation
                    ),
                },
            ).model_dump(mode="json")
            self._put_json_locked(
                target,
                payload,
                if_generation_match=0 if existing is None else generation,
            )

    def read_checkpoint(
        self,
        job_id: str,
        artifact_id: str,
    ) -> Optional[IngestCheckpoint]:
        try:
            data = self.get_json(_checkpoint_key(job_id, artifact_id))
        except ArtifactNotFound:
            return None
        return IngestCheckpoint.model_validate(data)

    def delete_checkpoints(
        self,
        job_id: str,
        *,
        artifact_ids: Optional[Sequence[str]] = None,
    ) -> None:
        import shutil

        if artifact_ids is None:
            directory = self._key_path(f"jobs/{_safe_fragment(job_id)}/checkpoints")
            shutil.rmtree(directory, ignore_errors=True)
            return
        for artifact_id in artifact_ids:
            self._key_path(_checkpoint_key(job_id, artifact_id)).unlink(
                missing_ok=True,
            )

    # -- internal helpers ---------------------------------------------------

    def _key_path(self, key: str) -> Path:
        safe = key.lstrip("/")
        return self.root_dir / safe

    def _put_json_locked(
        self,
        target: Path,
        data: Any,
        *,
        if_generation_match: Optional[int],
    ) -> int:
        """Write *data* and bump the generation. Caller must hold the lock."""
        current = _read_generation(target)
        if if_generation_match is not None:
            expected = if_generation_match
            actual = 0 if not target.exists() else current
            if expected != actual:
                raise _PreconditionFailed(
                    f"Generation mismatch for {target.name!r}: "
                    f"expected {expected}, found {actual}",
                )
        nxt = current + 1
        target.parent.mkdir(parents=True, exist_ok=True)
        # Replace via a temp file so a reader without the lock -- get_json takes
        # none, being a single read -- never observes a partial document.
        tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(
            json.dumps(data, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        os.replace(tmp, target)
        _generation_path(target).write_text(str(nxt), encoding="utf-8")
        return nxt

    def _read_json_locked(self, target: Path) -> tuple[Optional[Any], Optional[int]]:
        if not target.exists():
            return None, None
        data = json.loads(target.read_text(encoding="utf-8"))
        return data, _read_generation(target)

    def _read_lease_locked(
        self,
        target: Path,
    ) -> tuple[Optional[dict[str, Any]], Optional[int]]:
        data, generation = self._read_json_locked(target)
        if data is None:
            return None, None
        if not isinstance(data, dict):
            raise ValueError(f"Lease at {target} is not a JSON object")
        return data, generation

    def _refresh_locked(
        self,
        target: Path,
        *,
        key: str,
        owner_id: str,
        attempt_id: str,
        generation: Optional[int],
        ttl_seconds: int,
    ) -> LeaseRecord:
        payload, current = self._read_lease_locked(target)
        if payload is None:
            raise LeaseNotAcquired(f"Lease {key!r} disappeared before refresh")
        held = LeaseRecord(**payload, generation=current)
        if held.owner_id != owner_id or held.attempt_id != attempt_id:
            raise StaleLeaseError(
                f"Lease {key!r} is owned by {held.owner_id!r}/"
                f"{held.attempt_id!r}, not {owner_id!r}/{attempt_id!r}",
            )
        if generation is not None and current != generation:
            raise StaleLeaseError(
                f"Lease {key!r} generation changed from {generation} to {current}",
            )
        now = _utc_now()
        renewed = dict(payload)
        renewed["heartbeat_at"] = now.isoformat()
        renewed["expires_at"] = (now + timedelta(seconds=ttl_seconds)).isoformat()
        written = self._put_json_locked(
            target,
            renewed,
            if_generation_match=current,
        )
        return LeaseRecord(**renewed, generation=written)

    def _artifact_path(
        self,
        *,
        logical_path: str,
        table_id: str,
        artifact_format: str,
    ) -> Path:
        digest = hashlib.sha256(
            f"{logical_path}::{table_id}".encode("utf-8"),
        ).hexdigest()[:12]
        file_slug = _safe_fragment(Path(str(logical_path or "file")).stem or "file")
        table_slug = _safe_fragment(table_id or "table")
        return self.root_dir / file_slug / f"{table_slug}-{digest}.{artifact_format}"


def _safe_fragment(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "artifact"
    return "".join(
        char if char.isalnum() or char in ("-", "_") else "_" for char in text
    )


class _PreconditionFailed(RuntimeError):
    """A conditional write's generation precondition did not hold.

    Private because callers should never branch on it: it means another writer
    won the race, and the lease and checkpoint paths above already translate
    that into the domain answer (``LeaseNotAcquired`` / ``StaleLeaseError``).
    """


def _checkpoint_key(job_id: str, artifact_id: str) -> str:
    """Locate a checkpoint under its job, mirroring the hosted layout.

    Keeping the layout identical across bindings is what lets a run that began
    in-process be adopted by the worker fleet: the fleet looks where it always
    looks and finds progress it did not write.
    """
    return f"jobs/{_safe_fragment(job_id)}/checkpoints/{_safe_fragment(artifact_id)}"


def _generation_path(target: Path) -> Path:
    return target.with_name(f".{target.name}.gen")


def _read_generation(target: Path) -> int:
    marker = _generation_path(target)
    if not marker.exists():
        # A payload with no marker predates its sidecar or was written by a
        # plain overwrite. Generation 0 keeps conditional writes honest: an
        # if-match against a real generation fails rather than silently passing.
        return 0
    text = marker.read_text(encoding="utf-8").strip()
    return int(text) if text.isdigit() else 0


@contextmanager
def _exclusive(target: Path) -> Iterator[None]:
    """Serialise a read-modify-write on *target* across processes.

    The lock is taken on a sibling descriptor rather than on the payload itself
    so the payload can be atomically replaced by rename while held. ``flock`` is
    released by the kernel when the descriptor closes, so an interrupted writer
    cannot leave the key permanently locked.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_name(f".{target.name}.lock")
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _lease_payload(
    *,
    key: str,
    owner_id: str,
    attempt_id: str,
    stage: str,
    now: datetime,
    ttl_seconds: int,
    takeover_count: int = 0,
    previous_owner_id: str = "",
) -> dict[str, Any]:
    return {
        "key": key,
        "owner_id": owner_id,
        "attempt_id": attempt_id,
        "stage": stage,
        "acquired_at": now.isoformat(),
        "heartbeat_at": now.isoformat(),
        "expires_at": (now + timedelta(seconds=ttl_seconds)).isoformat(),
        "takeover_count": takeover_count,
        "previous_owner_id": previous_owner_id,
    }


def _reject_regression(
    incoming: IngestCheckpoint,
    prior: IngestCheckpoint,
    *,
    job_id: str,
    artifact_id: str,
) -> None:
    """Refuse a checkpoint that would move progress backwards.

    The failure this prevents is the quiet one: a superseded attempt resuming
    from its own stale count would lower the mark, and the next resume would
    re-insert rows that already landed while the run reported fewer than it
    holds. Monotonicity is the only reason a checkpoint can be trusted after a
    crash, so it is enforced at the write rather than hoped for.
    """
    if incoming.rows_committed < prior.rows_committed:
        raise ValueError(
            f"Refusing non-monotonic checkpoint for job={job_id} "
            f"artifact={artifact_id}: rows {incoming.rows_committed} "
            f"< {prior.rows_committed}",
        )
    if incoming.chunks_committed < prior.chunks_committed:
        raise ValueError(
            f"Refusing non-monotonic checkpoint for job={job_id} "
            f"artifact={artifact_id}: chunks {incoming.chunks_committed} "
            f"< {prior.chunks_committed}",
        )


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _parse_datetime(value: str) -> datetime:
    text = str(value or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _lease_is_expired(record: LeaseRecord, *, grace_seconds: int) -> bool:
    try:
        expires_at = _parse_datetime(record.expires_at)
    except ValueError:
        # An unparseable expiry cannot be honoured, and treating it as live
        # would strand the work permanently. Expired is the recoverable reading.
        return True
    return _utc_now() >= expires_at + timedelta(seconds=max(int(grace_seconds), 0))


def _path_from_file_uri(uri: str) -> str:
    from urllib.parse import unquote, urlparse

    return unquote(urlparse(uri).path)
