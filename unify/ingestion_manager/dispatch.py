"""Client for the pipeline control plane at ``/infra/pipeline/*``.

The assistant does not talk to the queue or the bucket directly. It posts a run
to the hosted control plane, which holds the cloud credentials and derives the
routing itself.

That indirection is the point rather than an accident of layering. Handing every
assistant pod the credentials to publish to the parse topic and write the
artifact bucket would make each one able to enqueue work for any tenant, and the
blast radius of a compromised pod would be the whole fleet. Posting a run to an
endpoint that authenticates the pod keeps that authority in one service.

**Submitting is two calls, and the bytes go around this client.** ``submit``
returns write targets; the sources and the staged request are uploaded straight
to the store; then ``publish`` emits one parse message per file. The control
plane never carries the payload, so a folder of large files is bounded by the
store's throughput rather than by an HTTP request-size ceiling. Where the store
cannot sign a URL -- self-host, whose store is a shared volume -- the same
targets come back unsigned and the bytes go through the plane's upload route
instead; the caller does not branch on which, because the target says.

The wire contract carries no identity. The control plane reads that from the
authenticated session, so a pod cannot dispatch, observe or recover another
tenant's work even by asking precisely for it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from unify.ingestion_manager.types.request import IngestionRequest
from unify.ingestion_manager.types.run import RetryScope

logger = logging.getLogger(__name__)

# Long enough for the control plane to record jobs and publish, short enough
# that a wedged plane surfaces as an error rather than as a hung submit. The
# manager records the run before dispatching, so a timeout leaves something with
# an id rather than lost work.
_TIMEOUT_S = 120.0

# Uploads get their own, longer budget: this one covers moving file bytes, and a
# ceiling tuned for a control-plane round trip would fail every large file.
_UPLOAD_TIMEOUT_S = 900.0


def _post(base_url: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(
        f"{base_url.rstrip('/')}/infra/pipeline/{path.lstrip('/')}",
        json=payload,
        headers=_auth_headers(),
        timeout=_TIMEOUT_S,
    )
    _raise_for_status(response)
    return response.json() if response.content else {}


def _get(
    base_url: str,
    path: str,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    response = requests.get(
        f"{base_url.rstrip('/')}/infra/pipeline/{path.lstrip('/')}",
        params=params,
        headers=_auth_headers(),
        timeout=_TIMEOUT_S,
    )
    _raise_for_status(response)
    return response.json() if response.content else {}


def _raise_for_status(response: Any) -> None:
    """Fail with the plane's own reason rather than a bare status code.

    The control plane refuses for reasons a caller can act on -- another
    recovery already owns this job, no such dispatch -- and losing that text
    would turn an actionable answer into "409".
    """
    if response.status_code < 400:
        return
    detail = ""
    try:
        body = response.json()
        if isinstance(body, dict):
            detail = str(body.get("detail") or "")
    except Exception:  # noqa: BLE001 -- a non-JSON error body is still an error
        detail = response.text[:300]
    raise RuntimeError(
        f"Pipeline control plane refused ({response.status_code}): "
        f"{detail or 'no detail given'}",
    )


def _auth_headers() -> Dict[str, str]:
    """Authenticate as this assistant, using its own key.

    Its own key rather than an admin one deliberately: the control plane scopes
    the dispatch to whoever asked, so a pod cannot enqueue or inspect work
    belonging to another tenant even if it tries.
    """
    key = _assistant_key()
    if not key:
        raise RuntimeError(
            "No assistant key available to authenticate a pipeline dispatch. "
            "The control plane scopes work to the caller, so an unauthenticated "
            "dispatch cannot be routed.",
        )
    return {"Authorization": f"Bearer {key}"}


def _assistant_key() -> str:
    from unify.session_details import SESSION_DETAILS

    return str(getattr(SESSION_DETAILS, "unify_key", "") or "")


def _assistant_id() -> int:
    """The assistant this pod is, which the plane verifies its key against."""
    from unify.session_details import SESSION_DETAILS

    agent_id = getattr(getattr(SESSION_DETAILS, "assistant", None), "agent_id", None)
    if agent_id is None:
        raise RuntimeError(
            "No assistant id available to authenticate a pipeline dispatch; the "
            "control plane verifies the caller's key against its own assistant.",
        )
    return int(agent_id)


def dispatch_run(
    *,
    base_url: str,
    run_key: str,
    request: IngestionRequest,
    request_key: str,
    request_payload: Dict[str, Any],
    paths: List[str],
    observability: Optional[Dict[str, str]] = None,
) -> str:
    """Stage a run's bytes and hand it to the fleet, returning its dispatch id.

    Three steps, in the only order that is safe: ask for write targets, upload
    everything, then publish. Publishing last means a message never names bytes
    that are not there yet -- a worker picking one up early would fail on a
    missing object and burn a delivery attempt for a run that was fine.

    ``run_key`` becomes the dispatch id rather than the plane minting one, so the
    checkpoints and leases the fleet writes land under the identity the manager
    already recorded. That shared identity is what makes a dispatched run
    resumable in process and an in-process run adoptable by the fleet.
    """
    assistant_id = _assistant_id()

    prepared = _post(
        base_url,
        "submit",
        {
            "assistant_id": assistant_id,
            "run_key": run_key,
            "request_key": request_key,
            "paths": list(paths),
        },
    )

    _upload(
        base_url,
        assistant_id=assistant_id,
        run_key=run_key,
        target=prepared.get("request_upload") or {},
        name="request.json",
        payload=_json_bytes(request_payload),
    )

    sources = prepared.get("sources") or []
    if len(sources) != len(paths):
        raise RuntimeError(
            f"The control plane returned {len(sources)} upload target(s) for "
            f"{len(paths)} file(s); refusing to dispatch a partial run.",
        )
    for target, path in zip(sources, paths):
        # Streamed from disk, never read whole. The dispatch tier exists so a
        # large file's bytes stay out of the assistant's process; loading them
        # here just to upload them would spend the very memory the boundary
        # protects, and it is exactly the multi-hundred-MB files that dispatch.
        with Path(path).open("rb") as source_file:
            _upload(
                base_url,
                assistant_id=assistant_id,
                run_key=run_key,
                target=target,
                name=Path(path).name,
                payload=source_file,
            )

    published = _post(
        base_url,
        "submit/publish",
        {
            "assistant_id": assistant_id,
            "run_key": run_key,
            "request_key": request_key,
            "logical_paths": list(paths),
            "source_uris": [str(target.get("object_uri") or "") for target in sources],
            # Documents stay whole for a collection; a table target merges the
            # tabular content of every file into one queryable context.
            "ingestion_mode": "fm" if request.target.kind == "collection" else "dm",
            "target_context": getattr(request.target, "context", "") or "",
            "destination": request.destination,
            "observability": observability,
        },
    )

    dispatch_id = str(published.get("dispatch_id") or run_key)
    logger.info(
        "Dispatched run %s to the fleet as %s (%d file(s), %d job(s))",
        run_key,
        dispatch_id,
        len(paths),
        int(published.get("jobs") or 0),
    )
    return dispatch_id


def _json_bytes(payload: Dict[str, Any]) -> bytes:
    import json

    return json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")


def _upload(
    base_url: str,
    *,
    assistant_id: int,
    run_key: str,
    target: Dict[str, Any],
    name: str,
    payload: Any,
) -> None:
    """Put one object where the control plane said to put it.

    A signed URL is written directly, so the bytes never touch the control
    plane. Without one the store cannot be reached from here, and the bytes go
    through the plane's upload route -- which is the self-host shape, where the
    plane and the workers share a volume. The caller does not choose: the target
    it was handed says which applies.

    ``payload`` is bytes or an open binary file; a file streams from disk with
    its length taken from the descriptor, so an upload's memory cost is a
    buffer, not the file.
    """
    upload_url = str(target.get("upload_url") or "")
    if upload_url:
        response = requests.put(
            upload_url,
            data=payload,
            headers={"Content-Type": "application/octet-stream"},
            timeout=_UPLOAD_TIMEOUT_S,
        )
        _raise_for_status(response)
        return

    response = requests.put(
        f"{base_url.rstrip('/')}/infra/pipeline/upload/{run_key}/{name}",
        params={"assistant_id": assistant_id},
        data=payload,
        headers={**_auth_headers(), "Content-Type": "application/octet-stream"},
        timeout=_UPLOAD_TIMEOUT_S,
    )
    _raise_for_status(response)


def fetch_status(*, base_url: str, dispatch_id: str) -> Dict[str, Any]:
    """Read the fleet's view of a dispatch.

    The fleet owns the truth about a dispatched run while it executes, so this
    is what the manager folds into the run row on read. It answers "is anything
    still working on it", which a checkpoint cannot -- a stalled run and a slow
    one look identical from progress alone.
    """
    return _get(
        base_url,
        f"status/{dispatch_id}",
        params={"assistant_id": _assistant_id()},
    )


def request_retry(
    *,
    base_url: str,
    dispatch_id: str,
    scope: RetryScope,
    files: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Ask the fleet to re-attempt part of a dispatch.

    Serialised by the control plane rather than here. Two recovery paths
    publishing at once -- a retry and a stale-recovery on the same job -- put
    two live attempts against one lease, and the loser's writes freeze the
    checkpoint: the run then under-ingests without reporting anything. The plane
    takes a per-job recovery lease, so one owner of the transition removes that
    race instead of a flag warning operators not to cause it.
    """
    payload: Dict[str, Any] = {
        "assistant_id": _assistant_id(),
        "dispatch_id": dispatch_id,
        "scope": scope,
    }
    if files:
        # Omitted entirely when absent, so a plane that does not know the key
        # keeps its whole-dispatch behaviour rather than receiving null and
        # having to interpret it.
        payload["files"] = list(files)
    return _post(base_url, "retry", payload)


def request_cancel(*, base_url: str, dispatch_id: str) -> Dict[str, Any]:
    """Abandon a dispatch's remaining work, keeping what already committed."""
    return _post(
        base_url,
        "cancel",
        {"assistant_id": _assistant_id(), "dispatch_id": dispatch_id},
    )


def request_pause(*, base_url: str, dispatch_id: str) -> Dict[str, Any]:
    """Stop a dispatch but keep its outstanding work.

    In-flight messages are parked rather than dropped, so the queue drains and
    the fleet can scale down while nothing is lost. Resume replays them.
    """
    return _post(
        base_url,
        "pause",
        {"assistant_id": _assistant_id(), "dispatch_id": dispatch_id},
    )


def request_resume(*, base_url: str, dispatch_id: str) -> Dict[str, Any]:
    """Replay a paused dispatch's parked work, in its original order."""
    return _post(
        base_url,
        "resume",
        {"assistant_id": _assistant_id(), "dispatch_id": dispatch_id},
    )


def probe(*, base_url: Optional[str]) -> bool:
    """Report whether a control plane is configured and answering.

    Used to decide whether the fleet is reachable at all. A deployment without
    one runs everything in process, which is safe because both tiers write the
    same layout -- a fleet configured later adopts whatever was left behind.

    Health is unauthenticated and reports whether the plane can actually reach
    its own backends, so a configured-but-broken plane reads as absent rather
    than accepting work that would go nowhere.
    """
    if not base_url:
        return False
    try:
        body = _get(base_url, "health")
    except Exception:
        logger.warning("Pipeline control plane at %s is not answering", base_url)
        return False
    if not body.get("ok"):
        logger.warning(
            "Pipeline control plane at %s is reachable but not usable: %s",
            base_url,
            body.get("detail") or "no backends configured",
        )
        return False
    return True
