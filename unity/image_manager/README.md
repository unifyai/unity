# Image Manager – Zero‑wait Producer/Consumer Flow

This README documents the only supported usage pattern right now: a two‑function, zero‑wait pipeline where one function produces `ImageHandle` objects immediately (without blocking on backend upload), and a second function consumes those handles right away (`raw()`/`ask()`), observes label updates as soon as they are set, and later awaits resolution to obtain real `image_id` values and local annotations for logging into a separate table.

Implementation lives in `unity/image_manager/`; representative tests live in `tests/image_manager/`.


## Motivation and guarantees

- Immediate handles: `ImageManager.add_images(..., synchronous=False, return_handles=True)` returns `List[Optional[ImageHandle]]` immediately. Each non‑None handle has a temporary pending id (an `int` ≥ 10**12). Upload is scheduled in the background.
- Instant access: `ImageHandle.raw() -> bytes` and `await ImageHandle.ask(question: str) -> str` work immediately from locally cached data; they do not wait on backend upload.
- Live label visibility: `ImageHandle.update_metadata(caption=..., ...) -> None` updates both the local in‑memory view and the manager’s local `DataStore` instantly. Other code using the same `ImageManager` can read `handle.caption` and get the updated label without notification or delay. When the handle resolves, the last pending updates are coalesced and persisted to the backend.
- Resolution to real ids: `await ImageHandle.wait_until_resolved(...) -> int` returns the real backend `image_id`. Each `ImageHandle` is also directly awaitable. You can `asyncio.gather` many handles to resolve in bulk while preserving input order.


## End‑to‑end flow (the motivating example)

There are two cooperating Python functions.

### Function A (producer/extractor)

Inputs
- Images as bytes or base64 strings; optional `timestamp` and initial `caption`.

Responsibilities
1) Thinly wrap raw images into `ImageHandle` objects without blocking on upload.
2) Optionally attach an initial label (caption) that becomes immediately visible to readers.
3) Pass the list of `ImageHandle` objects to the consumer.
4) Provide per‑image annotations via `handle.annotation = "..."` when they are ready.
   - Annotations are handle‑local and not stored in the `Images` context.
   - They may be ready after images are ready and even after the handles have already been passed to Function B (as long as both functions share the same handle objects).

Key API calls and types
- `ImageManager.add_images(items, synchronous=False, return_handles=True) -> List[Optional[ImageHandle]]`
- `ImageHandle.update_metadata(caption: Optional[str] = None, timestamp: Optional[datetime] = None, data: Optional[bytes|bytearray|str] = None) -> None`
- `ImageHandle.annotation: Optional[str]` (handle‑local, never persisted)

Example

```python
from __future__ import annotations

from typing import Iterable, List, Optional
from datetime import datetime, timezone

from unity.image_manager.image_manager import ImageManager, ImageHandle


def produce_image_handles(raw_images: Iterable[bytes]) -> List[ImageHandle]:
    """Return handles immediately; upload happens in the background."""
    manager = ImageManager()

    items = [
        {
            "timestamp": datetime.now(timezone.utc),
            "caption": None,               # label can be attached now or later
            "data": img_bytes,            # bytes | bytearray | base64 str
        }
        for img_bytes in raw_images
    ]

    handles_opt: List[Optional[ImageHandle]] = manager.add_images(
        items,
        synchronous=False,                 # do not block on backend upload
        return_handles=True,               # return handles, not ids
    )

    handles: List[ImageHandle] = [h for h in handles_opt if h is not None]

    # Optionally set/adjust labels immediately; readers will see these at once.
    for h in handles:
        h.update_metadata(caption="initial label")

    return handles
```


### Function B (consumer)

Inputs
- `handles: List[ImageHandle]` from Function A.

Responsibilities
1) Use `raw()` and `ask()` immediately (no backend wait).
2) See label updates as soon as Function A sets them.
3) Later, await both events per handle and then log elsewhere:
   - (a) resolution to obtain the real `image_id`
   - (b) readiness of the handle‑local `annotation`
   The downstream table populated by Function B should include both the resolved `image_id` and the `annotation` (even though `annotation` is not stored in `Images`).

Key API calls and types
- `ImageHandle.raw() -> bytes`
- `ImageHandle.ask(question: str) -> Awaitable[str]`
- `ImageHandle.caption -> Optional[str]`
- `ImageHandle.wait_until_resolved(timeout: Optional[float] = None) -> Awaitable[int]`
  - Each handle is also awaitable: `await handle` ≡ `await handle.wait_until_resolved()`
- `ImageHandle.wait_for_annotation(timeout: Optional[float] = None) -> Awaitable[Optional[str]]`
  - Await until a local annotation is set on this handle.
- Optional: `ImageHandle.wait_for_caption(timeout: Optional[float] = None) -> Awaitable[Optional[str]]`
  - Await until a non‑None caption/label exists (persisted upstream).

Example

```python
from __future__ import annotations

import asyncio
from typing import Iterable, List

from unity.image_manager.image_manager import ImageHandle


async def consume_and_log(handles: Iterable[ImageHandle]) -> List[int]:
    # Immediate use: raw bytes and vision question/answer
    for h in handles:
        img_bytes: bytes = h.raw()  # works from local cache immediately
        _ = img_bytes               # do something with the bytes

        answer: str = await h.ask("What do you notice in this image?")
        _ = answer                  # use the answer

        # Label is visible immediately if producer called update_metadata
        current_label = h.caption

    # Later: await both annotation readiness and resolution to real backend ids
    pairs = await asyncio.gather(
        *(asyncio.gather(h.wait_for_annotation(), h.wait_until_resolved()) for h in handles)
    )

    annotations: List[Optional[str]] = [a for (a, _rid) in pairs]
    real_ids: List[int] = [_rid for (_a, _rid) in pairs]

    # Log both (real_ids, annotations) into your separate table here
    # (Images does not store annotations; your downstream table should.)
    return real_ids
```


## APIs used (signatures and returns)

- `ImageManager.add_images(items: List[dict], *, synchronous: bool = True, return_handles: bool = False) -> List[int] | List[Optional[ImageHandle]]`
  - This workflow uses `synchronous=False, return_handles=True`.
  - Each non‑None `ImageHandle` has: `image_id: int` (pending id initially), `is_pending: bool`.

- `ImageHandle.raw() -> bytes`
  - Returns decoded bytes from locally cached base64; if the data is a GCS URL, it downloads the bytes once and caches the base64 locally for future reads.

- `ImageHandle.ask(question: str) -> Awaitable[str]`
  - Single vision call; returns a plain text answer.

- `ImageHandle.update_metadata(caption: Optional[str] = None, timestamp: Optional[datetime] = None, data: Optional[bytes|bytearray|str] = None) -> None`
  - Immediate local effect; if pending, updates are coalesced and persisted after resolution.

- `ImageHandle.wait_until_resolved(timeout: Optional[float] = None) -> Awaitable[int]`
  - Returns the real backend `image_id`. Also available via `await handle`.

- `ImageHandle.annotation: Optional[str]`
  - Handle‑local annotation, never persisted to `Images`.

- `ImageHandle.wait_for_annotation(timeout: Optional[float] = None) -> Awaitable[Optional[str]]`
  - Await until a non‑None annotation is set on the handle.

- `ImageHandle.wait_for_caption(timeout: Optional[float] = None) -> Awaitable[Optional[str]]`
  - Await until a non‑None caption exists; useful if caption is added later via `update_metadata`.

- Optional batch alternative: `ImageManager.await_pending(pending_ids: List[int]) -> Awaitable[Dict[int, int]]` to map multiple pending ids to real ids in one call.


## Why `ImageHandle` exists (and when to use the reference types)

`ImageHandle` is a convenient, session‑local wrapper around a single image that:

- Provides immediate, ergonomic access to the raw bytes (`raw()`), and single‑call vision Q&A (`await ask(...)`).
- Keeps local metadata in sync with reads and writes (`update_metadata(...)` coalesces and persists updates once the image resolves; readers see changes instantly).
- Exposes awaitables for workflow coordination: `await wait_until_resolved()` for real `image_id`, `await wait_for_annotation()` for handle‑local annotations, and `await wait_for_caption()` for labels persisted upstream.
- Preserves ordering and favors local cache to avoid unnecessary backend reads.

However, `ImageHandle` instances are Python objects that live only inside your process. When LLMs call tools, they cannot marshal these in‑memory objects directly; tool arguments must be simple, JSON‑serializable structures.

For that reason the module also provides JSON‑friendly reference types that carry only the data LLMs can pass across tool boundaries:

- `RawImageRef` → `{ image_id: int }`
- `AnnotatedImageRef` → `{ raw_image_ref: { image_id: int }, annotation: str }`
- Containers: `ImageRefs`, `RawImageRefs`, `AnnotatedImageRefs` for lists.

Typical pattern:

1) Use `ImageHandle` locally for extraction, labeling, Q&A, and awaiting resolution/annotation.
2) When an LLM (or any remote agent) needs to pass images into tools, convert to reference types so the payload is JSON‑compatible.

Example JSON payloads an LLM can produce:

```json
{
  "images": [
    { "image_id": 123 },
    { "raw_image_ref": { "image_id": 456 }, "annotation": "used in step X" }
  ]
}
```

In this design, `annotation` is context‑specific and not stored in the `Images` context; it travels alongside `image_id` wherever you log or pass the reference. The `Images` table remains the canonical store for image bytes and image‑level metadata (e.g., `caption`, `timestamp`).

## File locations

- Implementation: `unity/image_manager/`
- Tests (examples of this flow): `tests/image_manager/`
