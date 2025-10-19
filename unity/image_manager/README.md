## Image Manager – Pending Images and Resolution

This module lets you capture images, hand them off to another process immediately, use them for vision Q&A right away, and later resolve their real backend `image_id` after upload. It does this by issuing temporary, client‑side ids ("pending ids") and seeding a process‑local `DataStore` with the full base64 image content.

### Key Concepts
- **Pending id**: A temporary integer id (≥ 10^12) used before the backend assigns a real `image_id`. Check with `ImageHandle.is_pending` or `ImageManager.is_pending_id(...)`.
- **Local DataStore mirror**: On staging, the full row (`image_id`, `timestamp`, optional `caption`, `data` base64) is written locally, so other APIs can read the image immediately without waiting for upload.
- **temp_image_id (local‑only)**: A persistent local column capturing the original temporary id used at staging time. It remains in the local `DataStore` even after resolution and is never written upstream. Use it to correlate pre/post‑upload activity or to preserve provenance of an image across the id remap.
- **Resolution**: Calling `ImageManager.flush_pending([...])` uploads the pending rows and returns a mapping `{pending_id -> real_id}`. The local `DataStore` is re‑keyed to the real id.

## API Summary

### ImageManager
- `stage_image(*, timestamp: datetime | None, caption: str | None, data: bytes | bytearray | str) -> ImageHandle`
  - Creates a pending image with a temporary id and seeds the local `DataStore` with base64. Returns an `ImageHandle` that can be used immediately.
  - The staged row also includes `temp_image_id` (equal to the temporary id at staging).
- `flush_pending(pending_ids: list[int]) -> dict[int, int]`
  - Uploads staged rows to the backend and returns `{pending_id: real_id}`. Also re‑keys the local `DataStore` from pending to real ids.
  - The resolved row in the local `DataStore` retains the original `temp_image_id` for traceability.
- `get_images(image_ids: list[int]) -> list[ImageHandle]`
  - Returns handles for ids, preferring the local `DataStore` (works for both pending and resolved ids).
- `is_pending_id(image_id: int | str) -> bool`
  - Returns true if the id is a client‑side pending id.

### ImageHandle
- Properties: `image_id: int`, `is_pending: bool`, `caption: str | None`, `timestamp: datetime`
- Methods:
  - `raw() -> bytes`: Returns image bytes immediately (uses cached base64 or downloads if URL).
  - `ask(question: str) -> str` (async): Runs a single‑shot vision Q&A call using the image.
  - `update_metadata(*, caption: str | None, timestamp: datetime | None, data: bytes | bytearray | str | None) -> None`:
    - Always updates local cache; if resolved, also persists to backend.
  - `resolve(real_image_id: int) -> None`: Rebinds the handle to a known real id (useful after mapping).


## End‑to‑End Flow

### 1) Producer: capture and share immediately (no labels or upload yet)
```python
from unity.image_manager.image_manager import ImageManager

producer = ImageManager()

def on_image_captured(raw_bytes: bytes):
    handle = producer.stage_image(data=raw_bytes)  # caption/timestamp optional
    # share `handle` directly within the same Python session
```

Notes:
- `stage_image` assigns a pending id and seeds the `DataStore` with base64, so the consumer can use the image immediately.
- You don’t need a caption yet; add it later with `handle.update_metadata(caption=...)` in either process.

### 2) Consumer: use the image immediately (ask/raw) before labels or upload
```python
from unity.image_manager.image_manager import ImageManager

consumer = ImageManager()

def on_handle_received(h):
    b = h.raw()                  # bytes are available right away
    # or, asynchronously:
    # answer = await h.ask("What’s in this image?")
```

Notes:


### 3) Resolve to real ids and log them (two patterns)

Pick the pattern that fits your architecture best.

#### Pattern A – Consumer performs the upload and obtains real ids
The consumer owns the upload step and therefore has real ids immediately.
```python
def resolve_and_log(h):
    if h.is_pending:
        mapping = consumer.flush_pending([h.image_id])
        real_id = mapping[h.image_id]
    else:
        real_id = h.image_id
    # Log `real_id` to your table now

    # (optional) provenance: you can fetch the original temp id locally
    ds = DataStore.for_context(consumer._ctx, key_fields=("image_id",))
    temp_id = ds[real_id].get("temp_image_id")
    # Use `temp_id` to correlate any pre‑upload metadata/events to this resolved row
```

Why this is simple: `flush_pending` returns the mapping synchronously, and the `DataStore` is re‑keyed under `real_id`.

#### Pattern B – Producer uploads later; consumer labels/logs after resolution
If your producer is responsible for uploads, it can call `flush_pending([...])` when ready and propagate `{pending_id: real_id}` to consumers out‑of‑band. Consumers can then call `handle.resolve(real_id)` and proceed to log.
```python
# Producer after upload
mapping = producer.flush_pending([pending_id, ...])
# Send mapping to consumer

# Consumer upon receiving mapping
h.resolve(mapping[pending_id])  # rebinds handle to real id
real_id = h.image_id
# Log `real_id` to your table

# (optional) provenance: `temp_image_id` persists on the resolved row locally
ds = DataStore.for_context(consumer._ctx, key_fields=("image_id",))
assert ds[real_id]["temp_image_id"] == pending_id
```

## Common Tasks

### Add labels later (before or after resolution)
```python
h.update_metadata(caption="label set later")
# If pending → stored locally and included in a later flush_pending
# If resolved → also persisted upstream via update_images
```

### Work with multiple images
```python
pending_ids = [h.image_id for h in handles if h.is_pending]
id_map = consumer.flush_pending(pending_ids)  # {pending_id: real_id}
```

## Design Notes
- No backend schema changes: pending ids are a client‑side convention; `image_id` remains an `int`.
- No extra per‑handle local buffers: image data lives in the per‑context `DataStore` and is referenced by id.

### About `temp_image_id`
- `temp_image_id` is a local‑only column stored in the `DataStore` to preserve the original temporary id.
- It is useful for joining pre‑upload traces/metrics with post‑upload records.
- It is never sent upstream or persisted in the backend schema.
- After `flush_pending`, the resolved row keeps `temp_image_id`, while `image_id` becomes the real backend id.
- Upload responsibility is flexible: either process can call `flush_pending` and obtain real ids.
