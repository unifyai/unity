# Call Recording Architecture

This document is the ground truth for how voice calls and Unify Meets are recorded, stored, and linked to transcript data. All design decisions and implementation details are captured here. Every future decision regarding call recording should be based on this document.

## Table of Contents

- [Design Principles](#design-principles)
- [High-Level Flow](#high-level-flow)
- [Recording Mechanism: LiveKit Egress](#recording-mechanism-livekit-egress)
- [GCS Storage Layout](#gcs-storage-layout)
- [Pub/Sub Event: recording\_ready](#pubsub-event-recording_ready)
- [Transcript Integration](#transcript-integration)
- [Per-Utterance Timing](#per-utterance-timing)
- [End-to-End Walkthrough: Phone Call](#end-to-end-walkthrough-phone-call)
- [End-to-End Walkthrough: Unify Meet](#end-to-end-walkthrough-unify-meet)
- [Repos and Files Involved](#repos-and-files-involved)
- [Environment Variables](#environment-variables)
- [What Was Removed](#what-was-removed)
- [Infrastructure Prerequisites](#infrastructure-prerequisites)
- [Current Limitations and Known Gaps](#current-limitations-and-known-gaps)

---

## Design Principles

1. **One recording mechanism for all call types.** Both phone calls (Twilio/PSTN) and Unify Meets (browser WebRTC) are recorded via LiveKit Egress. There is no separate Twilio-specific recording path.

2. **Recording is a property of an exchange, not a separate entity.** A phone call or Unify Meet maps to a single transcript exchange. The recording URL lives on that exchange's metadata, making it self-contained. There is no separate recordings table or registry.

3. **LiveKit writes directly to GCS.** LiveKit Egress uploads the recording file to a GCS bucket. No service downloads or re-uploads the file.

4. **Asynchronous linking.** The recording URL is not available at call-end time. It arrives later via a LiveKit webhook. The session identifier (`conference_name` for phone calls, `room_name` for Unify Meets) is stored on the exchange at call-end time. The recording URL is added asynchronously when the `RecordingReady` event arrives.

5. **One recording per session.** Each call or meet produces a single MP3 file covering the full duration of the session. There is no chunk-by-chunk recording. The entire call is one continuous audio file.

---

## High-Level Flow

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────┐     ┌──────────────┐
│ Call starts   │────►│  Communication   │────►│ LiveKit     │────►│ GCS Bucket   │
│ (Twilio or   │     │  starts egress   │     │ Egress      │     │ (MP3 file)   │
│  WebRTC)     │     │  on LiveKit room │     │ records     │     │              │
└──────────────┘     └──────────────────┘     │ full call   │     └──────┬───────┘
                                               └──────┬──────┘            │
                                                      │                   │
                                                      │  egress complete  │
                                                      ▼                   │
                                               ┌──────────────────────┐   │
                                               │  Adapter             │   │
                                               │  /livekit/           │   │
                                               │  recording-complete  │   │
                                               │  (ensures job alive, │   │
                                               │   publishes Pub/Sub) │   │
                                               └──────┬───────────────┘   │
                                                      │                   │
                                                      │ Pub/Sub           │
                                                      │ recording_ready   │
                                                      ▼                   │
                                               ┌──────────────────┐       │
                                               │  Unity receives  │◄──────┘
                                               │  RecordingReady  │  (URL references
                                               │  event           │   the GCS file)
                                               └──────┬───────────┘
                                                      │
                                                      │ stores recording_url
                                                      │ on exchange metadata
                                                      ▼
                                               ┌──────────────────┐
                                               │  Exchange now    │
                                               │  has recording   │
                                               │  URL inline      │
                                               └──────────────────┘
```

---

## Recording Mechanism: LiveKit Egress

All recording uses **LiveKit Room Composite Egress** — a server-side feature where LiveKit mixes all audio tracks in a room and writes the result to a file.

### Why LiveKit Egress (not Twilio Conference Recording)

Phone calls flow through both Twilio and LiveKit: the PSTN side is handled by a Twilio Conference, which bridges via SIP to a LiveKit room where the AI agent lives. Previously, Twilio Conference's own `record="record-from-start"` feature was used. This was replaced with LiveKit Egress for several reasons:

- **Unified path.** Unify Meets only go through LiveKit (no Twilio). Using LiveKit Egress for both means one recording mechanism instead of two.
- **Direct GCS upload.** LiveKit Egress writes directly to GCS. Twilio recording required downloading from Twilio, then uploading to GCS, then notifying Orchestra — three hops instead of zero.
- **No dependency on Twilio for recording.** The Twilio Conference is still used for PSTN bridging, but recording is decoupled from it.

### Egress Configuration

The egress is started via `start_room_egress()` (or `create_room_and_dispatch_agent()` with `record=True`) in `common/livekit.py`:

- **Format:** MP3 (audio-only, `file_type=3`)
- **Mixing:** Room Composite — all audio tracks in the room are mixed into a single file
- **Upload target:** GCS bucket via `GCPUpload` with credentials from `LIVEKIT_EGRESS_GCS_CREDENTIALS`
- **Webhook:** On completion, LiveKit calls `{UNITY_ADAPTERS_URL}/livekit/recording-complete?assistant_id=X&room_name=Y`, signed with `LIVEKIT_API_KEY`

### When Egress Starts

| Call Type | Where Egress is Started | Code Location |
|-----------|------------------------|---------------|
| **Phone call** (Twilio inbound) | After Twilio conference setup in the adapter webhook | `adapters/main.py` line ~177: `await start_room_egress(room_name, assistant_id)` |
| **Unify Meet** (browser WebRTC) | Inside `create_room_and_dispatch_agent()` when `record=True` | `common/livekit.py` — called from the `/dispatch-livekit-agent` endpoint |

For phone calls, the LiveKit room name is `unity_{twilio_number}` (e.g. `unity_12025551234`). For Unify Meets, it defaults to `unity_{assistant_id}_web` or a custom name passed by the caller.

### Egress Lifecycle

1. **Start**: `start_room_egress()` calls LiveKit's `start_room_composite_egress` API
2. **Recording**: LiveKit Egress runs server-side, recording all audio in the room for the full duration
3. **End**: When the room closes (all participants leave), egress finalizes the file
4. **Upload**: LiveKit writes the MP3 to the configured GCS bucket
5. **Webhook**: LiveKit sends a POST to the adapter's `/livekit/recording-complete` endpoint

---

## GCS Storage Layout

**Bucket:** `unity-call-recordings` (configurable via `LIVEKIT_EGRESS_GCS_BUCKET`)

**File path pattern:** `{environment}/{assistant_id}/{room_name}.mp3`

| Environment | Prefix | Example Path |
|-------------|--------|--------------|
| Staging | `staging/` | `staging/25/unity_12025551234.mp3` |
| Production | `production/` | `production/25/unity_12025551234.mp3` |

The environment is determined by the `STAGING` env var in the Communication service. If `STAGING` is truthy, the prefix is `staging`; otherwise `production`.

**Public URL format:** `https://storage.googleapis.com/unity-call-recordings/{prefix}/{assistant_id}/{room_name}.mp3`

This is the URL stored in exchange metadata as `recording_url`. The Console's `AudioPlayer` component can take this URL and generate a signed URL for playback via the Console's own `/api/media/get` route.

---

## Pub/Sub Event: recording_ready

When LiveKit Egress completes and the webhook fires, the **adapter** (not the comms app) handles the webhook, ensures the assistant's Unity container is running, and publishes a Pub/Sub message to the assistant's topic.

### Message Format

```json
{
    "thread": "recording_ready",
    "event": {
        "assistant_id": "25",
        "conference_name": "unity_12025551234",
        "recording_url": "https://storage.googleapis.com/unity-call-recordings/staging/25/unity_12025551234.mp3"
    }
}
```

**Key fields:**
- `conference_name` — the LiveKit room name. This is the join key used to match the recording to its exchange. For phone calls this is the room name (e.g. `unity_12025551234`), for Unify Meets it's also the room name (e.g. `unity_25_web`).
- `recording_url` — the public GCS URL for the MP3 file.

### Publishing Logic

The `/livekit/recording-complete` adapter endpoint in `adapters/main.py`:
1. Verifies the LiveKit webhook signature via `verify_livekit_webhook()` (shared helper in `common/livekit.py`)
2. Calls `build_webhook_context()` with `validate_contact=False, ensure_job=True` to ensure the assistant's Unity container is alive (starts a new job if the container shut down while waiting for the recording)
3. Constructs the Pub/Sub topic name: `unity-{assistant_id}[-staging]`
4. Publishes the `recording_ready` JSON message to that topic

---

## Transcript Integration

Recording data is linked to transcripts via the **Exchange** abstraction. Each call/meet session maps to exactly one exchange, and the recording metadata lives on that exchange.

### Data Model

Exchanges support arbitrary `metadata` (a dict). The recording-related keys are:

| Key | Set When | Set By | Value |
|-----|----------|--------|-------|
| `conference_name` | Phone call ends (`PhoneCallEnded`) | Event handler in `event_handlers.py` | Twilio conference name, e.g. `Unity_12025551234_2026_02_18_10_30_00` |
| `room_name` | Unify Meet ends (`UnifyMeetEnded`) | Event handler in `event_handlers.py` | LiveKit room name, e.g. `unity_25_web` |
| `recording_url` | Recording is ready (`RecordingReady`) | Event handler in `event_handlers.py` | Full GCS public URL |

### How Exchange Metadata is Populated

**Step 1: At call/meet end** — The `PhoneCallEnded`/`UnifyMeetEnded` handler:
- Reads the `exchange_id` from `call_manager.call_exchange_id` or `call_manager.unify_meet_exchange_id`
- Reads the session identifier from `call_manager.conference_name` or `call_manager.room_name`
- Calls `transcript_manager.update_exchange_metadata(exchange_id, {"conference_name": ...})` (or `{"room_name": ...}`)
- Stashes the mapping `session_identifier -> exchange_id` in `cm._recording_exchange_ids` (an in-memory dict on the ConversationManager)

**Step 2: When recording arrives** — The `RecordingReady` handler:
- Pops the `exchange_id` from `cm._recording_exchange_ids` using the `conference_name` from the event
- Calls `transcript_manager.update_exchange_metadata(exchange_id, {"recording_url": ...})`

This two-step approach is necessary because the recording arrives asynchronously — often seconds or minutes after the call ends. The in-memory dict bridges the gap without requiring a database query.

### Exchange ID Assignment

Each call/meet session gets a single exchange ID, assigned when the first message (utterance) is logged:
- `call_exchange_id` for phone calls
- `unify_meet_exchange_id` for Unify Meets

These are set in the `LogMessageResponse` handler via `log_first_message_in_new_exchange()` and reset to `UNASSIGNED` (-1) at call end.

### Result

After both steps complete, the exchange for a call looks like:

```python
{
    "exchange_id": 42,
    "medium": "phone_call",  # or "unify_meet"
    "metadata": {
        "conference_name": "Unity_12025551234_2026_02_18_10_30_00",  # or "room_name" for meets
        "recording_url": "https://storage.googleapis.com/unity-call-recordings/staging/25/unity_12025551234.mp3"
    },
    "messages": [
        {"content": "Hello?", "sender_id": 1, "metadata": {"call_utterance_timestamp": "00.03"}},
        {"content": "Hi, how can I help?", "sender_id": 0, "metadata": {"call_utterance_timestamp": "00.07"}},
        ...
    ]
}
```

---

## Per-Utterance Timing

Each message logged during a call/meet includes a `call_utterance_timestamp` in its metadata. This is a timestamp offset from the start of the call, formatted as `MM.SS`.

### How It Works

In `managers_utils.py`, every time a call utterance is logged:
1. The call start time is read from `call_manager.call_start_timestamp` (phone) or `call_manager.unify_meet_start_timestamp` (meet)
2. The delta from call start to now is computed
3. For assistant utterances, 2 seconds are added to approximate TTS playback delay
4. The result is formatted as `MM.SS` (e.g. `"02.15"` = 2 minutes 15 seconds into the call)
5. This is stored in `message.metadata["call_utterance_timestamp"]`

### Timestamps Are Set at Call Start

The `PhoneCallStarted`/`UnifyMeetStarted` handler sets:
- `cm.call_manager.call_start_timestamp = event.timestamp` (for phone calls)
- `cm.call_manager.unify_meet_start_timestamp = event.timestamp` (for meets)

These are cleared at call end.

### Purpose

Per-utterance timestamps enable precise time-alignment of transcript text to audio playback. A consumer can take the `recording_url` from exchange metadata and the `call_utterance_timestamp` from each message to highlight which utterance is playing at any given point in the audio.

---

## End-to-End Walkthrough: Phone Call

1. **Incoming call** → Twilio webhook hits `adapters/main.py` `/twilio/call`
2. **Conference setup** → `create_conference_response()` creates a Twilio Conference (no recording flag — recording is handled by LiveKit)
3. **SIP bridge** → Twilio bridges the PSTN caller to a LiveKit room via SIP trunk. Room name: `unity_{twilio_number}`
4. **Start egress** → `await start_room_egress(room_name, assistant_id)` starts LiveKit Room Composite Egress on the room (fire-and-forget, errors are non-fatal)
5. **Pub/Sub** → Adapter publishes `call` thread event to `unity-{assistant_id}[-staging]` topic
6. **Unity receives call** → `CommsManager` routes to `PhoneCallReceived` event → sets `conference_name` on `call_manager`
7. **Call starts** → `PhoneCallStarted` event → sets `call_start_timestamp` on `call_manager`
8. **Utterances** → Each utterance is logged with `call_utterance_timestamp` in message metadata
9. **LiveKit Egress records** → Server-side, LiveKit mixes all audio tracks in the room into a single MP3, streaming to GCS
10. **Call ends** → `PhoneCallEnded` event handler:
    - Stores `conference_name` in exchange metadata
    - Stashes `conference_name -> exchange_id` in `_recording_exchange_ids`
    - Clears all session state (timestamps, exchange IDs, conference_name)
11. **Egress completes** → LiveKit finishes writing MP3 to GCS, fires webhook to `/livekit/recording-complete` on the adapters
12. **Adapter handler** → Verifies LiveKit signature, ensures Unity container is alive (starts a job if needed), constructs `recording_url`, publishes `recording_ready` Pub/Sub event
13. **Unity receives recording** → `CommsManager` routes to `RecordingReady` event → handler looks up `exchange_id` from `_recording_exchange_ids`, stores `recording_url` on exchange metadata

---

## End-to-End Walkthrough: Unify Meet

1. **User starts meet** → Console/frontend calls `/phone/dispatch-livekit-agent` with `record=True`, `assistant_id`, `room_name`
2. **Room creation + agent dispatch** → `create_room_and_dispatch_agent()` in `common/livekit.py` creates a LiveKit room and dispatches the AI agent
3. **Start egress** → Since `record=True`, `_start_room_egress()` is called immediately after agent dispatch
4. **Pub/Sub** → A `unify_meet` thread event is published to `unity-{assistant_id}[-staging]` topic
5. **Unity receives meet** → `CommsManager` routes to `UnifyMeetReceived` event → `call_manager.start_unify_meet()` sets `room_name` on `call_manager`
6. **Meet starts** → `UnifyMeetStarted` event → sets `unify_meet_start_timestamp` on `call_manager`
7. **Utterances** → Each utterance is logged with `call_utterance_timestamp` in message metadata
8. **LiveKit Egress records** → Same as phone calls — full room audio mixed to MP3
9. **Meet ends** → `UnifyMeetEnded` event handler:
    - Stores `room_name` in exchange metadata
    - Stashes `room_name -> exchange_id` in `_recording_exchange_ids`
    - Clears all session state
10. **Egress completes** → Same adapter webhook flow as phone calls (`/livekit/recording-complete`)
11. **Unity receives recording** → Same `RecordingReady` handler as phone calls

---

## Repos and Files Involved

### Communication (`communication/`)

| File | Role |
|------|------|
| `common/livekit.py` | Shared LiveKit utilities: `get_livekit_api()`, `create_room_and_dispatch_agent()`, `start_room_egress()`, `_start_room_egress()`, `verify_livekit_webhook()` |
| `adapters/main.py` | `/livekit/recording-complete` webhook handler (verifies signature, ensures job, publishes Pub/Sub); Twilio call webhook (starts egress after conference setup) |
| `adapters/helpers.py` | Imports `start_room_egress` from `common.livekit`, `create_conference_response()` (no recording flag) |
| `communication/phone/views.py` | `/dispatch-livekit-agent` endpoint (for Unify Meets) |

### Unity (`unity/`)

| File | Role |
|------|------|
| `unity/conversation_manager/events.py` | `RecordingReady` event dataclass |
| `unity/conversation_manager/comms_manager.py` | Routes `recording_ready` Pub/Sub thread to `RecordingReady` event |
| `unity/conversation_manager/domains/event_handlers.py` | `RecordingReady` handler (stores URL on exchange), `PhoneCallEnded`/`UnifyMeetEnded` handler (stores session ID on exchange) |
| `unity/conversation_manager/domains/call_manager.py` | `conference_name` and `room_name` attributes, `start_unify_meet()` |
| `unity/conversation_manager/domains/managers_utils.py` | `call_utterance_timestamp` computation and storage in message metadata |
| `unity/conversation_manager/conversation_manager.py` | `_recording_exchange_ids: dict[str, int]` in-memory mapping |

### Orchestra (`orchestra/`)

Orchestra has **no role** in call recording. The `assistant_call_recording` table, its DAO, service, endpoints, and schemas were all deleted. A migration (`2026-02-17-14-00_drop_assistant_call_recording.py`) drops the table. Recordings are stored in GCS and referenced from exchange metadata — Orchestra is not in the loop.

### Console (`console/`)

The Console already has an `AudioPlayer` component that handles GCS URLs. When a GCS public URL appears in interface data (e.g. from exchange metadata), the `AudioPlayer`:
1. Extracts bucket and path from the URL
2. Calls the Console's `/api/media/get` to generate a signed URL
3. Renders an `<audio controls>` element

No Console changes were needed for recording support.

---

## Environment Variables

### Required for recording to work

| Variable | Service | Purpose |
|----------|---------|---------|
| `LIVEKIT_EGRESS_GCS_CREDENTIALS` | Communication (adapters + comms app) | GCS service account JSON that LiveKit Egress uses to upload recordings to the bucket |
| `LIVEKIT_API_KEY` | Communication (adapters + comms app) | LiveKit API key — used to start egress and verify webhook signatures |
| `LIVEKIT_API_SECRET` | Communication (adapters) | LiveKit API secret — used alongside `LIVEKIT_API_KEY` for webhook verification |
| `LIVEKIT_URL` | Communication (adapters + comms app) | LiveKit server URL |
| `UNITY_ADAPTERS_URL` | Communication (adapters + comms app) | Public URL of the adapters service — used as the base for the egress webhook callback |
| `GCP_PROJECT_ID` | Communication (adapters) | GCP project ID — used for Pub/Sub topic path construction |
| `STAGING` | Communication (adapters + comms app) | If truthy, recordings go to `staging/` prefix; otherwise `production/` |

### Optional

| Variable | Service | Default | Purpose |
|----------|---------|---------|---------|
| `LIVEKIT_EGRESS_GCS_BUCKET` | Communication | `unity-call-recordings` | GCS bucket name for recordings |

---

## What Was Removed

The following were explicitly deleted as part of this consolidation:

### Orchestra
- `CallRecording` SQLAlchemy model and `recordings` relationship on `Assistant`
- `recording_dao.py` (DAO for recording CRUD)
- `call_recording_service.py` (service layer)
- `POST /assistant/{id}/recordings`, `GET /assistant/{id}/recordings`, `DELETE /assistant/{id}/recordings/{recording_id}` endpoints
- `RecordingCreate` and `RecordingInfo` Pydantic schemas
- `upload_recording()` and `assistant_recordings_bucket` from `BucketService`
- `test_assistant_recordings_audio_lifecycle` test
- `ORCHESTRA_GCP_ASSISTANT_RECORDINGS_BUCKET_NAME` env var from CI

### Communication
- Twilio `/recording` callback endpoint (the one that downloaded from Twilio, uploaded to Orchestra)
- `record="record-from-start"` and `recording_status_callback` from `create_conference_response()`
- `_get_recordings_bucket()` helper (for Twilio recording GCS upload)
- `/phone/egress-complete` endpoint and `_publish_recording_ready()` from the comms app (moved to the adapters as `/livekit/recording-complete`)

### Unity
- `debug_audio.py` script (stale debugging tool with hardcoded bucket paths and heavyweight desktop deps)
- Hardcoded `call_url` GCS URL construction in `managers_utils.py` (was computed but never stored)
- Commented-out `call_url` and `call_utterance_timestamp` lines from old event fields

---

## Infrastructure Prerequisites

For recording to work end-to-end, the following must be true:

1. **LiveKit Egress must be enabled** on the LiveKit Cloud project (or a self-hosted Egress service must be running). Without this, `start_room_composite_egress` API calls will fail.

2. **`GOOGLE_APPLICATION_CREDENTIALS`** must be set on both the adapters and communication Cloud Run services. This is a GCS service account JSON string with `Storage Object Creator` permissions on the `unity-call-recordings` bucket.

3. **`UNITY_ADAPTERS_URL` must be publicly reachable** from LiveKit's infrastructure, so the egress completion webhook can reach `/livekit/recording-complete` on the adapters.

4. **The `/livekit/recording-complete` adapter endpoint** has no application-level auth beyond LiveKit's own webhook signing (verified via `WebhookReceiver`/`TokenVerifier`). If there's infrastructure-level auth (API gateway, load balancer) that blocks unauthenticated requests to the adapters service, the webhook will be rejected.

5. **The GCS bucket `unity-call-recordings` must exist** in the GCP project (`responsive-city-458413-a2`), with the service account from `GOOGLE_APPLICATION_CREDENTIALS` having write access.

6. **GCP_PROJECT_ID** must be set (already required for all other Pub/Sub publishing).

### Failure modes

All egress-related code is wrapped in exception handlers. If any prerequisite is missing:
- Calls and meets still work normally — the recording code is fire-and-forget
- The recording simply won't be produced or linked
- Errors are logged to stdout (e.g. `[Egress] Non-fatal: failed to start egress for call: ...`)

---

## Current Limitations and Known Gaps

1. **`_recording_exchange_ids` is in-memory.** The adapter now ensures the container is alive before publishing `recording_ready`, which prevents the message from being lost to an empty Pub/Sub topic. However, if the original container shut down and a *new* container was started, the in-memory `_recording_exchange_ids` dict is empty and the handler cannot resolve the exchange. The recording file still exists in GCS but the URL will not be linked to its exchange. A persistent lookup (e.g. filtering exchanges by `metadata.conference_name`) would be needed to fully close this gap.

2. **No signed URL generation in the recording flow.** The `recording_url` stored on exchange metadata is a raw GCS public URL. The Console's `AudioPlayer` handles signed URL generation client-side. Direct API consumers would need to generate their own signed URLs or rely on the Console's `/api/media/get` route.

3. **Phone call room name reuse.** Phone call LiveKit room names are `unity_{twilio_number}` — they're tied to the assistant's phone number, not to the specific call. If two calls arrive on the same number in quick succession, the second call's egress could overwrite the first call's recording file (same filename). The `conference_name` (which includes a timestamp, e.g. `Unity_12025551234_2026_02_18_10_30_00`) is unique per call but is stored in exchange metadata, not used as the GCS filename.

4. **Egress start timing for phone calls.** Egress is started in the adapter webhook, which runs during Twilio's initial callback. The LiveKit room may not have audio tracks yet if the SIP bridge hasn't connected. LiveKit Egress should wait for tracks to appear, but this is an edge case that hasn't been validated in production.

5. **No retry on webhook failure.** If the `/livekit/recording-complete` webhook fails (network issue, adapters service down), LiveKit may retry depending on its configuration, but there's no application-level retry or dead-letter queue. The recording file would exist in GCS but the Pub/Sub event would never be published.

6. **`user_id` is accepted but unused.** The `/dispatch-livekit-agent` endpoint accepts a `user_id` parameter and passes it to `create_room_and_dispatch_agent()`, but the function doesn't use it. It could be useful for organizing recordings by user in the future.
