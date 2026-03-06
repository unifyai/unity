# Unity System Infrastructure Documentation

## 🏗️ Overview

The Unity system is a comprehensive multi-channel communication platform that dynamically provisions infrastructure for AI assistants. Each assistant gets its own dedicated communication channels, cloud resources, and notification systems.

## 📋 Table of Contents

- [System Architecture](#-system-architecture)
- [Deployment Components](#-deployment-components)
- [Pub/Sub Architecture](#-pubsub-architecture)
- [Container States & Lifecycle](#-container-states--lifecycle)
- [Hiring Flow](#-hiring-flow)
- [Inbound Communication Flow](#-inbound-communication-flow)
- [Outbound Communication Flow](#-outbound-communication-flow)
- [Inactivity & Idle Container Management](#-inactivity--idle-container-management)
- [Debugging](#-debugging)
- [Infrastructure Components](#️-infrastructure-components)
- [Webhook System](#-webhook-system)
- [Deployment & CI/CD](#-deployment--cicd)
- [Repository Structure](#-repository-structure)

## 🏛️ System Architecture

The system consists of three main repositories:

1. **Orchestra** (`@https://github.com/unifyai/orchestra`) - Main orchestration service with database containing assistants
2. **Communications** (`@https://github.com/unifyai/communication`) - Contains a web app for low-level comms and adapters that capture inbound and perform various tasks
3. **Unity** (current repo) - The container deployed on GKE (each job on GKE is a separate container)

### External Services

| Service | Purpose |
|---------|---------|
| **Twilio** | Phone calls and SMS messaging |
| **Gmail API** | Email send/receive for `@unify.ai` addresses |
| **Microsoft Graph** | Outlook email and Teams integration |
| **LiveKit** | Real-time audio/video calls (Unify Meet) |
| **Google Cloud Pub/Sub** | Message routing between adapters and containers |
| **Google Kubernetes Engine (GKE)** | Container orchestration for Unity jobs |

### Authentication

| Key | Used By | Purpose |
|-----|---------|---------|
| `ORCHESTRA_ADMIN_KEY` | Adapters, Communications | Admin access to Orchestra APIs |
| `api_key` (per user) | Unity containers | User-specific API key for logging |
| `SHARED_UNIFY_KEY` | Debug logger | Shared key for AssistantJobs project access |
| `GCP_SA_KEY` | All services | Google Cloud service account credentials |

## 📦 Deployment Components

### 1. Orchestra
- Hosts the database containing all assistants
- Provides APIs for assistant management (`POST /assistant`, `DELETE /assistant`, etc.)
- Provisions resources during hiring (phone numbers, emails, Pub/Sub topics)

### 2. Communication
Contains two main parts:

| Part | Purpose |
|------|---------|
| **`communication/` web app** | Low-level communications service with endpoints for Gmail, Twilio, etc. to make outbound calls, SMS, emails |
| **`adapters/`** | Webhook handlers that capture inbound from Twilio, Gmail, Microsoft, etc. and route to the appropriate container |

### 3. Unity
- Each job on GKE is a separate container
- Containers can be in **Idle** or **Live** state
- Handles the actual assistant logic, computer use, and conversation management

#### Key Components Inside a Unity Container

| Component | File | Purpose |
|-----------|------|---------|
| **CommsManager** | `comms_manager.py` | Handles Pub/Sub subscriptions, receives inbound messages, bridges external events to internal event broker |
| **ConversationManager** | `conversation_manager.py` | Main orchestrator—manages conversation state, inactivity detection, coordinates all managers |
| **EventBroker** | `event_broker.py` | Internal in-memory message bus for component communication |
| **DebugLogger** | `debug_logger.py` | Logs job lifecycle to AssistantJobs project |

> ⚠️ **Note**: `debug_logger.py` is misnamed—it's not just for debugging. It's the authoritative source for tracking which jobs are live. This file should be renamed to something like `job_tracker.py` or `assistant_jobs_logger.py`.

**Relationship**: CommsManager subscribes to Pub/Sub topics and translates external messages into internal events published to the EventBroker. ConversationManager listens to these events and orchestrates the appropriate response (triggering the actor, sending replies, etc.).

## 📡 Pub/Sub Architecture

### Assistant-Specific Topics
Every assistant has a dedicated Pub/Sub topic for receiving inbound messages:

| Environment | Topic Name Format | Example |
|-------------|-------------------|---------|
| Production | `unity-{assistant_id}` | `unity-6` |
| Staging | `unity-{assistant_id}-staging` | `unity-25-staging` |

### Startup Topics
Used to engage idle containers when an assistant needs to go live:

| Environment | Topic Name |
|-------------|------------|
| Production | `unity-startup` |
| Staging | `unity-startup-staging` |

### Subscriptions
Each Pub/Sub topic has a corresponding subscription (suffix `-sub`):
- Topic `unity-6` → Subscription `unity-6-sub`
- Topic `unity-startup-staging` → Subscription `unity-startup-staging-sub`

Subscriptions are created automatically when the topic is provisioned during hiring.

### Multiple Idle Containers & Startup Race Condition
When multiple idle containers are subscribed to the startup topic, only **one** receives each message (Pub/Sub delivers to one subscriber per subscription). The first container to acknowledge the message transitions to live; others continue waiting. This is handled automatically by Pub/Sub's message delivery semantics.

### Pub/Sub Message Format

All Pub/Sub messages follow this structure:

```json
{
    "thread": "<event_type>",
    "event": { /* event-specific payload */ }
}
```

**Common thread types**:

| Thread | Direction | Purpose |
|--------|-----------|---------|
| `startup` | Adapter → Container | Engage idle container with assistant details |
| `msg` | Adapter → Container | Incoming SMS |
| `email` | Adapter → Container | Incoming email |
| `call` | Adapter → Container | Incoming phone call |
| `unify_message` | Adapter → Container | Message from Unify console |
| `unify_meet` | Adapter → Container | LiveKit meeting invite |
| `unify_message_outbound` | Container → Frontend | Outbound message for UI |
| `assistant_update` | Adapter → Container | Config change for live assistant |

**Example startup message**:
```json
{
    "thread": "startup",
    "event": {
        "api_key": "...",
        "medium": "sms",
        "assistant_id": "25",
        "user_id": "123",
        "assistant_first_name": "Alice",
        "assistant_surname": "Smith",
        "assistant_email": "alice@unify.ai",
        "assistant_number": "+1234567890",
        "user_first_name": "John",
        "user_surname": "Doe",
        "user_email": "john@example.com",
        "user_number": "+0987654321",
        "voice_provider": "cartesia",
        "voice_id": "...",
    }
}
```

## 🔄 Container States & Lifecycle

### 1. Idle State
This is the default state of any newly created container on GKE:

- **No assistant identity**: `agent_id` is `None` to indicate it's not yet assigned
- **Subscribes to startup topics**: Listens to `unity-startup` or `unity-startup-staging`
- **Keep-alive pings**: Sends pings to itself every 30 seconds to avoid the inactivity timeout (6 minutes) of the conversation manager
- **Ready for engagement**: Waiting for a startup message to become live

The container detects it's in idle state by checking `SESSION_DETAILS.assistant.agent_id is None`.

### 2. Live State
When an inbound event occurs, the flow is:

1. **Inbound triggers adapter**: Whether it's wakeup (hiring), SMS, Unify messages, calls, emails, etc., an adapter in the `adapters/` folder gets called from orchestra/console/Twilio/Gmail/etc.
2. **Fetch assistant details**: The adapter fetches the details of that assistant from Orchestra
3. **Check if already live**: Queries the `AssistantJobs` project (shared with the team) to see if the assistant is already live. This is tracked via `debug_logger.py`
4. **Engage idle container if needed**: If not already live, sends a notification to the startup topic with assistant details
5. **Send inbound to assistant topic**: Regardless of step 4, sends a notification to the specific topic of that assistant with the inbound details (e.g., Twilio/LiveKit configs for calls, email/thread IDs for emails)
6. **Idle container receives startup**: When an idle container receives a startup message (handled in `comms_manager.py`):
   - Sets the assistant details in the conversation manager
   - Cancels the subscription to the startup topic
   - Subscribes to the specific topic of that assistant
   - Marks that job as live using `debug_logger.py`
7. **Container is now live**: Listens for any inbound on that assistant's topic until terminated on inactivity (currently 6 minutes)

### Message Ordering: Startup vs Inbound

The adapter sends **both** messages in sequence:
1. First: Startup message to `unity-startup` (if not already live)
2. Second: Inbound details to `unity-{assistant_id}`

The container handles this correctly because:
- The startup message triggers subscription to the assistant's topic
- The inbound message is already waiting in the assistant's topic queue
- As soon as the container subscribes, it receives the queued inbound

If the assistant was already live, only the inbound message is sent, and the existing container receives it immediately.

### Container Lifecycle Diagram

```
┌─────────────────┐     Startup Message     ┌─────────────────┐
│                 │ ◄────────────────────── │                 │
│   Idle State    │                         │    Adapter      │
│  (Pinging 30s)  │                         │   (Inbound)     │
└────────┬────────┘                         └─────────────────┘
         │
         │ Receive startup message
         ▼
┌─────────────────┐
│  Set assistant  │
│  details        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Unsubscribe from│
│ startup topic   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Subscribe to    │
│ assistant topic │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Mark job as     │
│ live (debug_log)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐     6 min inactivity    ┌─────────────────┐
│                 │ ──────────────────────► │                 │
│   Live State    │                         │   Shutdown      │
│ (Processing)    │                         │  (Job retained) │
└─────────────────┘                         └─────────────────┘
```

## 🎯 Hiring Flow

When a user hires an assistant:

1. **Request to Orchestra**: User calls `POST /assistant` endpoint in Orchestra
2. **Resource provisioning**: Orchestra provisions necessary resources:
   - Phone number via `/phone/create`
   - Email address via `/email/create`
   - Pub/Sub topic via `/infra/pubsub/topic`
3. **Wakeup call**: At the end of the hiring endpoint, Orchestra makes a call to `/assistant/wakeup` on the adapters
4. **Startup notification**: The wakeup endpoint sends a ping to `unity-startup` (or `unity-startup-staging`) so an assistant container goes live
5. **New idle container**: A new idle container is created to maintain availability

## 📥 Inbound Communication Flow

For inbound events after hiring (SMS, calls, emails, Unify messages):

```
External Service → Cloud Function/Adapter → Check if Live → Start if needed → Pub/Sub Topic → GKE Container
```

Detailed flow:

1. **External service triggers webhook**: Twilio, Gmail, Microsoft, etc. hit the appropriate adapter endpoint
2. **Adapter resolves assistant**:
   - For phone/SMS: Looks up assistant by the Twilio number that received the call/message (`GET /admin/assistant?phone=+1234567890`)
   - For email: Looks up assistant by the email address (`GET /admin/assistant?email=alice@unify.ai`)
   - For Unify messages: Assistant ID is passed directly in the request
3. **Contact validation**: The adapter verifies the sender is a known contact for this assistant (boss user or saved contact). Unknown senders receive an error response.
4. **Check running status**: Queries `AssistantJobs` project to check if a job is already running for this assistant
5. **Start container if needed**: If not running:
   - Publishes startup message to `unity-startup[-staging]` topic with full assistant details
   - Triggers creation of a new idle job (to replace the one being engaged)
6. **Publish to assistant topic**: Sends inbound details to `unity-{assistant_id}[-staging]`
7. **Container processes**: The `comms_manager.py` in the live container receives and processes the message

### What Happens If No Idle Container Is Available?

If there's no idle container when a startup message is published:
1. The message sits in the Pub/Sub queue (messages are retained for up to 7 days by default)
2. The adapter's reactive replenishment has already triggered `/scheduled/jobs/create`, which will spin up new containers to meet the target
3. As soon as a new idle container starts (~25-30s), it subscribes to the startup topic and picks up the queued message

In the worst case (e.g., GKE node provisioning required), this delay can be 30-60 seconds. The `UNITY_MIN_IDLE_JOBS` floor is the primary defense against this: set it to cover your expected maximum concurrent burst within that window.

### Supported Inbound Channels

| Channel | Adapter Endpoint | Description |
|---------|-----------------|-------------|
| Phone Call | `/twilio/call` | Incoming voice calls via Twilio |
| SMS | `/twilio/sms` | Incoming text messages |
| WhatsApp | `/twilio/whatsapp` | Incoming WhatsApp messages |
| Gmail | `/email/gmail` | Incoming emails via Gmail |
| Outlook | `/email/outlook` | Incoming emails via Microsoft |
| Teams | `/chat/teams` | Teams chat and channel messages |
| Unify Message | `/unify/message` | Internal platform messages |
| Unify Meet | `/unify/meet` | LiveKit meeting invites |

## 📤 Outbound Communication Flow

### Unify Message Outbound
- Directly handled in `comms_utils.py` in the unity container
- Publishes to the Pub/Sub topic with thread `unify_message_outbound`
- Frontend listens to this topic for real-time updates

### Other Outbound (SMS, Email, Calls)
- Goes through the `communication/` web app in the communication repo
- Endpoints available:

| Type | Endpoint | Description |
|------|----------|-------------|
| SMS | `/phone/send-text` | Send SMS via Twilio |
| Email | `/gmail/send` | Send email via Gmail |
| Call | `/phone/send-call` | Initiate outbound call |

## 🔋 Inactivity & Idle Container Management

### Inactivity Timeout
- Containers shut down after **6 minutes** of inactivity
- The `check_inactivity()` method in `conversation_manager.py` monitors activity
- Idle containers ping every **30 seconds** to stay alive (half the timeout)

### What Happens During Shutdown?

When a container hits the inactivity timeout:

1. **Graceful shutdown initiated**: The conversation manager sets the stop event
2. **Cleanup performed**: Active subscriptions are cancelled, resources released
3. **Job marked as done**: `debug_logger.mark_job_done()` updates AssistantJobs with `running: False`
4. **Service deleted**: The external service (for liveview) is deleted
5. **Container exits**: The process exits cleanly
6. **Job retained on GKE**: The job itself is NOT deleted—it remains for log access

**Important**: Any ongoing work is lost when the container shuts down. The assistant does not persist state between sessions. If the user contacts the assistant again, a new container is engaged and starts fresh (though conversation history is available via Unify logs).

### Idle Job Management Strategy

The system maintains a demand-aware pool of "warm" containers using three coordinated mechanisms:

#### Demand-Aware Target Calculation

All three mechanisms share a single function (`get_target_idle_count` in `adapters/helpers.py`) to determine how many idle jobs should exist:

```
target = max(UNITY_MIN_IDLE_JOBS, live_count // UNITY_IDLE_DEMAND_FACTOR)
```

| Env Var | Default | Purpose |
|---------|---------|---------|
| `UNITY_MIN_IDLE_JOBS` | `3` | Absolute floor — guarantees this many warm containers regardless of traffic |
| `UNITY_IDLE_DEMAND_FACTOR` | `5` | Proportional scaling — 1 idle job per N live assistants (e.g., 5 = 20% buffer) |

At small scale the floor dominates (e.g., 10 live assistants → target is 3). At larger scale the proportional buffer takes over (e.g., 100 live assistants with factor 5 → target is 20).

#### Inventory Discovery

Both the creator and cleaner use `get_unity_jobs_inventory()` (in `adapters/helpers.py`) which fetches all active Unity jobs in a **single GKE API call** using the label selector `app=unity,unity-status!=done`. The response includes the `labels` dict for each job, allowing categorization into `live` and `idle` buckets without additional requests.

#### 1. Reactive Replenishment (On Every Inbound)
- **Trigger**: Any inbound event (SMS, call, email, Unify message, hiring wakeup) that engages an idle container.
- **Purpose**: **Immediate Pool Recovery**. As soon as an idle job is consumed, `build_webhook_context` calls `create_job()` which hits the smart `/scheduled/jobs/create` endpoint. The endpoint checks inventory, calculates the gap to the target, and spins up exactly as many jobs as needed.
- **Why**: Ensures the pool is topped up at the exact moment it's most vulnerable — right after a job was consumed.

#### 2. Deployment-Triggered Refresh (CloudBuild)
- **Trigger**: Every successful build in `cloudbuild.yaml` or `cloudbuild-staging.yaml`.
- **Purpose**: **Image Freshness**. Ensures the idle pool picks up the latest `SHORT_SHA` image.
- **Action**: Calls `/scheduled/jobs/create` once (the endpoint is demand-aware, so a single call creates as many jobs as needed to reach the target), waits 30s for them to register, then calls `/scheduled/jobs/cleanup` to remove containers running older image versions.

#### 3. Scheduled Maintenance (Hourly Cron)
- **Trigger**: Cloud Scheduler hits `/scheduled/jobs/create` hourly.
- **Purpose**: **Pool Health & Self-Healing**.
  - **Self-Healing**: Replenishes the pool if containers have crashed or been consumed by high traffic between deployments.
  - **Garbage Collection**: The subsequent cleanup (10 min later) ensures no idle container lives longer than ~70 minutes, preventing memory leaks or stale state.

### Idle Job Creator
- **Adapter**: `/scheduled/jobs/create`
- **Schedule**: Runs hourly, during deployments, and reactively on every inbound engagement
- **Logic**:
  1. Fetches current inventory via `get_unity_jobs_inventory()` (single GKE request)
  2. Calculates the target via `get_target_idle_count(live_count)`
  3. If `current_idle >= target`, returns early ("pool is healthy")
  4. Otherwise, creates `target - current_idle` new jobs using the latest image

### Idle Job Cleaner
- **Adapter**: `/scheduled/jobs/cleanup`
- **Schedule**: Runs 10 minutes after idle job creation
- **Logic**:
  1. Fetches current inventory via `get_unity_jobs_inventory()` (single GKE request)
  2. Calculates the retention target via `get_target_idle_count(live_count)`
  3. Separates recently-created idle jobs (< 11 min old) from older ones
  4. Retains the N most recent idle jobs (where N = target), deletes the rest
  5. Uses `required_labels: {"unity-status": "idle"}` to guard against race conditions where a job transitions to live between the fetch and the delete
- **Important**: Does NOT delete jobs that were live at some point, even if they hit inactivity. This preserves logs for debugging.

### Why Retain Inactive Live Jobs?
Deleting jobs that were live and then went inactive would mean losing their GKE logs, making it harder to debug issues that occurred during their active period.

## 🔍 Debugging

### What is AssistantJobs?

`AssistantJobs` is a **Unify logging project** (not a GKE concept) used to track which containers are currently serving which assistants. It's accessible via the Unify console at `https://console.unify.ai` under the Projects section.

- **Project name**: `AssistantJobs`
- **Context**: `startup_events`
- **Purpose**: Maps job names to assistant IDs and tracks running status
- **Shared access**: The project is accessible to all team members via `SHARED_UNIFY_KEY`

### Finding Issues via AssistantJobs

When investigating an issue at a particular time:

1. **Open AssistantJobs project** on the Unify console (`https://console.unify.ai`)
2. **Filter logs** by `assistant_id` and/or timestamp to find relevant entries
3. **Get the job name** from the `job_name` field (e.g., `unity-2025-01-15-10-30-00-staging`)

### Accessing GKE Logs

1. Go to GKE workloads in Google Cloud Console
2. Find the job by name
3. Open the logs for that specific job
4. Analyze what went wrong during the container's lifecycle

### AssistantJobs Log Structure

Stored via `debug_logger.py`:

```python
{
    "job_name": "unity-2025-01-15-10-30-00-staging",
    "timestamp": "2025-01-15T10:30:00Z",
    "medium": "sms",  # or "call", "email", "wakeup", etc.
    "user_id": "123",
    "assistant_id": "25",
    "user_name": "John Doe",
    "assistant_name": "Alice Smith",
    "user_number": "+1234567890",
    "assistant_number": "+0987654321",
    "user_email": "john@example.com",
    "assistant_email": "alice@unify.ai",
    "running": True,  # False when job completes
    "liveview_url": "https://..."  # URL to view the assistant's desktop
}
```

### What is liveview_url?

Each assistant's VM runs a virtual desktop (VNC via noVNC) accessible over HTTPS through the Caddy reverse proxy. The `liveview_url` is the external URL to view this desktop in real-time:
- Format: `https://unity-assistant-{id}{-staging}.vm.unify.ai/desktop/custom.html`
- Set in `AssistantJobs` by the `AssistantDesktopReady` event handler once the VM's HTTPS endpoint is confirmed reachable (via `_probe_vm_https`)
- The Console polls `AssistantJobs` for this URL to enable the "Share assistant screen" button
- TLS is provided by the wildcard `*.vm.unify.ai` certificate (see "VM TLS Certificates" below)

## 🛠️ Infrastructure Components

Each assistant gets the following dedicated infrastructure:

### Communication Channels

| Component | Endpoint | Purpose |
|-----------|----------|---------|
| **Email Address** | `/email/create` | Dedicated email for the assistant |
| **Phone Number** | `/phone/create` | Voice and SMS communication |

### Cloud Infrastructure

| Component | Endpoint | Naming Convention | Purpose |
|-----------|----------|-------------------|---------|
| **Pub/Sub Topic** | `/infra/pubsub/topic` | `unity-{assistant_id}[-staging]` | Notification routing |
| **Pub/Sub Startup Topic** | `/infra/pubsub/startup` | `unity-startup[-staging]` | Container activation |
| **GKE Job** | `/infra/gke/job` | `unity-{timestamp}[-staging]` | Assistant runtime |

### VM TLS Certificates (Wildcard)

Each VM runs [Caddy](https://caddyserver.com/) as an HTTPS reverse proxy, terminating TLS for the agent-service API (`/api/*`) and noVNC desktop (`/desktop/*`). All VM hostnames follow the pattern `unity-assistant-{id}{-staging}.vm.unify.ai`.

Rather than each VM requesting its own Let's Encrypt certificate via ACME (which counts against LE's **50 certificates per registered domain per week** limit for `unify.ai`), a single **wildcard certificate** for `*.vm.unify.ai` is pre-provisioned and distributed to every VM:

```
Secret Manager                  VM metadata                 Caddyfile
┌──────────────────┐     ┌─────────────────────┐     ┌──────────────────────┐
│ VM_WILDCARD_      │ ──► │ tls-fullchain       │ ──► │ tls fullchain.pem    │
│   FULLCHAIN       │     │ tls-privkey         │     │     privkey.pem      │
│ VM_WILDCARD_      │     │                     │     │                      │
│   PRIVKEY         │     │ (read by startup    │     │ (Caddy uses cert     │
│                   │     │  script on boot)    │     │  directly, no ACME)  │
└──────────────────┘     └─────────────────────┘     └──────────────────────┘
```

**How it works:**

1. A `*.vm.unify.ai` wildcard cert is generated via `certbot` using a DNS-01 challenge against the `unifyai` Cloud DNS zone (project: `unify-dns-server`).
2. The cert (`fullchain.pem`) and key (`privkey.pem`) are stored as secrets `VM_WILDCARD_FULLCHAIN` and `VM_WILDCARD_PRIVKEY` in GCP Secret Manager (project: `unity-assistant-vms`).
3. During VM creation, `vm_helpers.py` fetches both secrets and passes them as GCP instance metadata (`tls-fullchain`, `tls-privkey`).
4. The VM startup script (Ubuntu or Windows) reads the metadata, writes the cert files to disk, and injects an explicit `tls <cert> <key>` directive into the Caddyfile — Caddy serves the wildcard cert immediately without any ACME requests.
5. If the secrets are absent (not yet provisioned, or Secret Manager unavailable), the startup scripts skip the `tls` directive and Caddy falls back to per-hostname ACME.

**Renewal:** LE wildcard certs expire after 90 days. A Cloud Scheduler job (`cert-renewal` / `cert-renewal-staging`) runs at 3 AM UTC on the 1st of each month, hitting `POST /scheduled/cert-renewal` on the adapters. The endpoint checks the current cert's expiry via Secret Manager; if it's within 30 days, it performs a DNS-01 challenge (creating a TXT record in Cloud DNS, completing the ACME exchange, then cleaning up) and adds new versions to both secrets. All subsequently created VMs pick up the new cert automatically; already-running VMs continue using their copy until stopped.

**Why not per-VM ACME:** With high assistant churn (each new `assistant_id` produces a never-before-seen hostname), per-VM ACME can exhaust the 50/week LE rate limit within days. The wildcard cert uses a single LE certificate slot regardless of how many VMs are created.

### Cleanup

All infrastructure can be removed via the `DELETE /assistant` endpoint in Orchestra, which reverses all provisioning steps.

## 🔗 Webhook System

### Adapters (Communication Repo)

Located in the `adapters/` folder, these handle incoming webhooks:

| Endpoint | Purpose |
|----------|---------|
| `/twilio/call` | Handle incoming voice calls |
| `/twilio/sms` | Process incoming SMS messages |
| `/twilio/whatsapp` | Handle WhatsApp messages |
| `/email/gmail` | Process Gmail notifications |
| `/email/outlook` | Process Outlook notifications |
| `/chat/teams` | Handle Teams chat/channel messages |
| `/microsoft/router` | Route Microsoft Graph notifications |
| `/assistant/wakeup` | Wake up an assistant (hiring) |
| `/assistant/update` | Update running assistant config |
| `/unify/message` | Handle internal messages |
| `/unify/meet` | Handle meet invites |
| `/scheduled/jobs/create` | Create new idle job (hourly) |
| `/scheduled/jobs/cleanup` | Clean old idle jobs |
| `/scheduled/email-watches` | Renew email subscriptions |
| `/scheduled/microsoft-tokens` | Refresh Microsoft OAuth tokens |
| `/scheduled/teams-watches` | Renew Teams subscriptions |

### Notification Flow

```
External Service → Adapter Webhook → Check/Start Job → Pub/Sub Topic → GKE Container → Process → Outbound Action
```

## 🚀 Deployment & CI/CD

### Unity Repository

| File | Purpose |
|------|---------|
| `cloudbuild.yaml` | Production environment deployment |
| `cloudbuild-staging.yaml` | Staging environment deployment |

### Communications Repository

| File/Directory | Purpose |
|----------------|---------|
| `cloudbuild/*.yaml` | Build configuration for adapters |
| `.github/deploy.yaml` | Build and deploy the communications Cloud Run service |

### Environment Workflows

- **Staging**: Uses `cloudbuild-staging.yaml` for testing and validation
- **Production**: Uses `cloudbuild.yaml` for live deployment
- **Scheduled Maintenance**:
  - Hourly idle job creation
  - Idle job cleanup 10 minutes after creation
  - Email watch renewals (see below)
  - Microsoft token refresh (every 30-45 minutes)
  - Teams subscription renewal (every 30-45 minutes)

### Email & Teams Watch Subscriptions

Email notifications require active "watches" that tell Gmail/Outlook to push notifications to our webhook:

| Service | Watch Endpoint | Expiry | Renewal |
|---------|---------------|--------|---------|
| Gmail | `/gmail/watch` | 7 days | Daily via `/scheduled/email-watches` |
| Outlook | `/outlook/watch` | 3 days | Daily via `/scheduled/email-watches` |
| Teams | `/teams/watch` | 60 minutes | Every 30-45 min via `/scheduled/teams-watches` |

Without these renewals, the assistant would stop receiving inbound emails/messages.

## 📁 Repository Structure

```
unity/
├── cloudbuild.yaml               # Production deployment
├── cloudbuild-staging.yaml       # Staging deployment
├── unity/
│   └── conversation_manager/
│       ├── main.py               # Container entry point
│       ├── comms_manager.py      # Pub/Sub subscription handler
│       ├── conversation_manager.py # Main conversation logic
│       ├── debug_logger.py       # AssistantJobs logging
│       └── domains/
│           └── comms_utils.py    # Outbound helpers

communications/
├── adapters/                     # Webhook handlers
│   ├── main.py                   # FastAPI app with all endpoints
│   └── helpers.py                # Shared utilities
├── communication/                # Low-level comms web app
│   ├── gmail/                    # Gmail integration
│   ├── phone/                    # Twilio integration
│   └── outlook/                  # Microsoft integration
├── cloudbuild/                   # Cloud function build configs
└── .github/deploy.yaml           # CI/CD for communications service

orchestra/
├── orchestra/web/api/
│   └── assistant/
│       └── views.py              # POST/DELETE /assistant endpoints
└── /assistant                    # Main orchestration endpoints
```

## 🔧 Technical Details

### Communication Flow

1. **Inbound**: External Service → Adapter → Pub/Sub → GKE Container
2. **Outbound**: GKE Container → Communications API → External Service

### Scalability

- GKE provides better resource utilization than individual Cloud Run instances
- Idle containers ensure immediate response to new requests
- Automatic container replacement maintains high availability
- Hourly job creation and cleanup prevents resource exhaustion

### Key Timeouts & Intervals

| Setting | Value | Purpose |
|---------|-------|---------|
| Inactivity timeout | 6 minutes (360s) | Shuts down inactive containers |
| Ping interval | 30 seconds | Keeps idle containers alive |
| Idle job creation | Hourly + on every inbound + on deploy | Demand-aware: fills pool to target |
| Idle job cleanup | 10 min after creation | Trims pool back to target |
| Microsoft token refresh | Every 30-45 min | Keeps OAuth tokens fresh |
| Teams subscriptions | Every 30-45 min | Renews before 60-min expiry |
| Wildcard TLS cert renewal | 1st of month, 3 AM UTC | Renews *.vm.unify.ai if <30 days to expiry |

### GCP Projects

Infrastructure is split across multiple GCP projects to isolate workloads and avoid shared API rate limits:

| Project | ID | Purpose |
|---|---|---|
| **Comms & GKE** | `responsive-city-458413-a2` | GKE Autopilot cluster, Pub/Sub, Cloud Run services, Artifact Registry |
| **Assistant VMs** | `unity-assistant-vms` | Compute Engine VMs for assistant desktops, static IPs, VM images, Secret Manager (TLS certs) |
| **DNS** | `unify-dns-server` | Cloud DNS zone (`unifyai`) for all `*.unify.ai` records |
| **Orchestra** | `saas-368716` | Orchestra and Console Cloud Run services |

#### Why VMs are in a separate project

The GKE Autopilot cluster in `responsive-city-458413-a2` dynamically creates and destroys node VMs via Node Auto-Provisioning (NAP). These node lifecycle operations consume the per-project GCE `instances.insert` API rate limit. When assistant VM creation (during hiring) coincides with a GKE NAP scaling burst, the shared rate limit can be exhausted, causing the hiring flow to fail with `403 Rate Limit Exceeded`.

This was observed in production on 2026-03-04: a GKE NAP burst at ~17:59 UTC exhausted the rate limit, and a subsequent hire attempt at ~18:29 UTC failed because `instances.insert` for `unity-ubuntu-617-staging` returned 403. The assistant VM creation has no retry logic (it's a synchronous step in the hiring flow), so the entire hire rolled back with a 500.

Separating assistant VMs into `unity-assistant-vms` gives each workload its own independent rate limit budget. GKE NAP can scale freely without affecting hiring, and vice versa.

#### What lives where

**`responsive-city-458413-a2`** (Comms & GKE):
- GKE Autopilot cluster (`unity`) — container orchestration for Unity jobs
- Pub/Sub topics — message routing between adapters and containers
- Artifact Registry — Docker images (`us-central1-docker.pkg.dev/responsive-city-458413-a2/unity/`)
- Cloud Run — adapters (`unity-adapters`, `unity-adapters-staging`) and comms app (`unity-comms-app`, `unity-comms-app-staging`)
- Remaining Secret Manager secrets (API keys, Twilio, LiveKit, etc.)

**`unity-assistant-vms`** (Assistant VMs):
- Compute Engine VMs — assistant desktops (Ubuntu and Windows)
- Static IPs — one per assistant VM
- VM images — custom golden images (`unity-ubuntu-vm`, `unity-windows-vm`, pool variants)
- Secret Manager — `VM_WILDCARD_FULLCHAIN`, `VM_WILDCARD_PRIVKEY`, `DEVBOT_GITHUB_TOKEN`
- Firewall rules — VM-specific port access (2222, 3000, 6080, 7000, WinRM, HTTPS)
- Tunnel server VM — shared relay for local machine tunnelling

#### Cross-project access

The communication service account (`comm-sa@responsive-city-458413-a2.iam.gserviceaccount.com`) has IAM bindings in `unity-assistant-vms`:
- `roles/compute.admin` — create, start, stop, delete VMs and static IPs
- `roles/secretmanager.admin` — read secrets during VM provisioning, write during cert renewal

DNS is already cross-project: VM A records are created in `unify-dns-server`'s `unifyai` zone.

#### Configuration

The project separation is controlled by two config files in the communication repo:

- `communication/infra/vm_config.py` — `VM_PROJECT_ID`, `WINDOWS_VM_IMAGE_PROJECT`, `UBUNTU_VM_IMAGE_PROJECT`
- `communication/infra/tunnel_config.py` — `TUNNEL_PROJECT_ID`

The GKE/PubSub/Artifact Registry project ID (`responsive-city-458413-a2`) is set separately in `communication/infra/views.py` as `GCP_PROJECT_ID`.

## 🐛 Common Issues & Troubleshooting

| Symptom | Likely Cause | How to Debug |
|---------|--------------|--------------|
| Assistant not responding | No idle container available | Check GKE for idle jobs; check `/scheduled/jobs/create` logs |
| Delayed response (minutes) | Startup message queued waiting for idle container | Check if idle jobs exist; manually trigger job creation |
| Email/SMS not received | Email watch expired or contact not validated | Check `/scheduled/email-watches` logs; verify sender is a saved contact |
| "This number is no longer active" | Sender not in assistant's contacts | Add sender as contact in Unify console |
| Container dies immediately | Crash during startup | Check GKE job logs for exceptions |
| `running: True` but assistant unresponsive | Container crashed without cleanup | Manually update AssistantJobs log to `running: False` |
| Liveview URL not working | Service not ready yet (< 60s) or already deleted | Wait or check if job is still running |
| `TLSV1_ALERT_INTERNAL_ERROR` on desktop session | VM has no TLS cert (wildcard secret missing or LE ACME rate-limited) | Check Secret Manager for `VM_WILDCARD_FULLCHAIN` in `unity-assistant-vms`; check `crt.sh/?q=%.vm.unify.ai` for rate limit status |
| "Share assistant screen" shows broken iframe | VM HTTPS unreachable but `liveview_url` was set from a stale record | Check `AssistantJobs` for stale `running: True` entries; the adapter's `_expire_stale_records` should prevent this |
| Hiring fails with 500 / "Rate Limit Exceeded" | GCE API rate limit exhausted (historically from shared project with GKE NAP) | Check `gcloud logging read` in the VM project for 403 errors on `instances.insert`. If VMs are still in a shared project with GKE, this can recur. |
