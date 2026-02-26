# Local Assistants

## Why Local Assistants Exist

The standard deployment model has three pieces working in concert:

1. **Orchestra** -- database and API for assistant management
2. **Adapters** (in `communication`) -- Cloud Run webhooks that capture inbound messages from Twilio, Gmail, Teams, etc.
3. **Unity** -- a GKE job (Kubernetes container) that runs the actual assistant logic

In production, when a message arrives the adapters resolve the assistant, start a Unity GKE job if one isn't already running, and route the message via Pub/Sub. This is great for production but painful for local development: you'd need a running GKE cluster just to iterate on Unity code.

**Local assistants** solve this by letting you run Unity on your own machine while keeping everything else in production. A local assistant is a real record in the Orchestra database with a flag (`is_local = True`) that tells the adapters to **skip GKE job creation and wakeup calls**. Inbound messages still flow through the production adapters and land on the assistant's Pub/Sub topic -- your local Unity instance subscribes to that topic and picks them up.

### What the `is_local` Flag Controls

| Concern | Production assistant | Local assistant |
|---|---|---|
| Database record in Orchestra | Yes | Yes |
| Email / phone / Pub/Sub provisioning | Via `create_infra` | Via `create_infra` |
| GKE job start on inbound message | Yes | **Skipped** |
| Wakeup call at creation time | Yes | **Skipped** |
| Contact validation | Enforced | **Bypassed** |
| Billing | Normal | Normal |

The flag is **disabled by default** and is not exposed in the frontend. It's purely a dev-tool for team members.

### How It Replaced "Default Assistants"

Before `is_local`, the adapters used brittle heuristics to avoid starting GKE jobs for certain assistants: checking `"default" in assistant_id` or `int(assistant_id) < 10`. These were fragile, undocumented, and semantically confusing (especially since Unity also uses `UNASSIGNED_ASSISTANT_ID` as a sentinel for idle containers, which is an entirely separate concept). The `is_local` flag replaces all of those heuristics with a single, explicit boolean on the assistant record.

## Creating a Local Assistant

### Using the Helper Script

The script `scripts/local_assistant.py` handles creation and env-var generation. You need your Unify API key (from console.unify.ai).

**Create a new local assistant:**

```bash
python scripts/local_assistant.py --api-key YOUR_KEY --name "Dev Assistant"
```

This calls `POST /v0/assistant` with `is_local=True` and `create_infra=False`, then prints the environment variables Unity needs. If an assistant with that name already exists under your account, the script fetches it instead of creating a duplicate.

**Fetch an existing assistant by ID:**

```bash
python scripts/local_assistant.py --api-key YOUR_KEY --id 42
```

### Sourcing Env Vars

The script prints `export` statements to stdout (status messages go to stderr), so you can source it directly:

```bash
source <(python scripts/local_assistant.py --api-key YOUR_KEY --name "Dev")
```

Or write to a `.env` file:

```bash
python scripts/local_assistant.py --api-key YOUR_KEY --name "Dev" > .env.local
```

### What the Script Outputs

The env vars match what `SessionDetails.export_to_env()` produces in production containers:

```
export ASSISTANT_ID="42"
export ASSISTANT_FIRST_NAME="Dev"
export ASSISTANT_SURNAME="Assistant"
export ASSISTANT_NAME="Dev Assistant"
export ASSISTANT_AGE=""
export ASSISTANT_NATIONALITY=""
export ASSISTANT_TIMEZONE=""
export ASSISTANT_ABOUT=""
export ASSISTANT_NUMBER=""
export ASSISTANT_EMAIL=""
export ASSISTANT_DESKTOP_MODE="ubuntu"
export ASSISTANT_DESKTOP_URL=""
export USER_ID="clx..."
export USER_FIRST_NAME="Your"
export USER_SURNAME="Name"
export USER_NAME="Your Name"
export USER_NUMBER=""
export USER_EMAIL="you@unify.ai"
export UNIFY_KEY="YOUR_KEY"
```

## Running Unity Locally

Once you have the env vars set, start Unity:

```bash
# Source the assistant env vars
source <(python scripts/local_assistant.py --api-key YOUR_KEY --name "Dev")

# Run unity (assumes all other env vars like GCP_SA_KEY, etc. are in your .env)
uv run python -m unity
```

Unity will subscribe to the assistant's Pub/Sub topic and begin processing inbound messages routed by the production adapters.

### With Infrastructure Provisioned

If you want inbound communication (email, phone, Pub/Sub) routed to your local instance through production adapters, create the assistant **with infrastructure**:

```bash
curl -X POST https://api.unify.ai/v0/assistant \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"first_name": "Dev", "surname": "Assistant", "is_local": true, "create_infra": true}'
```

Then use the script to fetch the env vars by ID:

```bash
source <(python scripts/local_assistant.py --api-key YOUR_KEY --id <agent_id>)
```

This provisions a real phone number, email address, and Pub/Sub topic. When someone sends a message to that number or email, the production adapters pick it up, see `is_local=True`, skip GKE job creation, and publish to the Pub/Sub topic. Your local Unity instance (subscribed to that topic) processes the message.

## Implementation Details

### Repositories Involved

| Repository | What changed |
|---|---|
| **Orchestra** | `is_local` column on `assistants` table, wired through DAO / schema / views. Wakeup skipped for local assistants. |
| **Communication** | Adapters read `is_local` from Orchestra API response. All old "default assistant" string heuristics replaced with `assistant_data.get("is_local", False)`. |
| **Unity** | `UNASSIGNED_ASSISTANT_ID` sentinel value changed from `"default-assistant"` to `"unassigned"` to avoid confusion with the local assistant concept. New `scripts/local_assistant.py` added. |

### Key Code Paths

- **Orchestra create endpoint** (`orchestra/web/api/assistant/views.py`): accepts `is_local` in the request body, stores it on the model, skips `wake_up_assistant()` when `True`.
- **Adapter webhook context** (`communication/adapters/helpers.py`, `build_webhook_context()`): reads `is_local` from the assistant data dict, sets `skip_auto_start = True` for local assistants, bypasses contact validation.
- **Adapter inbound handlers** (`communication/adapters/main.py`): Outlook email and Teams handlers check `is_local` instead of string-matching against `"default"`.
