# Stress Test Scripts

Scripts for stress-testing Unity assistants at scale — creating assistants in bulk, sending concurrent messages, and tracking running jobs.

## Prerequisites

- A `.env` file (or exported environment variables) with the required keys (see per-script sections below).
- Python dependencies: `aiohttp`, `python-dotenv`, `unify`.

## Scripts

### `create_assistants.py`

Provisions multiple assistants sequentially via the Orchestra API.

| Variable | Source | Description |
|---|---|---|
| `UNIFY_KEY` | env | Unify API key |
| `ORCHESTRA_URL` | env | Orchestra base URL |
| `NUM_ASSISTANTS` | in-script | Number of assistants to create (default: 10) |

```bash
python create_assistants.py
```

Each assistant is created with a random name suffix and default profile settings. Adjust `NUM_ASSISTANTS` or the payload in the script as needed.

### `unify_message_assistants.py`

Sends messages to multiple assistants **in parallel** via the adapters `/unify/message` endpoint.

| Variable | Source | Description |
|---|---|---|
| `ORCHESTRA_ADMIN_KEY` | env | Admin key for the adapters service |
| `ADAPTER_URL` | in-script | Adapters service URL |
| `ASSISTANT_IDS` | in-script | List of target assistant IDs |
| `CONTACT_ID` | in-script | Contact ID to send messages as |

```bash
python unify_message_assistants.py
```

Edit `ASSISTANT_IDS` and `CONTACT_ID` in the script to match your environment before running.

### `assistant_jobs_track.py`

Lists all currently-running assistant jobs by querying the `AssistantJobs` context in Unify.

| Variable | Source | Description |
|---|---|---|
| `SHARED_UNIFY_KEY` | env | Shared Unify API key |

```bash
python assistant_jobs_track.py
```

Prints each running job's name and assistant ID.
