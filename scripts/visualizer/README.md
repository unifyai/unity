# Unity Session Visualizer

Web-based tool for monitoring and analyzing Unity assistant sessions. Downloads session data from GCP and the Orchestra API, then serves an interactive dashboard for exploring timelines, LLM calls, API calls, system logs, guidance, functions, and uploaded files.

## Prerequisites

- **gcloud CLI** authenticated with access to the `responsive-city-458413-a2` project
- **uv** (Python package manager)
- `.env` file in `unity/` with:
  - `ORCHESTRA_ADMIN_KEY` — admin API key for assistant discovery
  - `SHARED_UNIFY_KEY` — shared key for AssistantJobs queries

## Quick start

```bash
cd unity/

# Download data for an organization
uv run python scripts/download_session_data.py --org-id 2

# Or for a single assistant
uv run python scripts/download_session_data.py --assistant-id 84

# Launch the visualizer
uv run python -m scripts.visualizer --data-dir colliers_healthcare_data
```

The visualizer opens at `http://localhost:8090`.

## Download script

```
scripts/download_session_data.py
```

Downloads all session data for an organization or individual assistant.

```bash
# By org (discovers all assistants automatically)
uv run python scripts/download_session_data.py --org-id 2
uv run python scripts/download_session_data.py --org-id 6

# By assistant
uv run python scripts/download_session_data.py --assistant-id 68

# With explicit name or output directory
uv run python scripts/download_session_data.py --org-id 2 --name colliers
uv run python scripts/download_session_data.py --org-id 2 --output-dir /path/to/data
```

The script is **incremental** — re-running it only downloads new sessions. Guidance, functions, and file records are always refreshed (they're small and may change).

### What it downloads

| Source | Content |
|--------|---------|
| Admin API | Assistant metadata, org info |
| Orchestra API | Guidance entries, stored functions, file records |
| GCS attachments bucket | Actual uploaded files (PDFs, DOCX, XLSX) |
| AssistantJobs | Session metadata (timestamps, user, medium) |
| GCP Cloud Logging | Terminal output per session (INFO+) |
| GCS pod logs | Full debug logs per session (LLM calls, API calls, framework logs) |

### Output structure

```
{name}_data/
  index.json                    # Session index with stats
  metadata/
    org.json                    # Org and assistant info
    sessions.json               # Raw session metadata
  assistants/
    {id}/
      info.json                 # Assistant metadata
      guidance.json             # Guidance entries
      functions.json            # Stored functions with code
      file_records.json         # File metadata with summaries
      files/                    # Actual binary files from GCS
  sessions/
    {job_name}/
      cloud_logging.txt         # Terminal output
      pod_logs/
        unillm/                 # LLM request/response logs
        unify/                  # Orchestra API call logs
        unity/                  # Framework debug logs
```

## Visualizer

```
scripts/visualizer/
  app.py          # FastAPI backend with REST API
  parsers.py      # Log parsing for all formats
  templates/
    index.html    # Single-page frontend
  __main__.py     # Entry point
```

```bash
uv run python -m scripts.visualizer --data-dir colliers_healthcare_data
uv run python -m scripts.visualizer --data-dir vantage_data --port 8091
```

### Views

- **Dashboard** — Session list grouped by day, assistant filter pills, assistant cards with stats
- **Assistant detail** — Guidance entries, stored functions (syntax-highlighted), file records with view/download
- **Session detail** — Stats bar, user messages, four tabs:
  - **Timeline** — Cloud logging events (high-level terminal output)
  - **LLM Calls** — Table of all LLM API calls, click to view full message thread
  - **API Calls** — Orchestra API calls with inline request/response detail
  - **System Logs** — Framework debug log with pagination
- **LLM Call detail** — Left sidebar for quick switching, collapsible messages, expandable tool calls, token usage

### API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/info` | Org name, assistants, session count |
| GET | `/api/assistants` | Assistant list with guidance/function/file counts |
| GET | `/api/assistants/{id}/guidance` | Guidance entries |
| GET | `/api/assistants/{id}/functions` | Stored functions |
| GET | `/api/assistants/{id}/files` | File records |
| GET | `/api/assistants/{id}/files/download/{name}` | Serve actual file |
| GET | `/api/sessions` | Session list |
| GET | `/api/sessions/{job}/summary` | Session overview stats |
| GET | `/api/sessions/{job}/cloud-log` | Parsed cloud logging events |
| GET | `/api/sessions/{job}/llm-calls` | LLM call list |
| GET | `/api/sessions/{job}/llm-calls/{file}` | Full LLM call detail |
| GET | `/api/sessions/{job}/api-calls` | API call list |
| GET | `/api/sessions/{job}/api-calls/{file}` | Full API call detail |
| GET | `/api/sessions/{job}/framework-log` | Paginated framework log |
| POST | `/api/refresh` | Clear cached data |
