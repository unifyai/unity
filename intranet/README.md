# Midland Heart Intranet RAG

> Lightweight Retrieval-Augmented Generation system for Midland Heart policy documents.
---

## 🗃️  Schema
`flat_schema.json` defines a single **`Content`** table.  Only the `summary` column is embedded (`_summary_emb`).  Retrieval is hierarchical:
sentence ➜ paragraph ➜ section ➜ document.

## 🚀 Quick start
1. **System packages** (Ubuntu/WSL2):
```bash
sudo apt update && sudo apt install -y \
    build-essential \
    pkg-config \
    python3-dev \
    libcairo2-dev \
    libgirepository1.0-dev \
    libglib2.0-dev \
    libffi-dev \
    libgdk-pixbuf2.0-dev \
    libgtk-3-dev \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libreoffice
```

2. **Env + Python deps** – follow the *Setup* section in the repository-root `README.md` (create virtualenv, `uv pip install -r requirements.txt`, add `.env` with credentials from **Haris**).

3. **Spacy model** – Once *Step 2* is completed, run `uv pip install $(spacy info en_core_web_sm --url)` to download the spaCy model used by the parser.

4. **Initialise** (build table + ingest docs + embeddings):
   ```bash
   python intranet/scripts/04_initialize_system.py
   ```

5. **Run API**:
   ```bash
   python intranet/scripts/05_start_api.py
   # → http://localhost:8000/docs
   ```

## 🧭 Start services (tmux)
Use the tmux-based launcher to run both the API and COMMS services with logs.

1) Make the script executable (one-time):
```bash
chmod +x intranet/scripts/start_intranet_service.sh
```

2) Start both services and open a logs window (detached session named "intranet"):
```bash
./intranet/scripts/start_intranet_service.sh --logs
```

3) Optional: attach to the tmux session to view API and COMMS logs side-by-side:
```bash
tmux attach -t intranet
```

Flags:
- `--logs`: Creates a `logs` window with side-by-side tails of `/tmp/intranet_api.log` and `/tmp/intranet_comms.log`.
- `--kill-port`: Frees port 8000 before starting if another process is listening.

Common examples:
```bash
# Clear logs, free port 8000 if needed, start services, create logs window
./intranet/scripts/start_intranet_service.sh --kill-port --logs

# Preserve existing logs, start and open logs window
./intranet/scripts/start_intranet_service.sh --no-clear-logs --logs
```

### Voice / CLI sandbox (optional)
```
python sandboxes/knowledge_manager/sandbox.py -P Intranet   # interactive demo
```
