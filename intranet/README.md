# Midland Heart Intranet RAG

> Lightweight Retrieval-Augmented Generation system for Midland Heart policy documents.
---

## 🗃️  Schema
`flat_schema.json` defines a single **`content`** table.  Only the `summary` column is embedded (`_summary_emb`).  Retrieval is hierarchical:
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
    poppler-utils
```

2. **Env + Python deps** – follow the *Setup* section in the repository-root `README.md` (create virtualenv, `uv pip install -r requirements.txt`, add `.env` with credentials from **Haris**).

3. **Initialise** (build table + ingest docs + embeddings):
   ```bash
   python intranet/scripts/04_initialize_system.py
   ```

4. **Run API**:
   ```bash
   python intranet/scripts/05_start_api.py
   # → http://localhost:8000/docs
   ```

### Voice / CLI sandbox (optional)
```
python sandboxes/knowledge_manager/sandbox.py -P Intranet   # interactive demo
```
