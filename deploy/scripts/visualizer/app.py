"""
FastAPI backend for the Unity session visualizer.

Usage:
    cd unity/
    uv run python -m scripts.visualizer --data-dir colliers_data
    uv run python -m scripts.visualizer --data-dir vantage_data
"""

from __future__ import annotations

import argparse
import json
import mimetypes
from functools import lru_cache
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse

from .parsers import (
    list_api_calls,
    list_llm_calls,
    parse_api_call,
    parse_cloud_log,
    parse_framework_log,
    parse_llm_call,
)

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

app = FastAPI(title="Unity Session Visualizer")

# Set at startup via CLI arg
_data_dir: Path = Path(".")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _dd() -> Path:
    return _data_dir


@lru_cache(maxsize=1)
def _load_index() -> list[dict]:
    p = _dd() / "index.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


@lru_cache(maxsize=1)
def _load_org_info() -> dict:
    p = _dd() / "metadata" / "org.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@lru_cache(maxsize=1)
def _load_sessions_meta() -> dict[str, dict]:
    p = _dd() / "metadata" / "sessions.json"
    if not p.exists():
        return {}
    sessions = json.loads(p.read_text(encoding="utf-8"))
    return {
        s.get("entries", {}).get("job_name", ""): s
        for s in sessions
        if s.get("entries", {}).get("job_name")
    }


def _session_dir(job_name: str) -> Path:
    d = _dd() / "sessions" / job_name
    if not d.is_dir():
        raise HTTPException(404, f"Session not found: {job_name}")
    return d


def _assistant_dir(aid: str) -> Path:
    d = _dd() / "assistants" / str(aid)
    if not d.is_dir():
        raise HTTPException(404, f"Assistant not found: {aid}")
    return d


def _load_assistant_json(aid: str, filename: str) -> list[dict]:
    p = _assistant_dir(aid) / filename
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Cached parser wrappers (data is static on disk)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=64)
def _cached_cloud_log(path: str) -> dict:
    return parse_cloud_log(Path(path))


@lru_cache(maxsize=64)
def _cached_llm_calls(path: str) -> list[dict]:
    return list_llm_calls(Path(path))


@lru_cache(maxsize=64)
def _cached_api_calls(path: str) -> list[dict]:
    return list_api_calls(Path(path))


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    p = TEMPLATE_DIR / "index.html"
    if not p.exists():
        raise HTTPException(500, "Template not found")
    return HTMLResponse(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# API: Org info
# ---------------------------------------------------------------------------


@app.get("/api/info")
def get_info():
    org = _load_org_info()
    index = _load_index()
    assistant_ids = {s.get("assistant_id") for s in index if s.get("assistant_id")}
    return {
        "org_name": org.get("org_name", "Unknown"),
        "org_id": org.get("org_id"),
        "assistants": org.get("assistants", []),
        "total_sessions": len(index),
        "assistant_ids_with_sessions": list(assistant_ids),
    }


# ---------------------------------------------------------------------------
# API: Assistants
# ---------------------------------------------------------------------------


@app.get("/api/assistants")
def get_assistants():
    org = _load_org_info()
    index = _load_index()
    assistants_meta = org.get("assistants", [])

    result = []
    for a in assistants_meta:
        aid = str(a.get("assistant_id", ""))
        ad = _dd() / "assistants" / aid

        guidance_count = 0
        functions_count = 0
        files_count = 0
        if ad.is_dir():
            for fname, key in [
                ("guidance.json", "guidance_count"),
                ("functions.json", "functions_count"),
                ("file_records.json", "files_count"),
            ]:
                p = ad / fname
                if p.exists():
                    try:
                        data = json.loads(p.read_text(encoding="utf-8"))
                        if key == "guidance_count":
                            guidance_count = len(data)
                        elif key == "functions_count":
                            functions_count = len(data)
                        else:
                            files_count = len(data)
                    except Exception:
                        pass

        session_count = sum(1 for s in index if str(s.get("assistant_id", "")) == aid)

        result.append(
            {
                **a,
                "guidance_count": guidance_count,
                "functions_count": functions_count,
                "files_count": files_count,
                "session_count": session_count,
            },
        )
    return {"assistants": result}


@app.get("/api/assistants/{aid}/guidance")
def get_guidance(aid: str):
    entries = _load_assistant_json(aid, "guidance.json")
    return {"entries": entries, "total": len(entries)}


@app.get("/api/assistants/{aid}/functions")
def get_functions(aid: str):
    entries = _load_assistant_json(aid, "functions.json")
    return {"entries": entries, "total": len(entries)}


@app.get("/api/assistants/{aid}/files")
def get_files(aid: str):
    entries = _load_assistant_json(aid, "file_records.json")
    return {"entries": entries, "total": len(entries)}


@app.get("/api/assistants/{aid}/files/download/{filename:path}")
def download_file(aid: str, filename: str):
    files_dir = _dd() / "assistants" / str(aid) / "files"
    file_path = files_dir / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(404, f"File not found: {filename}")
    mime, _ = mimetypes.guess_type(str(file_path))
    return FileResponse(
        file_path,
        media_type=mime or "application/octet-stream",
        filename=filename,
    )


# ---------------------------------------------------------------------------
# API: Sessions
# ---------------------------------------------------------------------------


@app.get("/api/sessions")
def get_sessions():
    index = _load_index()
    meta = _load_sessions_meta()
    sessions = []
    for entry in index:
        jn = entry.get("job_name", "")
        m = meta.get(jn, {}).get("entries", {})
        sessions.append(
            {
                **entry,
                "liveview_url": m.get("liveview_url", ""),
            },
        )
    sessions.sort(key=lambda s: s.get("timestamp", ""), reverse=True)
    return {"sessions": sessions, "total": len(sessions)}


@app.get("/api/sessions/{job_name}/summary")
def get_session_summary(job_name: str):
    sd = _session_dir(job_name)

    llm_dir = sd / "pod_logs" / "unillm"
    api_dir = sd / "pod_logs" / "unify"
    llm_count = len(list(llm_dir.glob("*.txt"))) if llm_dir.is_dir() else 0
    api_count = len(list(api_dir.glob("*.json"))) if api_dir.is_dir() else 0

    cloud_path = sd / "cloud_logging.txt"
    cloud_lines = 0
    if cloud_path.exists():
        try:
            cloud_lines = cloud_path.read_text(
                encoding="utf-8",
                errors="replace",
            ).count("\n")
        except Exception:
            pass

    return {
        "job_name": job_name,
        "cloud_log_lines": cloud_lines,
        "llm_call_count": llm_count,
        "api_call_count": api_count,
    }


@app.get("/api/sessions/{job_name}/cloud-log")
def get_cloud_log(job_name: str):
    sd = _session_dir(job_name)
    return _cached_cloud_log(str(sd / "cloud_logging.txt"))


@app.get("/api/sessions/{job_name}/llm-calls")
def get_llm_calls(job_name: str):
    sd = _session_dir(job_name)
    calls = _cached_llm_calls(str(sd / "pod_logs" / "unillm"))
    return {"calls": calls, "total": len(calls)}


@app.get("/api/sessions/{job_name}/llm-calls/{filename}")
def get_llm_call_detail(job_name: str, filename: str):
    sd = _session_dir(job_name)
    fp = sd / "pod_logs" / "unillm" / filename
    if not fp.exists():
        raise HTTPException(404, f"LLM call not found: {filename}")
    return parse_llm_call(fp)


@app.get("/api/sessions/{job_name}/api-calls")
def get_api_calls(job_name: str):
    sd = _session_dir(job_name)
    calls = _cached_api_calls(str(sd / "pod_logs" / "unify"))
    return {"calls": calls, "total": len(calls)}


@app.get("/api/sessions/{job_name}/api-calls/{filename:path}")
def get_api_call_detail(job_name: str, filename: str):
    sd = _session_dir(job_name)
    fp = sd / "pod_logs" / "unify" / filename
    if not fp.exists():
        raise HTTPException(404, f"API call not found: {filename}")
    return parse_api_call(fp)


@app.get("/api/sessions/{job_name}/framework-log")
def get_framework_log(
    job_name: str,
    level: str | None = Query(None),
    component: str | None = Query(None),
    offset: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=2000),
):
    sd = _session_dir(job_name)
    return parse_framework_log(
        sd / "pod_logs" / "unity" / "unity.log",
        level=level,
        component=component,
        offset=offset,
        limit=limit,
    )


@app.post("/api/refresh")
def refresh_data():
    _load_index.cache_clear()
    _load_org_info.cache_clear()
    _load_sessions_meta.cache_clear()
    _cached_cloud_log.cache_clear()
    _cached_llm_calls.cache_clear()
    _cached_api_calls.cache_clear()
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    global _data_dir

    parser = argparse.ArgumentParser(description="Unity Session Visualizer")
    parser.add_argument(
        "--data-dir",
        default="colliers_data",
        help="Path to org data directory",
    )
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()

    _data_dir = Path(args.data_dir).resolve()
    if not _data_dir.is_dir():
        print(f"Error: {_data_dir} not found")
        return

    org_info = {}
    org_json = _data_dir / "metadata" / "org.json"
    if org_json.exists():
        org_info = json.loads(org_json.read_text())

    print()
    print("=" * 50)
    print(f"  Unity Session Visualizer")
    print(f"  Org:  {org_info.get('org_name', args.data_dir)}")
    print(f"  Data: {_data_dir}")
    print(f"  URL:  http://localhost:{args.port}")
    print("=" * 50)
    print()
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
