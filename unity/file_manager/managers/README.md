## `managers/` — FileManager orchestration layer

This directory contains the FileManager implementations and their shared orchestration utilities.

## Core manager

- `file_manager.py`: the canonical implementation of FileManager behavior:
  - resolves identifiers (`file_id`, `file_path`, provider URIs)
  - exports files via adapters
  - invokes `file_parsers.FileParser`
  - adapts outputs via `parse_adapter`
  - orchestrates ingestion + embeddings via task graphs

Concrete managers (local, gdrive, etc.) are thin wrappers choosing an adapter.

## Utilities

- `utils/`: task graph builders, task functions, storage provisioning, search helpers.

## Design principles

- **Pure functions** for task execution (`utils/task_functions.py`) with explicit inputs/outputs.
- **Typed payloads** at boundaries: `FileParseResult` from parser, `FileRecordRow`/`FileContentRow` into ingestion.
- **Best-effort robustness**: per-file failures should not crash the whole batch.

## Notable architectural decisions

### 1) Orchestration is separated from pure task execution

`file_manager.py` orchestrates the overall workflow and constructs the task graph, but the actual “do work” units live as pure functions under `utils/` so they can be tested deterministically.

### 2) The parser is treated as a black box

Managers depend on `file_parsers.FileParser` and the parse boundary models only. They do **not** depend on Docling or any concrete parsing library types.

### 3) Server-managed IDs are not written client-side

`file_id` and `row_id` are server-assigned. Managers build typed *row payloads* (`FileRecordRow`, `FileContentRow`) and write them; the server assigns IDs.

## Extension tips

- Adding a new storage backend usually means:
  - implement a new `filesystem_adapters.*Adapter`
  - add a thin manager wrapper selecting that adapter
- Adding new ingestion behavior should usually be done by:
  - extending lowering in `parse_adapter/`
  - extending task graph construction in `utils/task_factory.py`
