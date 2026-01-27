## `file_parsers/` — format-aware parsing subsystem

This package implements the canonical parsing boundary for FileManager ingestion.

## Canonical boundary

- **Input**: `types/contracts.py::FileParseRequest`
- **Output**: `types/contracts.py::FileParseResult`

The FileParser is format-aware (PDF/DOCX/TXT vs CSV/XLSX) and delegates to backend implementations chosen by a registry.

## What “format-aware” means here

Format awareness is not just backend routing. It also imposes *output policy*:

- **PDF/DOCX/TXT** should populate:
  - `full_text`: extracted text (useful for debugging and enrichment)
  - `summary`: non-empty on success
  - `metadata`: non-empty on success (comma-separated strings)
- **CSV/XLSX** should populate:
  - `full_text`: a **bounded profile** (columns + bounded sample rows), never a full data dump
  - `summary`: non-empty on success
  - `metadata`: non-empty on success

This policy is centralized in `utils/format_policy.py` so it is easy to evolve.

## Key components

- `file_parser.py`: `FileParser` facade (backend selection, safety wrappers, trace + invariants)
- `registry.py`: lazy backend registry + configurable class-path mapping
- `implementations/`: concrete parsing backends (Docling, pure Python text)
- `types/`: strict Pydantic models and enums for the parser boundary + internal content graph
- `utils/`: shared helpers (token clipping, tracing, format policy, etc.)

## Backend contract

Every backend must inherit `types/backend.py::BaseFileParserBackend` and must:

- accept a `FileParseRequest`
- return a `FileParseResult` (never raw dicts)
- fill `logical_path` consistently (the facade will enforce this too)
- attach a `FileParseTrace` (or let the facade attach one)

Backends should prefer returning `FileParseResult(status='error')` for anticipated failures, but the facade will still catch unexpected exceptions to prevent catastrophic batch crashes.

## Registry and hot-swapping

`registry.py` maps `FileFormat` → backend class-path. Backends are lazily imported and cached by class path.

- Default mapping: `DEFAULT_BACKEND_CLASS_PATHS_BY_FORMAT`
- Override per FileManager call: `FilePipelineConfig.parse.backend_class_paths_by_format`

## Extending formats

To add a format:

1. Implement a backend under `implementations/<impl>/backends/` (subclass `BaseFileParserBackend`).
2. Register it in `DEFAULT_BACKEND_CLASS_PATHS_BY_FORMAT` (or provide config override).
3. Add tests under `tests/file_manager/file_parser/` that assert format-aware invariants and trace identity.
