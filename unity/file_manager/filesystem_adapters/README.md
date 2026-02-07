## `filesystem_adapters/` — filesystem adapters

This package contains **thin synchronous adapters** around concrete storage backends.

## Responsibilities

- Listing and lookup (`iter_files`, `get_file`, `exists`, `list`)
- Byte access (`open_bytes`)
- Export to local filesystem for parsing (`export_file`, `export_directory`)
- Optional mutations (rename/move/delete) guarded by `capabilities`

## Non-responsibilities

- No parsing
- No LLM usage
- No ingestion logic

## Key interfaces

- `BaseFileSystemAdapter` (`filesystem_adapters/base.py`): abstract contract
- `FileReference` (`types/filesystem.py`): typed file metadata returned by adapters

## Identity

Adapters must provide stable identifiers:

- `FileReference.path`: canonical “adapter path” used by FileManager as the **logical path**
- `FileReference.uri`: canonical provider URI (e.g. `local:///abs/path`) when available

## Export strategy

Parsing backends operate on **local paths**. For non-local stores, the FileManager exports to a temp directory first:

- `logical_path` stays stable (adapter path)
- `source_local_path` points at the exported local file

## Guidelines for implementing a new adapter

- Keep methods **purely I/O**: avoid parsing, enrichment, or ingestion concerns.
- Prefer stable, user-meaningful `FileReference.path` values (these become context keys).
- Populate `FileReference.uri` when you can (helps dedupe and provenance).
- Make export deterministic and safe:
  - export into the provided directory
  - do not mutate the remote source unless explicitly requested via a mutation method
  - ensure exported files preserve extension where possible (format routing)

## Common pitfalls

- Returning unstable `path` values (breaks record identity across runs).
- Exporting without file extensions (breaks `FileFormat` routing).
- Hiding errors: adapter exceptions should be explicit so FileManager can record them per-file.
