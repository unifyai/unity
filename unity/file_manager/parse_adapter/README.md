## `parse_adapter/` — FileParser → FileManager adaptation

This directory is the **explicit seam** between parsing and ingestion.

The FileParser returns a parse-only `FileParseResult` (parser-owned artifacts):

- optional `ContentGraph` (`file_parsers.types.graph`)
- extracted `tables` (`file_parsers.types.table.ExtractedTable`)
- `trace`/`metadata` (`file_parsers.types.contracts`)

The FileManager needs ingestion-ready payloads (FileManager-owned schemas):

- `/Content/` rows (`types.file.FileContentRow`)
- `/Tables/<label>` rows (raw JSON rows per `ExtractedTable.rows`)

This adapter exists to keep the parser ingestion-agnostic while keeping ingestion strictly typed.

## Key entry points

- `adapter.py`: `adapt_parse_result_for_file_manager(parse_result, config)`
- `lowering/`: graph/table lowering that is format-aware (document vs spreadsheet)

## Design rules (important invariants)

- **Parser must not depend on ingestion schemas**: `file_parsers/` must not import `unity.file_manager.types.file`.
- **Adapter may be format-aware but must stay library-agnostic**: it should not depend on Docling-specific objects.
- **Best-effort behavior**: adapter functions should never raise; failures return empty payloads so the batch can continue.
- **No `content_text` for sheet/table `/Content/` rows**: large tabular payloads do not belong in `/Content/`.
  - `/Content/` table rows are catalogs (title + summary).
  - raw table rows are stored under `/Tables/<label>`.

## Extensibility notes

- If you add a new `NodeKind` in the internal graph, decide whether it should lower into `/Content/` and update `lowering/content_rows.py`.
- If you change how tables are labeled, update both lowering and the FileManager table-context naming logic.
