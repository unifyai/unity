"""
File-parser settings (single env-var ingestion point).

Design goals
------------
- Centralize ALL parsing-related environment variables in one place.
- Use strictly typed, validated settings with sensible defaults.
- Keep settings decoupled from backend implementations; backends depend on this
  module rather than reading os.environ directly.

Notes
-----
This intentionally does not reuse `unity.settings.SETTINGS` because the file parser
has many knobs that are domain-specific and should remain modular. The settings
are still loaded from `.env` (same convention as the rest of Unity).
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class FileParserSettings(BaseSettings):
    # ---------------------------------------------------------------------
    # LLM settings used by parser enrichment steps (summaries, metadata)
    # ---------------------------------------------------------------------
    # Default model is aligned with project preference.
    SUMMARY_MODEL: str = "gpt-5.2@openai"

    # Summariser token accounting / backoff
    SUMMARY_ENCODING: str = "o200k_base"
    SUMMARY_MAX_TOKENS: int = 100000
    SUMMARY_REDUCE_STEP: int = 5000
    SUMMARY_MIN_TOKENS: int = 4000

    # Embedding constraints (used to clip summaries so they are embeddable)
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_ENCODING: str = "cl100k_base"
    EMBEDDING_MAX_INPUT_TOKENS: int = 8000

    # Skip expensive enrichment on trivially short text.
    ENRICHMENT_MIN_TEXT_CHARS: int = 200
    PARAGRAPH_SUMMARY_MIN_CHARS: int = 200
    MAX_PARAGRAPH_SUMMARY_CALLS: int = 250
    MAX_SECTION_SUMMARY_CALLS: int = 120

    # Tabular transport + bounded profiling
    TABULAR_SAMPLE_ROWS: int = 25
    TABULAR_INLINE_ROW_LIMIT: int = 1000
    TABULAR_PROFILE_MAX_TABLES: int = 12
    TABULAR_PROFILE_MAX_SAMPLE_ROWS: int = 5

    # ---------------------------------------------------------------------
    # Docling / extraction knobs
    # ---------------------------------------------------------------------
    # Run Docling's VLM picture-description pipeline on extracted images.
    # Off by default — it downloads a model and adds significant latency.
    PICTURE_DESCRIPTION_ENABLED: bool = False
    # OCR stays on by default for PDFs so scanned documents work out of the box.
    DOCLING_OCR_ENABLED: bool = True
    # Model repo used by Docling's picture description pipeline.
    PICTURE_DESCRIPTION_MODEL_REPO: str = "HuggingFaceTB/SmolVLM-500M-Instruct"

    # Docling's built-in per-document conversion timeout (PDF pipeline only).
    # When exceeded, Docling stops processing and returns partial results with
    # PARTIAL_SUCCESS status — no crash, no hang, just fewer pages extracted.
    # Set to None to disable.  Recommended: 300s for production.
    DOCLING_DOCUMENT_TIMEOUT: float | None = 300.0

    # ---------------------------------------------------------------------
    # Optional NLP utilities
    # ---------------------------------------------------------------------
    SPACY_MODEL: str = "en_core_web_sm"

    # ---------------------------------------------------------------------
    # Diagnostics / debugging
    # ---------------------------------------------------------------------
    SAVE_PARSED_RESULTS: bool = False
    PARSED_RESULTS_DIR: str = "logs/parsed_results"

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )


FILE_PARSER_SETTINGS = FileParserSettings()
