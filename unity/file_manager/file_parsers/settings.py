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

    # ---------------------------------------------------------------------
    # Docling / extraction knobs
    # ---------------------------------------------------------------------
    # Run Docling's VLM picture-description pipeline on extracted images.
    # Off by default — it downloads a model and adds significant latency.
    PICTURE_DESCRIPTION_ENABLED: bool = False
    # Model repo used by Docling's picture description pipeline.
    PICTURE_DESCRIPTION_MODEL_REPO: str = "HuggingFaceTB/SmolVLM-500M-Instruct"

    # ---------------------------------------------------------------------
    # Optional NLP utilities
    # ---------------------------------------------------------------------
    SPACY_MODEL: str = "en_core_web_sm"

    # ---------------------------------------------------------------------
    # Diagnostics / debugging
    # ---------------------------------------------------------------------
    SAVE_PARSED_RESULTS: bool = False  # replaces UNITY_SAVE_PARSED_RESULTS usage

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )


FILE_PARSER_SETTINGS = FileParserSettings()
