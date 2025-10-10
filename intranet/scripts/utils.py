#!/usr/bin/env python3
"""
Utility functions for intranet RAG system scripts.
Supports both the new single-table design and multi-table legacy mode.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import asdict
import json

# Add intranet to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def activate_project(project_name: str, overwrite: bool = False) -> None:
    """
    Activate *project_name* and re-initialise the global EventBus singleton so
    that all subsequent Unify contexts (including those automatically created
    by EventBus) belong to that project.  Call this immediately after handling
    CLI arguments and before any manager instances are constructed.
    """
    import unity
    from unity.events.event_bus import EVENT_BUS

    unity.init(
        project_name,
        overwrite=("contexts" if overwrite else False),
    )
    # Clears/initialises EventBus contexts. In multi-worker/server setups this
    # can race on first boot; make this best-effort and tolerate "already exists".
    try:
        EVENT_BUS.clear()
    except Exception as e:
        # Ignore context-exists and other benign initialisation races.
        # The global EventBus context will already be present.
        print(f"⚠️  EVENT_BUS clear skipped due to concurrency: {e}")

    import unify

    unify.set_trace_context("Traces")

    # Only delete Knowledge contexts if explicitly overwriting
    if overwrite:
        [
            unify.delete_context(table)
            for table in unify.get_contexts(prefix="Knowledge").keys()
        ]
        if "Traces" in unify.get_contexts():
            unify.delete_context("Traces")

    # Always ensure Traces context exists
    if "Traces" not in unify.get_contexts():
        unify.create_context("Traces")


def setup_project_path():
    """Add project root to Python path for imports."""

    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    return project_root


def load_environment():
    """Load environment variables from .env file."""

    try:
        from dotenv import load_dotenv

        # Look for .env file in project root (parent of intranet)
        project_root = Path(__file__).parent.parent.parent
        env_file = project_root / ".env"

        if env_file.exists():
            load_dotenv(env_file)
            print(f"✅ Loaded environment from: {env_file}")
            return True
        else:
            # Look for .env in current directory
            current_env = Path(".env")
            if current_env.exists():
                load_dotenv(current_env)
                print(f"✅ Loaded environment from: {current_env}")
                return True
            else:
                print(f"⚠️  No .env file found (checked {env_file} and {current_env})")
                print(f"   Using system environment variables")
                return False

    except ImportError:
        print(f"⚠️  python-dotenv not installed, using system environment variables")
        return False


def get_env_var(
    key: str,
    default: Optional[str] = None,
    required: bool = False,
) -> Optional[str]:
    """Get environment variable with better error handling."""

    value = os.getenv(key, default)

    if required and value is None:
        print(f"❌ Required environment variable '{key}' not set")
        if default:
            print(f"   Using default value: {default}")
            return default
        else:
            print(f"   Please set {key} in your .env file or environment")
            return None

    return value


def get_config_values() -> Dict[str, Any]:
    """Get configuration values from environment variables."""
    return {
        "unify_cache": get_env_var("UNIFY_CACHE", "true").lower() == "true",
        "unify_traced": get_env_var("UNIFY_TRACED", "true").lower() == "true",
        "log_level": get_env_var("LOG_LEVEL", "INFO"),
        "documents_path": get_env_var("DOCUMENTS_PATH", "intranet/policies"),
        "api_host": get_env_var("API_HOST", "0.0.0.0"),
        "api_port": int(get_env_var("API_PORT", "8000")),
        "dev_mode": get_env_var("DEV_MODE", "false").lower() == "true",
        "debug_mode": get_env_var("DEBUG_MODE", "false").lower() == "true",
        "use_tool_loops": get_env_var("RAG_USE_TOOL_LOOPS", "false").lower() == "true",
        "enable_debug": get_env_var("RAG_DEBUG", "false").lower() == "true",
        "single_table_mode": get_env_var("RAG_SINGLE_TABLE", "true").lower() == "true",
        "use_docling": get_env_var("RAG_USE_DOCLING", "true").lower() == "true",
    }


def initialize_script_environment() -> bool:
    """
    Initialize the script environment

    Returns:
        bool: True if initialization successful
    """
    if not load_environment():
        print("⚠️  Environment not loaded, continuing with system variables")

    setup_project_path()

    return True


async def initialize_single_table_schema(knowledge_manager):
    """Create the `Content` table based on *flat_schema.json* (v3)."""

    print("📋 Initializing single-table schema from flat_schema.json …")

    try:
        schema_path = Path(__file__).resolve().parent.parent / "flat_schema.json"
        with open(schema_path, "r", encoding="utf-8") as f:
            flat_schema = json.load(f)

        # Extract column → type mapping from JSON definition
        col_defs = flat_schema["tables"]["Content"]["columns"]
        columns = {name: info["type"] for name, info in col_defs.items()}

        knowledge_manager._create_table(
            name="Content",
            columns=columns,
        )

        print(
            "✅ Single-table schema created successfully (version",
            flat_schema.get("version"),
            ")",
        )
        return True

    except Exception as e:
        print(f"❌ Failed to create single-table schema: {e}")
        return False


async def initialize_schema_direct(knowledge_manager):
    """
    Initialize schema directly using KnowledgeManager.

    Args:
        knowledge_manager instance
    """
    config = get_config_values()

    if config["single_table_mode"]:
        return await initialize_single_table_schema(knowledge_manager)
    else:
        # Legacy multi-table initialization
        return await initialize_legacy_schema(knowledge_manager)


async def initialize_legacy_schema(knowledge_manager):
    """Initialize the legacy multi-table schema (for backward compatibility)."""
    print("📋 Initializing legacy multi-table schema...")

    try:
        # Create documents table
        knowledge_manager._create_table(
            name="documents",
            columns={
                "document_id": "str",
                "title": "str",
                "document_type": "str",
                "category": "str",
                "summary": "str",
                "key_topics": "list",
                "named_entities": "dict",
                "embedding_title": "str",
                "embedding_summary": "str",
                "schema_id": "str",
                "created_at": "str",
                "processed_at": "str",
                "metadata": "dict",
            },
            unique_key_name="document_id",
        )

        # Create sections table
        knowledge_manager._create_table(
            name="sections",
            columns={
                "section_id": "str",
                "document_id": "str",
                "title": "str",
                "summary": "str",
                "embedding": "str",
                "level": "str",
                "section_index": "str",
                "metadata": "dict",
            },
            unique_key_name="section_id",
        )

        # Create chunks table
        knowledge_manager._create_table(
            name="chunks",
            columns={
                "chunk_id": "str",
                "document_id": "str",
                "section_id": "str",
                "content": "str",
                "embedding": "str",
                "chunk_index": "str",
                "chunk_type": "str",
                "metadata": "dict",
            },
            unique_key_name="chunk_id",
        )

        print("✅ Legacy multi-table schema created successfully")
        return True

    except Exception as e:
        print(f"❌ Failed to create legacy schema: {e}")
        return False


async def ingest_documents_direct(
    knowledge_manager,
    documents_dir: Path,
    batch_size: int = 5,
    embed_along: bool = True,
    embedding_config: dict | None = None,
):
    """
    Ingest documents using the unified parser and Knowledge Manager's ingestion tool.

    Args:
        documents_dir: Directory containing documents to ingest
        knowledge_manager: Knowledge manager instance
        batch_size: Number of documents to process in parallel

    Returns:
        dict: Processing statistics with keys: processed, failed, total_files, errors
    """
    config = get_config_values()

    # Check if file manager is available
    if (
        not hasattr(knowledge_manager, "_file_manager")
        or not knowledge_manager._file_manager
    ):
        print("❌ FileManager not available in Knowledge Manager")
        return {
            "processed": 0,
            "failed": 0,
            "total_files": 0,
            "errors": [{"file": "system", "error": "FileManager not configured"}],
            "success": False,
        }

    # Find documents to process
    document_files = []
    for pattern in ["*.pdf", "*.docx", "*.html", "*.md", "*.txt"]:
        document_files.extend(documents_dir.glob(pattern))

    if not document_files:
        print(f"⚠️  No documents found in {documents_dir}")
        return {
            "processed": 0,
            "failed": 0,
            "total_files": 0,
            "errors": [],
            "success": True,
        }

    print(f"📄 Found {len(document_files)} documents to process")

    # First, import all documents into the FileManager
    print("📥 Importing documents into FileManager...")
    try:
        imported_docs = knowledge_manager._file_manager.import_directory(
            str(documents_dir),
        )
    except Exception as e:
        print(f"❌ Failed to import directory: {e}")
        return {
            "processed": 0,
            "failed": len(document_files),
            "total_files": len(document_files),
            "errors": [{"file": "import", "error": str(e)}],
            "success": False,
        }

    # Process documents
    processed_count = 0
    failed_count = 0
    errors = []

    # Use batch ingestion for all documents at once
    filenames = [doc_path.name for doc_path in document_files]

    print(f"📄 Processing {len(filenames)} documents with batch size {batch_size}...")

    try:
        # Use the Knowledge Manager's batch ingestion tool
        result = await knowledge_manager._ingest_documents(
            filenames=filenames,
            table="Content",
            replace_existing=True,
            batch_size=batch_size,
            embed_along=embed_along,
            embedding_config=embedding_config,
        )

        if result.get("success"):
            processed_count = result.get("successful_files", 0)
            failed_count = result.get("failed_files", 0)
            total_inserted = result.get("total_inserted", 0)
            total_deleted = result.get("total_deleted", 0)

            print(f"\n✅ Batch ingestion complete:")
            print(f"   - Successful files: {processed_count}")
            print(f"   - Failed files: {failed_count}")
            print(f"   - Total records inserted: {total_inserted}")
            print(f"   - Total records deleted: {total_deleted}")

            # Extract individual file errors from results
            for file_result in result.get("file_results", []):
                if not file_result.get("success"):
                    errors.append(
                        {
                            "file": file_result["filename"],
                            "error": file_result.get("error", "Unknown error"),
                        },
                    )
                    print(
                        f"   ❌ {file_result['filename']}: {file_result.get('error')}",
                    )
        else:
            # Complete failure
            failed_count = len(document_files)
            error_msg = result.get("error", "Batch ingestion failed")
            errors.append({"file": "batch_operation", "error": error_msg})
            print(f"❌ Batch ingestion failed: {str(result)}")

    except Exception as e:
        failed_count = len(document_files)
        errors.append({"file": "batch_operation", "error": str(e)})
        print(f"❌ Batch ingestion failed with exception: {str(e)}")

    return {
        "processed": processed_count,
        "failed": failed_count,
        "total_files": len(document_files),
        "errors": errors,
        "success": failed_count == 0,
    }


async def store_document_single_table(document, knowledge_manager):
    """
    Store document using the single-table design.

    Args:
        document: Parsed document from parser
        knowledge_manager instance
    """
    try:
        # Load allowed columns from schema v3
        schema_path = Path(__file__).resolve().parent.parent / "flat_schema.json"
        with open(schema_path, "r", encoding="utf-8") as f:
            schema_json = json.load(f)

        allowed_cols = set(schema_json["tables"]["Content"]["columns"].keys())

        # Convert document to flat records (assumed to include summary/title fields)
        flat_records = document.to_flat_records()

        schema_id = "midland-heart-single-table-v3.0"

        for rec in flat_records:
            # essential computed fields
            rec["schema_id"] = schema_id
            rec["confidence_score"] = rec.get("confidence_score", 1.0)

        # Strip unknown / deprecated keys
        cleaned = [
            {k: v for k, v in rec.items() if k in allowed_cols} for rec in flat_records
        ]

        # Store batch
        knowledge_manager._add_rows(table="Content", rows=cleaned)

        print(f"💾 Stored {len(cleaned)} content records in single table")

    except Exception as e:
        print(f"❌ Failed to store document in single table: {e}")
        raise


async def store_document_legacy(document, knowledge_manager):
    """
    Store document using legacy multi-table approach.

    Args:
        document: Parsed document from parser
        knowledge_manager instance
    """
    try:
        schema_id = "midland-heart-legacy-v1.0"

        # Store document record
        doc_record = {
            "document_id": str(document.document_id),
            "title": document.metadata.title,
            "document_type": document.metadata.document_type,
            "category": document.metadata.category,
            "summary": document.summary or document.full_text[:500] + "...",
            "key_topics": document.metadata.key_topics,
            "named_entities": document.metadata.named_entities,
            "content_tags": document.metadata.content_tags,
            "confidence_score": document.metadata.confidence_score,
            "schema_id": schema_id,
            "created_at": document.metadata.created_at,
            "processed_at": document.metadata.processed_at,
            "metadata": asdict(document.metadata),
        }

        knowledge_manager._add_rows(table="documents", rows=[doc_record])

        # Store sections and chunks
        section_records = []
        chunk_records = []

        for section in document.sections:
            section_record = {
                "section_id": str(section.section_id),
                "document_id": str(document.document_id),
                "title": section.title,
                "summary": section.summary or section.title,
                "level": str(section.level),
                "section_index": str(section.section_index),
                "metadata": section.metadata,
            }
            section_records.append(section_record)

            # Store paragraphs as chunks
            for paragraph in section.paragraphs:
                chunk_record = {
                    "chunk_id": str(paragraph.paragraph_id),
                    "document_id": str(document.document_id),
                    "section_id": str(section.section_id),
                    "content": paragraph.text,
                    "chunk_index": str(paragraph.paragraph_index),
                    "chunk_type": "paragraph",
                    "metadata": paragraph.metadata,
                }
                chunk_records.append(chunk_record)

        if section_records:
            knowledge_manager._add_rows(table="sections", rows=section_records)
        if chunk_records:
            knowledge_manager._add_rows(table="chunks", rows=chunk_records)

        print(
            f"💾 Stored document with {len(section_records)} sections and {len(chunk_records)} chunks",
        )

    except Exception as e:
        print(f"❌ Failed to store document in legacy format: {e}")
        raise
