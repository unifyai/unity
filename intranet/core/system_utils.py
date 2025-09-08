"""
System Initialization Utilities for Midland Heart RAG System
===========================================================

Provides modular, reusable utilities for system initialization including:
- Schema initialization
- Document ingestion
- Pre-embedding optimization
- Health checks

Follows dependency injection patterns to avoid multiple instantiation.
"""

import os
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Any, List, Optional
from unity.knowledge_manager.knowledge_manager import KnowledgeManager
from .rag_agent import IntranetRAGAgent, pre_embed_from_schema

logger = logging.getLogger(__name__)


def setup_logging():
    """Configure logging for the RAG system."""

    # Create logs directory
    logs_dir = Path("intranet/logs")
    logs_dir.mkdir(exist_ok=True)

    # Get log level from environment
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    simple_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler for all logs
    file_handler = logging.handlers.RotatingFileHandler(
        logs_dir / "rag_agent.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)

    # File handler for API logs
    api_handler = logging.handlers.RotatingFileHandler(
        logs_dir / "api.log",
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
    )
    api_handler.setLevel(logging.INFO)
    api_handler.setFormatter(simple_formatter)

    # Add handlers to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    # Add API-specific handler
    api_logger = logging.getLogger("intranet.core.api")
    api_logger.addHandler(api_handler)

    # Reduce noise from some external libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)

    return logging.getLogger(__name__)


async def initialize_schema(
    knowledge_manager: KnowledgeManager,
    use_tool_loops: bool = False,
    rag_agent: Optional[IntranetRAGAgent] = None,
) -> Dict[str, Any]:
    """
    Initialize the knowledge base schema.

    Args:
        knowledge_manager: KnowledgeManager instance
        use_tool_loops: Whether to use tool loops (slower) or direct method (faster)
        rag_agent: Optional RAG agent instance (required if use_tool_loops=True)

    Returns:
        Dict with initialization results
    """
    try:
        if use_tool_loops:
            if rag_agent is None:
                raise ValueError("RAG agent required when use_tool_loops=True")

            logger.info("📋 Initializing schema (tool loop method)...")
            init_result = await rag_agent.initialize_schema()

            if init_result.get("success"):
                logger.info("✅ Schema initialized successfully")
                return {"success": True, "method": "tool_loops", "result": init_result}
            else:
                error = init_result.get("error", "Unknown issue")
                logger.warning(f"⚠️  Schema initialization warning: {error}")
                return {"success": False, "method": "tool_loops", "error": error}
        else:
            logger.info("📋 Initializing schema (direct method)...")
            from intranet.scripts.utils import initialize_schema_direct

            await initialize_schema_direct(knowledge_manager)
            logger.info("✅ Schema initialized successfully")
            return {"success": True, "method": "direct"}

    except Exception as e:
        logger.error(f"❌ Schema initialization error: {e}")
        return {"success": False, "error": str(e)}


async def check_existing_data(
    knowledge_manager: KnowledgeManager,
    table_name: str = "content",
) -> bool:
    """
    Check if the knowledge base already contains data.

    Args:
        knowledge_manager: KnowledgeManager instance
        table_name: Table to check for existing data

    Returns:
        True if data exists, False otherwise
    """
    try:
        search_result = knowledge_manager._search(tables=[table_name], limit=1)
        return len(search_result.get(table_name, [])) > 0
    except Exception:
        return False


def find_documents_to_ingest(docs_path: Path) -> List[Path]:
    """
    Find documents available for ingestion.

    Args:
        docs_path: Directory to search for documents

    Returns:
        List of document file paths
    """
    if not docs_path.exists():
        return []

    doc_files = []
    for pattern in ["*.pdf", "*.docx", "*.doc", "*.txt"]:
        doc_files.extend(docs_path.glob(pattern))

    return doc_files


async def ingest_documents(
    docs_path: Path,
    knowledge_manager: KnowledgeManager,
    use_tool_loops: bool = False,
    rag_agent: Optional[IntranetRAGAgent] = None,
    batch_size: int = 5,
) -> Dict[str, Any]:
    """
    Ingest documents into the knowledge base.

    Args:
        docs_path: Directory containing documents to ingest
        knowledge_manager: KnowledgeManager instance
        use_tool_loops: Whether to use tool loops or direct method
        rag_agent: Optional RAG agent instance (required if use_tool_loops=True)

    Returns:
        Dict with ingestion results
    """
    doc_files = find_documents_to_ingest(docs_path)

    if not doc_files:
        logger.warning("⚠️  No documents found to ingest")
        logger.warning(f"   Add documents to {docs_path} for processing")
        return {
            "success": True,
            "processed": 0,
            "failed": 0,
            "total_files": 0,
            "message": "No documents found",
        }

    try:
        if use_tool_loops:
            if rag_agent is None:
                raise ValueError("RAG agent required when use_tool_loops=True")

            logger.info(
                f"📋 Ingesting {len(doc_files)} documents (tool loop method)...",
            )
            ingest_result = await rag_agent.ingest_documents(str(docs_path))

            processed = ingest_result.get("processed", 0)
            failed = ingest_result.get("failed", 0)
            total = ingest_result.get("total_files", 0)

            logger.info(f"✅ Ingested {processed}/{total} documents")
            if failed > 0:
                logger.warning(f"⚠️  {failed} documents failed to process")

            return ingest_result

        else:
            logger.info(f"📋 Ingesting {len(doc_files)} documents (direct method)...")
            from intranet.scripts.utils import ingest_documents_direct

            results = await ingest_documents_direct(
                knowledge_manager,
                docs_path,
                batch_size=batch_size,
            )

            processed = results.get("processed", 0)
            failed = results.get("failed", 0)
            total = results.get("total_files", 0)

            logger.info(f"✅ Ingested {processed}/{total} documents")
            if failed > 0:
                logger.warning(f"⚠️  {failed} documents failed to process")
                for error in results.get("errors", []):
                    logger.warning(f"   • {error['file']}: {error['error']}")

            return results

    except Exception as e:
        logger.error(f"❌ Document ingestion error: {e}")
        return {
            "success": False,
            "processed": 0,
            "failed": len(doc_files),
            "total_files": len(doc_files),
            "error": str(e),
        }


async def pre_embed_columns(
    knowledge_manager: KnowledgeManager,
    schema_path: str,
) -> Dict[str, Any]:
    """
    Pre-embed configured columns for optimal performance.

    Args:
        knowledge_manager: KnowledgeManager instance
        schema_path: Path to schema file containing embedding configuration

    Returns:
        Dict with embedding results
    """
    try:
        logger.info("🔮 Pre-embedding configured columns...")
        embedding_result = await pre_embed_from_schema(knowledge_manager, schema_path)

        if embedding_result.get("success"):
            success_count = embedding_result.get("success_count", 0)
            failed_count = embedding_result.get("failed_count", 0)
            duration = embedding_result.get("duration_seconds", 0)

            if success_count > 0:
                logger.info(
                    f"✅ Pre-embedded {success_count} columns in {duration:.2f}s",
                )
                if failed_count > 0:
                    logger.warning(f"⚠️  {failed_count} columns failed to embed")
            else:
                logger.info("ℹ️  No columns configured for pre-embedding")
        else:
            error = embedding_result.get("error", "Unknown error")
            logger.warning(f"⚠️  Pre-embedding failed: {error}")

        return embedding_result

    except Exception as e:
        logger.warning(f"⚠️  Pre-embedding error: {e}")
        return {
            "success": False,
            "error": str(e),
            "embedded_columns": [],
        }


class SystemInitializer:
    """
    System initializer with dependency injection to avoid multiple instantiation.
    """

    def __init__(
        self,
        knowledge_manager: Optional[KnowledgeManager] = None,
        rag_agent: Optional[IntranetRAGAgent] = None,
        use_tool_loops: bool = False,
    ):
        """
        Initialize system components.

        Args:
            knowledge_manager: Optional KnowledgeManager instance
            rag_agent: Optional RAG agent instance
            use_tool_loops: Whether to use tool loops for operations
        """
        self.use_tool_loops = use_tool_loops
        self.knowledge_manager = knowledge_manager or KnowledgeManager()
        self.rag_agent = rag_agent or IntranetRAGAgent()

    async def initialize_system(
        self,
        config: Dict[str, Any],
        overwrite: bool = False,
        batch_size: int = 5,
    ) -> Dict[str, Any]:
        """
        Complete system initialization workflow.

        Args:
            config: Configuration dict containing paths and settings
            overwrite: Whether to overwrite existing project data
            batch_size: Number of documents to process in parallel

        Returns:
            Dict with complete initialization results
        """
        results = {
            "schema_init": {},
            "document_ingestion": {},
            "pre_embedding": {},
            "success": False,
        }

        try:
            if overwrite:
                # Step 1: Initialize schema
                schema_result = await initialize_schema(
                    self.knowledge_manager,
                    self.use_tool_loops,
                    self.rag_agent,
                )
                results["schema_init"] = schema_result

                if not schema_result.get("success"):
                    raise Exception(
                        f"Schema initialization failed: {schema_result.get('error')}",
                    )

            # Step 2: Ingest documents if none exist (fast-path via Agent pipeline)
            docs_path = Path(
                config.get("documents_path", "intranet/data/documents"),
            )
            ingestion_result = await ingest_documents(
                docs_path,
                self.knowledge_manager,
                self.use_tool_loops,
                self.rag_agent,
                batch_size=batch_size,
            )
            results["document_ingestion"] = ingestion_result

            # Step 3: Pre-embed configured columns
            schema_path = config.get(
                "schema_path",
                str(Path(__file__).parent.parent / "flat_schema.json"),
            )
            embedding_result = await pre_embed_columns(
                self.knowledge_manager,
                schema_path,
            )
            results["pre_embedding"] = embedding_result

            results["success"] = True
            logger.info("🎉 System initialization completed successfully")

            return results

        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            results["error"] = str(e)
            return results
