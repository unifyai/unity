"""
Midland Heart Intranet RAG System - Core Components
==================================================

This module contains the core components of the RAG system
"""

# Core RAG system components

# Main RAG agent
from .rag_agent import IntranetRAGAgent

# Pydantic models for validation
try:
    from .models import RAGQueryResponse, SourceDocument
except ImportError:
    pass

# System utilities
try:
    from .system_utils import (
        SystemInitializer,
        initialize_schema,
        check_existing_data,
        find_documents_to_ingest,
        ingest_documents,
        pre_embed_columns,
    )
except ImportError:
    pass

# Evaluation components
try:
    from .semantic_validator import SemanticValidator, ValidationResult
    from .rag_http_client import RAGHTTPClient, RAGResponse
    from .qa_loader import QAManager, JSONQALoader, QAPair
    from .evaluation_results import (
        EvaluationResults,
        QuestionEvaluation,
        PerformanceMetrics,
        ErrorAnalysis,
        EvaluationSummary,
    )
except ImportError:
    pass

__all__ = [
    "IntranetRAGAgent",
    "RAGQueryResponse",
    "SourceDocument",
    "SystemInitializer",
    "initialize_schema",
    "check_existing_data",
    "find_documents_to_ingest",
    "ingest_documents",
    "pre_embed_columns",
    "SemanticValidator",
    "ValidationResult",
    "RAGHTTPClient",
    "RAGResponse",
    "QAManager",
    "JSONQALoader",
    "QAPair",
    "EvaluationResults",
    "QuestionEvaluation",
    "PerformanceMetrics",
    "ErrorAnalysis",
    "EvaluationSummary",
]
