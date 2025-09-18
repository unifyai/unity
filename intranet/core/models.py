"""
Pydantic models for the Midland Heart Intranet RAG system.
"""

from pydantic import BaseModel, Field, model_validator
from typing import List, Dict, Any, Optional
from datetime import datetime
import json


class SourceDocument(BaseModel):
    """Model for a source document/section reference."""

    document_id: Optional[str] = Field(None, description="Unique document identifier")
    section_id: Optional[str] = Field(None, description="Unique section identifier")
    title: Optional[str] = Field(None, description="Document or section title")
    content_text: Optional[str] = Field(None, description="Relevant content snippet")
    score: Optional[float] = Field(None, description="Relevance score", ge=0.0, le=1.0)
    document_type: Optional[str] = Field(
        None,
        description="Type of document (policy, procedure, etc.)",
    )
    department: Optional[str] = Field(None, description="Associated department")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata",
    )


# New simpler response model expected directly from LLM
class RAGLLMResponse(BaseModel):
    """Minimal response schema returned directly by the LLM.

    This subset is enough for end-users while keeping the JSON small. It is
    also what we pass to KnowledgeManager.ask via the `response_format`
    argument so the LLM knows exactly which keys to return.
    """

    answer: str = Field(
        description="The main answer to the user's question",
        min_length=1,
    )

    sources: List[SourceDocument] = Field(
        default_factory=list,
        description="List of source documents/sections that support the answer",
    )

    follow_up_questions: Optional[List[str]] = Field(
        default_factory=list,
        description="Optional suggested follow-up questions based on the query and results",
        max_items=5,
    )

    confidence: Optional[float] = Field(
        None,
        description="Confidence score for the answer (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )

    search_metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Metadata about the search process (strategy used, etc.)",
    )


# Old comprehensive response now *extends* the minimal LLM response
class RAGQueryResponse(RAGLLMResponse):
    """
    Comprehensive response model for RAG query results.

    Inherits the minimal set of fields from :class:`RAGLLMResponse` and adds
    conversation-level metadata plus validation helpers.
    """

    # Additional conversation / metadata fields
    conversation_id: str = Field(
        description="Unique identifier for this conversation thread",
    )

    user_id: Optional[str] = Field(
        None,
        description="Optional identifier for the user making the query",
    )

    timestamp: str = Field(
        description="ISO format timestamp of when the response was generated",
    )

    error: Optional[str] = Field(
        None,
        description="Error message if something went wrong during processing",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_timestamp(cls, data):
        """Ensure timestamp is in ISO format."""
        if isinstance(data, dict):
            timestamp = data.get("timestamp")
            if timestamp and isinstance(timestamp, str):
                try:
                    # Validate it's a proper ISO format
                    datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                except ValueError:
                    # If not valid ISO, set current timestamp
                    data["timestamp"] = datetime.now().isoformat()
            elif not timestamp:
                data["timestamp"] = datetime.now().isoformat()
        return data

    @model_validator(mode="after")
    def validate_answer_and_sources(self):
        """Ensure answer is meaningful and sources are properly formatted."""
        # Ensure answer is not just whitespace
        if not self.answer.strip():
            raise ValueError("Answer cannot be empty or just whitespace")

        # Validate source documents
        for i, source in enumerate(self.sources):
            if source.score is not None and (source.score < 0.0 or source.score > 1.0):
                raise ValueError(f"Source {i} score must be between 0.0 and 1.0")

        return self

    def to_legacy_format(self, generate_follow_up: bool = True) -> Dict[str, Any]:
        """Convert to the legacy dictionary format for backwards compatibility."""
        legacy_dict = {
            "answer": self.answer,
            "sources": [source.model_dump() for source in self.sources],
            "follow_up_questions": self.follow_up_questions,
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "search_metadata": self.search_metadata,
            "error": self.error,
        }
        if not generate_follow_up:
            del legacy_dict["follow_up_questions"]
        return legacy_dict

    @classmethod
    def from_unstructured_response(
        cls,
        result: str,
        query: str,
        conversation_id: str,
        user_id: Optional[str] = None,
        follow_up_generator=None,
    ) -> "RAGQueryResponse":
        """
        Create a RAGQueryResponse from an unstructured string response.

        Args:
            result: The raw LLM response string
            query: Original query text
            conversation_id: Conversation identifier
            user_id: Optional user identifier
            follow_up_generator: Optional function to generate follow-ups

        Returns:
            RAGQueryResponse instance
        """
        try:
            # Try to parse as JSON first
            if result.strip().startswith("{") and result.strip().endswith("}"):
                parsed_data = json.loads(result.strip())

                # Ensure required fields
                parsed_data.setdefault("conversation_id", conversation_id)
                parsed_data.setdefault("user_id", user_id)
                parsed_data.setdefault("timestamp", datetime.now().isoformat())

                return cls.model_validate(parsed_data)

        except (json.JSONDecodeError, ValueError) as e:
            print(f"⚠️ Could not parse structured response: {e}")

        # Fallback: create from unstructured text
        follow_ups = []
        if follow_up_generator:
            try:
                follow_ups = follow_up_generator(query, result, [])
            except Exception as e:
                print(f"⚠️ Could not generate follow-ups: {e}")

        return cls(
            answer=(
                result
                if isinstance(result, str) and result.strip()
                else "I encountered an issue processing your question."
            ),
            sources=[],
            follow_up_questions=follow_ups,
            conversation_id=conversation_id,
            user_id=user_id,
            timestamp=datetime.now().isoformat(),
        )
