"""
Midland Heart RAG Agent with Single-Table Architecture
==========================================================

This RAG agent implements the hierarchical document retrieval:

1. SINGLE TABLE DESIGN: All content stored in one table with hierarchical IDs
2. FLAT SEMANTIC SEARCH: Search across all sentences for precise retrieval
3. TREE EXPANSION: Easily expand from sentence -> paragraph -> section -> document
4. MULTI-LEVEL SEARCH: Search at different granularities as needed
5. HIERARCHICAL SUMMARIZATION: Bottom-up summary generation

Retrieval Strategy:
1) Semantic search across all sentences (~50 matches)
2) Expand to paragraphs for best matches
3) Search within best matching documents for context
4) Generate response with proper citations

This implementation uses Unity's tool-loop pattern while supporting the efficient single-table retrieval approach.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

from .prompt_builder import (
    build_intranet_ask_instructions,
    build_intranet_update_instructions,
)
from unity.file_manager.file_manager import FileManager

# Import Pydantic models
try:
    from .models import RAGQueryResponse, RAGLLMResponse

    MODELS_AVAILABLE = True
except ImportError:
    print("Warning: Models module not available, using fallback response structure")
    MODELS_AVAILABLE = False

# Import parser system
try:
    pass

    PARSERS_AVAILABLE = True
except ImportError:
    print("Warning: Parsers module not available")
    PARSERS_AVAILABLE = False

# Export functions for external use
__all__ = [
    "IntranetRAGAgent",
    "pre_embed_from_schema",
    "pre_embed_from_configuration",
    "load_embedding_configuration",
]


def load_embedding_configuration(schema_path: str) -> Dict[str, Any]:
    """
    Load embedding configuration from schema file.

    Args:
        schema_path: Path to schema file

    Returns:
        Dict containing embedding configuration
    """
    import json
    from pathlib import Path

    schema_path = Path(schema_path)
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    with open(schema_path, "r") as f:
        schema = json.load(f)

    return schema.get("embedding_configuration", {})


async def pre_embed_from_configuration(
    knowledge_manager,
    embedding_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Pre-embed columns based on embedding configuration.

    Args:
        knowledge_manager: KnowledgeManager instance
        embedding_config: Embedding configuration dict from schema

    Returns:
        Dict with embedding results and timing information
    """
    try:
        from datetime import datetime

        tables_config = embedding_config.get("tables", {})

        if not tables_config:
            return {
                "success": True,
                "message": "No embedding configuration found",
                "embedded_columns": [],
            }

        print("🔮 Pre-embedding configured columns...")
        start_time = datetime.now()
        embedded_columns = []

        for table_name, table_config in tables_config.items():
            columns_to_embed = table_config.get("columns_to_embed", [])

            for column_config in columns_to_embed:
                source_column = column_config["source_column"]
                target_column = column_config["target_column"]
                description = column_config.get("description", "")

                print(
                    f"   🎯 Embedding {table_name}.{source_column} -> {target_column}",
                )
                if description:
                    print(f"      {description}")

                try:
                    knowledge_manager._vectorize_column(
                        table=table_name,
                        source_column=source_column,
                        target_column_name=target_column,
                    )

                    embedded_columns.append(
                        {
                            "table": table_name,
                            "source_column": source_column,
                            "target_column": target_column,
                            "status": "success",
                        },
                    )
                    print(
                        f"      ✅ Successfully embedded {table_name}.{source_column}",
                    )

                except Exception as e:
                    print(f"      ❌ Failed to embed {table_name}.{source_column}: {e}")
                    embedded_columns.append(
                        {
                            "table": table_name,
                            "source_column": source_column,
                            "target_column": target_column,
                            "status": "failed",
                            "error": str(e),
                        },
                    )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        success_count = len(
            [col for col in embedded_columns if col["status"] == "success"],
        )
        failed_count = len(
            [col for col in embedded_columns if col["status"] == "failed"],
        )

        print(f"🔮 Pre-embedding completed in {duration:.2f}s")
        print(f"   ✅ Successfully embedded: {success_count} columns")
        if failed_count > 0:
            print(f"   ❌ Failed to embed: {failed_count} columns")

        return {
            "success": True,
            "embedded_columns": embedded_columns,
            "duration_seconds": duration,
            "success_count": success_count,
            "failed_count": failed_count,
            "timestamp": end_time.isoformat(),
        }

    except Exception as e:
        print(f"❌ Error during pre-embedding: {e}")
        return {
            "success": False,
            "error": str(e),
            "embedded_columns": [],
        }


async def pre_embed_from_schema(knowledge_manager, schema_path: str) -> Dict[str, Any]:
    """
    Pre-embed columns configured in schema file.

    Args:
        knowledge_manager: KnowledgeManager instance
        schema_path: Path to schema file

    Returns:
        Dict with embedding results and timing information
    """
    try:
        embedding_config = load_embedding_configuration(schema_path)
        return await pre_embed_from_configuration(knowledge_manager, embedding_config)
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "embedded_columns": [],
        }


class IntranetRAGAgent:
    """
    RAG Agent implementing single-table retrieval strategy.

    This agent provides intelligent document retrieval using:
    - Flat semantic search across all sentences
    - Hierarchical tree expansion for context
    - Multi-level search strategies
    - Conversation management with follow-ups
    """

    def __init__(
        self,
        enabled_tools: Optional[List[str]] = None,
        conv_context_length: int = 10,
        sandbox_mode: bool = False,
    ):
        """
        Initialize the RAG agent.

        Args:
            enabled_tools: List of tool categories to enable ('knowledge', 'search', etc.)
            conv_context_length: Number of previous messages to keep in context
            sandbox_mode: Whether to return the SteerableToolHandle instance
                          from the knowledge manager's public methods instead of the result
        """
        # Configuration
        self.conv_context_length = conv_context_length
        self.conversations = {}  # Store conversation history
        self.sandbox_mode = sandbox_mode

        # FileManager for session-scoped files with document parser
        from unity.file_manager.parser import DoclingParser

        parser = DoclingParser(
            use_llm_enrichment=True,  # Enable enrichment for RAG use case
            extract_images=True,
            extract_tables=True,
            use_hybrid_chunking=True,
        )
        # Create a new FileManager instance with custom parser for this agent
        self.file_manager = FileManager(parser=parser)

        # Initialize KnowledgeManager with custom FileManager
        from unity.knowledge_manager.knowledge_manager import KnowledgeManager

        self.knowledge_manager = KnowledgeManager(file_manager=self.file_manager)

        # Build enabled tools
        self.enabled_tools = self._build_enabled_tools(enabled_tools or ["knowledge"])

        print(f"🤖 RAG Agent initialized with {len(self.enabled_tools)} tools")
        print(f"💬 Conversation Context: {conv_context_length} messages")

    def _build_enabled_tools(self, tool_categories: List[str]) -> Dict[str, Any]:
        """
        Build the tool dictionary for the agent using Unity's methods_to_tool_dict.

        Args:
            tool_categories: Categories of tools to enable

        Returns:
            Dict mapping tool names to tool functions
        """
        available_tools = {}

        # Knowledge Management Tools
        from unity.common.llm_helpers import methods_to_tool_dict

        if "knowledge" in tool_categories:
            km_tools = methods_to_tool_dict(
                self.knowledge_manager.ask,
                self.knowledge_manager.update,
                self.knowledge_manager.refactor,
            )
            # Expose FileManager tools for ingestion/reflection flows
            fm_tools = methods_to_tool_dict(
                self.file_manager.list,
                self.file_manager.exists,
                self.file_manager.ask,
                self.file_manager.parse,
            )
            available_tools.update({**km_tools, **fm_tools})

        return available_tools

    async def ask(
        self,
        query_text: str,
        *,
        conversation_context: Optional[List[Dict[str, str]]] = None,
        conversation_id: str = "default",
        user_id: Optional[str] = None,
        generate_follow_up: bool = False,
        _return_reasoning_steps: bool = False,
        parent_chat_context: list[dict] | None = None,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
        rolling_summary_in_prompts: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Delegate to KnowledgeManager.ask with optional case-specific instructions.
        """
        try:
            print(f"🔍 Processing query: {query_text[:100]}...")
            print(f"👤 User: {user_id or 'anonymous'}, Conversation: {conversation_id}")

            if conversation_context is None:
                conversation_context = self._get_conversation_context(conversation_id)

            print("🔍 Calling KnowledgeManager.ask directly for retrieval")

            case_specific_instructions = build_intranet_ask_instructions(
                generate_follow_up=generate_follow_up,
            )

            ask_handle = await self.knowledge_manager.ask(
                query_text,
                case_specific_instructions=case_specific_instructions,
                _return_reasoning_steps=_return_reasoning_steps,
                parent_chat_context=conversation_context,
                clarification_up_q=clarification_up_q,
                clarification_down_q=clarification_down_q,
                rolling_summary_in_prompts=rolling_summary_in_prompts,
                response_format=RAGLLMResponse,
            )

            if self.sandbox_mode:
                return ask_handle

            result = await ask_handle.result()

            response = self._structure_response(
                result,
                query_text,
                conversation_id,
                user_id,
                generate_follow_up,
            )

            self._update_conversation(conversation_id, query_text, response, user_id)
            print(f"✅ Query processed successfully")
            return response

        except Exception as e:
            print(f"❌ Error processing query: {e}")
            return {
                "answer": f"I encountered an error processing your question: {str(e)}",
                "sources": [],
                "follow_up_questions": [],
                "conversation_id": conversation_id,
                "user_id": user_id,
                "error": str(e),
            }

    async def update(
        self,
        update_prompt: str,
        *,
        _return_reasoning_steps: bool = False,
        conversation_context: Optional[List[Dict[str, str]]] = None,
        clarification_up_q: asyncio.Queue[str] | None = None,
        clarification_down_q: asyncio.Queue[str] | None = None,
        rolling_summary_in_prompts: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Delegate update requests to KnowledgeManager with enhanced instructions.
        The KnowledgeManager has access to FileManager tools and can handle:
        - File ingestion from directories
        - Specific document updates
        - Content deprecation/removal
        """
        try:
            if conversation_context is None:
                conversation_context = self._get_conversation_context("default")

            # Build enhanced instructions for file-aware updates
            case_specific_instructions = build_intranet_update_instructions()

            # Delegate to KnowledgeManager's update method
            update_handle = await self.knowledge_manager.update(
                update_prompt,
                _return_reasoning_steps=_return_reasoning_steps,
                parent_chat_context=conversation_context,
                clarification_up_q=clarification_up_q,
                clarification_down_q=clarification_down_q,
                rolling_summary_in_prompts=rolling_summary_in_prompts,
                case_specific_instructions=case_specific_instructions,
            )

            if self.sandbox_mode:
                return update_handle

            result = await update_handle.result()
            return {"status": "ok", "result": result}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def generate_follow_up_questions(
        self,
        query: str,
        answer: str,
        sources: List[Dict],
    ) -> List[str]:
        """
        Generate relevant follow-up questions based on the query and results.

        Args:
            query: Original query
            answer: Generated answer
            sources: Source documents/sections used

        Returns:
            List of follow-up question strings
        """
        try:
            # Extract topics from sources
            topics = set()
            departments = set()

            for source in sources[:5]:  # Limit to top 5 sources
                # Extract topics from metadata
                if source.get("key_topics"):
                    topics.update(source["key_topics"][:3])  # Top 3 topics per source

                if source.get("department"):
                    departments.add(source["department"])

            # Generate follow-ups based on context
            follow_ups = []

            # Topic-based follow-ups
            for topic in list(topics)[:2]:
                follow_ups.append(f"What else should I know about {topic}?")

            # Department-specific follow-ups
            for dept in list(departments)[:1]:
                follow_ups.append(
                    f"Are there other {dept} policies I should be aware of?",
                )

            # Generic follow-ups
            follow_ups.extend(
                [
                    "Can you provide more specific examples?",
                    "What are the key requirements I need to follow?",
                    "Are there any recent updates to this information?",
                ],
            )

            return follow_ups[:3]  # Return top 3 follow-ups

        except Exception as e:
            print(f"❌ Follow-up generation failed: {e}")
            return [
                "Can you tell me more about this topic?",
                "Are there related policies I should know about?",
                "What are the key points I need to remember?",
            ]

    def _get_conversation_context(self, conversation_id: str) -> List[Dict]:
        """Get conversation history for context."""
        return self.conversations.get(conversation_id, [])[-self.conv_context_length :]

    async def get_conversation_history(
        self,
        conversation_id: str,
        user_id: Optional[str] = None,
    ) -> List[Dict]:
        """Get full conversation history for a given conversation ID.

        Args:
            conversation_id: ID of the conversation to retrieve
            user_id: Optional user ID for additional filtering/validation

        Returns:
            List of conversation turns
        """
        try:
            conversation_turns = self.conversations.get(conversation_id, [])

            # Optional: Filter by user_id if provided
            if user_id:
                # For now, just log the user_id for potential future filtering
                print(
                    f"📜 Retrieving conversation {conversation_id} for user {user_id}",
                )
                # In the future, you could filter turns by user_id if needed

            return conversation_turns

        except Exception as e:
            print(f"❌ Error retrieving conversation history: {e}")
            return []

    async def cleanup_old_conversations(self, max_age_hours: int = 24) -> None:
        """Clean up old conversations to manage memory.

        Args:
            max_age_hours: Maximum age of conversations to keep (in hours)
        """
        try:
            from datetime import datetime, timedelta

            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

            conversations_to_remove = []
            for conv_id, turns in self.conversations.items():
                if turns:
                    # Check the last turn's timestamp
                    last_turn = turns[-1]
                    last_timestamp = datetime.fromisoformat(
                        last_turn.get("timestamp", ""),
                    )
                    if last_timestamp < cutoff_time:
                        conversations_to_remove.append(conv_id)

            for conv_id in conversations_to_remove:
                del self.conversations[conv_id]

            if conversations_to_remove:
                print(f"🧹 Cleaned up {len(conversations_to_remove)} old conversations")

        except Exception as e:
            print(f"❌ Error cleaning up conversations: {e}")

    def _update_conversation(
        self,
        conversation_id: str,
        query: str,
        response: Dict,
        user_id: Optional[str] = None,
    ):
        """Update conversation history."""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []

        self.conversations[conversation_id].append(
            {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "answer": response.get("answer", ""),
                "sources_count": len(response.get("sources", [])),
                "user_id": user_id,
            },
        )

        # Keep only recent conversations
        if len(self.conversations[conversation_id]) > self.conv_context_length * 2:
            self.conversations[conversation_id] = self.conversations[conversation_id][
                -self.conv_context_length :
            ]

    def _structure_response(
        self,
        result: str,
        query: str,
        conversation_id: str,
        user_id: Optional[str] = None,
        generate_follow_up: bool = False,
    ) -> Dict[str, Any]:
        """Structure the LLM response using Pydantic validation."""
        try:
            if MODELS_AVAILABLE:
                # Use Pydantic model for validation
                print("🔍 Validating response with Pydantic model...")

                # Try to create a validated response
                response_model = RAGQueryResponse.from_unstructured_response(
                    result=result,
                    query=query,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    follow_up_generator=(
                        self.generate_follow_up_questions
                        if generate_follow_up
                        else None
                    ),
                )

                print("✅ Response successfully validated with Pydantic")
                return response_model.to_legacy_format(generate_follow_up)

            else:
                # Fallback to original logic if models not available
                print("⚠️ Pydantic models not available, using fallback")
                return self._fallback_structure_response(
                    result,
                    query,
                    conversation_id,
                    user_id,
                    generate_follow_up,
                )

        except Exception as e:
            print(f"❌ Response structuring/validation failed: {e}")

            # Create a minimal valid response on error
            if MODELS_AVAILABLE:
                try:
                    error_response = RAGQueryResponse(
                        answer=(
                            result
                            if isinstance(result, str) and result.strip()
                            else "I encountered an issue processing your question."
                        ),
                        conversation_id=conversation_id,
                        user_id=user_id,
                        timestamp=datetime.now().isoformat(),
                        error=str(e),
                    )
                    return error_response.to_legacy_format()
                except Exception as nested_e:
                    print(f"❌ Even error response creation failed: {nested_e}")

            # Ultimate fallback
            return self._fallback_structure_response(
                result,
                query,
                conversation_id,
                user_id,
                str(e),
                generate_follow_up,
            )

    def _fallback_structure_response(
        self,
        result: str,
        query: str,
        conversation_id: str,
        user_id: Optional[str] = None,
        error: Optional[str] = None,
        generate_follow_up: bool = False,
    ) -> Dict[str, Any]:
        """Fallback response structuring when Pydantic models are not available."""
        try:
            # Try to parse structured response if available
            if result.strip().startswith("{") and result.strip().endswith("}"):
                parsed = json.loads(result.strip())
                # Ensure required fields
                parsed.setdefault("conversation_id", conversation_id)
                parsed.setdefault("user_id", user_id)
                parsed.setdefault("timestamp", datetime.now().isoformat())
                if error:
                    parsed["error"] = error
                return parsed

            # Fallback: structure unstructured response
            follow_up_questions = (
                self.generate_follow_up_questions(
                    query,
                    result,
                    [],
                )
                if generate_follow_up
                else []
            )

            fallback_response = {
                "answer": (
                    result
                    if isinstance(result, str) and result.strip()
                    else "I encountered an issue processing your question."
                ),
                "sources": [],
                "follow_up_questions": follow_up_questions,
                "conversation_id": conversation_id,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "error": error,
            }
            if not generate_follow_up:
                del fallback_response["follow_up_questions"]
            return fallback_response

        except Exception as e:
            print(f"❌ Fallback response structuring failed: {e}")
            return {
                "answer": "I encountered an issue processing your question.",
                "sources": [],
                "follow_up_questions": [],
                "conversation_id": conversation_id,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "error": error or str(e),
            }

    # Tool-loop compatible methods for Unity integration

    async def pre_embed_configured_columns(self, schema_path: str) -> Dict[str, Any]:
        """
        Pre-embed columns configured in schema file for optimal performance.
        This should be called once after document ingestion to prepare embeddings.

        Args:
            schema_path: Path to schema file containing embedding configuration

        Returns:
            Dict with embedding results and timing information
        """
        return await pre_embed_from_schema(self.knowledge_manager, schema_path)

    async def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        try:
            # Get raw stats from knowledge manager
            stats_prompt = "Provide statistics about the current knowledge base including document counts by type, departments, and search performance. Return the information in a structured format."

            from unity.common.llm_helpers import start_async_tool_use_loop

            stats_handle = start_async_tool_use_loop(
                self.client,
                stats_prompt,
                {"ask": self.knowledge_manager.ask},
                loop_id=f"{self.__class__.__name__}.get_statistics",
            )
            raw_stats = await stats_handle.result()

            # Try to get actual counts from the knowledge manager
            try:
                # Get documents count
                doc_results = self.knowledge_manager._search(
                    tables=["content"],
                    filters={"content_type": "document"},
                    limit=1000,
                )
                total_documents = len(doc_results.get("content", []))

                # Get sections count
                section_results = self.knowledge_manager._search(
                    tables=["content"],
                    filters={"content_type": "section"},
                    limit=1000,
                )
                total_sections = len(section_results.get("content", []))

                # Count policies (documents with type 'policy')
                policy_results = self.knowledge_manager._search(
                    tables=["content"],
                    filters={"content_type": "document", "document_type": "policy"},
                    limit=1000,
                )
                total_policies = len(policy_results.get("content", []))

                # Get available tables
                available_tables = [
                    "content",
                ]  # We know we have at least the content table

            except Exception as e:
                print(f"⚠️ Could not get detailed stats: {e}")
                total_documents = 0
                total_sections = 0
                total_policies = 0
                available_tables = []

            return {
                "total_documents": total_documents,
                "total_sections": total_sections,
                "total_policies": total_policies,
                "system_status": "operational" if total_documents > 0 else "no_data",
                "available_tables": available_tables,
                "raw_stats": raw_stats,
            }

        except Exception as e:
            print(f"❌ Error getting statistics: {e}")
            return {
                "total_documents": 0,
                "total_sections": 0,
                "total_policies": 0,
                "system_status": "error",
                "available_tables": [],
                "error": str(e),
            }
