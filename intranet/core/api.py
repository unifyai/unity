"""
Query Interface for Midland Heart RAG System
===========================================

User-friendly interface for interacting with the RAG system, featuring:
1. Natural language query processing
2. Conversation context management
3. Response enhancement and formatting
4. Query suggestions and autocomplete
5. Source attribution and citations
6. User feedback collection
"""

import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid
import re
from contextlib import asynccontextmanager

# RAG Agent components
from .rag_agent import IntranetRAGAgent

# Import new Pydantic models
try:
    MODELS_AVAILABLE = True
except ImportError:
    print("Warning: RAG models not available, using fallback models")
    MODELS_AVAILABLE = False

# AI/ML
import unify

# Web framework (FastAPI example)
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Setup logging first
from .system_utils import setup_logging

logger = setup_logging()


@dataclass
class ConversationTurn:
    """Single turn in a conversation."""

    turn_id: str
    user_query: str
    system_response: str
    sources: List[Dict[str, Any]]
    confidence: float
    timestamp: datetime
    response_time: float
    feedback: Optional[Dict[str, Any]] = None


@dataclass
class Conversation:
    """Complete conversation session."""

    conversation_id: str
    user_id: Optional[str]
    turns: List[ConversationTurn]
    started_at: datetime
    last_activity: datetime
    metadata: Dict[str, Any]


# Removed custom ConversationManager - now using Unity conversation manager in RAG agent


class ResponseEnhancer:
    """Enhances responses with formatting, citations, and additional context."""

    def __init__(self):
        # Initialize LLM for response enhancement
        self.llm_client = unify.AsyncUnify(
            "o4-mini@openai",
            cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            traced=json.loads(os.environ.get("UNIFY_TRACED", "true")),
        )

    async def enhance_response(
        self,
        response: str,
        sources: List[Dict[str, Any]],
        query: str,
        conversation_context: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Enhance response with formatting and citations."""

        enhanced = {
            "answer": response,
            "formatted_answer": await self._format_response(response),
            "citations": await self._generate_citations(sources),
            "source_summary": await self._summarize_sources(sources),
            "confidence_explanation": await self._explain_confidence(response, sources),
            "related_topics": await self._suggest_related_topics(query, response),
            "metadata": {
                "response_length": len(response),
                "num_sources": len(sources),
                "has_policy_references": await self._has_policy_references(response),
                "answer_type": await self._classify_answer_type(response),
            },
        }

        return enhanced

    async def _format_response(self, response: str) -> str:
        """Format response with proper structure and emphasis."""

        # Add markdown formatting for better readability
        formatted = response

        # Emphasize important terms
        important_terms = [
            "policy",
            "procedure",
            "requirement",
            "must",
            "should",
            "contact",
        ]
        for term in important_terms:
            pattern = rf"\b({term})\b"
            formatted = re.sub(pattern, r"**\1**", formatted, flags=re.IGNORECASE)

        # Format lists better
        formatted = re.sub(r"^-\s+", "• ", formatted, flags=re.MULTILINE)

        return formatted

    async def _generate_citations(
        self,
        sources: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """Generate properly formatted citations."""

        citations = []

        for i, source in enumerate(sources, 1):
            citation = {
                "id": str(i),
                "title": source.get("title", "Unknown Document"),
                "type": source.get("document_type", "document"),
                "source_path": source.get("source_path", ""),
                "relevance": source.get("relevance_score", 0.0),
            }
            citations.append(citation)

        return citations

    async def _summarize_sources(self, sources: List[Dict[str, Any]]) -> str:
        """Create a summary of source documents."""

        if not sources:
            return "No source documents found."

        doc_types = {}
        departments = {}

        for source in sources:
            doc_type = source.get("document_type", "unknown")
            department = source.get("department", "unknown")

            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            if department != "unknown":
                departments[department] = departments.get(department, 0) + 1

        summary_parts = [f"Found {len(sources)} relevant documents"]

        if doc_types:
            type_list = [f"{count} {doc_type}" for doc_type, count in doc_types.items()]
            summary_parts.append(f"including {', '.join(type_list)}")

        if departments:
            dept_list = list(departments.keys())
            summary_parts.append(f"from {', '.join(dept_list)} department(s)")

        return ". ".join(summary_parts) + "."

    async def _explain_confidence(
        self,
        response: str,
        sources: List[Dict[str, Any]],
    ) -> str:
        """Explain confidence level in the response."""

        if not sources:
            return "Low confidence - no source documents found"

        avg_relevance = sum(
            source.get("relevance_score", 0) for source in sources
        ) / len(sources)

        if avg_relevance > 0.8:
            return "High confidence - answer is well-supported by relevant documents"
        elif avg_relevance > 0.6:
            return "Medium confidence - answer is supported by moderately relevant documents"
        else:
            return "Low confidence - answer is based on loosely related documents"

    async def _suggest_related_topics(self, query: str, response: str) -> List[str]:
        """Suggest related topics the user might be interested in."""

        self.llm_client.reset_messages()

        topics_prompt = f"""
        Based on this query and response about company policies, suggest 3 related topics or questions that might be useful.

        Query: {query}
        Response: {response[:300]}...

        Suggest 3 related topics that an employee at Midland Heart might want to know about:
        """

        try:
            topics_response = await self.llm_client.generate(
                messages=[{"role": "user", "content": topics_prompt}],
            )

            # Parse topics
            topics = []
            for line in topics_response.strip().split("\n"):
                if line.strip() and not line.startswith("Based on"):
                    topic = line.strip().lstrip("123456789.-• ")
                    if topic:
                        topics.append(topic)

            return topics[:3]

        except Exception as e:
            logger.warning(f"Failed to generate related topics: {e}")
            return []

    async def _has_policy_references(self, response: str) -> bool:
        """Check if response contains policy references."""

        policy_indicators = [
            "policy",
            "procedure",
            "guideline",
            "handbook",
            "regulation",
        ]
        response_lower = response.lower()

        return any(indicator in response_lower for indicator in policy_indicators)

    async def _classify_answer_type(self, response: str) -> str:
        """Classify the type of answer provided."""

        response_lower = response.lower()

        if any(
            word in response_lower
            for word in ["step", "process", "procedure", "how to"]
        ):
            return "procedural"
        elif any(
            word in response_lower for word in ["policy", "rule", "requirement", "must"]
        ):
            return "policy"
        elif any(
            word in response_lower
            for word in ["contact", "phone", "email", "department"]
        ):
            return "contact_info"
        elif "error" in response_lower or "sorry" in response_lower:
            return "error"
        else:
            return "informational"


# Removed QueryInterface - using RAG agent directly with Unity conversation manager

# Global RAG agent (to be initialized)
rag_agent: Optional[IntranetRAGAgent] = None


async def ensure_system_initialized():
    """Initialize system components if needed (schema + documents)."""

    import os

    if os.environ.get("RAG_SKIP_INIT", "false").lower() == "true":
        logger.info("⏩ RAG_SKIP_INIT is true – Skipping system initialization.")
        from intranet.scripts.utils import activate_project

        # Set up Unity project context
        import sys
        from pathlib import Path

        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root))
        # Best-effort activation without forcing context creation. Avoid raising
        # on races when multiple workers initialize simultaneously.
        try:
            activate_project("Intranet", overwrite=False)
        except Exception as e:
            logger.warning(f"activate_project best-effort failed (continuing): {e}")
        return True

    try:
        import sys
        from pathlib import Path
        from .system_utils import SystemInitializer
        from intranet.scripts.utils import activate_project, get_config_values

        logger.info("🔍 Checking system initialization...")

        # Set up Unity project context
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root))
        try:
            activate_project("Intranet", overwrite=True)
        except Exception as e:
            # When multiple workers start together, one may have already created
            # contexts. Treat duplicate-creation as benign and continue.
            logger.warning(f"activate_project overwrite best-effort: {e}")

        # Get configuration
        config = get_config_values()
        use_tool_loops = os.environ.get("RAG_USE_TOOL_LOOPS", "false").lower() == "true"

        # Add schema path to config
        config["schema_path"] = str(Path(__file__).parent.parent / "flat_schema.json")

        logger.info("🧹 Ensuring clean initialization...")

        # Initialize system using the new modular approach
        initializer = SystemInitializer(use_tool_loops=use_tool_loops)
        # allow runtime override via env
        embed_along = os.environ.get("RAG_EMBED_ALONG", "true").lower() != "false"
        # Serialize expensive initialize_system across workers using a simple
        # file lock to prevent duplicate context creation during startup.
        from pathlib import Path as _P
        import time as _t

        lock_path = _P("/tmp/intranet_init.lock")
        got_lock = False
        for _ in range(60):
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.close(fd)
                got_lock = True
                break
            except FileExistsError:
                _t.sleep(0.25)

        try:
            results = await initializer.initialize_system(
                config,
                embed_along=embed_along,
            )
        finally:
            if got_lock:
                try:
                    os.remove(str(lock_path))
                except Exception:
                    pass

        if results.get("success"):
            logger.info("🎉 System initialization completed successfully")
            return True
        else:
            error = results.get("error", "Unknown error")
            logger.error(f"❌ System initialization failed: {error}")
            raise Exception(f"System initialization failed: {error}")

    except Exception as e:
        logger.error(f"❌ Error during system initialization: {e}")
        import traceback

        traceback.print_exc()
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application."""
    # Startup
    logger.info("🚀 RAG API starting up...")
    logger.info("🐛 DEBUG MODE: Breakpoints should be active!")
    print("🐛 DEBUG: FastAPI server starting up - breakpoints should work now!")

    global rag_agent

    try:
        # Initialize Unity environment and system components
        await ensure_system_initialized()

        # Initialize the RAG system with Unity conversation manager
        logger.info("🧠 Initializing RAG agent with Unity conversation manager...")
        rag_agent = IntranetRAGAgent()

        logger.info("✅ RAG system with Unity conversation manager initialized")
        logger.info("🌐 API server ready to accept connections on port 8000")
        logger.info("🔧 Use intranet/scripts/simple_debug_client.py to test debugging")
        print("🌐 API server ready! Visit http://localhost:8000/health to test")

    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG system: {e}")
        raise

    yield  # Application runs here

    # Shutdown
    logger.info("🛑 RAG API shutting down...")

    # Cleanup resources
    if rag_agent:
        # Clean up conversations and resources
        await rag_agent.cleanup_old_conversations(
            max_age_hours=1,
        )  # Clean up after 1 hour
        logger.info("🧹 Cleaning up RAG system resources")

    logger.info("✅ RAG API shutdown complete")


# FastAPI Application with lifespan
app = FastAPI(
    title="Midland Heart Intranet RAG API",
    version="1.0.0",
    description="Talk to your data - RAG system for Midland Heart policy documents",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class QueryRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None


class APIQueryResponse(BaseModel):
    """API-specific response model that includes additional API metadata."""

    # Core RAG response fields
    answer: str
    sources: List[Dict[str, Any]]
    follow_up_questions: List[str]
    conversation_id: str
    user_id: Optional[str] = None
    timestamp: str
    confidence: Optional[float] = None
    search_metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    # API-specific fields
    query: str
    response_time: float
    turn_id: Optional[str] = None
    context_used: Optional[bool] = None


class FeedbackRequest(BaseModel):
    helpful: bool
    accurate: bool
    complete: bool
    comments: Optional[str] = None


class SystemStats(BaseModel):
    total_documents: int
    total_sections: int
    total_policies: int
    system_status: str
    available_tables: List[str]


@app.post("/query", response_model=APIQueryResponse)
async def query_endpoint(request: QueryRequest):
    """Main query endpoint for processing user questions with validated Pydantic responses."""

    logger.info(
        f"📝 Query received: '{request.query[:100]}...' from user {request.user_id}",
    )

    if not rag_agent:
        logger.error("RAG system not initialized")
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        start_time = datetime.now()

        # Use RAG agent which now returns a validated dictionary
        rag_response = await rag_agent.ask(
            query_text=request.query,
            conversation_id=request.conversation_id or f"conv_{uuid.uuid4().hex[:8]}",
            user_id=request.user_id,
        )

        response_time = (datetime.now() - start_time).total_seconds()

        # Log response details safely
        confidence = rag_response.get("confidence", 0.0)
        if confidence:
            logger.info(
                f"✅ Query processed in {response_time:.2f}s, confidence: {confidence:.2f}",
            )
        else:
            logger.info(f"✅ Query processed in {response_time:.2f}s")

        # Create API response using the validated RAG response
        api_response = APIQueryResponse(
            # Core fields from RAG response
            answer=rag_response.get("answer", ""),
            sources=rag_response.get("sources", []),
            follow_up_questions=rag_response.get("follow_up_questions", []),
            conversation_id=rag_response.get("conversation_id", ""),
            user_id=rag_response.get("user_id"),
            timestamp=rag_response.get("timestamp", datetime.now().isoformat()),
            confidence=rag_response.get("confidence"),
            search_metadata=rag_response.get("search_metadata", {}),
            error=rag_response.get("error"),
            # API-specific fields
            query=request.query,
            response_time=response_time,
            turn_id=f"turn_{uuid.uuid4().hex[:8]}",  # Generate unique turn ID
            context_used=True,  # Assume context was used
        )

        # Validate the response
        if MODELS_AVAILABLE:
            logger.debug("🔍 Validating API response structure...")
            # The response is already validated by Pydantic when we create APIQueryResponse
            logger.debug("✅ API response validation successful")

        return api_response

    except Exception as e:
        logger.error(f"❌ Query endpoint error: {e}")

        # Create error response
        try:
            error_response = APIQueryResponse(
                answer=f"I encountered an error processing your question: {str(e)}",
                sources=[],
                follow_up_questions=[],
                conversation_id=request.conversation_id
                or f"conv_{uuid.uuid4().hex[:8]}",
                user_id=request.user_id,
                timestamp=datetime.now().isoformat(),
                error=str(e),
                query=request.query,
                response_time=(
                    (datetime.now() - start_time).total_seconds()
                    if "start_time" in locals()
                    else 0.0
                ),
                turn_id=f"turn_{uuid.uuid4().hex[:8]}",
            )
            return error_response
        except Exception as nested_e:
            logger.error(f"❌ Failed to create error response: {nested_e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback/{conversation_id}/{turn_id}")
async def feedback_endpoint(
    conversation_id: str,
    turn_id: str,
    feedback: FeedbackRequest,
):
    """Add feedback to a response."""

    logger.info(
        f"👍 Feedback received for conversation {conversation_id}, turn {turn_id}: helpful={feedback.helpful}",
    )

    if not rag_agent:
        logger.error("RAG system not initialized for feedback")
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        # Note: Feedback storage can be implemented in the future
        # For now, just log the feedback
        feedback_data = feedback.model_dump()
        logger.info(f"📝 Feedback logged: {feedback_data}")

        logger.info(f"✅ Feedback recorded successfully")
        return {
            "message": "Feedback recorded",
            "conversation_id": conversation_id,
            "turn_id": turn_id,
        }

    except ValueError as e:
        logger.warning(f"⚠️  Feedback validation error: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Feedback endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history using Unity conversation manager."""

    logger.info(f"📜 Conversation history requested for {conversation_id}")

    if not rag_agent:
        logger.error("RAG system not initialized for conversation retrieval")
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        history = await rag_agent.get_conversation_history(conversation_id)

        turns_count = len(history)
        logger.info(f"✅ Retrieved conversation with {turns_count} turns")

        return {
            "conversation_id": conversation_id,
            "turns": history,
            "turn_count": turns_count,
            "timestamp": datetime.now().isoformat(),
        }

    except ValueError as e:
        logger.warning(f"⚠️  Conversation not found: {conversation_id}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Conversation endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""

    logger.debug("🔍 Health check requested")

    status = "healthy" if rag_agent is not None else "degraded"

    health_data = {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "rag_agent_initialized": rag_agent is not None,
        "unity_conversation_manager": True,
        "version": "1.0.0",
    }

    logger.debug(f"Health status: {status}")
    return health_data


@app.get("/system/stats", response_model=SystemStats)
async def get_system_stats():
    """Get system statistics and status."""

    logger.info("📊 System stats requested")

    if not rag_agent:
        logger.error("RAG system not initialized for stats")
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        # Get stats from the existing RAG agent
        stats = await rag_agent.get_statistics()

        logger.info(f"📈 Stats retrieved: {stats.get('total_documents', 0)} documents")

        return SystemStats(
            total_documents=stats.get("total_documents", 0),
            total_sections=stats.get("total_sections", 0),
            total_policies=stats.get("total_policies", 0),
            system_status=stats.get("system_status", "unknown"),
            available_tables=stats.get("available_tables", []),
        )

    except Exception as e:
        logger.error(f"❌ System stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations/active")
async def get_active_conversations():
    """Get list of active conversations."""

    logger.info("💬 Active conversations requested")

    if not rag_agent:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        active_convs = list(rag_agent.conversations.keys())
        logger.info(f"Found {len(active_convs)} active conversations")

        return {
            "active_conversations": active_convs,
            "count": len(active_convs),
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"❌ Active conversations error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/{conversation_id}")
async def websocket_endpoint(websocket: WebSocket, conversation_id: str):
    """WebSocket endpoint for real-time interaction."""

    logger.info(f"🔗 WebSocket connection requested for conversation {conversation_id}")
    await websocket.accept()
    logger.info(f"✅ WebSocket connection established for {conversation_id}")

    try:
        while True:
            # Receive query from client
            data = await websocket.receive_text()
            query_data = json.loads(data)

            query_text = query_data.get("query", "")
            logger.info(
                f"📱 WebSocket query received: '{query_text[:50]}...' in {conversation_id}",
            )

            if not rag_agent:
                error_msg = {"error": "RAG system not initialized"}
                await websocket.send_text(json.dumps(error_msg))
                logger.error("❌ WebSocket query failed: RAG system not initialized")
                continue

            # Process query using RAG agent
            start_time = datetime.now()
            rag_response = await rag_agent.ask(
                query_text=query_text,
                conversation_id=conversation_id,
                user_id=query_data.get("user_id"),
            )

            response_time = (datetime.now() - start_time).total_seconds()

            # Create response using the new validated format
            response = {
                "conversation_id": rag_response.get("conversation_id", conversation_id),
                "turn_id": f"turn_{uuid.uuid4().hex[:8]}",
                "query": query_text,
                "answer": rag_response.get("answer", ""),
                "confidence": rag_response.get("confidence", 0.0),
                "sources": rag_response.get("sources", []),
                "follow_up_questions": rag_response.get("follow_up_questions", []),
                "response_time": response_time,
                "context_used": True,
                "timestamp": rag_response.get("timestamp", datetime.now().isoformat()),
                "error": rag_response.get("error"),
            }

            logger.info(f"✅ WebSocket query processed in {response_time:.2f}s")

            # Send response
            await websocket.send_text(json.dumps(response, default=str))

    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket disconnected for conversation {conversation_id}")
    except json.JSONDecodeError as e:
        logger.error(f"❌ WebSocket JSON decode error: {e}")
        await websocket.send_text(json.dumps({"error": "Invalid JSON format"}))
    except Exception as e:
        logger.error(f"❌ WebSocket error for {conversation_id}: {e}")
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
        except Exception:
            logger.error("Failed to send error message over WebSocket")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
