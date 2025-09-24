"""
HTTP Client for RAG API Evaluation
==================================

This module provides an HTTP client for interacting with the RAG API
for evaluation purposes with proper error handling and response parsing.
"""

import aiohttp
import asyncio
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class RAGResponse:
    """Response from the RAG API."""

    answer: Optional[str]
    sources: List[Dict[str, Any]]
    follow_up_questions: List[str]
    conversation_id: str
    user_id: Optional[str]
    timestamp: str
    confidence: Optional[float]
    response_time: float
    error: Optional[str] = None
    success: bool = False

    # API-specific fields
    query: Optional[str] = None
    turn_id: Optional[str] = None
    context_used: Optional[bool] = None

    @classmethod
    def from_api_response(
        cls,
        api_data: Dict[str, Any],
        measured_response_time: float,
    ) -> "RAGResponse":
        """Create RAGResponse from API JSON response."""
        return cls(
            answer=api_data.get("answer"),
            sources=api_data.get("sources", []),
            follow_up_questions=api_data.get("follow_up_questions", []),
            conversation_id=api_data.get("conversation_id", ""),
            user_id=api_data.get("user_id"),
            timestamp=api_data.get("timestamp", datetime.now().isoformat()),
            confidence=api_data.get("confidence"),
            response_time=measured_response_time,
            error=api_data.get("error"),
            success=bool(api_data.get("success", False)),
            query=api_data.get("query"),
            turn_id=api_data.get("turn_id"),
            context_used=api_data.get("context_used"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "answer": self.answer,
            "sources": self.sources,
            "follow_up_questions": self.follow_up_questions,
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "response_time": self.response_time,
            "error": self.error,
            "success": self.success,
            "query": self.query,
            "turn_id": self.turn_id,
            "context_used": self.context_used,
        }


class RAGHTTPClient:
    """
    HTTP client for the RAG API with proper error handling and response parsing.
    """

    def __init__(
        self,
        base_url: str = "http://0.0.0.0:8000",
        timeout: Optional[float] = None,
        max_retries: int = 2,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the RAG HTTP client.

        Args:
            base_url: Base URL of the RAG API
            timeout: Request timeout in seconds (default: None for no timeout limit)
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        print(f"🌐 RAG HTTP client initialized for {self.base_url}")

    async def health_check(self) -> Dict[str, Any]:
        """Check if the API is healthy and responding."""
        try:
            async with aiohttp.ClientSession() as session:
                # Use shorter timeout for health check, but make it configurable
                health_timeout = min(self.timeout or 30, 30)  # Max 30s for health check
                timeout = aiohttp.ClientTimeout(total=health_timeout)
                async with session.get(
                    f"{self.base_url}/health",
                    timeout=timeout,
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        print(
                            f"✅ API health check passed: {data.get('status', 'unknown')}",
                        )
                        return data
                    else:
                        print(f"⚠️ API health check returned status {response.status}")
                        return {"status": "unhealthy", "status_code": response.status}
        except Exception as e:
            print(f"❌ API health check failed: {e}")
            return {"status": "error", "error": str(e)}

    async def query(
        self,
        question: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        *,
        retreival_mode: str = "tool_loop",
        session: Optional[aiohttp.ClientSession] = None,
    ) -> RAGResponse:
        """
        Send a query to the RAG API.

        Args:
            question: The question to ask
            conversation_id: Optional conversation ID
            user_id: Optional user ID

        Returns:
            RAGResponse object with the API response
        """
        request_data = {
            "query": question,
            "conversation_id": conversation_id,
            "user_id": user_id,
            "retreival_mode": (retreival_mode or "tool_loop"),
        }

        # Remove None values
        request_data = {k: v for k, v in request_data.items() if v is not None}

        for attempt in range(self.max_retries):
            try:
                start_time = time.time()

                timeout_config = None
                if self.timeout:
                    timeout_config = aiohttp.ClientTimeout(total=self.timeout)

                # Reuse provided session when available for connection pooling
                if session is None:
                    async with aiohttp.ClientSession() as _session:
                        async with _session.post(
                            f"{self.base_url}/query",
                            json=request_data,
                            timeout=timeout_config,
                        ) as response:
                            response_time = time.time() - start_time

                            if response.status == 200:
                                response_data = await response.json()
                                return RAGResponse.from_api_response(
                                    response_data,
                                    response_time,
                                )
                            else:
                                error_text = await response.text()
                                print(
                                    f"❌ API returned status {response.status}: {error_text}",
                                )

                                return RAGResponse(
                                    answer=None,
                                    sources=[],
                                    follow_up_questions=[],
                                    conversation_id=conversation_id or "",
                                    user_id=user_id,
                                    timestamp=datetime.now().isoformat(),
                                    confidence=None,
                                    response_time=response_time,
                                    error=f"HTTP {response.status}: {error_text}",
                                    success=False,
                                    query=question,
                                )
                else:
                    async with session.post(
                        f"{self.base_url}/query",
                        json=request_data,
                        timeout=timeout_config,
                    ) as response:
                        response_time = time.time() - start_time

                        if response.status == 200:
                            response_data = await response.json()
                            return RAGResponse.from_api_response(
                                response_data,
                                response_time,
                            )
                        else:
                            error_text = await response.text()
                            print(
                                f"❌ API returned status {response.status}: {error_text}",
                            )

                            return RAGResponse(
                                answer=None,
                                sources=[],
                                follow_up_questions=[],
                                conversation_id=conversation_id or "",
                                user_id=user_id,
                                timestamp=datetime.now().isoformat(),
                                confidence=None,
                                response_time=response_time,
                                error=f"HTTP {response.status}: {error_text}",
                                success=False,
                                query=question,
                            )

            except asyncio.TimeoutError:
                print(
                    f"⏰ Request timeout (attempt {attempt + 1}/{self.max_retries + 1})",
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    continue
                else:
                    return RAGResponse(
                        answer=None,
                        sources=[],
                        follow_up_questions=[],
                        conversation_id=conversation_id or "",
                        user_id=user_id,
                        timestamp=datetime.now().isoformat(),
                        confidence=None,
                        response_time=self.timeout or 0.0,
                        error="Request timeout",
                        success=False,
                        query=question,
                    )

            except Exception as e:
                print(
                    f"❌ Request error (attempt {attempt + 1}/{self.max_retries + 1}): {e}",
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    continue
                else:
                    return RAGResponse(
                        answer=None,
                        sources=[],
                        follow_up_questions=[],
                        conversation_id=conversation_id or "",
                        user_id=user_id,
                        timestamp=datetime.now().isoformat(),
                        confidence=None,
                        response_time=0.0,
                        error=str(e),
                        success=False,
                        query=question,
                    )

    async def query_batch(
        self,
        questions: List[str],
        max_concurrent: int = 5,
        conversation_prefix: str = "eval",
    ) -> List[RAGResponse]:
        """
        Send a batch of queries to the RAG API with concurrency control.

        Args:
            questions: List of questions to ask
            max_concurrent: Maximum concurrent requests
            conversation_prefix: Prefix for conversation IDs

        Returns:
            List of RAGResponse objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        # Create a shared connector/session to maximise connection reuse
        connector_limit = max(100, int(max_concurrent))
        timeout_config = None
        if self.timeout:
            timeout_config = aiohttp.ClientTimeout(total=self.timeout)

        async with aiohttp.TCPConnector(
            limit=connector_limit,
            limit_per_host=connector_limit,
        ) as connector:
            async with aiohttp.ClientSession(
                connector=connector,
                timeout=timeout_config,
            ) as session:

                async def query_with_semaphore(
                    question: str,
                    index: int,
                ) -> RAGResponse:
                    async with semaphore:
                        conversation_id = f"{conversation_prefix}_{index:04d}"
                        user_id = f"eval_user_{index:04d}"

                        print(
                            f"📋 Querying {index + 1}/{len(questions)}: {question[:60]}...",
                        )
                        response = await self.query(
                            question,
                            conversation_id,
                            user_id,
                            session=session,
                        )

                        if response.error:
                            print(f"❌ Query {index + 1} failed: {response.error}")
                        else:
                            print(
                                f"✅ Query {index + 1} completed in {response.response_time:.2f}s",
                            )

                        return response

                # Create tasks for all queries
                tasks = [
                    query_with_semaphore(question, i)
                    for i, question in enumerate(questions)
                ]

                print(
                    f"🚀 Starting batch of {len(questions)} queries with max {max_concurrent} concurrent...",
                )

                # Execute all tasks
                start_time = time.time()
                responses = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time

        # Handle any exceptions
        processed_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                print(f"❌ Query {i + 1} failed with exception: {response}")
                error_response = RAGResponse(
                    answer="",
                    sources=[],
                    follow_up_questions=[],
                    conversation_id=f"{conversation_prefix}_{i:04d}",
                    user_id=f"eval_user_{i:04d}",
                    timestamp=datetime.now().isoformat(),
                    confidence=None,
                    response_time=0.0,
                    error=str(response),
                    query=questions[i],
                )
                processed_responses.append(error_response)
            else:
                processed_responses.append(response)

        successful = sum(1 for r in processed_responses if not r.error)
        failed = len(processed_responses) - successful

        print(f"✅ Batch completed in {total_time:.2f}s:")
        print(f"   • Successful: {successful}/{len(questions)}")
        print(f"   • Failed: {failed}/{len(questions)}")

        return processed_responses

    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics from the API."""
        try:
            async with aiohttp.ClientSession() as session:
                # Use configurable timeout, defaulting to 30s for stats
                stats_timeout = min(self.timeout or 30, 60)  # Max 60s for stats
                timeout = aiohttp.ClientTimeout(total=stats_timeout)
                async with session.get(
                    f"{self.base_url}/system/stats",
                    timeout=timeout,
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        print(
                            f"📊 System stats retrieved: {data.get('total_documents', 0)} documents",
                        )
                        return data
                    else:
                        print(f"❌ Failed to get system stats: HTTP {response.status}")
                        return {}
        except Exception as e:
            print(f"❌ Error getting system stats: {e}")
            return {}
