from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..common.llm_helpers import SteerableToolHandle


class BaseWebSearch(ABC):
    """Abstract base class for WebSearch managers."""

    @abstractmethod
    async def ask(
        self,
        text: str,
        *,
        _return_reasoning_steps: bool = False,
        parent_chat_context: Optional[List[Dict[str, Any]]] = None,
        _call_id: Optional[str] = None,
    ) -> SteerableToolHandle:  # pragma: no cover - interface only
        """Ask a question about the web and return a steerable handle."""
        raise NotImplementedError
