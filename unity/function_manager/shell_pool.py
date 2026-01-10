"""
Shell session pool management.

Analogous to VenvPool but for shell languages. Manages persistent shell
sessions keyed by (language, session_id) pairs, allowing multiple independent
stateful execution contexts.
"""

from __future__ import annotations

import asyncio
from typing import Dict, Optional, Tuple

from .shell_session import ShellExecutionResult, ShellLanguage, ShellSession


class ShellPool:
    """
    Pool of persistent shell sessions.

    Manages shell sessions keyed by (language, session_id). Multiple sessions
    allow independent stateful execution contexts for the same language,
    similar to having multiple terminal tabs open.

    Usage:
        pool = ShellPool()

        # Execute in default session (session_id=0)
        result = await pool.execute(language="bash", command="export FOO=bar")
        result = await pool.execute(language="bash", command="echo $FOO")

        # Execute in a different session (independent state)
        result = await pool.execute(
            language="bash",
            command="echo $FOO",  # Will be empty - different session
            session_id=1,
        )

        # Clean up
        await pool.close()
    """

    def __init__(self):
        """Initialize an empty shell pool."""
        self._sessions: Dict[Tuple[ShellLanguage, int], ShellSession] = {}
        self._lock = asyncio.Lock()

    async def execute(
        self,
        *,
        language: ShellLanguage,
        command: str,
        session_id: int = 0,
        timeout: Optional[float] = 30.0,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
    ) -> ShellExecutionResult:
        """
        Execute a command in a persistent shell session.

        Creates the session on first use for this (language, session_id) pair.
        Subsequent calls with the same key reuse the existing session, preserving
        state from previous commands.

        Args:
            language: Shell language ("bash", "zsh", "sh", "powershell").
            command: The shell command to execute.
            session_id: Session identifier within the language (default 0).
                        Different session_ids have independent state.
            timeout: Maximum time to wait for command completion (default 30s).
            env: Environment variables for new sessions (ignored if session exists).
            cwd: Working directory for new sessions (ignored if session exists).

        Returns:
            ShellExecutionResult with stdout, stderr, exit_code, and error.
        """
        key = (language, session_id)

        async with self._lock:
            if key not in self._sessions:
                session = ShellSession(language=language, env=env, cwd=cwd)
                await session.start()
                self._sessions[key] = session

        session = self._sessions[key]
        return await session.execute(command, timeout=timeout)

    async def get_session(
        self,
        *,
        language: ShellLanguage,
        session_id: int = 0,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
    ) -> ShellSession:
        """
        Get or create a shell session.

        This is useful when you need direct access to the session object,
        for example to check if it's running or to use advanced features.

        Args:
            language: Shell language ("bash", "zsh", "sh", "powershell").
            session_id: Session identifier within the language (default 0).
            env: Environment variables for new sessions (ignored if session exists).
            cwd: Working directory for new sessions (ignored if session exists).

        Returns:
            The ShellSession instance for the given key.
        """
        key = (language, session_id)

        async with self._lock:
            if key not in self._sessions:
                session = ShellSession(language=language, env=env, cwd=cwd)
                await session.start()
                self._sessions[key] = session

        return self._sessions[key]

    async def close_session(
        self,
        *,
        language: ShellLanguage,
        session_id: int = 0,
    ) -> bool:
        """
        Close a specific session.

        Args:
            language: Shell language.
            session_id: Session identifier.

        Returns:
            True if a session was closed, False if no such session existed.
        """
        key = (language, session_id)

        async with self._lock:
            if key in self._sessions:
                await self._sessions[key].close()
                del self._sessions[key]
                return True
            return False

    async def close(self) -> None:
        """
        Close all sessions in the pool.

        Safe to call multiple times.
        """
        async with self._lock:
            for session in self._sessions.values():
                await session.close()
            self._sessions.clear()

    def get_active_sessions(self) -> list[Tuple[ShellLanguage, int]]:
        """
        List all active session keys.

        Returns:
            List of (language, session_id) tuples for all active sessions.
        """
        return list(self._sessions.keys())

    def has_session(
        self,
        *,
        language: ShellLanguage,
        session_id: int = 0,
    ) -> bool:
        """
        Check if a session exists in the pool.

        Args:
            language: Shell language.
            session_id: Session identifier.

        Returns:
            True if the session exists and is active.
        """
        key = (language, session_id)
        return key in self._sessions and self._sessions[key].is_running

    async def __aenter__(self) -> "ShellPool":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
