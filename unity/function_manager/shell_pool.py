"""
Shell session pool management.

Analogous to VenvPool but for shell languages. Manages persistent shell
sessions keyed by (language, session_id) pairs, allowing multiple independent
stateful execution contexts.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Any, Dict, Optional, Tuple

from .shell_session import ShellExecutionResult, ShellLanguage, ShellSession


@dataclass
class SessionMetadata:
    language: str
    session_id: int
    created_at: datetime
    last_used: datetime


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

    def __init__(self, *, max_total_sessions: int = 20):
        """Initialize an empty shell pool.

        Args:
            max_total_sessions: Maximum number of concurrent sessions across all
                (language, session_id) keys for this pool.
        """
        self._sessions: Dict[Tuple[ShellLanguage, int], ShellSession] = {}
        self._metadata: Dict[Tuple[ShellLanguage, int], SessionMetadata] = {}
        self._lock = asyncio.Lock()
        self._max_total_sessions = int(max_total_sessions)

    def _active_session_count(self) -> int:
        return sum(
            1 for s in self._sessions.values() if getattr(s, "is_running", False)
        )

    def get_all_sessions(self) -> list[dict[str, Any]]:
        """Return a list of all active shell sessions with metadata."""
        out: list[dict[str, Any]] = []
        for (language, session_id), session in list(self._sessions.items()):
            if not session.is_running:
                continue
            md = self._metadata.get((language, session_id))
            if md is None:
                now = datetime.now(timezone.utc)
                md = SessionMetadata(
                    language=str(language),
                    session_id=int(session_id),
                    created_at=now,
                    last_used=now,
                )
                self._metadata[(language, session_id)] = md
            out.append(
                {
                    "language": str(language),
                    "session_id": int(session_id),
                    "created_at": md.created_at.isoformat(),
                    "last_used": md.last_used.isoformat(),
                    # State can be inspected via get_session_state(); keep listing cheap.
                    "state_summary": "active",
                },
            )
        return out

    async def get_session_state(
        self,
        *,
        language: ShellLanguage,
        session_id: int,
        detail: str = "summary",
    ) -> Dict[str, Any]:
        """Inspect the state of an active shell session.

        Delegates to ShellSession.snapshot_state() and formats output.
        """
        key = (language, int(session_id))
        sess = self._sessions.get(key)
        if sess is None or not sess.is_running:
            return {
                "error": f"Shell session {(str(language), int(session_id))} not found",
                "error_type": "validation",
            }

        snap = await sess.snapshot_state()

        def _line_count(s: str) -> int:
            return len([ln for ln in (s or "").splitlines() if ln.strip()])

        def _parse_var_names(snapshot_vars: str) -> list[str]:
            names: set[str] = set()
            for ln in (snapshot_vars or "").splitlines():
                ln = ln.strip()
                # bash declare -p: declare -- VAR="x" / declare -x VAR="x"
                m = re.match(r"^(?:declare|typeset)\b.*\s([A-Za-z_][A-Za-z0-9_]*)=", ln)
                if m:
                    names.add(m.group(1))
                    continue
                # sh export -p: export VAR='x'
                m2 = re.match(r"^export\s+([A-Za-z_][A-Za-z0-9_]*)=", ln)
                if m2:
                    names.add(m2.group(1))
            return sorted(names)

        def _parse_alias_names(snapshot_aliases: str) -> list[str]:
            names: set[str] = set()
            for ln in (snapshot_aliases or "").splitlines():
                ln = ln.strip()
                # alias ll='ls -la'
                m = re.match(r"^alias\s+([A-Za-z_][A-Za-z0-9_]*)=", ln)
                if m:
                    names.add(m.group(1))
            return sorted(names)

        def _parse_function_names(snapshot_functions: str) -> list[str]:
            names: set[str] = set()
            for ln in (snapshot_functions or "").splitlines():
                ln = ln.strip()
                # bash: foo () { ... }  OR foo() { ... }
                m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(\))?\s*\{", ln)
                if m:
                    names.add(m.group(1))
            return sorted(names)

        cwd = (snap.get("cwd") or "").strip()
        vars_blob = snap.get("variables") or ""
        funcs_blob = snap.get("functions") or ""
        aliases_blob = snap.get("aliases") or ""
        options_blob = snap.get("options") or ""

        if detail in ("summary", "names"):
            var_names = _parse_var_names(vars_blob)
            fn_names = _parse_function_names(funcs_blob)
            alias_names = _parse_alias_names(aliases_blob)
            return {
                "cwd": cwd,
                "variables": var_names,
                "functions": fn_names,
                "aliases": alias_names,
                "options_count": _line_count(options_blob),
                "summary": (
                    f"cwd={cwd!r}, {len(var_names)} vars, {len(fn_names)} funcs, {len(alias_names)} aliases"
                ),
            }

        if detail == "full":
            # Return full (potentially large) blobs; caller is expected to truncate for display.
            return {
                "cwd": cwd,
                "variables": vars_blob,
                "functions": funcs_blob,
                "aliases": aliases_blob,
                "options": options_blob,
            }

        return {
            "error": f"Unsupported detail level: {detail!r}",
            "error_type": "validation",
        }

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
