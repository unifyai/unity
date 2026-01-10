"""
Persistent shell session management.

Provides a way to keep a shell process alive and send commands to it,
accumulating state across multiple command executions. This is analogous
to VenvPool's persistent connections but for shell languages.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

ShellLanguage = Literal["bash", "zsh", "sh", "powershell"]


@dataclass
class ShellExecutionResult:
    """Result from executing a command in a shell session."""

    stdout: str
    stderr: str
    exit_code: int
    error: Optional[str] = None


class ShellSession:
    """
    A persistent shell session that maintains state across command executions.

    This is analogous to VenvPool's persistent connections but for shell languages.
    Commands are sent to a long-running shell subprocess, and state (environment
    variables, functions, aliases, working directory) persists between calls.

    Usage:
        session = ShellSession(language="bash")
        await session.start()
        result = await session.execute("export FOO=bar")
        result = await session.execute("echo $FOO")  # prints "bar"
        await session.close()
    """

    def __init__(
        self,
        language: ShellLanguage = "bash",
        *,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
    ):
        """
        Initialize a shell session.

        Args:
            language: Shell interpreter to use ("bash", "zsh", "sh", "powershell").
            env: Optional environment variables to set in the shell.
            cwd: Optional working directory to start in.
        """
        self.language = language
        self.initial_env = env
        self.initial_cwd = cwd
        self._process: Optional[asyncio.subprocess.Process] = None
        self._marker = f"__UNITY_SHELL_DONE_{uuid.uuid4().hex[:8]}__"
        self._started = False

    def _get_shell_command(self) -> List[str]:
        """Get the shell interpreter command for the configured language."""
        commands = {
            "bash": ["/bin/bash", "--norc", "--noprofile", "-i"],
            "zsh": ["/bin/zsh", "--no-rcs", "-i"],
            "sh": ["/bin/sh", "-i"],
            "powershell": ["pwsh", "-NoProfile", "-NoLogo", "-Command", "-"],
        }
        if self.language not in commands:
            raise ValueError(f"Unsupported shell language: {self.language}")
        return commands[self.language]

    def _build_env(self) -> Optional[Dict[str, str]]:
        """Build the environment dict for the subprocess."""
        if self.initial_env is None:
            return None
        import os

        env = os.environ.copy()
        env.update(self.initial_env)
        return env

    async def start(self) -> None:
        """
        Start the shell subprocess.

        Raises:
            RuntimeError: If the session is already started.
        """
        if self._started:
            raise RuntimeError("Session already started")

        shell_cmd = self._get_shell_command()

        self._process = await asyncio.create_subprocess_exec(
            *shell_cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,  # Merge stderr into stdout
            env=self._build_env(),
            cwd=self.initial_cwd,
        )
        self._started = True

    async def execute(self, command: str) -> ShellExecutionResult:
        """
        Execute a command in the persistent session.

        Args:
            command: The shell command to execute.

        Returns:
            ShellExecutionResult with stdout, stderr, exit_code, and error.

        Raises:
            RuntimeError: If the session is not started.
        """
        # Placeholder - will be implemented in Step 1.2
        raise NotImplementedError("Command execution will be implemented in Step 1.2")

    async def close(self) -> None:
        """
        Terminate the shell session.

        Safe to call multiple times or on an unstarted session.
        """
        if self._process is not None:
            if self._process.returncode is None:
                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    self._process.kill()
                    await self._process.wait()
            self._process = None
        self._started = False

    @property
    def is_running(self) -> bool:
        """Check if the session is active."""
        return (
            self._started
            and self._process is not None
            and self._process.returncode is None
        )

    async def __aenter__(self) -> "ShellSession":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
