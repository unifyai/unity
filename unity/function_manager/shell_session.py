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
        """Get the shell interpreter command for the configured language.

        We use non-interactive mode to avoid prompts and command echoing
        which would interfere with our marker-based output parsing.
        """
        commands = {
            # Non-interactive bash - no prompts, no rc files
            "bash": ["/bin/bash", "--norc", "--noprofile"],
            # Non-interactive zsh - no prompts, no rc files
            "zsh": ["/bin/zsh", "--no-rcs", "--no-globalrcs"],
            # POSIX sh - minimal shell
            "sh": ["/bin/sh"],
            # PowerShell - read from stdin
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

    async def execute(
        self,
        command: str,
        *,
        timeout: Optional[float] = 30.0,
    ) -> ShellExecutionResult:
        """
        Execute a command in the persistent session.

        Uses a marker-based approach to detect command completion:
        1. Send the command
        2. Send an echo of a unique marker with the exit code
        3. Read output until we see the marker
        4. Parse exit code from marker line

        Args:
            command: The shell command to execute.
            timeout: Maximum time to wait for command completion (default 30s).
                     Set to None for no timeout.

        Returns:
            ShellExecutionResult with stdout, stderr, exit_code, and error.

        Raises:
            RuntimeError: If the session is not started.
        """
        if not self._started or self._process is None:
            raise RuntimeError("Session not started. Call start() first.")

        if self._process.returncode is not None:
            return ShellExecutionResult(
                stdout="",
                stderr="",
                exit_code=-1,
                error="Shell process has terminated",
            )

        # Construct command with completion marker
        # The marker line format: __MARKER__ <exit_code>
        # We use a subshell to capture the exit code reliably
        wrapped_command = (
            f"{command}\n" f"__unity_ec__=$?\n" f'echo "{self._marker} $__unity_ec__"\n'
        )

        try:
            self._process.stdin.write(wrapped_command.encode())
            await self._process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as e:
            return ShellExecutionResult(
                stdout="",
                stderr="",
                exit_code=-1,
                error=f"Failed to send command: {e}",
            )

        # Read until we see the marker
        stdout_lines: List[str] = []
        exit_code = 0

        async def read_until_marker() -> ShellExecutionResult:
            nonlocal exit_code
            while True:
                try:
                    line = await self._process.stdout.readline()
                except Exception as e:
                    return ShellExecutionResult(
                        stdout="".join(stdout_lines),
                        stderr="",
                        exit_code=-1,
                        error=f"Error reading output: {e}",
                    )

                if not line:
                    # Process ended unexpectedly
                    return ShellExecutionResult(
                        stdout="".join(stdout_lines),
                        stderr="",
                        exit_code=-1,
                        error="Shell process terminated unexpectedly",
                    )

                decoded = line.decode()

                # Check if this line contains our marker
                if self._marker in decoded:
                    # Parse exit code from marker line
                    # Format: "__MARKER__ <exit_code>"
                    parts = decoded.strip().split()
                    if len(parts) >= 2:
                        try:
                            exit_code = int(parts[-1])
                        except ValueError:
                            exit_code = 0
                    break

                stdout_lines.append(decoded)

            return ShellExecutionResult(
                stdout="".join(stdout_lines),
                stderr="",  # Merged into stdout
                exit_code=exit_code,
                error=None,
            )

        # Apply timeout if specified
        if timeout is not None:
            try:
                return await asyncio.wait_for(read_until_marker(), timeout=timeout)
            except asyncio.TimeoutError:
                return ShellExecutionResult(
                    stdout="".join(stdout_lines),
                    stderr="",
                    exit_code=-1,
                    error=f"Command timed out after {timeout} seconds",
                )
        else:
            return await read_until_marker()

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
