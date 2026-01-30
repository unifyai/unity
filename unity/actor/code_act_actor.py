import asyncio
import base64
import contextlib
import contextvars
import functools
import inspect
import io
import traceback
import ast
import copy
import uuid
import sys
from datetime import datetime, timezone
from secrets import token_hex as _token_hex
import logging
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Optional,
    Callable,
    Awaitable,
    Type,
    TYPE_CHECKING,
    Tuple,
    Literal,
    Union,
)
from pydantic import BaseModel, Field

from unity.actor.base import BaseCodeActActor
from unity.actor.handle import ActorHandle
from unity.common.async_tool_loop import SteerableToolHandle, start_async_tool_loop
from unity.common.llm_client import new_llm_client
from unity.function_manager.primitives import ComputerPrimitives
from unity.actor.prompt_builders import build_code_act_prompt
from unity.events.manager_event_logging import log_manager_call
from unity.image_manager.types.image_refs import ImageRefs
from unity.image_manager.types.raw_image_ref import RawImageRef
from unity.image_manager.types.annotated_image_ref import AnnotatedImageRef
from unity.common._async_tool.loop_config import TOOL_LOOP_LINEAGE
from unity.common.hierarchical_logger import (
    build_hierarchy_label,
    log_boundary_event,
)
from unity.events.manager_event_logging import (
    new_call_id,
    publish_manager_method_event,
)

if TYPE_CHECKING:
    from unity.actor.environments.base import BaseEnvironment
    from unity.function_manager.function_manager import FunctionManager


_CURRENT_SANDBOX: contextvars.ContextVar["PythonExecutionSession"] = (
    contextvars.ContextVar(
        "code_act_current_sandbox",
    )
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured output types for sandbox execution (Pydantic discriminated union)
# ---------------------------------------------------------------------------


class TextPart(BaseModel):
    """A text output part from sandbox execution."""

    type: Literal["text"] = "text"
    text: str

    def to_llm_content(self) -> dict:
        """Convert to LLM content block format."""
        return {"type": "text", "text": self.text}


class ImagePart(BaseModel):
    """An image output part from sandbox execution (e.g., from display())."""

    type: Literal["image"] = "image"
    mime: str = "image/png"
    data: str  # base64 encoded

    def to_llm_content(self) -> dict:
        """Convert to LLM content block format."""
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{self.mime};base64,{self.data}"},
        }


# Discriminated union - Pydantic auto-parses based on `type` field
OutputPart = Annotated[Union[TextPart, ImagePart], Field(discriminator="type")]


def parts_to_text(parts: List[Union[TextPart, ImagePart]]) -> str:
    """Convert a list of OutputPart to a plain text string.

    Useful for backward compatibility and simple text extraction.
    Only TextPart parts are included; ImagePart parts are skipped.
    """
    return "".join(p.text for p in parts if isinstance(p, TextPart))


def parts_to_llm_content(parts: List[Union[TextPart, ImagePart]]) -> List[dict]:
    """Convert a list of OutputParts to LLM content blocks, preserving order.

    This function maintains the original interleaving of text and images,
    unlike the legacy approach which collected all images at the end.

    Adjacent TextParts are merged into a single text block for cleaner output.
    """
    if not parts:
        return []

    blocks: List[dict] = []
    pending_text = ""

    for part in parts:
        if isinstance(part, TextPart):
            pending_text += part.text
        elif isinstance(part, ImagePart):
            # Flush any pending text before the image
            if pending_text:
                blocks.append({"type": "text", "text": pending_text})
                pending_text = ""
            blocks.append(part.to_llm_content())

    # Flush any remaining text
    if pending_text:
        blocks.append({"type": "text", "text": pending_text})

    return blocks


class ExecutionResult(BaseModel):
    """Result from sandbox code execution, implementing FormattedToolResult protocol.

    This model gives the sandbox full control over how its output is formatted
    for the LLM, preserving the original interleaving of text and images from
    print() and display() calls.
    """

    stdout: List[Union[TextPart, ImagePart]] = Field(default_factory=list)
    stderr: List[Union[TextPart, ImagePart]] = Field(default_factory=list)
    result: Any = None
    error: Optional[str] = None
    browser_used: bool = False
    browser_state: Optional[Dict[str, Any]] = None
    language: Optional[str] = None
    state_mode: Optional[str] = None
    session_id: Optional[int] = None
    session_name: Optional[str] = None
    venv_id: Optional[int] = None
    session_created: Optional[bool] = None
    duration_ms: Optional[int] = None

    model_config = {"arbitrary_types_allowed": True}

    def to_llm_content(self) -> List[dict]:
        """Format this execution result for the LLM, preserving output order.

        Implements the FormattedToolResult protocol, giving the sandbox full
        control over how its output appears in the LLM transcript.
        """
        blocks: List[dict] = []

        # Build metadata section (non-stdout/stderr fields)
        meta: Dict[str, Any] = {}
        if self.result is not None:
            meta["result"] = self.result
        if self.error is not None:
            meta["error"] = self.error
        if self.language is not None:
            meta["language"] = self.language
        if self.state_mode is not None:
            meta["state_mode"] = self.state_mode
        if self.session_id is not None:
            meta["session_id"] = self.session_id
        if self.session_name is not None:
            meta["session_name"] = self.session_name
        if self.venv_id is not None:
            meta["venv_id"] = self.venv_id
        if self.session_created is not None:
            meta["session_created"] = self.session_created
        if self.duration_ms is not None:
            meta["duration_ms"] = self.duration_ms
        if self.browser_used:
            meta["browser_used"] = True
        if self.browser_state is not None:
            meta["browser_state"] = self.browser_state

        # Add metadata block if present
        if meta:
            import json

            meta_text = json.dumps(meta, indent=2, default=str)
            blocks.append({"type": "text", "text": meta_text})

        # Add stdout with preserved ordering (interleaved text/images)
        if self.stdout:
            has_content = any(
                (isinstance(p, TextPart) and p.text.strip()) or isinstance(p, ImagePart)
                for p in self.stdout
            )
            if has_content:
                if blocks:  # Add separator if we have metadata
                    blocks.append({"type": "text", "text": "\n--- stdout ---\n"})
                blocks.extend(parts_to_llm_content(self.stdout))

        # Add stderr with preserved ordering (if non-empty)
        if self.stderr:
            has_content = any(
                (isinstance(p, TextPart) and p.text.strip()) or isinstance(p, ImagePart)
                for p in self.stderr
            )
            if has_content:
                blocks.append({"type": "text", "text": "\n--- stderr ---\n"})
                blocks.extend(parts_to_llm_content(self.stderr))

        # Ensure we always return at least something
        if not blocks:
            blocks.append({"type": "text", "text": "(no output)"})

        return blocks


# ---------------------------------------------------------------------------
# ContextVars for per-execution stream isolation
# ---------------------------------------------------------------------------
_stdout_parts: contextvars.ContextVar[List[Union[TextPart, ImagePart]]] = (
    contextvars.ContextVar(
        "sandbox_stdout_parts",
    )
)
_stderr_parts: contextvars.ContextVar[List[Union[TextPart, ImagePart]]] = (
    contextvars.ContextVar(
        "sandbox_stderr_parts",
    )
)
_current_stdout: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "sandbox_current_stdout",
)
_current_stderr: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "sandbox_current_stderr",
)


# ---------------------------------------------------------------------------
# Stream capture classes
# ---------------------------------------------------------------------------
class StreamLike:
    """Captures output to a parts list, supporting text and images."""

    def __init__(
        self,
        parts_var: contextvars.ContextVar[List[Union[TextPart, ImagePart]]],
    ):
        self._parts_var = parts_var

    def write(self, obj: str) -> int:
        parts = self._parts_var.get()
        # Merge consecutive text writes into a single TextPart
        if parts and isinstance(parts[-1], TextPart):
            # TextPart is immutable (Pydantic), so we need to replace it
            last = parts[-1]
            parts[-1] = TextPart(text=last.text + obj)
        else:
            parts.append(TextPart(text=obj))
        return len(obj)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


class StreamRouter:
    """Routes writes to the current context's stream, falls back to original stream.

    Uses __getattr__ to forward ALL unknown attributes/methods to the current stream,
    ensuring compatibility with Jupyter's introspection (e.g., _ipython_* methods),
    and any future stream methods we haven't explicitly handled.
    """

    def __init__(
        self,
        context_var: contextvars.ContextVar[Any],
        fallback: Any,
    ):
        # Use object.__setattr__ to avoid triggering our __getattr__
        object.__setattr__(self, "_context_var", context_var)
        object.__setattr__(self, "_fallback", fallback)

    def _get_stream(self) -> Any:
        try:
            return self._context_var.get()
        except LookupError:
            return self._fallback

    def write(self, s: str) -> int:
        return self._get_stream().write(s)

    def flush(self) -> None:
        return self._get_stream().flush()

    def __getattr__(self, name: str) -> Any:
        """Forward any unknown attribute to the current stream."""
        return getattr(self._get_stream(), name)


# ---------------------------------------------------------------------------
# Lazy StreamRouter installation (installed on first sandbox use)
# ---------------------------------------------------------------------------
# We install the StreamRouter lazily (on first sandbox use) rather than at
# module load to avoid conflicts with pytest and other test frameworks that
# replace sys.stdout after imports. By installing on first use, we capture
# whatever stdout is current at that moment (e.g., pytest's capture) as our
# fallback, ensuring proper output routing.
_stream_router_installed = False
_original_stdout: Any = None
_original_stderr: Any = None


def _ensure_stream_router_installed() -> None:
    """Install StreamRouters for sys.stdout/stderr if not already installed.

    This is called at the start of each sandbox execution. We check if
    sys.stdout is actually a StreamRouter (not just a flag) because pytest
    and other frameworks may replace sys.stdout between tests.
    """
    global _stream_router_installed, _original_stdout, _original_stderr

    # Check if sys.stdout is still our StreamRouter (pytest may have replaced it)
    if isinstance(sys.stdout, StreamRouter):
        return  # Already installed

    # Install StreamRouter, capturing current stdout as fallback
    _original_stdout = sys.stdout
    _original_stderr = sys.stderr
    sys.stdout = StreamRouter(_current_stdout, _original_stdout)  # type: ignore[assignment]
    sys.stderr = StreamRouter(_current_stderr, _original_stderr)  # type: ignore[assignment]
    _stream_router_installed = True


# ---------------------------------------------------------------------------
# Display function for rich output (images, etc.)
# ---------------------------------------------------------------------------
def _make_display(
    parts_var: contextvars.ContextVar[List[Union[TextPart, ImagePart]]],
) -> Callable[[Any], None]:
    """Create a display function that adds images to output parts."""

    def display(obj: Any) -> None:
        try:
            from PIL import Image
        except ImportError:
            Image = None  # type: ignore[misc, assignment]

        parts = parts_var.get()

        if Image is not None and isinstance(obj, Image.Image):
            buf = io.BytesIO()
            obj.save(buf, format="PNG")
            b64_data = base64.b64encode(buf.getvalue()).decode("ascii")
            parts.append(ImagePart(mime="image/png", data=b64_data))
        elif isinstance(obj, str):
            parts.append(TextPart(text=obj + "\n"))
        else:
            # Fallback: convert to string
            parts.append(TextPart(text=str(obj) + "\n"))

    return display


# ---------------------------------------------------------------------------
# Context manager for sandbox output capture
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def capture_sandbox_output():
    """Context manager that sets up stream capture for a sandbox execution.

    Yields (stdout_parts, stderr_parts, display_fn) tuple.
    All ContextVars are properly reset on exit.

    The StreamRouter is installed lazily on first use (not at module load)
    to avoid conflicts with pytest and other test frameworks that replace
    sys.stdout after imports.
    """
    # Ensure StreamRouter is installed (lazy, once per process)
    _ensure_stream_router_installed()

    stdout_parts: List[Union[TextPart, ImagePart]] = []
    stderr_parts: List[Union[TextPart, ImagePart]] = []

    # Set up ContextVars
    stdout_token = _stdout_parts.set(stdout_parts)
    stderr_token = _stderr_parts.set(stderr_parts)

    # Create StreamLike instances for this execution
    stdout_stream = StreamLike(_stdout_parts)
    stderr_stream = StreamLike(_stderr_parts)

    stdout_stream_token = _current_stdout.set(stdout_stream)
    stderr_stream_token = _current_stderr.set(stderr_stream)

    display_fn = _make_display(_stdout_parts)

    try:
        yield stdout_parts, stderr_parts, display_fn
    finally:
        _stdout_parts.reset(stdout_token)
        _stderr_parts.reset(stderr_token)
        _current_stdout.reset(stdout_stream_token)
        _current_stderr.reset(stderr_stream_token)


SupportedShellLanguage = Literal["bash", "zsh", "sh", "powershell"]
SupportedLanguage = Literal["python", "bash", "zsh", "sh", "powershell"]
StateMode = Literal["stateful", "read_only", "stateless"]
SessionKey = Tuple[str, Optional[int], int]  # (language, venv_id, session_id)


def _validation_error(
    *,
    message: str,
    suggestion: str,
    state_mode: str,
    session_id: int | None,
    session_name: str | None,
    language: str,
    venv_id: int | None = None,
) -> dict:
    return {
        "error": message,
        "error_type": "validation",
        "suggestion": suggestion,
        "received": {
            "state_mode": state_mode,
            "session_id": session_id,
            "session_name": session_name,
            "language": language,
            "venv_id": venv_id,
        },
    }


def _validate_execution_params(
    *,
    state_mode: str,
    session_id: int | None,
    session_name: str | None,
    language: str,
    venv_id: int | None = None,
    supported_languages: tuple[str, ...] = (
        "python",
        "bash",
        "zsh",
        "sh",
        "powershell",
    ),
    # Name resolution/lookup is actor-owned, so validation accepts callables.
    resolve_session_name: Optional[Callable[[str], Optional[SessionKey]]] = None,
    get_session_name_for_id: Optional[
        Callable[[str, Optional[int], int], Optional[str]]
    ] = None,
    session_exists: Optional[Callable[[str, Optional[int], int], bool]] = None,
    max_sessions_total: Optional[int] = None,
    active_session_count: Optional[int] = None,
) -> dict | None:
    """
    Validate state_mode + session selection rules for execute_code.

    Returns:
        None if valid, otherwise a structured validation error dict.

    Notes:
        This function intentionally returns structured errors (not exceptions)
        so the LLM can self-correct deterministically.
    """
    if language not in supported_languages:
        return _validation_error(
            message=f"Unsupported language: {language!r}",
            suggestion=f"Use one of: {sorted(supported_languages)}",
            state_mode=state_mode,
            session_id=session_id,
            session_name=session_name,
            language=language,
            venv_id=venv_id,
        )

    if state_mode not in ("stateful", "read_only", "stateless"):
        return _validation_error(
            message=f"Unsupported state_mode: {state_mode!r}",
            suggestion="Use one of: 'stateful', 'read_only', 'stateless'",
            state_mode=state_mode,
            session_id=session_id,
            session_name=session_name,
            language=language,
            venv_id=venv_id,
        )

    # Stateless must not reference sessions.
    if state_mode == "stateless" and (
        session_id is not None or session_name is not None
    ):
        return _validation_error(
            message="Cannot use state_mode='stateless' with a session.",
            suggestion="Remove session_id/session_name or switch to state_mode='stateful' or 'read_only'.",
            state_mode=state_mode,
            session_id=session_id,
            session_name=session_name,
            language=language,
            venv_id=venv_id,
        )

    # Read-only requires an existing session.
    if state_mode == "read_only" and (session_id is None and session_name is None):
        return _validation_error(
            message="Cannot use state_mode='read_only' without specifying a session.",
            suggestion="Provide session_id or session_name (must refer to an existing session), or use state_mode='stateless'.",
            state_mode=state_mode,
            session_id=session_id,
            session_name=session_name,
            language=language,
            venv_id=venv_id,
        )

    # If both are present, ensure they match.
    if session_id is not None and session_name is not None:
        if resolve_session_name is None:
            return _validation_error(
                message="Cannot validate session_name against session_id (no resolver configured).",
                suggestion="Specify only one of session_id or session_name.",
                state_mode=state_mode,
                session_id=session_id,
                session_name=session_name,
                language=language,
                venv_id=venv_id,
            )
        key = resolve_session_name(session_name)
        if key is None:
            # For stateful, the caller may choose to create+bind; for read_only it must exist.
            if state_mode == "read_only":
                return _validation_error(
                    message=f"Session name {session_name!r} not found for read_only execution.",
                    suggestion="Use an existing session_name (see list_sessions) or specify an existing session_id.",
                    state_mode=state_mode,
                    session_id=session_id,
                    session_name=session_name,
                    language=language,
                    venv_id=venv_id,
                )
        else:
            resolved_language, resolved_venv_id, resolved_session_id = key
            if (
                resolved_language != language
                or resolved_venv_id != venv_id
                or resolved_session_id != session_id
            ):
                return _validation_error(
                    message=(
                        f"session_id and session_name refer to different sessions. "
                        f"{session_name!r} resolves to {(resolved_language, resolved_venv_id, resolved_session_id)} "
                        f"but received {(language, venv_id, session_id)}."
                    ),
                    suggestion="Specify only one of session_id or session_name, or make them consistent.",
                    state_mode=state_mode,
                    session_id=session_id,
                    session_name=session_name,
                    language=language,
                    venv_id=venv_id,
                )

    # If session_name is provided alone:
    if session_name is not None and session_id is None:
        if resolve_session_name is None:
            return _validation_error(
                message="Cannot resolve session_name (no resolver configured).",
                suggestion="Specify session_id instead, or configure a session registry.",
                state_mode=state_mode,
                session_id=session_id,
                session_name=session_name,
                language=language,
                venv_id=venv_id,
            )
        key = resolve_session_name(session_name)
        if key is None and state_mode == "read_only":
            return _validation_error(
                message=f"Session name {session_name!r} not found for read_only execution.",
                suggestion="Use an existing session_name (see list_sessions) or specify an existing session_id.",
                state_mode=state_mode,
                session_id=session_id,
                session_name=session_name,
                language=language,
                venv_id=venv_id,
            )

    # Optional: enforce a global session cap (actor-owned pools, so this is per-actor).
    # Note: `stateful` with no session specified means "use the default session" (session_id=0),
    # so it does NOT create a new session and should not be rejected by the global cap here.
    #
    # We only apply the cap when the call would CREATE a new stateful session:
    # - session_name is provided but does not resolve (new session would be allocated)
    if (
        max_sessions_total is not None
        and active_session_count is not None
        and state_mode == "stateful"
        and session_id is None
        and session_name is not None
        and resolve_session_name is not None
        and resolve_session_name(session_name) is None
        and active_session_count >= max_sessions_total
    ):
        return _validation_error(
            message=f"Session limit exceeded: {active_session_count} active sessions (max {max_sessions_total}).",
            suggestion="Close an existing session (close_session) or reuse an existing session_id/session_name.",
            state_mode=state_mode,
            session_id=session_id,
            session_name=session_name,
            language=language,
            venv_id=venv_id,
        )

    # If an explicit session_id is provided for stateful execution, enforce limits when it would
    # CREATE a new session (important for in-process Python sessions where pool-level limits may not apply).
    if (
        max_sessions_total is not None
        and active_session_count is not None
        and state_mode == "stateful"
        and session_id is not None
        and session_exists is not None
        and active_session_count >= max_sessions_total
    ):
        try:
            exists = bool(session_exists(language, venv_id, session_id))
        except Exception:
            exists = False
        if not exists:
            return _validation_error(
                message=f"Session limit exceeded: {active_session_count} active sessions (max {max_sessions_total}).",
                suggestion="Close an existing session (close_session) or reuse an existing session_id/session_name.",
                state_mode=state_mode,
                session_id=session_id,
                session_name=session_name,
                language=language,
                venv_id=venv_id,
            )

    # If session_id is provided and we can validate existence for read_only, do so.
    if (
        state_mode == "read_only"
        and session_id is not None
        and session_exists is not None
    ):
        if not session_exists(language, venv_id, session_id):
            name_hint = None
            if get_session_name_for_id is not None:
                name_hint = get_session_name_for_id(language, venv_id, session_id)
            hint = f" (known name: {name_hint!r})" if name_hint else ""
            return _validation_error(
                message=f"Session {(language, venv_id, session_id)} does not exist for read_only execution{hint}.",
                suggestion="Use list_sessions to find an existing session, or switch to state_mode='stateful' to create a new session.",
                state_mode=state_mode,
                session_id=session_id,
                session_name=session_name,
                language=language,
                venv_id=venv_id,
            )

    return None


class _CodeActEntrypointHandle(SteerableToolHandle):  # type: ignore[abstract-method]
    """Execute a FunctionManager entrypoint function without invoking the CodeAct LLM loop.

    TaskScheduler delegates task execution to an actor via:
    `actor.act(task_description, entrypoint=<function_id>, persist=False)`.

    When an `entrypoint` is provided, CodeActActor resolves the function by id,
    injects it into the sandbox namespace, and executes it in an asyncio task.
    """

    def __init__(
        self,
        *,
        entrypoint_id: int,
        execution_task: asyncio.Task[Any],
        on_finally: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> None:
        self._entrypoint_id = int(entrypoint_id)
        self._execution_task = execution_task
        self._completion_event = asyncio.Event()
        self._result_str: Optional[str] = None
        self._stopped = False
        self._on_finally = on_finally

        asyncio.create_task(self._monitor_execution())

    async def _monitor_execution(self) -> None:
        try:
            out = await self._execution_task
            if not self._stopped:
                self._result_str = str(out) if out is not None else ""
        except asyncio.CancelledError:
            self._stopped = True
            self._result_str = f"Entrypoint {self._entrypoint_id} was cancelled."
        except Exception as e:
            self._result_str = f"Error: {e}"
        finally:
            if self._on_finally is not None:
                try:
                    await self._on_finally()
                except Exception:
                    pass
            self._completion_event.set()

    async def ask(
        self,
        question: str,
        *,
        _parent_chat_context: list[dict] | None = None,
        images: list | dict | None = None,
    ) -> SteerableToolHandle:
        status = "completed" if self.done() else "still running"
        client = new_llm_client()
        client.set_system_message(
            "You are an AI assistant answering a status question about an in-flight entrypoint execution. "
            "Be brief and factual.",
        )
        msg = (
            f"Entrypoint {self._entrypoint_id} status: {status}.\n\n"
            f"User question: {question}"
        )
        return start_async_tool_loop(
            client=client,
            message=msg,
            tools={},
            loop_id=f"EntrypointQuestion({self._entrypoint_id})",
            max_consecutive_failures=1,
            timeout=30,
        )

    async def interject(
        self,
        message: str,
        *,
        _parent_chat_context_cont: list[dict] | None = None,
        images: list | dict | None = None,
    ) -> Optional[str]:
        # No-op for non-LLM entrypoint execution.
        return None

    async def stop(
        self,
        reason: Optional[str] = None,
    ) -> Optional[str]:
        if self._completion_event.is_set():
            return self._result_str
        self._stopped = True
        self._execution_task.cancel()
        try:
            await asyncio.wait_for(self._completion_event.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            pass
        return (
            f"Entrypoint {self._entrypoint_id} stopped."
            if not reason
            else f"Entrypoint {self._entrypoint_id} stopped: {reason}"
        )

    async def pause(self) -> Optional[str]:
        return None

    async def resume(self) -> Optional[str]:
        return None

    def done(self) -> bool:
        return self._completion_event.is_set()

    async def result(self) -> str:
        await self._completion_event.wait()
        return self._result_str or ""

    async def next_clarification(self) -> dict:
        await asyncio.Event().wait()
        return {}

    async def next_notification(self) -> dict:
        await asyncio.Event().wait()
        return {}

    async def answer_clarification(self, call_id: str, answer: str) -> None:
        return None


class PythonExecutionSession:
    """
    A stateful execution environment for running Python code asynchronously.

    This class maintains a persistent global state across multiple executions,
    capturing stdout, stderr, return values, and exceptions in a structured format.

    It can optionally use pools for persistent subprocess connections (VenvPool
    for Python venvs, ShellPool for shell sessions), enabling state to be
    preserved across multiple function calls.
    """

    def __init__(
        self,
        computer_primitives: Optional[ComputerPrimitives] = None,
        environments: Optional[Dict[str, "BaseEnvironment"]] = None,
        venv_pool: Optional[Any] = None,
        shell_pool: Optional[Any] = None,
    ):
        """
        Initializes the execution environment.

        Args:
            computer_primitives: An instance of ComputerPrimitives to be injected into the
                             global state, making browser tools available.
            environments: Optional mapping of environment namespaces to environments. If
                provided, each environment instance is injected into globals.
            venv_pool: Optional VenvPool for persistent Python venv connections.
                If provided, venv-backed functions will use persistent connections
                that maintain state across calls.
            shell_pool: Optional ShellPool for persistent shell session connections.
                If provided, shell functions will use persistent sessions.
        """
        from unity.function_manager.execution_env import create_execution_globals

        self.id: str = str(uuid.uuid4())
        self.global_state: Dict[str, Any] = create_execution_globals()
        self._browser_used: bool = False

        # Expose sandbox metadata to user code (best-effort; callers may ignore).
        self.global_state["__sandbox_id__"] = self.id

        # Notification queue is injected dynamically by execute_code when it
        # receives a _notification_up_q from the async tool loop:
        # sandbox.global_state["__notification_up_q__"] = <asyncio.Queue>
        #
        # Provide a user-driven progress helper:
        #   notify({"type": "...", ...})
        # This helper is intentionally synchronous; it uses put_nowait.
        # Notifications bubble up through the async tool loop to the outer handle.
        def notify(payload: dict) -> None:
            try:
                q = self.global_state.get("__notification_up_q__")
                if q is None:
                    return
                # Queue is expected to be an asyncio.Queue[dict]
                q.put_nowait(payload)
            except Exception:
                return

        self.global_state["notify"] = notify

        # Inject pools into namespace (for function proxies to use)
        if venv_pool is not None:
            self.global_state["__venv_pool__"] = venv_pool
        if shell_pool is not None:
            self.global_state["__shell_pool__"] = shell_pool

        class _UsageTrackingProxy:
            def __init__(self, target: Any, on_use: Callable[[], None]):
                self._target = target
                self._on_use = on_use

            def __getattr__(self, name: str) -> Any:
                # Treat any access as potential "use" since callers may invoke nested objects
                # like `computer_primitives.computer.get_screenshot()`.
                self._on_use()
                attr = getattr(self._target, name)
                if callable(attr):

                    async def _async_wrapper(*args, **kwargs):
                        self._on_use()
                        return await attr(*args, **kwargs)

                    def _sync_wrapper(*args, **kwargs):
                        self._on_use()
                        return attr(*args, **kwargs)

                    # Preserve sync vs async callable behavior.
                    if asyncio.iscoroutinefunction(attr):
                        return _async_wrapper
                    return _sync_wrapper
                return attr

        def _mark_browser_used() -> None:
            self._browser_used = True

        if environments:
            for namespace, env in environments.items():
                try:
                    # Use get_sandbox_instance() if available (for filtered primitives),
                    # otherwise fall back to get_instance()
                    if hasattr(env, "get_sandbox_instance"):
                        instance = env.get_sandbox_instance()
                    else:
                        instance = env.get_instance()
                    if namespace == "computer_primitives":
                        instance = _UsageTrackingProxy(instance, _mark_browser_used)
                    self.global_state[namespace] = instance
                except Exception:
                    # Keep sandbox usable even if a non-critical environment fails to inject.
                    continue

        # Backward-compat: allow direct injection when environments weren't provided.
        if computer_primitives and "computer_primitives" not in self.global_state:
            self.global_state["computer_primitives"] = _UsageTrackingProxy(
                computer_primitives,
                _mark_browser_used,
            )

    async def close(self) -> None:
        """
        Best-effort cleanup for an ephemeral sandbox instance.

        Notes
        -----
        - Pools (venv/shell) are owned by the actor and are not closed here.
        - This method is safe to call multiple times.
        """
        try:
            self.global_state.clear()
        except Exception as e:
            try:
                logger.warning(
                    f"PythonExecutionSession.close() failed: {e}",
                    exc_info=True,
                )
            except Exception:
                pass

    async def execute(self, code: str) -> dict:
        """
        Executes a string of Python code within the sandbox's stateful environment.

        Returns a dict with:
            stdout: list[OutputPart] - structured output parts (TextPart, ImagePart)
            stderr: list[OutputPart] - structured error output parts
            result: Any - return value of the last expression
            error: str | None - traceback if an exception occurred
            browser_used: bool - whether browser primitives were accessed
        """
        # Reset per-execution usage flags.
        self._browser_used = False
        result = None
        error = None

        with capture_sandbox_output() as (stdout_parts, stderr_parts, display_fn):
            # Inject display function into globals
            self.global_state["display"] = display_fn

            try:
                # Guardrails: prevent agent code from accidentally shadowing critical
                # injected environment globals (common failure mode in LLM-generated code).
                #
                # We do this via an AST rewrite (not brittle string heuristics):
                # rewrite any assignment targets named `primitives` / `computer_primitives`
                # to `_primitives_local` / `_computer_primitives_local`.
                #
                # This preserves the injected globals for the rest of the session.
                try:

                    class _ShadowingGuard(ast.NodeTransformer):
                        _REMAP = {
                            "primitives": "_primitives_local",
                            "computer_primitives": "_computer_primitives_local",
                        }

                        def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802
                            # Only rewrite *assignments* (Store context). Loads are preserved.
                            if (
                                isinstance(node.ctx, ast.Store)
                                and node.id in self._REMAP
                            ):
                                return ast.copy_location(
                                    ast.Name(id=self._REMAP[node.id], ctx=node.ctx),
                                    node,
                                )
                            return node

                        def visit_Global(
                            self,
                            node: ast.Global,
                        ) -> ast.AST:  # noqa: N802
                            node.names = [self._REMAP.get(n, n) for n in node.names]
                            return node

                        def visit_Nonlocal(
                            self,
                            node: ast.Nonlocal,
                        ) -> ast.AST:  # noqa: N802
                            node.names = [self._REMAP.get(n, n) for n in node.names]
                            return node

                    tree = ast.parse(code)
                    tree = _ShadowingGuard().visit(tree)  # type: ignore[assignment]
                    ast.fix_missing_locations(tree)
                    code = ast.unparse(tree)
                except Exception:
                    # Best-effort only; if rewriting fails, proceed with original code.
                    pass

                is_empty_or_comment_only = all(
                    line.strip() == "" or line.strip().startswith("#")
                    for line in code.splitlines()
                )
                if is_empty_or_comment_only:
                    code += "\npass"

                tree = ast.parse(code)
                top_level_assign_targets = set()
                for node in tree.body:
                    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                        targets = []
                        if isinstance(node, ast.Assign):
                            targets.extend(node.targets)
                        else:
                            targets.append(node.target)

                        for target in targets:
                            if isinstance(target, ast.Name):
                                top_level_assign_targets.add(target.id)
                            elif isinstance(target, ast.Tuple):
                                for elt in target.elts:
                                    if isinstance(elt, ast.Name):
                                        top_level_assign_targets.add(elt.id)

                    elif isinstance(node, (ast.Import, ast.ImportFrom)):
                        for alias in node.names:
                            top_level_assign_targets.add(
                                alias.asname or alias.name.split(".")[0],
                            )

                    elif isinstance(
                        node,
                        (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                    ):
                        top_level_assign_targets.add(node.name)

                async_code = "async def __exec_wrapper():\n"
                if top_level_assign_targets:
                    async_code += f"    global {', '.join(sorted(list(top_level_assign_targets)))}\n"

                async_code += "".join(f"    {line}\n" for line in code.splitlines())

                exec(async_code, self.global_state)
                result = await self.global_state["__exec_wrapper"]()

            except Exception:
                error = traceback.format_exc()
            finally:
                if "__exec_wrapper" in self.global_state:
                    del self.global_state["__exec_wrapper"]

        return {
            "stdout": stdout_parts,
            "stderr": stderr_parts,
            "result": result,
            "error": error,
            "browser_used": self._browser_used,
        }


class SessionExecutor:
    """
    Unified execution engine for multi-language, multi-session CodeAct execution.

    Notes
    -----
    - Python (in-process) sessions are backed by persistent PythonExecutionSession instances.
    - Shell sessions use ShellPool (persistent) and ephemeral subprocesses for stateless.
    - Venv-backed Python sessions are supported when venv_id is provided.
    """

    def __init__(
        self,
        *,
        venv_pool: Any,
        shell_pool: Any,
        environments: Optional[Dict[str, "BaseEnvironment"]] = None,
        computer_primitives: Optional[ComputerPrimitives] = None,
        function_manager: Optional["FunctionManager"] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self._venv_pool = venv_pool
        self._shell_pool = shell_pool
        self._environments = environments or {}
        self._computer_primitives = computer_primitives
        self._function_manager = function_manager
        self._timeout = timeout

        # In-process Python sessions keyed by (venv_id=None, session_id).
        self._python_sessions: Dict[
            Tuple[Optional[int], int],
            PythonExecutionSession,
        ] = {}
        self._python_session_meta: Dict[Tuple[Optional[int], int], dict[str, str]] = {}

    def has_python_session(
        self,
        *,
        session_id: int,
        venv_id: int | None = None,
    ) -> bool:
        return (venv_id, session_id) in self._python_sessions

    def list_in_process_python_sessions(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for (venv_id, session_id), sb in list(self._python_sessions.items()):
            meta = self._python_session_meta.get((venv_id, session_id)) or {}
            out.append(
                {
                    "language": "python",
                    "venv_id": venv_id,
                    "session_id": int(session_id),
                    "created_at": meta.get("created_at"),
                    "last_used": meta.get("last_used"),
                    "state_summary": "active",
                },
            )
        return out

    async def close_in_process_python_session(
        self,
        *,
        session_id: int,
        venv_id: int | None = None,
    ) -> bool:
        key = (venv_id, int(session_id))
        sb = self._python_sessions.pop(key, None)
        self._python_session_meta.pop(key, None)
        if sb is None:
            return False
        try:
            await sb.close()
        except Exception:
            pass
        return True

    async def close(self) -> None:
        # Close in-process python sandboxes; pools are owned by the actor.
        for sb in list(self._python_sessions.values()):
            try:
                await sb.close()
            except Exception:
                pass
        self._python_sessions.clear()
        self._python_session_meta.clear()

    async def execute(
        self,
        *,
        code: str,
        language: SupportedLanguage,
        state_mode: StateMode,
        session_id: int | None,
        venv_id: int | None,
        primitives: Any = None,
        computer_primitives: Any = None,
    ) -> Dict[str, Any]:
        started = datetime.now(timezone.utc)
        t0 = started.timestamp()

        # Default: use actor computer primitives (if any).
        if computer_primitives is None:
            computer_primitives = self._computer_primitives

        # ─── Python ────────────────────────────────────────────────────────
        if language == "python":
            # Special-case: session 0 is the *current bound sandbox* when present.
            # This preserves legacy CodeAct behavior (one sandbox per act() handle)
            # while enabling state sharing for execute_code(..., session_id=0) when a sandbox is bound.
            if state_mode == "stateful" and venv_id is None and session_id == 0:
                try:
                    sb0 = _CURRENT_SANDBOX.get()
                    res = await sb0.execute(code)
                    return {
                        **res,
                        "language": language,
                        "state_mode": state_mode,
                        "session_id": 0,
                        "venv_id": None,
                        "session_created": False,
                        "duration_ms": int(
                            (datetime.now(timezone.utc).timestamp() - t0) * 1000,
                        ),
                    }
                except Exception:
                    # If no sandbox is bound, fall back to executor-managed session 0.
                    pass
            # Stateless: fresh in-process sandbox per call.
            if state_mode == "stateless":
                sb = PythonExecutionSession(
                    computer_primitives=computer_primitives,
                    environments=self._environments,
                    venv_pool=self._venv_pool,
                    shell_pool=self._shell_pool,
                )
                try:
                    res = await sb.execute(code)
                finally:
                    try:
                        await sb.close()
                    except Exception:
                        pass
                return {
                    **res,
                    "language": language,
                    "state_mode": state_mode,
                    "session_id": None,
                    "venv_id": venv_id,
                    "session_created": False,
                    "duration_ms": int(
                        (datetime.now(timezone.utc).timestamp() - t0) * 1000,
                    ),
                }

            # If a venv_id is provided, use persistent subprocess sessions.
            if venv_id is not None:
                if session_id is None:
                    raise ValueError(
                        "session_id is required for venv-backed python execution",
                    )
                # Determine whether this is a new persistent session.
                existed_before = (int(venv_id), int(session_id)) in set(
                    self._venv_pool.list_active_sessions(),
                )
                # Wrap arbitrary code in a function definition so venv_runner can execute it.
                implementation = _wrap_code_as_async_function(code)
                if state_mode == "stateful":
                    out = await self._venv_pool.execute_in_venv(
                        venv_id=int(venv_id),
                        implementation=implementation,
                        call_kwargs={},
                        is_async=True,
                        session_id=int(session_id),
                        primitives=primitives,
                        computer_primitives=computer_primitives,
                        function_manager=self._function_manager,
                        timeout=self._timeout,
                    )
                    return {
                        **out,
                        "language": language,
                        "state_mode": state_mode,
                        "session_id": session_id,
                        "venv_id": venv_id,
                        "session_created": not existed_before,
                        "duration_ms": int(
                            (datetime.now(timezone.utc).timestamp() - t0) * 1000,
                        ),
                    }

                if state_mode == "read_only":
                    # Snapshot state from persistent session, then run in one-shot subprocess.
                    if self._function_manager is None:
                        raise RuntimeError(
                            "function_manager is required for venv read_only execution",
                        )
                    initial_state = await self._venv_pool.get_connection_state(
                        venv_id=int(venv_id),
                        function_manager=self._function_manager,
                        session_id=int(session_id),
                        timeout=10.0,
                    )
                    out = await self._function_manager.execute_in_venv(
                        venv_id=int(venv_id),
                        implementation=implementation,
                        call_kwargs={},
                        is_async=True,
                        initial_state=initial_state,
                        primitives=primitives,
                        computer_primitives=computer_primitives,
                    )
                    return {
                        **out,
                        "language": language,
                        "state_mode": state_mode,
                        "session_id": session_id,
                        "venv_id": venv_id,
                        "session_created": False,
                        "duration_ms": int(
                            (datetime.now(timezone.utc).timestamp() - t0) * 1000,
                        ),
                    }

                raise ValueError(
                    f"Unsupported state_mode for python venv: {state_mode}",
                )

            # In-process persistent sessions (venv_id is None).
            if session_id is None:
                raise ValueError(
                    "session_id is required for in-process python stateful/read_only execution",
                )

            key = (venv_id, int(session_id))
            if state_mode == "stateful":
                created = False
                if key not in self._python_sessions:
                    self._python_sessions[key] = PythonExecutionSession(
                        computer_primitives=computer_primitives,
                        environments=self._environments,
                        venv_pool=self._venv_pool,
                        shell_pool=self._shell_pool,
                    )
                    created = True
                    now = datetime.now(timezone.utc).isoformat()
                    self._python_session_meta[key] = {
                        "created_at": now,
                        "last_used": now,
                    }
                sb = self._python_sessions[key]
                res = await sb.execute(code)
                meta = self._python_session_meta.get(key)
                if meta is not None:
                    meta["last_used"] = datetime.now(timezone.utc).isoformat()
                return {
                    **res,
                    "language": language,
                    "state_mode": state_mode,
                    "session_id": session_id,
                    "venv_id": venv_id,
                    "session_created": created,
                    "duration_ms": int(
                        (datetime.now(timezone.utc).timestamp() - t0) * 1000,
                    ),
                }

            if state_mode == "read_only":
                # Create a throwaway sandbox seeded with current state.
                if key not in self._python_sessions:
                    raise ValueError(
                        f"Python session {key} not found for read_only execution",
                    )
                base = self._python_sessions[key]
                sb = PythonExecutionSession(
                    computer_primitives=computer_primitives,
                    environments=self._environments,
                    venv_pool=self._venv_pool,
                    shell_pool=self._shell_pool,
                )
                try:
                    # Shallow copy globals to allow read access while avoiding persistence.
                    sb.global_state.update(dict(base.global_state))
                    res = await sb.execute(code)
                finally:
                    try:
                        await sb.close()
                    except Exception:
                        pass
                return {
                    **res,
                    "language": language,
                    "state_mode": state_mode,
                    "session_id": session_id,
                    "venv_id": venv_id,
                    "session_created": False,
                    "duration_ms": int(
                        (datetime.now(timezone.utc).timestamp() - t0) * 1000,
                    ),
                }

            raise ValueError(
                f"Unsupported state_mode for python in-process: {state_mode}",
            )

        # ─── Shell ─────────────────────────────────────────────────────────
        # Stateless: ephemeral subprocess (no pool/session).
        if state_mode == "stateless":
            out = await _execute_shell_stateless(language=language, command=code)
            return {
                **out,
                "language": language,
                "state_mode": state_mode,
                "session_id": None,
                "venv_id": None,
                "session_created": False,
                "duration_ms": int(
                    (datetime.now(timezone.utc).timestamp() - t0) * 1000,
                ),
            }

        if session_id is None:
            raise ValueError(
                "session_id is required for shell stateful/read_only execution",
            )

        # Persistent shell session.
        if state_mode == "stateful":
            existed_before = self._shell_pool.has_session(
                language=language,  # type: ignore[arg-type]
                session_id=int(session_id),
            )
            res = await self._shell_pool.execute(
                language=language,  # type: ignore[arg-type]
                command=code,
                session_id=int(session_id),
                timeout=self._timeout,
            )
            return {
                "stdout": res.stdout,
                "stderr": res.stderr,
                "result": res.exit_code,
                "error": res.error,
                "browser_used": False,
                "language": language,
                "state_mode": state_mode,
                "session_id": session_id,
                "venv_id": None,
                "session_created": not existed_before,
                "duration_ms": int(
                    (datetime.now(timezone.utc).timestamp() - t0) * 1000,
                ),
            }

        if state_mode == "read_only":
            # Snapshot persistent state, restore into ephemeral session, execute, then discard.
            from unity.function_manager.shell_session import ShellSession

            sess = await self._shell_pool.get_session(
                language=language,  # type: ignore[arg-type]
                session_id=int(session_id),
            )
            snap = await sess.snapshot_state()
            tmp = ShellSession(language=language)  # type: ignore[arg-type]
            await tmp.start()
            try:
                restore_res = await tmp.restore_state(snap)
                if restore_res.error:
                    return {
                        "stdout": restore_res.stdout,
                        "stderr": restore_res.stderr,
                        "result": restore_res.exit_code,
                        "error": restore_res.error,
                        "browser_used": False,
                        "language": language,
                        "state_mode": state_mode,
                        "session_id": session_id,
                        "venv_id": None,
                        "session_created": False,
                        "duration_ms": int(
                            (datetime.now(timezone.utc).timestamp() - t0) * 1000,
                        ),
                    }
                res = await tmp.execute(code, timeout=self._timeout)
                return {
                    "stdout": res.stdout,
                    "stderr": res.stderr,
                    "result": res.exit_code,
                    "error": res.error,
                    "browser_used": False,
                    "language": language,
                    "state_mode": state_mode,
                    "session_id": session_id,
                    "venv_id": None,
                    "session_created": False,
                    "duration_ms": int(
                        (datetime.now(timezone.utc).timestamp() - t0) * 1000,
                    ),
                }
            finally:
                try:
                    await tmp.close()
                except Exception:
                    pass

        raise ValueError(f"Unsupported state_mode for shell: {state_mode}")


def _wrap_code_as_async_function(code: str) -> str:
    """
    Wrap an arbitrary code snippet into a single async function definition.

    This is required for the venv runner protocol, which expects exactly one function
    definition in the provided source.
    """
    # Ensure non-empty body.
    body = code if code.strip() else "pass"
    indented = "\n".join(
        ("    " + line) if line.strip() else "    " for line in body.splitlines()
    )
    return "async def __unity_code_act__():\n" + indented + "\n"


async def _execute_shell_stateless(
    *,
    language: SupportedLanguage,
    command: str,
) -> Dict[str, Any]:
    """
    Execute shell code in an ephemeral subprocess (stateless).
    """
    if language == "python":
        raise ValueError("Shell stateless executor called with language='python'")

    # Build command line consistent with ShellSession's non-interactive choices.
    if language == "bash":
        argv = ["/bin/bash", "--norc", "--noprofile", "-c", command]
    elif language == "zsh":
        argv = ["/bin/zsh", "--no-rcs", "--no-globalrcs", "-c", command]
    elif language == "sh":
        argv = ["/bin/sh", "-c", command]
    elif language == "powershell":
        argv = ["pwsh", "-NoProfile", "-NoLogo", "-Command", command]
    else:
        raise ValueError(f"Unsupported shell language: {language}")

    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_b, stderr_b = await proc.communicate()
        return {
            "stdout": (stdout_b or b"").decode(errors="replace"),
            "stderr": (stderr_b or b"").decode(errors="replace"),
            "result": int(proc.returncode or 0),
            "error": None,
            "browser_used": False,
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": "",
            "result": None,
            "error": f"{type(e).__name__}: {e}",
            "browser_used": False,
        }


class CodeActActor(BaseCodeActActor):
    """
    An actor that uses a conversational tool loop and a stateful code execution
    sandbox to accomplish tasks. It acts as a baseline for code-centric agents.
    """

    def __init__(
        self,
        session_connect_url: Optional[str] = None,
        headless: bool = False,
        computer_mode: str = "magnitude",
        timeout: float = 1000,
        agent_mode: str = "browser",
        agent_server_url: str = "http://localhost:3000",
        computer_primitives: Optional["ComputerPrimitives"] = None,
        environments: Optional[list["BaseEnvironment"]] = None,
        function_manager: Optional["FunctionManager"] = None,
        can_compose: bool = True,
        can_store: bool = True,
    ):
        """
        Initializes the CodeActActor.

        Args:
            computer_primitives: Optional existing ComputerPrimitives instance to reuse.
                           If provided, other browser-related params are ignored.
            environments: Optional list of execution environments. If None, defaults to
                [ComputerEnvironment, StateManagerEnvironment].
            function_manager: Manages a library of reusable functions. Exposes read-only tools
                (list_functions, search_functions, filter_functions) to the LLM.
                The LLM can call these tools to discover and retrieve reusable function implementations.
            agent_server_url: URL for the agent server. For desktop mode, pass the
                external VM's URL.
        """
        super().__init__(
            environments=environments,
            computer_primitives=computer_primitives,
            function_manager=function_manager,
            session_connect_url=session_connect_url,
            headless=headless,
            computer_mode=computer_mode,
            agent_mode=agent_mode,
            agent_server_url=agent_server_url,
        )

        # Create persistent pools that survive across act() calls
        from unity.function_manager.function_manager import VenvPool
        from unity.function_manager.shell_pool import ShellPool

        self._venv_pool = VenvPool()
        self._shell_pool = ShellPool()
        self._session_executor = SessionExecutor(
            venv_pool=self._venv_pool,
            shell_pool=self._shell_pool,
            environments=self.environments,
            computer_primitives=self._computer_primitives,
            function_manager=self.function_manager,
            timeout=timeout,
        )

        # Session name registry: name -> (language, venv_id, session_id)
        self._session_names: Dict[str, SessionKey] = {}
        # Reverse map: (language, venv_id, session_id) -> set(names)
        self._session_names_rev: Dict[SessionKey, set[str]] = {}
        # Actor-level session cap (global across languages for this actor instance).
        self._max_sessions_total: int = 20
        self._next_session_id: dict[tuple[str, Optional[int]], int] = {}

        self._timeout = timeout
        self.can_compose: bool = bool(can_compose)
        self.can_store: bool = bool(can_store)
        self._browser_tools = self._get_browser_tools()
        # Register stable tools once; per-call sandboxes are bound via _CURRENT_SANDBOX.
        self.add_tools("act", self._build_tools())

        self._main_event_loop: Optional[asyncio.AbstractEventLoop] = None
        try:
            self._main_event_loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

        # Concurrency guard: limit active sandboxes per actor instance.
        self._act_semaphore = asyncio.Semaphore(20)
        # Timeout used when acquiring the semaphore (prevents unbounded waits).
        self._act_semaphore_timeout_s: float = 30.0

    # ───────────────────────── Session name registry ─────────────────────── #

    def _register_session_name(
        self,
        *,
        name: str,
        language: str,
        venv_id: int | None,
        session_id: int,
    ) -> None:
        key: SessionKey = (language, venv_id, int(session_id))
        existing = self._session_names.get(name)
        if existing is not None and existing != key:
            raise ValueError(
                f"Session name {name!r} is already bound to {existing}, cannot rebind to {key}.",
            )
        self._session_names[name] = key
        self._session_names_rev.setdefault(key, set()).add(name)

    def _resolve_session_name(self, name: str) -> SessionKey | None:
        return self._session_names.get(name)

    def _get_session_name(
        self,
        *,
        language: str,
        venv_id: int | None,
        session_id: int,
    ) -> str | None:
        key: SessionKey = (language, venv_id, int(session_id))
        names = self._session_names_rev.get(key)
        if not names:
            return None
        # Prefer stable ordering for determinism.
        return sorted(names)[0]

    def _unregister_session_name(self, name: str) -> None:
        key = self._session_names.pop(name, None)
        if key is None:
            return
        names = self._session_names_rev.get(key)
        if names is not None:
            names.discard(name)
            if not names:
                self._session_names_rev.pop(key, None)

    def _unregister_all_names_for_session(self, *, key: SessionKey) -> None:
        names = self._session_names_rev.pop(key, None)
        if not names:
            return
        for n in list(names):
            self._session_names.pop(n, None)

    def _count_active_sessions_total(self) -> int:
        # Count unique in-process python sessions + persistent pool sessions.
        n = 0
        try:
            n += len(
                self._session_executor._python_sessions,
            )  # pylint: disable=protected-access
        except Exception:
            pass
        try:
            n += len(self._shell_pool.get_active_sessions())
        except Exception:
            pass
        try:
            n += len(self._venv_pool.list_active_sessions())
        except Exception:
            pass
        return n

    def _session_exists(
        self,
        *,
        language: str,
        venv_id: int | None,
        session_id: int,
    ) -> bool:
        if language == "python":
            if venv_id is None:
                return self._session_executor.has_python_session(
                    session_id=int(session_id),
                    venv_id=None,
                )
            # venv-backed python session exists if pool has it active
            try:
                return (int(venv_id), int(session_id)) in set(
                    self._venv_pool.list_active_sessions(),
                )
            except Exception:
                return False
        # shell
        try:
            return self._shell_pool.has_session(
                language=language,  # type: ignore[arg-type]
                session_id=int(session_id),
            )
        except Exception:
            return False

    def _validate_execution_params(
        self,
        *,
        state_mode: str,
        session_id: int | None,
        session_name: str | None,
        language: str,
        venv_id: int | None = None,
    ) -> dict | None:
        return _validate_execution_params(
            state_mode=state_mode,
            session_id=session_id,
            session_name=session_name,
            language=language,
            venv_id=venv_id,
            resolve_session_name=self._resolve_session_name,
            get_session_name_for_id=lambda l, v, s: self._get_session_name(
                language=l,
                venv_id=v,
                session_id=s,
            ),
            session_exists=lambda l, v, s: self._session_exists(
                language=l,
                venv_id=v,
                session_id=s,
            ),
            max_sessions_total=self._max_sessions_total,
            active_session_count=self._count_active_sessions_total(),
        )

    def _get_browser_tools(self) -> Dict[str, Callable]:
        """Extracts browser-related methods from the ComputerPrimitives."""
        if not self._computer_primitives:
            return {}
        return {
            "navigate": self._computer_primitives.navigate,
            "act": self._computer_primitives.act,
            "observe": self._computer_primitives.observe,
        }

    def _build_tools(self) -> Dict[str, Callable[..., Awaitable[Any]]]:
        """Builds the dictionary of tools available to the LLM."""

        async def execute_code(
            thought: str,
            code: Optional[str] = None,
            *,
            language: str,
            state_mode: str = "stateless",
            session_id: int | None = None,
            session_name: str | None = None,
            venv_id: int | None = None,
            _notification_up_q: asyncio.Queue[dict] | None = None,
        ) -> Any:
            """
            Execute code in a specified language and state mode, optionally within a session.

            Use this tool for BOTH Python and shell execution. This is the unified
            "brain execution" tool for CodeActActor.

            Key concepts
            -----------
            - **language**: "python" | "bash" | "zsh" | "sh" | "powershell"
            - **state_mode**:
              - "stateless": no session; clean execution; no persistence
              - "stateful": persistent session; state accumulates
              - "read_only": reads from an existing session but does not persist changes
            - **session_id/session_name**:
              - only meaningful for stateful/read_only
              - for stateful: if omitted, defaults to **session_id=0** (the default session)
              - to create an additional stateful session, provide a fresh `session_name` (recommended)
                or an explicit `session_id` > 0
              - **Python session_id=0** is special:
                - If this tool is called from inside a running CodeAct `act()` loop, session 0 maps to the
                  **current per-call Python sandbox** (shared via the ContextVar binding).
                - If no sandbox is bound (e.g. calling the tool directly in a unit test), session 0 behaves
                  like a normal in-process Python session managed by the SessionExecutor.

            Best practices
            --------------
            - Use **stateful** when doing multi-step work (cd then ls; load data then analyze).
            - Use **stateless** for one-off checks or when you need isolation.
            - Use **read_only** to "peek" without mutating state (what-if exploration).
            - Use `list_sessions()` and `inspect_state()` to decide which session to use.

            Output
            ------
            Returns either a dict or an ExecutionResult object with the following fields:

            - **stdout**: For in-process Python, a List[TextPart | ImagePart] preserving
              rich output (text and images from print()/display()). For shell or venv
              execution, a plain string.
            - **stderr**: Same format as stdout (list for in-process Python, string otherwise).
            - **result**: The evaluated result of the last expression (Any), or None.
            - **error**: Error message string if execution failed, otherwise None.
            - **language**: The language used for execution.
            - **state_mode**: The state mode used ("stateless", "stateful", or "read_only").
            - **session_id**: The session ID (int) if stateful/read_only, otherwise None.
            - **session_name**: The session name alias if one was assigned, otherwise None.
            - **venv_id**: The virtual environment ID if applicable, otherwise None.
            - **session_created**: True if a new session was created by this call.
            - **duration_ms**: Execution duration in milliseconds.
            - **browser_used**: True if browser tools were invoked during execution.
            - **browser_state** (optional): Only present when browser_used is True and a
              browser environment is available. Contains {"url": str, "screenshot": str}
              or {"error": str} on failure.

            For in-process Python execution with rich output, the result is wrapped in an
            ExecutionResult object (a Pydantic model implementing FormattedToolResult).
            """
            _ = thought  # Thought is logged by the LLM; not used programmatically.
            if code is None or code.strip() == "":
                return {
                    "stdout": "",
                    "stderr": "",
                    "result": None,
                    "error": None,
                    "language": language,
                    "state_mode": state_mode,
                    "session_id": session_id,
                    "session_name": session_name,
                    "venv_id": venv_id,
                    "session_created": False,
                    "duration_ms": 0,
                    "browser_used": False,
                }

            # ──────────────────────────────────────────────────────────────
            # Boundary wrapper: execute_code (lineage + events + terminal log)
            # ──────────────────────────────────────────────────────────────

            _suffix = _token_hex(2)
            _call_id = new_call_id()
            _parent = TOOL_LOOP_LINEAGE.get([])
            _parent_lineage = list(_parent) if isinstance(_parent, list) else []
            _hierarchy = [*_parent_lineage, "execute_code"]
            _hierarchy_label = build_hierarchy_label(_hierarchy, _suffix)
            # Establish a boundary lineage frame so nested calls (e.g., FunctionManager-injected
            # functions calling state managers) keep a consistent parent->child chain.
            _lineage_token = TOOL_LOOP_LINEAGE.set(_hierarchy)

            async def _pub_safe(**payload: Any) -> None:
                try:
                    await publish_manager_method_event(
                        _call_id,
                        "CodeActActor",
                        "execute_code",
                        hierarchy=_hierarchy,
                        hierarchy_label=_hierarchy_label,
                        **payload,
                    )
                except Exception as e:
                    log_boundary_event(
                        _hierarchy_label,
                        f"Warning: failed to publish event: {type(e).__name__}: {e}",
                        icon="⚠️",
                        level="warning",
                    )

            try:
                await _pub_safe(phase="incoming")
            except Exception:
                pass
            log_boundary_event(_hierarchy_label, "Executing code...", icon="🛠️")

            out: dict[str, Any] | None = None
            tb_str: str | None = None
            exec_exc: Exception | None = None

            notification_q = _notification_up_q
            sandbox_id = None
            try:
                # Resolve / allocate sessions for stateful.
                if state_mode == "stateful":
                    if session_name:
                        resolved = self._resolve_session_name(session_name)
                        if resolved is not None:
                            language, venv_id, session_id = resolved
                        elif session_id is None:
                            # Allocate a new session id (reserve 0 for default sandbox).
                            key = (
                                str(language),
                                int(venv_id) if venv_id is not None else None,
                            )
                            next_id = self._next_session_id.get(key, 1)
                            session_id = next_id
                            self._next_session_id[key] = next_id + 1
                            self._register_session_name(
                                name=session_name,
                                language=str(language),
                                venv_id=venv_id,
                                session_id=int(session_id),
                            )
                    elif session_id is None:
                        # Default stateful session for each language/venv is session_id=0.
                        # This is especially important for Python because FunctionManager-injected
                        # callables are available in the default/bound Python sandbox (session 0).
                        session_id = 0

                # If name + id are both set but not registered yet, register alias.
                if state_mode == "stateful" and session_name and session_id is not None:
                    if self._resolve_session_name(session_name) is None:
                        self._register_session_name(
                            name=session_name,
                            language=str(language),
                            venv_id=venv_id,
                            session_id=int(session_id),
                        )

                # Validate.
                err = self._validate_execution_params(
                    state_mode=state_mode,
                    session_id=session_id,
                    session_name=session_name,
                    language=str(language),
                    venv_id=venv_id,
                )
                if err is not None:
                    out = err
                    return out

                # Inject per-tool notification queue into bound sandbox so notify() works.
                try:
                    sb_for_notifs = _CURRENT_SANDBOX.get()
                    sandbox_id = getattr(sb_for_notifs, "id", None)
                    if notification_q is not None:
                        sb_for_notifs.global_state["__notification_up_q__"] = (
                            notification_q
                        )
                except Exception:
                    pass

                if notification_q is not None and str(language) == "python":
                    try:
                        await notification_q.put(
                            {
                                "type": "execution_started",
                                "sandbox_id": sandbox_id,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            },
                        )
                    except Exception:
                        pass

                # Execute via SessionExecutor. Route primitives if available in current sandbox.
                primitives = None
                computer_primitives = self._computer_primitives
                try:
                    sb = _CURRENT_SANDBOX.get()
                    primitives = sb.global_state.get("primitives")
                    computer_primitives = sb.global_state.get(
                        "computer_primitives",
                        computer_primitives,
                    )
                except Exception:
                    pass

                try:
                    out = await self._session_executor.execute(
                        code=code,
                        language=str(language),  # type: ignore[arg-type]
                        state_mode=state_mode,  # type: ignore[arg-type]
                        session_id=session_id,
                        venv_id=venv_id,
                        primitives=primitives,
                        computer_primitives=computer_primitives,
                    )
                except Exception as e:
                    exec_exc = e
                    tb = traceback.format_exc()
                    tb_str = tb
                    if notification_q is not None and str(language) == "python":
                        try:
                            await notification_q.put(
                                {
                                    "type": "execution_error",
                                    "sandbox_id": sandbox_id,
                                    "error_kind": "exception",
                                    "traceback_preview": tb[:2000],
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                },
                            )
                        except Exception:
                            pass
                    out = {
                        "stdout": "",
                        "stderr": "",
                        "result": None,
                        "error": tb,
                        "language": language,
                        "state_mode": state_mode,
                        "session_id": session_id,
                        "session_name": session_name,
                        "venv_id": venv_id,
                        "session_created": False,
                        "duration_ms": 0,
                        "browser_used": False,
                    }

                # Enrich with session name.
                if out.get("session_id") is not None:
                    out["session_name"] = self._get_session_name(
                        language=str(out.get("language")),
                        venv_id=out.get("venv_id"),
                        session_id=int(out["session_id"]),
                    )
                else:
                    out["session_name"] = None

                # Attach browser state for Python runs when browser tools were used.
                if (
                    str(out.get("language")) == "python"
                    and out.get("browser_used")
                    and self._computer_primitives is not None
                ):
                    try:
                        url = await self._computer_primitives.computer.get_current_url()
                        screenshot_b64 = (
                            await self._computer_primitives.computer.get_screenshot()
                        )
                        out["browser_state"] = {
                            "url": url,
                            "screenshot": screenshot_b64,
                        }
                    except Exception as e:
                        out["browser_state"] = {"error": str(e)}

                if notification_q is not None and str(language) == "python":
                    try:
                        await notification_q.put(
                            {
                                "type": "execution_finished",
                                "sandbox_id": sandbox_id,
                                "status": ("ok" if not out.get("error") else "error"),
                                "stdout_len": len(out.get("stdout") or ""),
                                "stderr_len": len(out.get("stderr") or ""),
                                "browser_used": bool(out.get("browser_used")),
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            },
                        )
                    except Exception:
                        pass

                # Wrap in-process Python results in ExecutionResult for proper LLM
                # image formatting. In-process Python has stdout as List[OutputPart];
                # venv/shell have strings.
                if out.get("language") == "python" and isinstance(
                    out.get("stdout"),
                    list,
                ):
                    out = ExecutionResult(**out)

                return out
            finally:
                try:
                    if out is not None and out.get("error"):
                        await _pub_safe(
                            phase="outgoing",
                            status="error",
                            error=str(out.get("error")),
                            error_type=(
                                type(exec_exc).__name__
                                if exec_exc is not None
                                else "Error"
                            ),
                            traceback=(tb_str or "")[:2000],
                        )
                    else:
                        await _pub_safe(phase="outgoing", status="ok")
                except Exception:
                    pass
                try:
                    TOOL_LOOP_LINEAGE.reset(_lineage_token)
                except Exception:
                    pass

        tools: Dict[str, Callable[..., Awaitable[Any]]] = {
            "execute_code": execute_code,
        }

        # Add FunctionManager tools (auto-inject callables into sandbox) if available.
        #
        # IMPORTANT:
        # These tools are called via JSON tool calls (not inside Python). They return
        # metadata to the LLM while injecting the matching function callables into the
        # sandbox global namespace so they can be executed immediately in Python code.
        if self.function_manager:

            async def FunctionManager_search_functions(
                query: str,
                n: int = 5,
            ) -> Any:
                """
                Search for functions by semantic similarity to a natural-language query.

                Functions are automatically injected into your Python sandbox namespace,
                so you can execute them immediately after searching.
                """
                result = self.function_manager.search_functions(
                    query=query,
                    n=n,
                    return_callable=True,
                    namespace=_CURRENT_SANDBOX.get().global_state,
                    also_return_metadata=True,
                )
                return result["metadata"]

            async def FunctionManager_filter_functions(
                filter: Optional[str] = None,
                offset: int = 0,
                limit: int = 100,
            ) -> Any:
                """
                Filter functions using a Python-like filter expression.

                Functions are automatically injected into your Python sandbox namespace,
                so you can execute them immediately after filtering.
                """
                result = self.function_manager.filter_functions(
                    filter=filter,
                    offset=offset,
                    limit=limit,
                    return_callable=True,
                    namespace=_CURRENT_SANDBOX.get().global_state,
                    also_return_metadata=True,
                )
                return result["metadata"]

            async def FunctionManager_list_functions(
                include_implementations: bool = False,
            ) -> Any:
                """
                List available functions.

                Functions are automatically injected into your Python sandbox namespace,
                so you can execute them immediately after listing.
                """
                result = self.function_manager.list_functions(
                    include_implementations=include_implementations,
                    return_callable=True,
                    namespace=_CURRENT_SANDBOX.get().global_state,
                    also_return_metadata=True,
                )
                return result["metadata"]

            tools["FunctionManager_search_functions"] = FunctionManager_search_functions
            tools["FunctionManager_filter_functions"] = FunctionManager_filter_functions
            tools["FunctionManager_list_functions"] = FunctionManager_list_functions

            async def FunctionManager_add_functions(
                implementations: str | list[str],
                *,
                language: str = "python",
                overwrite: bool = False,
                verify: Optional[dict[str, bool]] = None,
                preconditions: Optional[dict[str, dict]] = None,
            ) -> Any:
                """
                Add/store new functions into the FunctionManager.

                Notes
                -----
                - This tool is gated by CodeActActor's `can_store` flag (and can be disabled per-call).
                - Prefer using existing functions (search first) before adding new ones.
                """
                fm = self.function_manager
                if fm is None:
                    raise RuntimeError(
                        "FunctionManager is not configured on this actor.",
                    )
                return fm.add_functions(
                    implementations=implementations,
                    language=language,  # type: ignore[arg-type]
                    overwrite=bool(overwrite),
                    verify=(verify or {}),
                    preconditions=(preconditions or {}),
                )

            tools["FunctionManager_add_functions"] = FunctionManager_add_functions

        # ───────────────────────── Session management tools ────────────────── #

        async def list_sessions(detail: str = "summary") -> Dict[str, Any]:
            """
            List all active sessions across all languages (Python + shell).

            Use this tool whenever you need to choose which session to use for a
            subsequent `execute_code(..., state_mode="stateful"/"read_only")` call.

            Parameters
            ----------
            detail:
                Controls how much information is returned per session:
                - "summary": metadata + a short `state_summary` string (default)
                - "full": best-effort enrichment using cheap inspection where available

            Returns
            -------
            dict:
                {"sessions": [ ... ]} where each entry includes (best-effort):
                - language: "python" | "bash" | "zsh" | "sh" | "powershell"
                - session_id: int (scoped per language + venv_id)
                - venv_id: int | None (Python only)
                - session_name: optional human-friendly alias (if registered)
                - created_at / last_used: timestamps when available
                - state_summary: a short human-readable summary (e.g. "3 names", "cwd=/repo")

            Notes
            -----
            - Session IDs are **scoped per (language, venv_id)**, so `python` session 0 and
              `bash` session 0 can coexist.
            - The default per-call Python sandbox is exposed as `python` session_id=0 (venv_id=None)
              when it is bound for the current call.
            """
            detail = (detail or "summary").strip()

            sessions: list[dict[str, Any]] = []

            # Default sandbox (current act sandbox) as python session 0 (venv_id=None).
            try:
                sb = _CURRENT_SANDBOX.get()
                sessions.append(
                    {
                        "language": "python",
                        "session_id": 0,
                        "venv_id": None,
                        "session_name": self._get_session_name(
                            language="python",
                            venv_id=None,
                            session_id=0,
                        ),
                        "created_at": None,
                        "last_used": None,
                        "state_summary": f"{len(sb.global_state)} globals",
                    },
                )
            except Exception:
                pass

            # In-process python sessions created via SessionExecutor.
            for s in self._session_executor.list_in_process_python_sessions():
                s = dict(s)
                s["session_name"] = self._get_session_name(
                    language="python",
                    venv_id=s.get("venv_id"),
                    session_id=int(s["session_id"]),
                )
                sessions.append(s)

            # Venv sessions.
            try:
                for s in self._venv_pool.get_all_sessions():
                    s = dict(s)
                    s["session_name"] = self._get_session_name(
                        language="python",
                        venv_id=s.get("venv_id"),
                        session_id=int(s["session_id"]),
                    )
                    sessions.append(s)
            except Exception:
                pass

            # Shell sessions.
            try:
                for s in self._shell_pool.get_all_sessions():
                    s = dict(s)
                    s["session_name"] = self._get_session_name(
                        language=str(s.get("language")),
                        venv_id=None,
                        session_id=int(s["session_id"]),
                    )
                    sessions.append(s)
            except Exception:
                pass

            if detail == "full":
                # Best-effort enrich state_summary with inspection where cheap.
                for s in sessions:
                    try:
                        if (
                            s.get("language") == "python"
                            and s.get("venv_id") is not None
                        ):
                            st = await self._venv_pool.get_session_state(
                                venv_id=int(s["venv_id"]),
                                session_id=int(s["session_id"]),
                                function_manager=self.function_manager,
                                detail="summary",
                            )
                            if isinstance(st, dict) and "count" in st:
                                s["state_summary"] = f'{st["count"]} names'
                        elif s.get("language") in ("bash", "zsh", "sh", "powershell"):
                            st = await self._shell_pool.get_session_state(
                                language=s["language"],
                                session_id=int(s["session_id"]),
                                detail="summary",
                            )
                            if isinstance(st, dict) and "summary" in st:
                                s["state_summary"] = st["summary"]
                    except Exception:
                        continue

            return {"sessions": sessions}

        async def inspect_state(
            session_name: str | None = None,
            session_id: int | None = None,
            language: str | None = None,
            venv_id: int | None = None,
            detail: str = "summary",
        ) -> Dict[str, Any]:
            """
            Inspect the state of a specific session (Python or shell).

            This tool is for debugging and for deciding whether to:
            - continue in the same session (stateful)
            - start a fresh session (stateful without session_id/session_name)
            - run a one-off (stateless)
            - do a what-if (read_only)

            Parameters
            ----------
            session_name:
                Optional human-friendly alias for a session (preferred when available).
            session_id + language (+ optional venv_id):
                Directly identify a session. `session_id` is scoped per (language, venv_id).
            detail:
                "summary" | "names" | "full"
                - Prefer "summary" when you just need quick context.
                - Use "names" to see what variables exist without dumping values.
                - Use "full" sparingly (can be large; values are truncated/redacted best-effort).

            Defaults
            --------
            If no session selector is provided, this inspects the **current per-call Python sandbox**
            (python session_id=0, venv_id=None) when bound.

            Returns
            -------
            dict with:
            - session: {language, session_id, session_name, venv_id}
            - state: implementation-specific state representation (Python vars; shell cwd/env/functions/aliases)
            """
            detail = (detail or "summary").strip()

            # Resolve session.
            resolved: SessionKey | None = None
            if session_name:
                resolved = self._resolve_session_name(session_name)
                if resolved is None:
                    return {
                        "error": f"Session {session_name!r} not found",
                        "error_type": "validation",
                    }
            elif session_id is not None and language is not None:
                resolved = (str(language), venv_id, int(session_id))

            # Default: current sandbox.
            if resolved is None:
                try:
                    sb = _CURRENT_SANDBOX.get()
                except Exception as e:
                    return {
                        "error": f"No sandbox bound: {type(e).__name__}",
                        "error_type": "internal",
                    }

                names: list[str] = []
                full_map: dict[str, str] = {}
                for k, v in sb.global_state.items():
                    if not isinstance(k, str) or k.startswith("_"):
                        continue
                    if callable(v) or isinstance(v, type):
                        continue
                    names.append(k)
                    if detail == "full":
                        try:
                            s = repr(v)
                            if len(s) > 500:
                                s = s[:500] + "..."
                        except Exception:
                            s = f"<{type(v).__name__}>"
                        full_map[k] = s

                names = sorted(names)
                state_obj: dict[str, Any]
                if detail == "full":
                    state_obj = {"variables": full_map, "functions": []}
                else:
                    state_obj = {"variables": names, "functions": []}

                return {
                    "session": {
                        "language": "python",
                        "session_id": 0,
                        "session_name": self._get_session_name(
                            language="python",
                            venv_id=None,
                            session_id=0,
                        ),
                        "venv_id": None,
                    },
                    "state": state_obj,
                }

            lang, resolved_venv_id, sid = resolved

            # Python venv-backed
            if lang == "python" and resolved_venv_id is not None:
                st = await self._venv_pool.get_session_state(
                    venv_id=int(resolved_venv_id),
                    session_id=int(sid),
                    function_manager=self.function_manager,
                    detail=detail,
                )
                return {
                    "session": {
                        "language": "python",
                        "session_id": int(sid),
                        "session_name": self._get_session_name(
                            language="python",
                            venv_id=int(resolved_venv_id),
                            session_id=int(sid),
                        ),
                        "venv_id": int(resolved_venv_id),
                    },
                    "state": st,
                }

            # Python in-process session (SessionExecutor)
            if lang == "python" and resolved_venv_id is None:
                key = (None, int(sid))
                sb = self._session_executor._python_sessions.get(
                    key,
                )  # pylint: disable=protected-access
                if sb is None:
                    return {
                        "error": f"Python session {sid} not found",
                        "error_type": "validation",
                    }
                names: list[str] = []
                full_map: dict[str, str] = {}
                for k, v in sb.global_state.items():
                    if not isinstance(k, str) or k.startswith("_"):
                        continue
                    if callable(v) or isinstance(v, type):
                        continue
                    names.append(k)
                    if detail == "full":
                        try:
                            s = repr(v)
                            if len(s) > 500:
                                s = s[:500] + "..."
                        except Exception:
                            s = f"<{type(v).__name__}>"
                        full_map[k] = s
                names = sorted(names)
                state_obj = {
                    "variables": full_map if detail == "full" else names,
                    "functions": [],
                }
                return {
                    "session": {
                        "language": "python",
                        "session_id": int(sid),
                        "session_name": self._get_session_name(
                            language="python",
                            venv_id=None,
                            session_id=int(sid),
                        ),
                        "venv_id": None,
                    },
                    "state": state_obj,
                }

            # Shell
            st = await self._shell_pool.get_session_state(
                language=lang,  # type: ignore[arg-type]
                session_id=int(sid),
                detail=detail,
            )
            return {
                "session": {
                    "language": str(lang),
                    "session_id": int(sid),
                    "session_name": self._get_session_name(
                        language=str(lang),
                        venv_id=None,
                        session_id=int(sid),
                    ),
                    "venv_id": None,
                },
                "state": st,
            }

        async def close_session(
            session_name: str | None = None,
            session_id: int | None = None,
            language: str | None = None,
            venv_id: int | None = None,
        ) -> Dict[str, Any]:
            """
            Close a specific session and free resources.

            Use this to proactively manage resources when you are done with a session.
            This operation is **idempotent**: closing an already-closed/non-existent session
            returns `closed=False, reason="not_found"` rather than raising.

            Parameters
            ----------
            session_name:
                Preferred: close by human-friendly alias.
            session_id + language (+ optional venv_id):
                Close by canonical identity.

            Returns
            -------
            dict:
                - closed: bool
                - reason: "success" | "not_found" | "error"
                - session: {language, session_id, session_name}
            """
            resolved: SessionKey | None = None
            if session_name:
                resolved = self._resolve_session_name(session_name)
                if resolved is None:
                    return {
                        "closed": False,
                        "reason": "not_found",
                        "session": {
                            "language": language,
                            "session_id": session_id,
                            "session_name": session_name,
                        },
                    }
            elif session_id is not None and language is not None:
                resolved = (str(language), venv_id, int(session_id))
            else:
                return {
                    "closed": False,
                    "reason": "error",
                    "error": "Must provide session_name or (language + session_id).",
                }

            lang, resolved_venv_id, sid = resolved
            closed = False

            if lang == "python" and resolved_venv_id is not None:
                closed = await self._venv_pool.close_session(
                    venv_id=int(resolved_venv_id),
                    session_id=int(sid),
                )
            elif lang == "python" and resolved_venv_id is None:
                closed = await self._session_executor.close_in_process_python_session(
                    session_id=int(sid),
                    venv_id=None,
                )
            else:
                closed = await self._shell_pool.close_session(language=lang, session_id=int(sid))  # type: ignore[arg-type]

            # Unregister all aliases for this session.
            self._unregister_all_names_for_session(
                key=(str(lang), resolved_venv_id, int(sid)),
            )

            return {
                "closed": bool(closed),
                "reason": "success" if closed else "not_found",
                "session": {
                    "language": str(lang),
                    "session_id": int(sid),
                    "session_name": session_name
                    or self._get_session_name(
                        language=str(lang),
                        venv_id=resolved_venv_id,
                        session_id=int(sid),
                    ),
                },
            }

        async def close_all_sessions() -> Dict[str, Any]:
            """
            Close all active sessions across all languages.

            This is a blunt cleanup tool. Prefer `close_session(...)` when you only
            want to discard a specific polluted/unused session.

            Returns
            -------
            dict:
                - closed_count: int
                - languages: list[str] (languages that had sessions closed)
                - details: per-language counts
            """
            closed_counts: dict[str, int] = {
                "python": 0,
                "bash": 0,
                "zsh": 0,
                "sh": 0,
                "powershell": 0,
            }

            # Close in-process python sessions.
            for s in list(self._session_executor.list_in_process_python_sessions()):
                sid = int(s.get("session_id", 0))
                if await self._session_executor.close_in_process_python_session(
                    session_id=sid,
                    venv_id=None,
                ):
                    closed_counts["python"] += 1
                    self._unregister_all_names_for_session(key=("python", None, sid))

            # Close venv python sessions.
            for vid, sid in list(self._venv_pool.list_active_sessions()):
                if await self._venv_pool.close_session(
                    venv_id=int(vid),
                    session_id=int(sid),
                ):
                    closed_counts["python"] += 1
                    self._unregister_all_names_for_session(
                        key=("python", int(vid), int(sid)),
                    )

            # Close shell sessions.
            for lang, sid in list(self._shell_pool.get_active_sessions()):
                if await self._shell_pool.close_session(
                    language=lang,
                    session_id=int(sid),
                ):
                    closed_counts[str(lang)] = closed_counts.get(str(lang), 0) + 1
                    self._unregister_all_names_for_session(
                        key=(str(lang), None, int(sid)),
                    )

            # Clear any remaining aliases.
            self._session_names.clear()
            self._session_names_rev.clear()

            closed_total = sum(closed_counts.values())
            langs = [k for k, v in closed_counts.items() if v > 0]
            return {
                "closed_count": closed_total,
                "languages": langs,
                "details": closed_counts,
            }

        tools["list_sessions"] = list_sessions
        tools["inspect_state"] = inspect_state
        tools["close_session"] = close_session
        tools["close_all_sessions"] = close_all_sessions

        return tools

    @functools.wraps(BaseCodeActActor.act, updated=())
    @log_manager_call("CodeActActor", "act", payload_key="description")
    async def act(
        self,
        description: str | dict | list[str | dict],
        *,
        clarification_enabled: bool = True,
        response_format: Optional[Type[BaseModel]] = None,
        _parent_chat_context: list[dict] | None = None,
        _clarification_up_q: Optional[asyncio.Queue[str]] = None,
        _clarification_down_q: Optional[asyncio.Queue[str]] = None,
        _call_id: Optional[str] = None,
        images: Optional[ImageRefs | list[RawImageRef | AnnotatedImageRef]] = None,
        entrypoint: Optional[int] = None,
        entrypoint_args: Optional[list[Any]] = None,
        entrypoint_kwargs: Optional[dict[str, Any]] = None,
        persist: Optional[bool] = None,
        can_compose: Optional[bool] = None,
        can_store: Optional[bool] = None,
        **kwargs,
    ) -> SteerableToolHandle:
        if not self._main_event_loop:
            self._main_event_loop = asyncio.get_running_loop()

        effective_can_compose = (
            self.can_compose if can_compose is None else bool(can_compose)
        )
        effective_can_store = self.can_store if can_store is None else bool(can_store)

        # can_compose=False mode: do not run an LLM tool loop or allow arbitrary code execution.
        # Instead, semantic-search for a stored function and execute it directly.
        if entrypoint is None and not effective_can_compose:
            # Validate description is a string for can_compose=False mode
            # (semantic search and SingleFunctionActorHandle require string input)
            if not isinstance(description, str):
                raise TypeError(
                    "can_compose=False requires description to be a string, "
                    f"got {type(description).__name__}",
                )

            from unity.actor.single_function_actor import SingleFunctionActorHandle

            fm = self.function_manager
            if fm is None:
                raise RuntimeError(
                    "CodeActActor cannot run with can_compose=False: function_manager is None",
                )

            matches = fm.search_functions(
                query=str(description or ""),
                n=1,
                include_implementations=True,
            )
            if not matches:

                async def _fail() -> Any:
                    raise RuntimeError(
                        "can_compose=False: no matching functions found via semantic search.",
                    )

                return SingleFunctionActorHandle(
                    function_name="(no_match)",
                    function_id=None,
                    execution_task=asyncio.create_task(_fail()),
                    is_primitive=False,
                    verify=False,
                    goal=description,
                )

            fn_name = matches[0].get("name")
            if not isinstance(fn_name, str) or not fn_name.strip():
                raise RuntimeError(
                    "can_compose=False: semantic search returned a function without a valid name.",
                )

            primitives = None
            try:
                env = self.environments.get("primitives")
                if env is not None:
                    primitives = env.get_instance()
            except Exception:
                primitives = None

            async def _run_found() -> Any:
                out = await fm.execute_function(
                    function_name=fn_name,
                    primitives=primitives,
                    computer_primitives=self._computer_primitives,
                    venv_pool=self._venv_pool,
                    shell_pool=self._shell_pool,
                    state_mode="stateless",
                )
                if isinstance(out, dict) and out.get("error"):
                    raise RuntimeError(str(out.get("error")))
                if isinstance(out, dict):
                    return out.get("result")
                return out

            return SingleFunctionActorHandle(
                function_name=fn_name,
                function_id=(
                    matches[0].get("function_id")
                    if isinstance(matches[0], dict)
                    else None
                ),
                execution_task=asyncio.create_task(_run_found()),
                is_primitive=False,
                verify=False,
                goal=description,
                docstring=(
                    matches[0].get("docstring")
                    if isinstance(matches[0], dict)
                    else None
                ),
            )

        initial_prompt = (
            "This is an interactive session. Acknowledge that you are ready and "
            "wait for the user to provide instructions via interjection."
        )

        # Clarification queues:
        # - When enabled, we ensure the handle has queues (either provided by caller or newly created).
        # - When disabled, we do not provide queues and we do not wire queue injection into environments.
        clarification_up_q: Optional[asyncio.Queue[str]]
        clarification_down_q: Optional[asyncio.Queue[str]]
        if clarification_enabled:
            clarification_up_q = _clarification_up_q or asyncio.Queue()
            clarification_down_q = _clarification_down_q or asyncio.Queue()
        else:
            clarification_up_q = None
            clarification_down_q = None

        # Create per-call environments so clarification queues are not stored on shared actor environments.
        sandbox_envs: Dict[str, "BaseEnvironment"] = {}
        try:
            from unity.actor.environments import (
                ComputerEnvironment as _ComputerEnvironment,
                StateManagerEnvironment as _StateManagerEnvironment,
            )
        except Exception:
            _ComputerEnvironment = None  # type: ignore
            _StateManagerEnvironment = None  # type: ignore

        for ns, env in self.environments.items():
            # Prefer explicit reconstruction for known env types.
            try:
                if _ComputerEnvironment is not None and isinstance(
                    env,
                    _ComputerEnvironment,
                ):
                    sandbox_envs[ns] = _ComputerEnvironment(
                        env.get_instance(),
                        clarification_up_q=clarification_up_q,
                        clarification_down_q=clarification_down_q,
                    )
                    continue
                if _StateManagerEnvironment is not None and isinstance(
                    env,
                    _StateManagerEnvironment,
                ):
                    sandbox_envs[ns] = _StateManagerEnvironment(
                        env.get_instance(),
                        exposed_managers=getattr(env, "_exposed_managers", None),
                        clarification_up_q=clarification_up_q,
                        clarification_down_q=clarification_down_q,
                    )
                    continue
            except Exception:
                pass

            # Fallback: shallow-copy and set private queue attrs on the copy only.
            try:
                env_copy = copy.copy(env)
                if hasattr(env_copy, "_clarification_up_q"):
                    setattr(env_copy, "_clarification_up_q", clarification_up_q)
                if hasattr(env_copy, "_clarification_down_q"):
                    setattr(env_copy, "_clarification_down_q", clarification_down_q)
                sandbox_envs[ns] = env_copy
            except Exception:
                sandbox_envs[ns] = env

        # Concurrency/backpressure guard. If we can't acquire within 30s, treat as resource exhaustion.
        try:
            await asyncio.wait_for(
                self._act_semaphore.acquire(),
                timeout=float(getattr(self, "_act_semaphore_timeout_s", 30.0)),
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                "CodeActActor is at capacity (too many concurrent sessions). "
                "Try again later or reduce concurrency.",
            )
        sandbox = PythonExecutionSession(
            computer_primitives=self._computer_primitives,
            environments=sandbox_envs,
            venv_pool=self._venv_pool,
            shell_pool=self._shell_pool,
        )
        # Note: __notification_up_q__ is injected dynamically by execute_code
        # when it receives a _notification_up_q from the async tool loop.
        token = _CURRENT_SANDBOX.set(sandbox)

        async def _cleanup() -> None:
            try:
                # Best-effort cleanup
                if hasattr(sandbox, "close") and callable(getattr(sandbox, "close")):
                    await sandbox.close()  # type: ignore[misc]
            except Exception:
                pass
            try:
                _CURRENT_SANDBOX.reset(token)
            except Exception:
                pass
            try:
                self._act_semaphore.release()
            except Exception:
                pass

        # If an explicit FunctionManager entrypoint is provided (e.g., TaskScheduler task execution),
        # bypass the CodeAct LLM loop and run the function directly.
        if entrypoint is not None:
            entrypoint_id = int(entrypoint)
            args = list(entrypoint_args or [])
            kwargs_for_entrypoint = dict(entrypoint_kwargs or {})

            async def _run_entrypoint() -> Any:
                fm = self.function_manager
                if fm is None:
                    raise RuntimeError(
                        "CodeActActor cannot execute entrypoint: function_manager is None",
                    )

                out = fm.filter_functions(
                    filter=f"function_id == {entrypoint_id}",
                    return_callable=True,
                    namespace=sandbox.global_state,
                    also_return_metadata=True,
                )
                metadata = []
                if isinstance(out, dict):
                    metadata = list(out.get("metadata") or [])
                if not metadata:
                    raise ValueError(
                        f"Entrypoint function_id {entrypoint_id} not found in FunctionManager.",
                    )
                fn_name = metadata[0].get("name")
                if not isinstance(fn_name, str) or not fn_name.strip():
                    raise ValueError(
                        f"Entrypoint {entrypoint_id} has no valid function name.",
                    )
                fn = sandbox.global_state.get(fn_name)
                if fn is None:
                    raise ValueError(
                        f"Entrypoint {entrypoint_id} ({fn_name}) was not injected into the sandbox namespace.",
                    )

                res = fn(*args, **kwargs_for_entrypoint)
                if inspect.isawaitable(res):
                    res = await res
                return res

            entry_task = asyncio.create_task(_run_entrypoint())
            entry_handle = _CodeActEntrypointHandle(
                entrypoint_id=entrypoint_id,
                execution_task=entry_task,
                on_finally=_cleanup,
            )
            return entry_handle

        system_prompt = build_code_act_prompt(
            environments=sandbox_envs,
            tools=dict(self.get_tools("act")),
        )

        # Tool policy controls which tools are visible per turn, and whether a tool call
        # is required. We use this for two concerns:
        # 1) If can_store is disabled, hide persistence tools.
        # 2) If FunctionManager tools are present, enforce "function-first" by preventing
        #    the very first turn from being `execute_code` (push the model to search/list
        #    and inject memoized functions first).
        _all_tools_for_policy = dict(self.get_tools("act"))
        _has_fm_tools = any(
            isinstance(k, str) and k.startswith("FunctionManager_")
            for k in _all_tools_for_policy.keys()
        )

        def _tool_policy(step: int, tools: Dict[str, Any]):
            filtered = dict(tools)

            # 1) Hide persistence tool if disabled
            if not effective_can_store:
                filtered.pop("FunctionManager_add_functions", None)

            # 2) Function-first: on the first model turn, require a FunctionManager call
            # (search/filter/list) when those tools exist. This avoids the model skipping
            # the function library and going straight to `execute_code`.
            if step == 0 and _has_fm_tools:
                fm_only = {
                    k: v
                    for k, v in filtered.items()
                    if isinstance(k, str)
                    and k.startswith("FunctionManager_")
                    and k != "FunctionManager_add_functions"
                }
                if fm_only:
                    return "required", fm_only

            return "auto", filtered

        tool_policy = _tool_policy

        handle = ActorHandle(
            task_description=description or initial_prompt,
            tools=dict(self.get_tools("act")),
            parent_chat_context=_parent_chat_context,
            clarification_up_q=clarification_up_q,
            clarification_down_q=clarification_down_q,
            call_id=_call_id,
            on_finally=_cleanup,
            main_event_loop=self._main_event_loop,
            timeout=self._timeout,
            persist=persist,
            custom_system_prompt=system_prompt,
            tool_policy=tool_policy,
            computer_primitives=self._computer_primitives,
            images=images,
            response_format=response_format,
        )
        return handle

    async def close(self):
        """Shuts down the actor and its associated resources gracefully."""
        # Close any in-process session sandboxes owned by the session executor.
        try:
            await self._session_executor.close()
        except Exception:
            pass

        # Clear session name registry.
        try:
            self._session_names.clear()
            self._session_names_rev.clear()
        except Exception:
            pass

        # Close the pools (terminates persistent subprocess/session connections)
        await self._venv_pool.close()
        await self._shell_pool.close()

        if self._computer_primitives:
            self._computer_primitives.computer.stop()
