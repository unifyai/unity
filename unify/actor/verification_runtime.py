"""Verifier passes for stored functions on the trust ramp.

Four independent LLM passes — static review (once per trust hash), argument
review, precondition probe and post-execution review — plus the read-only
probe they may run. Each pass builds a three-block prompt (constant prefix,
per-call-site stable block, per-call volatile block), asks a verifier client
tagged ``purpose="verification"`` for a ``Verdict``, and appends the verdict
to ``Functions/Verifications``. Passes never repair anything and never see
the conversation transcript or the CodeAct trajectory.

The intent chain lives on ``current_verification_frames``: wrappers push a
``Frame`` on entry and pop it on exit, so a pass for a leaf sees every frame
from the root of the run down to the call under review.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

from pydantic import ValidationError

from unify.actor.prompt_builders import (
    build_args_review_prompt,
    build_call_stable_block,
    build_post_probe_prompt,
    build_precondition_probe_prompt,
    build_static_review_prompt,
)
from unify.common.async_tool_loop import start_async_tool_loop
from unify.common.llm_client import (
    new_llm_client,
    pydantic_to_json_schema_response_format,
)
from unify.function_manager.types.verification import (
    StaticReviewRecord,
    Verdict,
    VerdictKind,
    VerificationRow,
)
from unify.function_manager.verification.ledger import args_signature, utcnow
from unify.function_manager.verification.source_labels import (
    executed_source_lines,
    function_name_from_filename,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Intent chain
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Frame:
    """One stored-function activation on the path from the root to a call."""

    function_id: int
    name: str
    docstring: str
    effect_class: str
    call_site_line: str
    args_repr: str

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


current_verification_frames: contextvars.ContextVar[tuple[Frame, ...]] = (
    contextvars.ContextVar("current_verification_frames", default=())
)


@contextmanager
def pushed_frame(frame: Frame) -> Iterator[tuple[Frame, ...]]:
    """Push ``frame`` for the duration of a call; yields the full chain."""
    chain = (*current_verification_frames.get(), frame)
    token = current_verification_frames.set(chain)
    try:
        yield chain
    finally:
        current_verification_frames.reset(token)


@dataclass(frozen=True)
class CallSite:
    """Where in a stored function the current call was made from."""

    parent_name: Optional[str]
    line_number: Optional[int]
    line_text: Optional[str]

    @property
    def label(self) -> str:
        return self.parent_name or "root"


def locate_call_site() -> CallSite:
    """Find the nearest enclosing stored-function frame on the Python stack.

    Stored functions are compiled under ``<function:NAME>``; the innermost
    such frame above the caller is the function whose body made this call.
    """
    frame = sys._getframe(1)
    while frame is not None:
        name = function_name_from_filename(frame.f_code.co_filename)
        if name is not None:
            lineno = frame.f_lineno
            lines = executed_source_lines(name)
            text = None
            if lines and 0 < lineno <= len(lines):
                text = lines[lineno - 1].rstrip("\n")
            return CallSite(parent_name=name, line_number=lineno, line_text=text)
        frame = frame.f_back
    return CallSite(parent_name=None, line_number=None, line_text=None)


def executed_source(name: str) -> Optional[str]:
    lines = executed_source_lines(name)
    return "".join(lines) if lines else None


# ---------------------------------------------------------------------------
# Read-only probe
# ---------------------------------------------------------------------------

PROBE_TIMEOUT_S = 60.0
PROBE_OUTPUT_LIMIT = 20_000


async def run_probe(code: str) -> str:
    """Execute a short read-only Python diagnosis snippet in a fresh subprocess.

    Use this to observe the CURRENT behavior of the external interfaces a
    function reads — for example, fetch the endpoint it ingests and print the
    response's shape, keys, and a sample record — so a judgement is grounded
    in observed reality rather than in assumptions or in the function's own
    error messages. The snippet runs in a fresh isolated interpreter with the
    standard library only and must print its observations to stdout.

    Strictly read-only diagnosis: fetch and inspect inputs only. Never
    perform a function's side effects (no writes, deliveries, or state
    mutations on external systems). Output is truncated after 20,000
    characters; the subprocess is killed after 60 seconds.
    """
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        "-c",
        str(code),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        raw, _ = await asyncio.wait_for(proc.communicate(), timeout=PROBE_TIMEOUT_S)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return f"Probe timed out after {int(PROBE_TIMEOUT_S)} seconds and was killed."
    output = raw.decode("utf-8", errors="replace")
    if len(output) > PROBE_OUTPUT_LIMIT:
        output = output[:PROBE_OUTPUT_LIMIT] + "\n... (output truncated)"
    return (
        f"exit_code={proc.returncode}\n{output}"
        if output.strip()
        else f"exit_code={proc.returncode} (no output printed)"
    )


# ---------------------------------------------------------------------------
# Verifier passes
# ---------------------------------------------------------------------------


@dataclass
class PassUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0


def _usage_from_completion(completion: Any) -> PassUsage:
    usage = getattr(completion, "usage", None)
    if usage is None and isinstance(completion, dict):
        usage = completion.get("usage")
    if usage is None:
        return PassUsage()

    def _get(name: str) -> Any:
        if isinstance(usage, dict):
            return usage.get(name)
        return getattr(usage, name, None)

    cost = _get("cost")
    return PassUsage(
        prompt_tokens=int(_get("prompt_tokens") or 0),
        completion_tokens=int(_get("completion_tokens") or 0),
        cost=float(cost) if isinstance(cost, (int, float)) else 0.0,
    )


#: Reasons that mean "no verdict was obtained", as opposed to a judgement of
#: UNSURE. A model that could not be reached, answered unparseably, or did not
#: answer in time has said nothing about the function -- so these must not be
#: recorded as evidence, and must not be read as the verifier declining to
#: clear the call. They are faults in the channel, and the only honest
#: responses to them are retry and, failing that, say so plainly.
VERDICT_FAULT_REASONS = frozenset({"unparseable_verdict", "llm_error", "timeout"})


def verdict_is_fault(verdict: Optional[Verdict]) -> bool:
    """Whether *verdict* stands in for a verdict never obtained."""

    if verdict is None:
        return True
    return verdict.verdict == "UNSURE" and verdict.reason in VERDICT_FAULT_REASONS


def _parse_verdict(text: Any) -> Optional[Verdict]:
    if isinstance(text, Verdict):
        return text
    if isinstance(text, dict):
        try:
            return Verdict.model_validate(text)
        except ValidationError:
            return None
    if not isinstance(text, str):
        return None
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.startswith("json"):
            candidate = candidate[4:]
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            payload = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            return None
    try:
        return Verdict.model_validate(payload)
    except ValidationError:
        return None


class VerifierPasses:
    """The four verifier passes for one run, sharing goal, guidance and settings."""

    def __init__(
        self,
        *,
        function_manager: Any,
        guidance_manager: Any = None,
        goal: str = "",
        run_key: Optional[str] = None,
        task_id: Optional[int] = None,
        model: Optional[str] = None,
    ) -> None:
        self.fm = function_manager
        self.gm = guidance_manager
        self.goal = goal
        self.run_key = run_key
        self.task_id = task_id
        self.settings = function_manager.verification_settings
        self.model = model if model is not None else self.settings.model
        self._hash_cache: Dict[int, str] = {}
        self._guidance_cache: Dict[int, Optional[Dict[str, Any]]] = {}
        self._row_cache: Dict[str, Optional[Dict[str, Any]]] = {}

    # ---- shared helpers -------------------------------------------------

    def trust_hash(self, row: Mapping[str, Any]) -> str:
        function_id = int(row["function_id"])
        cached = self._hash_cache.get(function_id)
        if cached is None:
            cached = self.fm.function_trust_hash(dict(row))
            self._hash_cache[function_id] = cached
        return cached

    def forget_hash(self, function_id: int) -> None:
        self._hash_cache.pop(int(function_id), None)

    def row_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        if name not in self._row_cache:
            self._row_cache[name] = self.fm._get_function_data_by_name(name=name)
        return self._row_cache[name]

    def guidance_for(self, row: Mapping[str, Any]) -> List[Dict[str, Any]]:
        """Bodies of the guidance entries linked to ``row``, capped by settings."""
        if self.gm is None:
            return []
        entries: List[Dict[str, Any]] = []
        linked = list(row.get("guidance_ids") or [])
        if not linked and row.get("function_id") is not None:
            linked = self.fm._get_guidance_ids_for_function(
                function_id=int(row["function_id"]),
            )
        for guidance_id in linked:
            gid = int(guidance_id)
            if gid not in self._guidance_cache:
                try:
                    guidance = self.gm.get_guidance(guidance_id=gid)
                    self._guidance_cache[gid] = {
                        "title": getattr(guidance, "title", None),
                        "content": getattr(guidance, "content", ""),
                    }
                except Exception:
                    self._guidance_cache[gid] = None
            entry = self._guidance_cache[gid]
            if entry is not None:
                entries.append(entry)
        return entries

    def dependencies_of(self, row: Mapping[str, Any]) -> List[Dict[str, Any]]:
        deps: List[Dict[str, Any]] = []
        for name in row.get("depends_on") or []:
            if not isinstance(name, str) or "." in name:
                continue
            dep = self.row_by_name(name)
            if dep is not None:
                deps.append(dep)
        return deps

    def stable_block(
        self,
        row: Mapping[str, Any],
        *,
        frames: Sequence[Frame],
        call_site: CallSite,
        root_row: Optional[Mapping[str, Any]] = None,
    ) -> str:
        guidance = list(self.guidance_for(root_row)) if root_row is not None else []
        if root_row is None or root_row.get("function_id") != row.get("function_id"):
            guidance.extend(self.guidance_for(row))
        parent_source = (
            executed_source(call_site.parent_name) if call_site.parent_name else None
        )
        return build_call_stable_block(
            goal=self.goal,
            guidance=guidance,
            frames=[frame.as_dict() for frame in frames],
            leaf=dict(row),
            parent_source=parent_source,
            call_line=call_site.line_number,
            children=self.dependencies_of(row),
            max_guidance_chars=self.settings.max_guidance_chars,
        )

    def _client(self, origin: str):
        return new_llm_client(self.model, purpose="verification", origin=origin)

    async def _judge(
        self,
        *,
        prefix: str,
        stable: str,
        volatile: str,
        origin: str,
    ) -> tuple[Verdict, PassUsage]:
        """One structured verdict; JSON parse failure retries once, transport failure is UNSURE."""
        client = self._client(origin)
        client.set_system_message(prefix)
        message = stable if not volatile else f"{stable}\n\n{volatile}"
        usage = PassUsage()
        messages = [{"role": "user", "content": message}]
        for attempt in (1, 2):
            try:
                completion = await client.generate(
                    messages=messages,
                    return_full_completion=True,
                    response_format=pydantic_to_json_schema_response_format(Verdict),
                )
            except Exception as exc:
                logger.warning(
                    "Verifier pass %s failed to reach the model: %s",
                    origin,
                    exc,
                )
                return (
                    Verdict(verdict="UNSURE", reason="llm_error", fault=None),
                    usage,
                )
            step = _usage_from_completion(completion)
            usage.prompt_tokens += step.prompt_tokens
            usage.completion_tokens += step.completion_tokens
            usage.cost += step.cost
            content = None
            try:
                content = completion.choices[0].message.content
            except (AttributeError, IndexError):
                content = str(completion)
            verdict = _parse_verdict(content)
            if verdict is not None:
                return verdict, usage
            if attempt == 1:
                messages = messages + [
                    {"role": "assistant", "content": str(content)},
                    {
                        "role": "user",
                        "content": (
                            "That was not a valid verdict object. Reply with exactly one "
                            'JSON object of the form {"verdict": ..., "reason": ..., "fault": ...}.'
                        ),
                    },
                ]
        return (
            Verdict(verdict="UNSURE", reason="unparseable_verdict", fault=None),
            usage,
        )

    def _record(
        self,
        row: Mapping[str, Any],
        *,
        kind: VerdictKind,
        verdict: Verdict,
        call_site: str,
        kwargs: Optional[Mapping[str, Any]],
        usage: PassUsage,
        wall_ms: int,
    ) -> VerificationRow:
        if verdict_is_fault(verdict):
            # A model that could not be reached or could not be parsed has
            # said nothing about this function. Recording it would put a
            # verdict nobody made into the evidence `derive_verify` reads,
            # and every such row pushes the function further from trust for
            # a reason that is not about the function at all.
            logger.warning(
                "Verifier %s pass produced no verdict (%s); not recorded",
                kind,
                verdict.reason,
            )
            return VerificationRow(
                function_id=int(row["function_id"]),
                function_hash=self.trust_hash(row),
                kind=kind,
                verdict=verdict.verdict,
                reason=verdict.reason,
                fault=verdict.fault,
                call_site=call_site,
                run_key=self.run_key,
                task_id=self.task_id,
                created_at=utcnow(),
            )
        entry = VerificationRow(
            function_id=int(row["function_id"]),
            function_hash=self.trust_hash(row),
            kind=kind,
            verdict=verdict.verdict,
            reason=verdict.reason,
            fault=verdict.fault,
            call_site=call_site,
            args_signature=args_signature(kwargs) if kwargs is not None else None,
            run_key=self.run_key,
            task_id=self.task_id,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cost=usage.cost,
            wall_ms=wall_ms,
            created_at=utcnow(),
        )
        self.fm.record_verification_nowait(entry)
        return entry

    # ---- passes ---------------------------------------------------------

    async def static_review(self, row: Mapping[str, Any]) -> Verdict:
        """Judge the source alone; cached per trust hash on the row."""
        current = self.trust_hash(row)
        cached = row.get("static_review")
        if isinstance(cached, dict) and cached.get("function_hash") == current:
            return Verdict(
                verdict=cached.get("verdict", "UNSURE"),
                reason=str(cached.get("reason") or ""),
                fault=("leaf" if cached.get("verdict") == "FAIL" else None),
            )
        prefix, stable, _ = build_static_review_prompt(
            row,
            dependencies=self.dependencies_of(row),
        )
        started = time.perf_counter()
        verdict, usage = await self._judge(
            prefix=prefix,
            stable=stable,
            volatile="",
            origin=f"Verifier.static({row.get('name')})",
        )
        if verdict.verdict == "FAIL":
            verdict = Verdict(verdict="FAIL", reason=verdict.reason, fault="leaf")
        wall_ms = int((time.perf_counter() - started) * 1000)
        self._record(
            row,
            kind=VerdictKind.static,
            verdict=verdict,
            call_site="root",
            kwargs=None,
            usage=usage,
            wall_ms=wall_ms,
        )
        record = StaticReviewRecord(
            verdict=verdict.verdict,
            reason=verdict.reason,
            function_hash=current,
            reviewed_at=utcnow(),
            model=self.model,
        )
        if isinstance(row, dict):
            row["static_review"] = record.model_dump(mode="json")
        self.fm.persist_static_review_nowait(dict(row), record)
        return verdict

    async def args_review(
        self,
        row: Mapping[str, Any],
        *,
        kwargs: Mapping[str, Any],
        stable_block: str,
        call_site: str = "root",
        tier0: Optional[str] = None,
        sibling_results: Sequence[Mapping[str, Any]] = (),
    ) -> Verdict:
        prefix, stable, volatile = build_args_review_prompt(
            stable_block=stable_block,
            kwargs=kwargs,
            tier0=tier0,
            sibling_results=sibling_results,
        )
        started = time.perf_counter()
        verdict, usage = await self._judge(
            prefix=prefix,
            stable=stable,
            volatile=volatile,
            origin=f"Verifier.args({row.get('name')})",
        )
        self._record(
            row,
            kind=VerdictKind.args,
            verdict=verdict,
            call_site=call_site,
            kwargs=kwargs,
            usage=usage,
            wall_ms=int((time.perf_counter() - started) * 1000),
        )
        return verdict

    async def precondition_probe(
        self,
        row: Mapping[str, Any],
        *,
        kwargs: Mapping[str, Any],
        stable_block: str,
        call_site: str = "root",
        sibling_results: Sequence[Mapping[str, Any]] = (),
        max_steps: int = 6,
    ) -> Verdict:
        """Judge whether the world satisfies the precondition; may run read-only probes."""
        prefix, stable, volatile = build_precondition_probe_prompt(
            stable_block=stable_block,
            precondition=row.get("precondition"),
            kwargs=kwargs,
            sibling_results=sibling_results,
        )
        started = time.perf_counter()
        usage = PassUsage()
        # Retried on an unreadable answer, for the same reason ``_judge``
        # retries: a malformed reply is a fault in the channel, and holding a
        # run on the first one throws away work over a formatting accident.
        # This pass is the one that needed it most -- it is the only pass that
        # declares a tool, which is exactly the shape providers most often
        # answer in a call of their own naming.
        verdict: Optional[Verdict] = None
        for attempt in (1, 2):
            client = self._client(f"Verifier.precondition({row.get('name')})")
            client.set_system_message(prefix)
            message = f"{stable}\n\n{volatile}"
            if attempt == 2:
                message += (
                    "\n\nYour previous reply was not a readable verdict. Reply with "
                    'exactly one JSON object of the form {"verdict": ..., '
                    '"reason": ..., "fault": ...} and nothing else.'
                )
            try:
                handle = start_async_tool_loop(
                    client=client,
                    message=message,
                    tools={"run_probe": run_probe},
                    loop_id=f"VerifierPrecondition({row.get('name')})",
                    max_consecutive_failures=1,
                    max_steps=max_steps,
                    response_format=Verdict,
                    log_steps=False,
                )
                verdict = _parse_verdict(await handle.result())
            except Exception as exc:
                logger.warning(
                    "Precondition probe for %s failed: %s",
                    row.get("name"),
                    exc,
                )
                verdict = Verdict(verdict="UNSURE", reason="llm_error", fault=None)
                break
            if verdict is not None:
                break
        if verdict is None:
            verdict = Verdict(
                verdict="UNSURE",
                reason="unparseable_verdict",
                fault=None,
            )
        self._record(
            row,
            kind=VerdictKind.precondition,
            verdict=verdict,
            call_site=call_site,
            kwargs=kwargs,
            usage=usage,
            wall_ms=int((time.perf_counter() - started) * 1000),
        )
        return verdict

    async def post_probe(
        self,
        row: Mapping[str, Any],
        *,
        kwargs: Mapping[str, Any],
        result: Any,
        stable_block: str,
        call_site: str = "root",
        tier0: Optional[str] = None,
        sibling_results: Sequence[Mapping[str, Any]] = (),
        probe_output: Optional[str] = None,
        interactions: Sequence[Mapping[str, Any]] = (),
        kind: VerdictKind = VerdictKind.post,
    ) -> Verdict:
        prefix, stable, volatile = build_post_probe_prompt(
            stable_block=stable_block,
            kwargs=kwargs,
            result=result,
            tier0=tier0,
            sibling_results=sibling_results,
            probe_output=probe_output,
            interactions=interactions,
        )
        started = time.perf_counter()
        verdict, usage = await self._judge(
            prefix=prefix,
            stable=stable,
            volatile=volatile,
            origin=f"Verifier.post({row.get('name')})",
        )
        self._record(
            row,
            kind=kind,
            verdict=verdict,
            call_site=call_site,
            kwargs=kwargs,
            usage=usage,
            wall_ms=int((time.perf_counter() - started) * 1000),
        )
        return verdict


__all__ = [
    "CallSite",
    "Frame",
    "PassUsage",
    "VerifierPasses",
    "current_verification_frames",
    "executed_source",
    "locate_call_site",
    "pushed_frame",
    "run_probe",
]


# ---------------------------------------------------------------------------
# Run supervision: pending verdicts, barrier, memo, rewind, hold
# ---------------------------------------------------------------------------

import concurrent.futures
import functools
import inspect
import random
import threading
import traceback
from collections import OrderedDict

from unify.function_manager.steering import _PASSTHROUGH_TYPES, _is_async_callable
from unify.function_manager.types.verification import SideEffectClass
from unify.function_manager.verification.source_labels import (
    function_name_from_filename as _fn_from_filename,
)
from unify.function_manager.verification.tier0 import (
    ContractViolation,
    Tier0Checker,
    bind_call_kwargs,
    signature_from_source,
)


class RewindRequested(BaseException):
    """A verdict failed (or a leaf raised): cancel the run and repair the target."""

    def __init__(
        self,
        *,
        target_function_id: Optional[int],
        target_name: Optional[str],
        frames: Sequence[Frame],
        verdict: Optional[Verdict] = None,
        exception: Optional[BaseException] = None,
        ordinal: int = 0,
    ) -> None:
        self.target_function_id = target_function_id
        self.target_name = target_name
        self.frames = tuple(frames)
        self.verdict = verdict
        self.exception = exception
        self.ordinal = ordinal
        what = verdict.reason if verdict is not None else repr(exception)
        super().__init__(f"rewind requested for {target_name!r}: {what}")


class HoldRequested(BaseException):
    """An effect must not run (or must not be repeated): the run is held for the owner."""

    def __init__(
        self,
        *,
        code: str,
        leaf_name: Optional[str],
        reason: str,
        verdict: Optional[Verdict] = None,
        frames: Sequence[Frame] = (),
    ) -> None:
        self.code = code
        self.leaf_name = leaf_name
        self.reason = reason
        self.verdict = verdict
        self.frames = tuple(frames)
        super().__init__(f"hold ({code}) at {leaf_name!r}: {reason}")


@dataclass
class HeldOutcome:
    """What the owner is told when a run is held, plus the payload kept on the row."""

    code: str
    leaf_name: Optional[str]
    reason: str
    payload: Any = None
    message: str = ""


class VerifierExecutor:
    """A background event loop that runs verifier passes.

    Verdict tasks must never depend on the loop a stored function runs on:
    a synchronous effectful leaf called from an async root blocks that loop
    while it waits for its verdicts, and a sync root runs on a worker thread
    with no loop at all. Passes therefore run here, and callers wait on
    ``concurrent.futures.Future`` objects from whichever context they are in.
    """

    _instance: Optional["VerifierExecutor"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run,
            name="unify-verifier",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    @classmethod
    def shared(cls) -> "VerifierExecutor":
        with cls._lock:
            if cls._instance is None or not cls._instance._thread.is_alive():
                cls._instance = cls()
            return cls._instance

    def submit(
        self,
        factory,
        *,
        timeout_s: Optional[float],
        label: str = "",
    ) -> concurrent.futures.Future:
        """Run ``await factory()`` on the verifier loop; a timeout lands as UNSURE."""
        context = contextvars.copy_context()
        future: concurrent.futures.Future = concurrent.futures.Future()

        async def _runner() -> Verdict:
            try:
                if timeout_s is None:
                    return await factory()
                return await asyncio.wait_for(factory(), timeout_s)
            except asyncio.TimeoutError:
                return Verdict(verdict="UNSURE", reason="timeout", fault=None)

        def _start() -> None:
            if future.cancelled():
                return
            task = self.loop.create_task(_runner(), context=context, name=label or None)

            def _done(done_task: "asyncio.Task[Verdict]") -> None:
                if future.cancelled():
                    return
                if done_task.cancelled():
                    future.cancel()
                    return
                error = done_task.exception()
                if error is not None:
                    future.set_exception(error)
                else:
                    future.set_result(done_task.result())

            task.add_done_callback(_done)
            future.add_done_callback(
                lambda f: (
                    self.loop.call_soon_threadsafe(task.cancel)
                    if f.cancelled() and not task.done()
                    else None
                ),
            )

        self.loop.call_soon_threadsafe(_start)
        return future


@dataclass
class PendingVerdict:
    ordinal: int
    kind: VerdictKind
    frame: Frame
    row: Dict[str, Any]
    kwargs: Dict[str, Any]
    call_id: int
    future: concurrent.futures.Future
    verdict: Optional[Verdict] = None
    # A spot check on a trusted function informs — it never gates a later
    # effect and never rewinds; a FAIL invalidates trust and tells the owner.
    blocking: bool = True

    @property
    def landed(self) -> bool:
        return self.verdict is not None


class InteractionRecorder:
    """Wrap ``primitives`` so every call below it is appended to a run-scoped log."""

    __slots__ = ("_target", "_log", "_path")

    def __init__(
        self,
        target: Any,
        log: List[Dict[str, Any]],
        path: str = "primitives",
    ) -> None:
        while isinstance(target, InteractionRecorder):
            target = target._target
        self._target = target
        self._log = log
        self._path = path

    def __dir__(self):
        return dir(self._target)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._target, name)
        if not callable(attr):
            if isinstance(attr, _PASSTHROUGH_TYPES):
                return attr
            return InteractionRecorder(attr, self._log, f"{self._path}.{name}")
        tool = f"{self._path}.{name}"
        log = self._log

        def _entry(args: tuple, kwargs: dict) -> Dict[str, Any]:
            named: Dict[str, Any] = dict(kwargs)
            if args:
                named["*args"] = list(args)
            return {"name": tool, "args": _clip(named), "result": None}

        if _is_async_callable(attr):

            @functools.wraps(attr)
            async def _dispatch(*args: Any, **kwargs: Any) -> Any:
                entry = _entry(args, kwargs)
                log.append(entry)
                result = attr(*args, **kwargs)
                if inspect.isawaitable(result):
                    result = await result
                entry["result"] = _clip(result)
                return result

        else:

            @functools.wraps(attr)
            def _dispatch(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
                entry = _entry(args, kwargs)
                log.append(entry)
                result = attr(*args, **kwargs)
                if inspect.isawaitable(result):

                    async def _finish() -> Any:
                        value = await result
                        entry["result"] = _clip(value)
                        return value

                    return _finish()
                entry["result"] = _clip(result)
                return result

        return _dispatch


def _clip(value: Any, limit: int = 2000) -> Any:
    """A JSON-safe, size-bounded copy of ``value`` for prompts and rows."""
    try:
        text = json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)
    except Exception:
        text = repr(value)
    if len(text) <= limit:
        try:
            return json.loads(text)
        except Exception:
            return text
    return text[:limit] + f"… ({len(text) - limit} more characters)"


def _row_class(row: Mapping[str, Any]) -> SideEffectClass:
    return SideEffectClass(
        str(row.get("side_effect_class") or SideEffectClass.unsafe_effectful),
    )


def _innermost_stored_function(exc: BaseException) -> Optional[str]:
    """The name of the innermost stored function in ``exc``'s traceback, if any."""
    innermost: Optional[str] = None
    for frame_summary in traceback.extract_tb(exc.__traceback__):
        name = _fn_from_filename(frame_summary.filename)
        if name is not None:
            innermost = name
    return innermost


class RunVerificationSupervisor:
    """Pending verdicts, barrier, memo and rewind state for one entrypoint run."""

    def __init__(
        self,
        *,
        passes: VerifierPasses,
        settings: Any,
        root_row: Mapping[str, Any],
        rows_by_name: Mapping[str, Dict[str, Any]],
        executor: Optional[VerifierExecutor] = None,
        memo: Optional[Dict[tuple, Any]] = None,
        verdict_counts: Optional[Dict[str, int]] = None,
    ) -> None:
        self.passes = passes
        self.settings = settings
        self.root_row = dict(root_row)
        self.rows_by_name: Dict[str, Dict[str, Any]] = {
            k: dict(v) for k, v in rows_by_name.items()
        }
        self.executor = executor or VerifierExecutor.shared()
        # Every verdict of the current attempt in launch order, landed or not;
        # ``pending`` is the subset still in flight.
        self.verdicts: "OrderedDict[int, PendingVerdict]" = OrderedDict()
        self.pending: "OrderedDict[int, PendingVerdict]" = OrderedDict()
        self.landed: List[PendingVerdict] = []
        # The memo outlives one attempt: a rewind must not re-run what already
        # passed or what a trusted function already did.
        self.memo: Dict[tuple, Any] = memo if memo is not None else {}
        self.sibling_results: List[Dict[str, Any]] = []
        self.interactions: List[Dict[str, Any]] = []
        self.rewind: Optional[RewindRequested] = None
        self.verdict_counts: Dict[str, int] = (
            verdict_counts
            if verdict_counts is not None
            else {"PASS": 0, "FAIL": 0, "UNSURE": 0}
        )
        self.rewinds = 0
        self.attempts = 0
        self.tasks_created = 0
        self.repair_targets: Dict[str, int] = {}
        self._ordinal = 0
        self._call_seq = 0
        self._call_pending: Dict[int, List[PendingVerdict]] = {}
        self._call_meta: Dict[int, Dict[str, Any]] = {}
        self._entry_task: Optional["asyncio.Task[Any]"] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock = threading.Lock()
        # Called with (pending, verdict) when a spot check on a trusted
        # function fails; installed by the run so the owner hears about it.
        self.on_spot_check_fail = None

    # ---- attempt lifecycle ---------------------------------------------

    def begin_attempt(
        self,
        entry_task: "asyncio.Task[Any]",
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self.attempts += 1
        self.verdicts.clear()
        self.pending.clear()
        self.landed.clear()
        self.sibling_results.clear()
        self.interactions.clear()
        self.rewind = None
        self._call_pending.clear()
        self._call_meta.clear()
        self._entry_task = entry_task
        self._loop = loop

    def new_call(
        self,
        *,
        row: Mapping[str, Any],
        args_signature: str,
        effectful: bool,
    ) -> int:
        self._call_seq += 1
        call_id = self._call_seq
        self._call_pending[call_id] = []
        self._call_meta[call_id] = {
            "row": row,
            "args_signature": args_signature,
            "effectful": effectful,
            "result": None,
            "executed": False,
            "tier0_failed": False,
            # Set once every verdict for the call has been launched; the memo
            # never fills before the post verdict exists to be waited on.
            "closed": False,
        }
        return call_id

    def memo_key(self, row: Mapping[str, Any], args_signature: str) -> tuple:
        return (int(row["function_id"]), self.passes.trust_hash(row), args_signature)

    # ---- verdict tasks --------------------------------------------------

    def launch(
        self,
        *,
        kind: VerdictKind,
        frame: Frame,
        row: Mapping[str, Any],
        kwargs: Mapping[str, Any],
        call_id: int,
        factory,
        blocking: bool = True,
    ) -> PendingVerdict:
        """Start a verifier pass as a background verdict, in call order."""
        with self._lock:
            self._ordinal += 1
            ordinal = self._ordinal
        self.tasks_created += 1
        future = self.executor.submit(
            factory,
            timeout_s=float(self.settings.pending_verdict_timeout_s),
            label=f"verdict:{kind.value}:{frame.name}",
        )
        pending = PendingVerdict(
            ordinal=ordinal,
            kind=kind,
            frame=frame,
            row=dict(row),
            kwargs=dict(kwargs),
            call_id=call_id,
            future=future,
            blocking=blocking,
        )
        with self._lock:
            self.verdicts[ordinal] = pending
            self.pending[ordinal] = pending
            self._call_pending.setdefault(call_id, []).append(pending)
        future.add_done_callback(lambda f, p=pending: self._on_future_done(p, f))
        return pending

    def _on_future_done(
        self,
        pending: PendingVerdict,
        future: concurrent.futures.Future,
    ) -> None:
        if future.cancelled():
            return
        error = future.exception()
        if error is not None:
            verdict = Verdict(verdict="UNSURE", reason="llm_error", fault=None)
        else:
            verdict = future.result()
        self._land(pending, verdict)

    def _land(self, pending: PendingVerdict, verdict: Verdict) -> None:
        with self._lock:
            if pending.landed:
                return
            pending.verdict = verdict
            self.verdict_counts[verdict.verdict] = (
                self.verdict_counts.get(verdict.verdict, 0) + 1
            )
            self.pending.pop(pending.ordinal, None)
            self.landed.append(pending)
        if verdict.verdict == "FAIL":
            if pending.blocking:
                self.fail(verdict, pending.frame, pending.row, ordinal=pending.ordinal)
            elif self.on_spot_check_fail is not None:
                self.on_spot_check_fail(pending, verdict)
        self._maybe_memoise(pending.call_id)

    def note_sync_verdict(self, pending_like: PendingVerdict, verdict: Verdict) -> None:
        """Record a verdict that was awaited to completion inline (blocking passes)."""
        self._land(pending_like, verdict)

    def close_call(self, call_id: int) -> None:
        """Every verdict for ``call_id`` has been launched; the memo may now fill."""
        meta = self._call_meta.get(call_id)
        if meta is None:
            return
        meta["closed"] = True
        self._maybe_memoise(call_id)

    def _maybe_memoise(self, call_id: int) -> None:
        meta = self._call_meta.get(call_id)
        if (
            meta is None
            or not meta["executed"]
            or meta["tier0_failed"]
            or not meta["closed"]
        ):
            return
        pendings = self._call_pending.get(call_id, [])
        if any(not p.landed for p in pendings):
            return
        verdicts = [p.verdict.verdict for p in pendings if p.verdict is not None]
        if any(v == "FAIL" for v in verdicts):
            return
        row = meta["row"]
        # An effect that ran is a fact whatever the verifier could tell about
        # it, so it is memoised on UNSURE too; a read is only memoised once
        # its verdicts passed — re-reading is harmless and yields fresh data.
        if any(v == "UNSURE" for v in verdicts) and not meta["effectful"]:
            return
        self.memo[self.memo_key(row, meta["args_signature"])] = meta["result"]

    def record_result(self, call_id: int, *, result: Any, name: str) -> None:
        meta = self._call_meta[call_id]
        meta["result"] = result
        meta["executed"] = True
        self.sibling_results.append({"name": name, "result": _clip(result)})
        self._maybe_memoise(call_id)

    def record_tier0_failure(self, call_id: int) -> None:
        self._call_meta[call_id]["tier0_failed"] = True

    # ---- failure, barrier, drain -----------------------------------------

    def fail(
        self,
        verdict: Verdict,
        frame: Frame,
        row: Mapping[str, Any],
        *,
        ordinal: int = 0,
        exception: Optional[BaseException] = None,
    ) -> RewindRequested:
        """Record a failure, cancel later verdicts and the entrypoint task."""
        frames = current_verification_frames.get() or (frame,)
        if verdict is not None and verdict.fault == "caller":
            chain = list(frames)
            index = next(
                (i for i, f in enumerate(chain) if f.function_id == frame.function_id),
                None,
            )
            target = chain[index - 1] if index is not None and index > 0 else frame
        else:
            target = frame
        rewind = RewindRequested(
            target_function_id=target.function_id,
            target_name=target.name,
            frames=frames,
            verdict=verdict,
            exception=exception,
            ordinal=ordinal,
        )
        with self._lock:
            if self.rewind is None or (0 < ordinal < self.rewind.ordinal):
                self.rewind = rewind
            later = (
                [p for o, p in self.pending.items() if o > ordinal] if ordinal else []
            )
        for pending in later:
            self._cancel_pending(pending, reason="cancelled")
        entry_task, loop = self._entry_task, self._loop
        if entry_task is not None and loop is not None and not entry_task.done():
            loop.call_soon_threadsafe(entry_task.cancel)
        return rewind

    def _cancel_pending(self, pending: PendingVerdict, *, reason: str) -> None:
        if pending.landed:
            return
        if not pending.future.cancel():
            return
        verdict = Verdict(verdict="UNSURE", reason=reason, fault=None)
        with self._lock:
            pending.verdict = verdict
            self.pending.pop(pending.ordinal, None)
            self.landed.append(pending)
            self.verdict_counts["UNSURE"] += 1
        self.passes._record(
            pending.row,
            kind=pending.kind,
            verdict=verdict,
            call_site=pending.frame.call_site_line and "root" or "root",
            kwargs=pending.kwargs,
            usage=PassUsage(),
            wall_ms=0,
        )

    def cancel_all(self, *, reason: str = "cancelled") -> None:
        for pending in list(self.pending.values()):
            self._cancel_pending(pending, reason=reason)

    def _earlier(self, before_ordinal: Optional[int]) -> List[PendingVerdict]:
        """Every verdict of this attempt launched before ``before_ordinal`` (landed or not)."""
        with self._lock:
            return [
                p
                for o, p in self.verdicts.items()
                if p.blocking and (before_ordinal is None or o < before_ordinal)
            ]

    def _outstanding(self) -> List[PendingVerdict]:
        with self._lock:
            return list(self.pending.values())

    def _check(self, pending: PendingVerdict) -> None:
        verdict = pending.verdict or Verdict(
            verdict="UNSURE",
            reason="cancelled",
            fault=None,
        )
        if verdict.verdict == "FAIL":
            raise self.rewind or RewindRequested(
                target_function_id=pending.frame.function_id,
                target_name=pending.frame.name,
                frames=current_verification_frames.get(),
                verdict=verdict,
                ordinal=pending.ordinal,
            )
        if verdict.verdict == "UNSURE":
            raise HoldRequested(
                code=("verdict_unavailable" if verdict_is_fault(verdict) else "unsure"),
                leaf_name=pending.frame.name,
                reason=verdict.reason,
                verdict=verdict,
                frames=current_verification_frames.get(),
            )

    async def barrier(self, *, before_ordinal: Optional[int] = None) -> None:
        """Wait, in order, for every earlier verdict to land as PASS."""
        for pending in self._earlier(before_ordinal):
            if not pending.landed:
                try:
                    await asyncio.wrap_future(pending.future)
                except (concurrent.futures.CancelledError, asyncio.CancelledError):
                    pass
                except Exception:
                    pass
            self._check(pending)
        if self.rewind is not None:
            raise self.rewind

    def barrier_sync(self, *, before_ordinal: Optional[int] = None) -> None:
        for pending in self._earlier(before_ordinal):
            if not pending.landed:
                try:
                    pending.future.result()
                except (concurrent.futures.CancelledError, Exception):
                    pass
            self._check(pending)
        if self.rewind is not None:
            raise self.rewind

    async def drain(self) -> List[PendingVerdict]:
        """Wait for every pending verdict; return everything that landed."""
        while True:
            outstanding = self._outstanding()
            if not outstanding:
                break
            for pending in outstanding:
                if not pending.landed:
                    try:
                        await asyncio.wrap_future(pending.future)
                    except (
                        concurrent.futures.CancelledError,
                        asyncio.CancelledError,
                        Exception,
                    ):
                        pass
        return list(self.landed)


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------


def _unwrap_tier0(fn: Any) -> tuple[Any, Optional[Tier0Checker]]:
    inner = getattr(fn, "__tier0_inner__", None)
    checker = getattr(fn, "__tier0_checker__", None)
    if inner is not None:
        return inner, checker
    return fn, None


def _tier0_text(reason: Optional[str]) -> str:
    return "PASS: contract satisfied" if reason is None else f"FAIL: {reason}"


class VerifiedCall:
    """Per-call verification around one untrusted stored function.

    Built by :func:`verified_call`; the callable it produces is a plain
    ``def``/``async def`` so ``inspect`` reports the wrapped function's shape.
    """

    def __init__(
        self,
        *,
        inner: Any,
        raw: Any,
        row: Dict[str, Any],
        supervisor: RunVerificationSupervisor,
        checker: Optional[Tier0Checker],
        signature: Optional[inspect.Signature],
    ) -> None:
        self.inner = inner
        self.raw = raw
        self.row = row
        self.supervisor = supervisor
        self.checker = checker
        self.signature = signature
        self.name = str(row.get("name"))
        self.klass = _row_class(row)
        self.effectful = self.klass.is_effectful
        self.trusted = not bool(row.get("verify", True))

    # -- shared pieces ------------------------------------------------------

    def _frame(self, named: Mapping[str, Any]) -> Frame:
        site = locate_call_site()
        return Frame(
            function_id=int(self.row["function_id"]),
            name=self.name,
            docstring=str(self.row.get("docstring") or ""),
            effect_class=self.klass.value,
            call_site_line=site.line_text or "",
            args_repr=json.dumps(
                _clip(dict(named), 1000),
                default=str,
                ensure_ascii=False,
            ),
        )

    def _site(self) -> CallSite:
        return locate_call_site()

    def _stable(self, frames: Sequence[Frame], site: CallSite) -> str:
        return self.supervisor.passes.stable_block(
            self.row,
            frames=frames,
            call_site=site,
            root_row=self.supervisor.root_row,
        )

    def _needs_static(self) -> bool:
        cached = self.row.get("static_review")
        return not (
            isinstance(cached, dict)
            and cached.get("function_hash")
            == self.supervisor.passes.trust_hash(self.row)
        )

    def _needs_precondition(self) -> bool:
        """Whether this call needs the world checked before its effect runs.

        Only when there is something to check. An effectful function with no
        declared precondition used to be probed anyway, which asked a judge
        to confirm a criterion nobody had written: the prompt substitutes
        "(none declared -- judge from the goal, the source and the
        arguments)", and for anything whose job is to go and read the world
        that is unanswerable without doing the work. An inconclusive answer
        then held the run, so a function was blocked by the absence of a
        precondition rather than by anything about its state.

        The other passes still cover it: static review reads the source,
        args review reads the call, tier-0 checks the contract, and the post
        pass checks what the effect produced.
        """

        return self.row.get("precondition") is not None

    def _tier0_input(
        self,
        named: Mapping[str, Any],
        call_id: int,
        frame: Frame,
    ) -> Optional[str]:
        if self.checker is None or not self.checker.active:
            return None
        try:
            self.checker.check_input(named)
        except ContractViolation as violation:
            self.supervisor.record_tier0_failure(call_id)
            raise self.supervisor.fail(
                violation.verdict,
                frame,
                self.row,
            ) from violation
        return None

    def _tier0_output(
        self,
        named: Mapping[str, Any],
        result: Any,
        call_id: int,
        frame: Frame,
    ) -> None:
        if self.checker is None or not self.checker.active:
            return
        try:
            self.checker.check_output(result=result, kwargs=named)
        except ContractViolation as violation:
            self.supervisor.record_tier0_failure(call_id)
            if self.klass is SideEffectClass.unsafe_effectful:
                raise HoldRequested(
                    code="effect_verification_failed",
                    leaf_name=self.name,
                    reason=violation.verdict.reason,
                    verdict=violation.verdict,
                    frames=current_verification_frames.get(),
                ) from violation
            raise self.supervisor.fail(
                violation.verdict,
                frame,
                self.row,
            ) from violation

    def _launch_pre(
        self,
        *,
        frames: Sequence[Frame],
        site: CallSite,
        named: Mapping[str, Any],
        call_id: int,
    ) -> None:
        """Static/args/precondition as background verdicts (non-effectful leaves)."""
        passes = self.supervisor.passes
        frame = frames[-1]
        stable = self._stable(frames, site)
        if self._needs_static():
            row = self.row
            self.supervisor.launch(
                kind=VerdictKind.static,
                frame=frame,
                row=row,
                kwargs={},
                call_id=call_id,
                factory=lambda: passes.static_review(row),
            )
        siblings = list(self.supervisor.sibling_results)
        row = self.row
        self.supervisor.launch(
            kind=VerdictKind.args,
            frame=frame,
            row=row,
            kwargs=named,
            call_id=call_id,
            factory=lambda: passes.args_review(
                row,
                kwargs=named,
                stable_block=stable,
                call_site=site.label,
                tier0=_tier0_text(None),
                sibling_results=siblings,
            ),
        )
        if self._needs_precondition():
            self.supervisor.launch(
                kind=VerdictKind.precondition,
                frame=frame,
                row=row,
                kwargs=named,
                call_id=call_id,
                factory=lambda: passes.precondition_probe(
                    row,
                    kwargs=named,
                    stable_block=stable,
                    call_site=site.label,
                    sibling_results=siblings,
                ),
            )

    def _blocking_pre_factories(
        self,
        *,
        frames: Sequence[Frame],
        site: CallSite,
        named: Mapping[str, Any],
    ):
        passes = self.supervisor.passes
        stable = self._stable(frames, site)
        siblings = list(self.supervisor.sibling_results)
        row = self.row
        steps = []
        if self._needs_static():
            steps.append((VerdictKind.static, lambda: passes.static_review(row)))
        steps.append(
            (
                VerdictKind.args,
                lambda: passes.args_review(
                    row,
                    kwargs=named,
                    stable_block=stable,
                    call_site=site.label,
                    tier0=_tier0_text(None),
                    sibling_results=siblings,
                ),
            ),
        )
        steps.append(
            (
                VerdictKind.precondition,
                lambda: passes.precondition_probe(
                    row,
                    kwargs=named,
                    stable_block=stable,
                    call_site=site.label,
                    sibling_results=siblings,
                ),
            ),
        )
        return steps

    def _after_blocking(
        self,
        pending: PendingVerdict,
        verdict: Verdict,
        frame: Frame,
    ) -> None:
        self.supervisor.note_sync_verdict(pending, verdict)
        if verdict.verdict == "FAIL":
            raise self.supervisor.rewind or self.supervisor.fail(
                verdict,
                frame,
                self.row,
                ordinal=pending.ordinal,
            )
        if verdict.verdict == "UNSURE":
            raise HoldRequested(
                code=("verdict_unavailable" if verdict_is_fault(verdict) else "unsure"),
                leaf_name=self.name,
                reason=verdict.reason,
                verdict=verdict,
                frames=current_verification_frames.get(),
            )

    def _launch_post(
        self,
        *,
        frames: Sequence[Frame],
        site: CallSite,
        named: Mapping[str, Any],
        result: Any,
        call_id: int,
        interactions: List[Dict[str, Any]],
        tier0: Optional[str],
    ) -> None:
        if self.klass is SideEffectClass.safe_noop:
            return
        passes = self.supervisor.passes
        stable = self._stable(frames, site)
        siblings = list(self.supervisor.sibling_results)
        row = self.row
        self.supervisor.launch(
            kind=VerdictKind.post,
            frame=frames[-1],
            row=row,
            kwargs=named,
            call_id=call_id,
            factory=lambda: passes.post_probe(
                row,
                kwargs=named,
                result=result,
                stable_block=stable,
                call_site=site.label,
                tier0=_tier0_text(tier0),
                sibling_results=siblings,
                interactions=interactions,
            ),
        )

    # -- the call -------------------------------------------------------------

    def _prepare(self, args: tuple, kwargs: Mapping[str, Any]):
        named = bind_call_kwargs(self.signature, args, kwargs)
        if named is None:
            named = dict(kwargs)
        signature = args_signature(named)
        site = self._site()
        frame = self._frame(named)
        call_id = self.supervisor.new_call(
            row=self.row,
            args_signature=signature,
            effectful=self.effectful,
        )
        return named, signature, site, frame, call_id

    async def call_async(self, *args: Any, **kwargs: Any) -> Any:
        named, signature, site, frame, call_id = self._prepare(args, kwargs)
        with pushed_frame(frame) as frames:
            self._tier0_input(named, call_id, frame)
            key = self.supervisor.memo_key(self.row, signature)
            if key in self.supervisor.memo:
                return self.supervisor.memo[key]
            if self.effectful:
                await self.supervisor.barrier()
                for kind, factory in self._blocking_pre_factories(
                    frames=frames,
                    site=site,
                    named=named,
                ):
                    pending = self.supervisor.launch(
                        kind=kind,
                        frame=frame,
                        row=self.row,
                        kwargs=named,
                        call_id=call_id,
                        factory=factory,
                    )
                    try:
                        verdict = await asyncio.wrap_future(pending.future)
                    except (concurrent.futures.CancelledError, asyncio.CancelledError):
                        raise
                    self._after_blocking(pending, verdict, frame)
            else:
                self._launch_pre(frames=frames, site=site, named=named, call_id=call_id)
            start = len(self.supervisor.interactions)
            result = self.inner(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            interactions = list(self.supervisor.interactions[start:])
            tier0_reason: Optional[str] = None
            try:
                self._tier0_output(named, result, call_id, frame)
            except (RewindRequested, HoldRequested):
                raise
            self.supervisor.record_result(call_id, result=result, name=self.name)
            self._launch_post(
                frames=frames,
                site=site,
                named=named,
                result=result,
                call_id=call_id,
                interactions=interactions,
                tier0=tier0_reason,
            )
            self.supervisor.close_call(call_id)
            return result

    def call_sync(self, *args: Any, **kwargs: Any) -> Any:
        named, signature, site, frame, call_id = self._prepare(args, kwargs)
        with pushed_frame(frame) as frames:
            self._tier0_input(named, call_id, frame)
            key = self.supervisor.memo_key(self.row, signature)
            if key in self.supervisor.memo:
                return self.supervisor.memo[key]
            if self.effectful:
                self.supervisor.barrier_sync()
                for kind, factory in self._blocking_pre_factories(
                    frames=frames,
                    site=site,
                    named=named,
                ):
                    pending = self.supervisor.launch(
                        kind=kind,
                        frame=frame,
                        row=self.row,
                        kwargs=named,
                        call_id=call_id,
                        factory=factory,
                    )
                    verdict = pending.future.result()
                    self._after_blocking(pending, verdict, frame)
            else:
                self._launch_pre(frames=frames, site=site, named=named, call_id=call_id)
            start = len(self.supervisor.interactions)
            result = self.inner(*args, **kwargs)
            if inspect.isawaitable(result):

                async def _finish() -> Any:
                    value = await result
                    interactions = list(self.supervisor.interactions[start:])
                    with pushed_frame(frame):
                        self._tier0_output(named, value, call_id, frame)
                        self.supervisor.record_result(
                            call_id,
                            result=value,
                            name=self.name,
                        )
                        self._launch_post(
                            frames=frames,
                            site=site,
                            named=named,
                            result=value,
                            call_id=call_id,
                            interactions=interactions,
                            tier0=None,
                        )
                        self.supervisor.close_call(call_id)
                    return value

                return _finish()
            interactions = list(self.supervisor.interactions[start:])
            self._tier0_output(named, result, call_id, frame)
            self.supervisor.record_result(call_id, result=result, name=self.name)
            self._launch_post(
                frames=frames,
                site=site,
                named=named,
                result=result,
                call_id=call_id,
                interactions=interactions,
                tier0=None,
            )
            self.supervisor.close_call(call_id)
            return result


def verified_call(
    fn: Any,
    *,
    row: Dict[str, Any],
    supervisor: RunVerificationSupervisor,
) -> Any:
    """Wrap a namespace callable of an untrusted function for a supervised run."""
    inner, checker = _unwrap_tier0(fn)
    raw = getattr(fn, "__wrapped__", None)
    signature = None
    if raw is not None:
        try:
            signature = inspect.signature(raw)
        except (TypeError, ValueError):
            signature = None
    if signature is None:
        signature = signature_from_source(row.get("implementation"))
    call = VerifiedCall(
        inner=inner,
        raw=raw,
        row=row,
        supervisor=supervisor,
        checker=checker,
        signature=signature,
    )
    target = raw if callable(raw) else inner
    if raw is not None and inspect.iscoroutinefunction(raw):

        @functools.wraps(target)
        async def _async(*args: Any, **kwargs: Any) -> Any:
            return await call.call_async(*args, **kwargs)

        _async.__verified_call__ = call  # type: ignore[attr-defined]
        return _async

    @functools.wraps(target)
    def _sync(*args: Any, **kwargs: Any) -> Any:
        return call.call_sync(*args, **kwargs)

    _sync.__verified_call__ = call  # type: ignore[attr-defined]
    return _sync


def memoised_call(
    fn: Any,
    *,
    row: Dict[str, Any],
    supervisor: RunVerificationSupervisor,
) -> Any:
    """Wrap a trusted function's callable so a rewind never re-runs its effect."""
    raw = getattr(fn, "__wrapped__", None)
    signature = None
    if raw is not None:
        try:
            signature = inspect.signature(raw)
        except (TypeError, ValueError):
            signature = None
    if signature is None:
        signature = signature_from_source(row.get("implementation"))
    name = str(row.get("name"))

    def _key(args: tuple, kwargs: Mapping[str, Any]) -> tuple:
        named = bind_call_kwargs(signature, args, kwargs)
        if named is None:
            named = dict(kwargs)
        return supervisor.memo_key(row, args_signature(named))

    def _remember(key: tuple, result: Any) -> Any:
        supervisor.memo[key] = result
        supervisor.sibling_results.append({"name": name, "result": _clip(result)})
        return result

    target = raw if callable(raw) else fn
    if raw is not None and inspect.iscoroutinefunction(raw):

        @functools.wraps(target)
        async def _async(*args: Any, **kwargs: Any) -> Any:
            key = _key(args, kwargs)
            if key in supervisor.memo:
                return supervisor.memo[key]
            return _remember(key, await fn(*args, **kwargs))

        return _async

    @functools.wraps(target)
    def _sync(*args: Any, **kwargs: Any) -> Any:
        key = _key(args, kwargs)
        if key in supervisor.memo:
            return supervisor.memo[key]
        result = fn(*args, **kwargs)
        if inspect.isawaitable(result):

            async def _finish() -> Any:
                return _remember(key, await result)

            return _finish()
        return _remember(key, result)

    return _sync


def spot_checked_call(
    fn: Any,
    *,
    row: Dict[str, Any],
    supervisor: RunVerificationSupervisor,
    rate: float,
) -> Any:
    """Memoise a trusted effectful function and sample its calls for a post probe.

    The probe never gates anything and never rewinds: a FAIL puts the
    function back on the ramp and tells the owner what was seen.
    """
    memoised = memoised_call(fn, row=row, supervisor=supervisor)
    raw = getattr(fn, "__wrapped__", None)
    signature = None
    if raw is not None:
        try:
            signature = inspect.signature(raw)
        except (TypeError, ValueError):
            signature = None
    if signature is None:
        signature = signature_from_source(row.get("implementation"))
    name = str(row.get("name"))
    klass = _row_class(row)

    def _maybe_probe(
        args: tuple,
        kwargs: Mapping[str, Any],
        result: Any,
        start: int,
    ) -> None:
        if random.random() >= rate:
            return
        named = bind_call_kwargs(signature, args, kwargs)
        if named is None:
            named = dict(kwargs)
        site = locate_call_site()
        frame = Frame(
            function_id=int(row["function_id"]),
            name=name,
            docstring=str(row.get("docstring") or ""),
            effect_class=klass.value,
            call_site_line=site.line_text or "",
            args_repr=json.dumps(
                _clip(dict(named), 1000),
                default=str,
                ensure_ascii=False,
            ),
        )
        frames = (*current_verification_frames.get(), frame)
        passes = supervisor.passes
        stable = passes.stable_block(
            row,
            frames=frames,
            call_site=site,
            root_row=supervisor.root_row,
        )
        interactions = list(supervisor.interactions[start:])
        siblings = list(supervisor.sibling_results)
        call_id = supervisor.new_call(
            row=row,
            args_signature=args_signature(named),
            effectful=True,
        )
        supervisor.launch(
            kind=VerdictKind.spot_check,
            frame=frame,
            row=row,
            kwargs=named,
            call_id=call_id,
            blocking=False,
            factory=lambda: passes.post_probe(
                row,
                kwargs=named,
                result=result,
                stable_block=stable,
                call_site=site.label,
                sibling_results=siblings,
                interactions=interactions,
                kind=VerdictKind.spot_check,
            ),
        )

    target = raw if callable(raw) else fn
    if raw is not None and inspect.iscoroutinefunction(raw):

        @functools.wraps(target)
        async def _async(*args: Any, **kwargs: Any) -> Any:
            start = len(supervisor.interactions)
            result = await memoised(*args, **kwargs)
            _maybe_probe(args, kwargs, result, start)
            return result

        return _async

    @functools.wraps(target)
    def _sync(*args: Any, **kwargs: Any) -> Any:
        start = len(supervisor.interactions)
        result = memoised(*args, **kwargs)
        if inspect.isawaitable(result):

            async def _finish() -> Any:
                value = await result
                _maybe_probe(args, kwargs, value, start)
                return value

            return _finish()
        _maybe_probe(args, kwargs, result, start)
        return result

    return _sync


def install_wrappers(
    namespace: Dict[str, Any],
    *,
    rows_by_name: Mapping[str, Dict[str, Any]],
    supervisor: RunVerificationSupervisor,
) -> int:
    """Replace closure callables in ``namespace``; returns how many were verified-wrapped."""
    from unify.function_manager.verification.policy import spot_check_rate

    wrapped = 0
    for name, row in rows_by_name.items():
        fn = namespace.get(name)
        if not callable(fn):
            continue
        if row.get("verify", True):
            namespace[name] = verified_call(fn, row=row, supervisor=supervisor)
            wrapped += 1
        else:
            rate = spot_check_rate(row, supervisor.settings)
            if rate > 0:
                namespace[name] = spot_checked_call(
                    fn,
                    row=row,
                    supervisor=supervisor,
                    rate=rate,
                )
            else:
                namespace[name] = memoised_call(fn, row=row, supervisor=supervisor)
    primitives = namespace.get("primitives")
    if primitives is not None:
        namespace["primitives"] = InteractionRecorder(
            primitives,
            supervisor.interactions,
        )
    return wrapped


def closure_rows(
    function_manager: Any,
    root_row: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Full rows of the root and every compositional function it depends on, transitively.

    Catalogue reads strip ledger internals, so the root is re-read by name to
    get the row the trust policy needs.
    """
    root_name = str(root_row["name"])
    full_root = function_manager._get_function_data_by_name(name=root_name)
    rows: Dict[str, Dict[str, Any]] = {root_name: dict(full_root or root_row)}
    queue = [
        d
        for d in (rows[root_name].get("depends_on") or [])
        if isinstance(d, str) and "." not in d
    ]
    while queue:
        name = queue.pop()
        if name in rows:
            continue
        row = function_manager._get_function_data_by_name(name=name)
        if row is None:
            continue
        rows[name] = dict(row)
        queue.extend(
            d
            for d in (row.get("depends_on") or [])
            if isinstance(d, str) and "." not in d
        )
    return rows


def closure_is_trusted(rows_by_name: Mapping[str, Mapping[str, Any]]) -> bool:
    return all(not row.get("verify", True) for row in rows_by_name.values())


def closure_needs_supervision(
    rows_by_name: Mapping[str, Mapping[str, Any]],
    settings: Any,
) -> bool:
    """Whether a run over this closure needs a supervisor at all.

    Untrusted members do; so does a trusted effectful member without an
    output contract, because its calls are sampled for spot checks.
    """
    from unify.function_manager.verification.policy import spot_check_rate

    for row in rows_by_name.values():
        if row.get("verify", True):
            return True
        if spot_check_rate(row, settings) > 0:
            return True
    return False


def rederive_trust(
    function_manager: Any,
    rows_by_name: Dict[str, Dict[str, Any]],
    *,
    settings: Any,
) -> None:
    """Recompute ``verify`` for every closure row from its ledger and current content.

    The stored flag can lag a content change elsewhere in the closure; the
    run must see the derived value, and a row whose stored trust no longer
    holds is refolded so the store catches up.
    """
    from unify.function_manager.verification.ledger import function_trust_hash
    from unify.function_manager.verification.policy import derive_verify

    for name, row in rows_by_name.items():
        if row.get("is_primitive"):
            continue
        current = function_trust_hash(
            row,
            resolve_row=lambda dep, _rows=rows_by_name: _rows.get(dep)
            or function_manager._get_function_data_by_name(name=dep),
            resolve_venv=lambda venv_id: function_manager.get_venv(venv_id=venv_id),
        )
        derived = derive_verify(row, settings=settings, current_hash=current)
        stored = bool(row.get("verify", True))
        row["verify"] = derived
        if derived != stored:
            function_manager.refresh_trust(int(row["function_id"]))


def held_message(task_name: str, outcome: HeldOutcome) -> str:
    """The owner-facing sentence for a held run: what was held and why, never internals."""
    leaf = outcome.leaf_name or "the task"
    if outcome.code == "deployment_owned_function_failed":
        return (
            f"Holding {task_name}: {leaf} failed and is owned by the deployment, so it "
            f"was not changed ({outcome.reason}). Nothing was sent or changed. "
            "Payload retained on the execution row."
        )
    if outcome.code == "effect_verification_failed":
        return (
            f"Holding {task_name}: {leaf} ran but could not be verified afterwards "
            f"({outcome.reason}). It was not repeated. Payload retained on the execution row."
        )
    if outcome.code == "verdict_unavailable":
        return (
            f"Holding {task_name}: the check on {leaf} did not return a readable "
            f"result ({outcome.reason}), so nothing was sent or changed. This is a "
            "fault in the check, not a judgement about the work. Payload retained "
            "on the execution row."
        )
    if outcome.code == "exhausted":
        return (
            f"Holding {task_name}: repairs to {leaf} did not verify ({outcome.reason}). "
            "Nothing further was sent or changed. Payload retained on the execution row."
        )
    return (
        f"Holding {task_name}: could not verify {leaf} ({outcome.reason}). "
        "Nothing was sent or changed. Payload retained on the execution row."
    )


def correction_message(
    task_name: str,
    *,
    delivered_at: str,
    reason: str,
    result: Any,
) -> str:
    return (
        f"Correction to {task_name} delivered at {delivered_at}: {reason}. "
        f"Corrected result: {_clip(result, 2000)}"
    )


# ---------------------------------------------------------------------------
# The run: attempts, repair, hold, delivery
# ---------------------------------------------------------------------------


class RepairRefused(Exception):
    """A repair could not be attempted (deployment-owned function, no manager, …)."""


@dataclass
class EntrypointOutcome:
    """What one entrypoint run produced, for the handle and the execution row."""

    result: Any = None
    held: Optional[HeldOutcome] = None
    rewinds: int = 0
    attempts: int = 1
    verdict_counts: Dict[str, int] = None  # type: ignore[assignment]
    verifier_tasks: int = 0
    follow_up: Optional["asyncio.Task[Any]"] = None

    def __post_init__(self) -> None:
        if self.verdict_counts is None:
            self.verdict_counts = {"PASS": 0, "FAIL": 0, "UNSURE": 0}


def _held_from(hold: HoldRequested, *, task_name: str) -> HeldOutcome:
    payload = None
    if hold.frames:
        last = hold.frames[-1]
        payload = {"function": last.name, "arguments": last.args_repr}
    outcome = HeldOutcome(
        code=hold.code,
        leaf_name=hold.leaf_name,
        reason=hold.reason,
        payload=payload,
    )
    outcome.message = held_message(task_name, outcome)
    return outcome


async def run_verified_entrypoint(
    *,
    settings: Any,
    task_name: str,
    resolve,
    invoke,
    make_passes,
    repair,
    notify=None,
    executor: Optional[VerifierExecutor] = None,
) -> EntrypointOutcome:
    """Run an entrypoint under verification: rewind on FAIL, hold on UNSURE, deliver after drain.

    ``resolve()`` returns ``(root_row, rows_by_name)`` for the current content
    of the closure; ``await invoke(rows_by_name, supervisor)`` injects the
    closure, installs wrappers when a supervisor is given, and calls the root;
    ``make_passes()`` builds the run's :class:`VerifierPasses`; ``await
    repair(rewind)`` repairs the rewind's target or raises ``RepairRefused``;
    ``notify(message)`` delivers an owner-facing follow-up.

    A fully trusted closure runs with no supervisor: no wrappers, no verdict
    tasks, no barrier — an exception still gets the bounded repair loop.
    """
    max_attempts = 1 + int(settings.max_rewinds_per_run)
    outcome = EntrypointOutcome()
    supervisor: Optional[RunVerificationSupervisor] = None
    last_exception: Optional[BaseException] = None
    repair_counts: Dict[str, int] = {}
    delivered_early = False
    memo: Dict[tuple, Any] = {}
    spot_check_failures: List[tuple[PendingVerdict, Verdict]] = []

    async def _settle_spot_checks(fm: Any) -> None:
        """A failed spot check puts the function back on the ramp and tells the owner."""
        while spot_check_failures:
            pending, verdict = spot_check_failures.pop(0)
            fm.invalidate_trust([int(pending.row["function_id"])])
            if notify is not None:
                await notify(
                    f"Spot check of {task_name}: {pending.frame.name} ran but failed "
                    f"verification afterwards ({verdict.reason}). It was not repeated; "
                    "the function is back under verification until it earns trust again.",
                )

    async def _repair_for(rewind: RewindRequested) -> Optional[HeldOutcome]:
        """Repair the rewind's target, escalating one frame on a repeat; None on success."""
        target_name = rewind.target_name
        if target_name is not None and repair_counts.get(target_name, 0) >= 1:
            chain = list(rewind.frames)
            index = next(
                (i for i, f in enumerate(chain) if f.name == target_name),
                None,
            )
            if index is not None and index > 0:
                parent = chain[index - 1]
                rewind = RewindRequested(
                    target_function_id=parent.function_id,
                    target_name=parent.name,
                    frames=rewind.frames,
                    verdict=rewind.verdict,
                    exception=rewind.exception,
                    ordinal=rewind.ordinal,
                )
                target_name = parent.name
        if target_name is not None:
            repair_counts[target_name] = repair_counts.get(target_name, 0) + 1
        try:
            await repair(rewind)
        except RepairRefused as refused:
            held = HeldOutcome(
                code="deployment_owned_function_failed",
                leaf_name=target_name,
                reason=str(refused),
            )
            held.message = held_message(task_name, held)
            return held
        return None

    for attempt in range(1, max_attempts + 1):
        outcome.attempts = attempt
        root_row, rows_by_name = resolve()
        if not closure_needs_supervision(rows_by_name, settings):
            try:
                outcome.result = await invoke(rows_by_name, None)
                return outcome
            except (RewindRequested, HoldRequested):
                raise
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_exception = exc
                if attempt == max_attempts:
                    raise
                target = _innermost_stored_function(exc) or str(root_row.get("name"))
                target_row = rows_by_name.get(target) or root_row
                rewind = RewindRequested(
                    target_function_id=int(target_row["function_id"]),
                    target_name=str(target_row.get("name")),
                    frames=(),
                    exception=exc,
                )
                outcome.rewinds += 1
                held = await _repair_for(rewind)
                if held is not None:
                    outcome.held = held
                    return outcome
                continue

        passes = make_passes()
        supervisor = RunVerificationSupervisor(
            passes=passes,
            settings=settings,
            root_row=root_row,
            rows_by_name=rows_by_name,
            executor=executor,
            memo=memo,
            verdict_counts=outcome.verdict_counts,
        )
        loop = asyncio.get_running_loop()
        entry_task = asyncio.current_task()
        supervisor.begin_attempt(entry_task, loop)  # type: ignore[arg-type]
        supervisor.on_spot_check_fail = (
            lambda pending, verdict: spot_check_failures.append(
                (pending, verdict),
            )
        )
        rewind: Optional[RewindRequested] = None
        try:
            result = await invoke(rows_by_name, supervisor)
        except asyncio.CancelledError:
            if supervisor.rewind is None:
                supervisor.cancel_all(reason="cancelled")
                raise
            entry_task.uncancel()  # type: ignore[union-attr]
            rewind = supervisor.rewind
        except RewindRequested as requested:
            entry_task.uncancel()  # type: ignore[union-attr]
            rewind = supervisor.rewind or requested
        except HoldRequested as hold:
            supervisor.cancel_all(reason="cancelled")
            outcome.held = _held_from(hold, task_name=task_name)
            outcome.verifier_tasks += supervisor.tasks_created
            return outcome
        except Exception as exc:
            last_exception = exc
            supervisor.cancel_all(reason="cancelled")
            outcome.verifier_tasks += supervisor.tasks_created
            if attempt == max_attempts:
                raise
            target = _innermost_stored_function(exc) or str(root_row.get("name"))
            target_row = rows_by_name.get(target) or root_row
            rewind = RewindRequested(
                target_function_id=int(target_row["function_id"]),
                target_name=str(target_row.get("name")),
                frames=current_verification_frames.get(),
                exception=exc,
            )
        else:
            if settings.deliver_before_root_verdict and not delivered_early:
                delivered_early = True
                outcome.result = result

                async def _follow_up(
                    sup: RunVerificationSupervisor = supervisor,
                ) -> None:
                    """Finish the verdicts; on a FAIL, repair, re-run without early delivery, notify."""
                    await sup.drain()
                    outcome.verifier_tasks += sup.tasks_created
                    if sup.rewind is None:
                        return
                    corrected = await _continue_after_early_delivery(sup)
                    if corrected is not None and notify is not None:
                        await notify(corrected)

                async def _continue_after_early_delivery(
                    sup: RunVerificationSupervisor,
                ) -> Optional[str]:
                    rewind_local = sup.rewind
                    held = await _repair_for(rewind_local)
                    if held is not None:
                        return held.message
                    inner = await run_verified_entrypoint(
                        settings=_without_early_delivery(settings),
                        task_name=task_name,
                        resolve=resolve,
                        invoke=invoke,
                        make_passes=make_passes,
                        repair=repair,
                        notify=notify,
                        executor=executor,
                    )
                    outcome.rewinds += 1 + inner.rewinds
                    if inner.held is not None:
                        return inner.held.message
                    return correction_message(
                        task_name,
                        delivered_at=utcnow().isoformat(timespec="seconds"),
                        reason=(
                            rewind_local.verdict.reason
                            if rewind_local.verdict
                            else "verification failed"
                        ),
                        result=inner.result,
                    )

                outcome.follow_up = asyncio.create_task(_follow_up())
                outcome.verifier_tasks += supervisor.tasks_created
                return outcome

            await supervisor.drain()
            outcome.verifier_tasks += supervisor.tasks_created
            await _settle_spot_checks(passes.fm)
            if supervisor.rewind is None:
                outcome.result = result
                return outcome
            rewind = supervisor.rewind

        # A rewind: settle what is still pending, then repair and try again.
        await supervisor.drain()
        outcome.verifier_tasks += supervisor.tasks_created
        if attempt == max_attempts:
            held = HeldOutcome(
                code="exhausted",
                leaf_name=rewind.target_name if rewind else None,
                reason=(
                    rewind.verdict.reason
                    if rewind and rewind.verdict
                    else (
                        str(rewind.exception)
                        if rewind and rewind.exception
                        else "rewind budget exhausted"
                    )
                ),
            )
            held.message = held_message(task_name, held)
            outcome.held = held
            return outcome
        outcome.rewinds += 1
        held = await _repair_for(rewind)
        if held is not None:
            outcome.held = held
            return outcome
        for row in rows_by_name.values():
            passes.forget_hash(int(row["function_id"]))

    held = HeldOutcome(
        code="exhausted",
        leaf_name=None,
        reason="rewind budget exhausted",
    )
    held.message = held_message(task_name, held)
    outcome.held = held
    return outcome


class _SettingsView:
    """Read-through view of settings with ``deliver_before_root_verdict`` forced off."""

    def __init__(self, base: Any) -> None:
        self._base = base

    def __getattr__(self, name: str) -> Any:
        if name == "deliver_before_root_verdict":
            return False
        return getattr(self._base, name)


def _without_early_delivery(settings: Any) -> Any:
    return _SettingsView(settings)
