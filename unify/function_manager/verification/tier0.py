"""Tier-0 boundary: contract checks and fixture capture around a stored function.

Deterministic, microseconds, no model. Applied to every compositional
callable handed to a plan namespace or executed by name while the function
is untrusted, and kept on trusted ``read_only``/effectful functions when
``tier0_always`` is set. A trusted ``safe_noop`` function runs bare.
"""

from __future__ import annotations

import ast
import concurrent.futures
import functools
import inspect
import logging
from typing import Any, Callable, Dict, Mapping, Optional, Protocol

from ..settings import VerificationSettings
from ..types.verification import (
    SideEffectClass,
    Verdict,
    VerdictKind,
    VerificationRow,
)
from .contracts import check_input, check_output
from .fixtures import add_fixture, make_fixture
from .ledger import args_signature

logger = logging.getLogger(__name__)


class ContractViolation(RuntimeError):
    """A tier-0 check failed; the call did not (input) or must not stand (output)."""

    def __init__(self, *, function_name: str, verdict: Verdict) -> None:
        self.function_name = function_name
        self.verdict = verdict
        super().__init__(f"{function_name}: {verdict.reason}")


class LedgerWriter(Protocol):
    """What the boundary needs from the function store."""

    @property
    def verification_settings(self) -> VerificationSettings: ...

    def record_verification_nowait(
        self,
        row: VerificationRow,
    ) -> concurrent.futures.Future[None]: ...

    def function_trust_hash(self, fn: Dict[str, Any]) -> str: ...

    def capture_fixture_nowait(
        self,
        fn: Dict[str, Any],
        fixture_payload: dict,
    ) -> concurrent.futures.Future[None]: ...


def _class_of(row: Mapping[str, Any]) -> SideEffectClass:
    return SideEffectClass(
        str(row.get("side_effect_class") or SideEffectClass.unsafe_effectful),
    )


def tier0_applies(row: Mapping[str, Any], settings: VerificationSettings) -> bool:
    """Whether tier-0 checks run for a call of ``row`` right now."""
    if row.get("verify", True):
        return True
    return (
        bool(settings.tier0_always) and _class_of(row) is not SideEffectClass.safe_noop
    )


def input_verdict(
    row: Mapping[str, Any],
    kwargs: Mapping[str, Any],
) -> Optional[Verdict]:
    """FAIL (fault=caller) when ``kwargs`` violate the input schema, else None."""
    reason = check_input(row.get("contract"), kwargs)
    if reason is None:
        return None
    return Verdict(verdict="FAIL", reason=reason, fault="caller")


def output_verdict(
    row: Mapping[str, Any],
    *,
    result: Any,
    kwargs: Mapping[str, Any],
) -> Optional[Verdict]:
    """FAIL (fault=leaf) when ``result`` violates the output schema or a postcondition."""
    reason = check_output(row.get("contract"), result=result, kwargs=kwargs)
    if reason is None:
        return None
    return Verdict(verdict="FAIL", reason=reason, fault="leaf")


def signature_from_source(implementation: Optional[str]) -> Optional[inspect.Signature]:
    """Signature of the single top-level function in ``implementation`` (names only)."""
    if not implementation:
        return None
    try:
        node = ast.parse(implementation).body[0]
    except (SyntaxError, IndexError):
        return None
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    params: list[inspect.Parameter] = []
    for arg in node.args.posonlyargs:
        params.append(inspect.Parameter(arg.arg, inspect.Parameter.POSITIONAL_ONLY))
    positional = list(node.args.args)
    defaults = [None] * (len(positional) - len(node.args.defaults)) + list(
        node.args.defaults,
    )
    for arg, default in zip(positional, defaults):
        params.append(
            inspect.Parameter(
                arg.arg,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=inspect.Parameter.empty if default is None else None,
            ),
        )
    if node.args.vararg is not None:
        params.append(
            inspect.Parameter(node.args.vararg.arg, inspect.Parameter.VAR_POSITIONAL),
        )
    for arg in node.args.kwonlyargs:
        params.append(
            inspect.Parameter(arg.arg, inspect.Parameter.KEYWORD_ONLY, default=None),
        )
    if node.args.kwarg is not None:
        params.append(
            inspect.Parameter(node.args.kwarg.arg, inspect.Parameter.VAR_KEYWORD),
        )
    try:
        return inspect.Signature(params)
    except ValueError:
        return None


def bind_call_kwargs(
    signature: Optional[inspect.Signature],
    args: tuple,
    kwargs: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """Name positional arguments so contracts see one shape.

    Returns None when positional arguments cannot be named; a check against
    keyword arguments alone would then report spurious missing-parameter
    failures, so callers skip the input check instead.
    """
    if not args:
        return dict(kwargs)
    if signature is None:
        return None
    try:
        return dict(signature.bind_partial(*args, **kwargs).arguments)
    except TypeError:
        return None


class Tier0Checker:
    """Stateless per-function tier-0 checks that write their verdicts to the ledger."""

    def __init__(
        self,
        *,
        row: Mapping[str, Any],
        writer: LedgerWriter,
        call_site: str = "root",
        run_key: Optional[str] = None,
        task_id: Optional[int] = None,
    ) -> None:
        self.row = dict(row)
        self.writer = writer
        self.call_site = call_site
        self.run_key = run_key
        self.task_id = task_id
        self.name = str(self.row.get("name") or "function")

    @property
    def active(self) -> bool:
        return tier0_applies(self.row, self.writer.verification_settings)

    def _record(self, verdict: Verdict, signature: str) -> None:
        self.writer.record_verification_nowait(
            VerificationRow(
                function_id=int(self.row["function_id"]),
                function_hash=None,
                kind=VerdictKind.tier0,
                verdict=verdict.verdict,
                reason=verdict.reason,
                fault=verdict.fault,
                call_site=self.call_site,
                args_signature=signature,
                run_key=self.run_key,
                task_id=self.task_id,
            ),
        )

    def check_input(self, kwargs: Mapping[str, Any]) -> None:
        """Raise ``ContractViolation`` when the arguments break the input contract."""
        verdict = input_verdict(self.row, kwargs)
        if verdict is None:
            return
        self._record(verdict, args_signature(kwargs))
        raise ContractViolation(function_name=self.name, verdict=verdict)

    def check_output(self, *, result: Any, kwargs: Mapping[str, Any]) -> None:
        """Record the tier-0 outcome for the call; raise on an output/postcondition failure."""
        signature = args_signature(kwargs)
        verdict = output_verdict(self.row, result=result, kwargs=kwargs)
        if verdict is not None:
            self._record(verdict, signature)
            raise ContractViolation(function_name=self.name, verdict=verdict)
        if self.row.get("verify", True):
            # Passes matter only while the function is on the ramp; a trusted
            # function's tier-0 pass is not evidence anyone needs to keep.
            self._record(
                Verdict(verdict="PASS", reason="tier-0 contract satisfied"),
                signature,
            )
            if _class_of(self.row) is SideEffectClass.safe_noop:
                self._capture(kwargs, result)

    def _capture(self, kwargs: Mapping[str, Any], result: Any) -> None:
        settings = self.writer.verification_settings
        fixture = make_fixture(
            args=kwargs,
            result=result,
            max_bytes=settings.max_fixture_bytes,
            run_key=self.run_key,
        )
        if fixture is None:
            return
        existing = list(self.row.get("fixtures") or [])
        if any(
            item.get("args_signature") == fixture.args_signature for item in existing
        ):
            return
        merged = add_fixture(existing, fixture, cap=settings.max_fixtures_per_function)
        self.row["fixtures"] = merged
        self.writer.capture_fixture_nowait(self.row, {"fixtures": merged})


def tier0_boundary(
    inner: Callable[..., Any],
    *,
    raw: Optional[Callable[..., Any]],
    checker: Tier0Checker,
    signature: Optional[inspect.Signature] = None,
) -> Callable[..., Any]:
    """Wrap ``inner`` (a namespace callable) with tier-0 checks.

    ``raw`` is the underlying function used for the signature and asyncness;
    the wrapper is a plain ``def``/``async def`` so ``inspect`` reports the
    same shape as the function it stands for. ``signature`` overrides the
    one read from ``raw`` (venv proxies have none). The wrapper exposes
    ``__tier0_inner__`` so a fuller verification wrapper can take over the
    call without running the checks twice.
    """
    if signature is None and raw is not None:
        try:
            signature = inspect.signature(raw)
        except (TypeError, ValueError):
            signature = None
    target = raw if raw is not None else inner

    if raw is not None and inspect.iscoroutinefunction(raw):

        @functools.wraps(target)
        async def _async_call(*args: Any, **kwargs: Any) -> Any:
            if not checker.active:
                return await inner(*args, **kwargs)
            named = bind_call_kwargs(signature, args, kwargs)
            if named is not None:
                checker.check_input(named)
            result = await inner(*args, **kwargs)
            checker.check_output(
                result=result,
                kwargs=named if named is not None else dict(kwargs),
            )
            return result

        _async_call.__tier0_inner__ = inner  # type: ignore[attr-defined]
        _async_call.__tier0_checker__ = checker  # type: ignore[attr-defined]
        return _async_call

    @functools.wraps(target)
    def _call(*args: Any, **kwargs: Any) -> Any:
        if not checker.active:
            return inner(*args, **kwargs)
        named = bind_call_kwargs(signature, args, kwargs)
        if named is not None:
            checker.check_input(named)
        seen = named if named is not None else dict(kwargs)
        result = inner(*args, **kwargs)
        if inspect.isawaitable(result):

            async def _finish() -> Any:
                value = await result
                checker.check_output(result=value, kwargs=seen)
                return value

            return _finish()
        checker.check_output(result=result, kwargs=seen)
        return result

    _call.__tier0_inner__ = inner  # type: ignore[attr-defined]
    _call.__tier0_checker__ = checker  # type: ignore[attr-defined]
    return _call
