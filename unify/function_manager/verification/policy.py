"""Trust policy: derive ``Function.verify`` from ledger evidence.

``derive_verify`` is the only writer of ``verify``. It is a pure function of
the row and the settings; the model that produced or repaired the code never
grants trust, and the librarian's policy can only raise the bar.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..settings import VerificationSettings
from ..types.verification import (
    SideEffectClass,
    StaticReviewRecord,
    VerdictKind,
    VerificationPolicy,
    VerificationSummary,
)


def _coerce_class(value: Any) -> SideEffectClass:
    if isinstance(value, SideEffectClass):
        return value
    return SideEffectClass(str(value or SideEffectClass.unsafe_effectful))


def _coerce_policy(value: Any) -> VerificationPolicy:
    if isinstance(value, VerificationPolicy):
        return value
    return VerificationPolicy.model_validate(value or {})


def _coerce_summary(value: Any) -> VerificationSummary:
    if isinstance(value, VerificationSummary):
        return value
    return VerificationSummary.model_validate(value or {})


def _coerce_static_review(value: Any) -> Optional[StaticReviewRecord]:
    if value is None or isinstance(value, StaticReviewRecord):
        return value
    return StaticReviewRecord.model_validate(value)


def policy_class(fn: Mapping[str, Any]) -> SideEffectClass:
    """The class policy applies to ``fn``: the effective class, or unsafe while an
    unconfirmed third-party inference stands."""
    effective = _coerce_class(fn.get("side_effect_class"))
    if fn.get("class_source") == "inferred_third_party":
        return SideEffectClass.unsafe_effectful
    return effective


def required_passes(fn: Mapping[str, Any], settings: VerificationSettings) -> int:
    """LLM ``args``/``post`` passes each required before trust (policy may raise)."""
    klass = policy_class(fn)
    base = int(settings.required_passes.get(klass, 0))
    override = _coerce_policy(fn.get("verification_policy")).required_passes
    if override is not None:
        return max(base, int(override))
    return base


def min_distinct_inputs(fn: Mapping[str, Any], settings: VerificationSettings) -> int:
    klass = policy_class(fn)
    base = int(settings.min_distinct_inputs.get(klass, 1))
    override = _coerce_policy(fn.get("verification_policy")).min_distinct_inputs
    if override is not None:
        return max(base, int(override))
    return base


def spot_check_rate(fn: Mapping[str, Any], settings: VerificationSettings) -> float:
    """Sampling rate for post probes on trusted effectful calls without an output contract."""
    if not settings.enabled:
        return 0.0
    klass = policy_class(fn)
    if not klass.is_effectful:
        return 0.0
    contract = fn.get("contract") or {}
    has_output_schema = (
        contract.get("output_schema") is not None
        if isinstance(contract, dict)
        else getattr(contract, "output_schema", None) is not None
    )
    base = 0.0 if has_output_schema else float(settings.spot_check_rate.get(klass, 0.0))
    override = _coerce_policy(fn.get("verification_policy")).spot_check_rate
    if override is not None:
        return max(base, float(override))
    return base


def uses_llm_passes(fn: Mapping[str, Any], settings: VerificationSettings) -> bool:
    """Whether the class runs LLM passes at all (``safe_noop`` relies on tier-0)."""
    return required_passes(fn, settings) > 0


def derive_verify(
    fn: Mapping[str, Any],
    *,
    settings: VerificationSettings,
    current_hash: str,
) -> bool:
    """Return the ``verify`` flag for ``fn`` given its current trust hash.

    With the master switch off (``settings.enabled``), always ``False``.
    Trusted (``verify=False``) iff the summary applies to the current hash, the
    static review passed, ``args`` and ``post`` passes (or ``tier0`` for
    ``safe_noop``) meet the class requirement after policy raises, enough
    distinct inputs were seen, no verdict for this hash failed, and the
    policy does not pin verification on forever.
    """
    if not settings.enabled:
        # The master switch outranks every per-function pin: off means the
        # verification subsystem does not exist, not that it defaults low.
        return False
    policy = _coerce_policy(fn.get("verification_policy"))
    if policy.always_verify:
        return True
    if fn.get("verified_hash") != current_hash:
        return True
    static = _coerce_static_review(fn.get("static_review"))
    if (
        static is None
        or static.verdict != "PASS"
        or static.function_hash != current_hash
    ):
        return True
    summary = _coerce_summary(fn.get("ledger"))
    if summary.fails > 0:
        return True
    needed = required_passes(fn, settings)
    if needed > 0:
        if summary.pass_count(VerdictKind.args) < needed:
            return True
        if summary.pass_count(VerdictKind.post) < needed:
            return True
    else:
        if summary.pass_count(VerdictKind.tier0) < 1:
            return True
    if len(summary.distinct_args_signatures) < min_distinct_inputs(fn, settings):
        return True
    return False
