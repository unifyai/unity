from pydantic import Field, field_validator
from typing import List, Optional, Dict, Any, Literal

from unify.common.authorship import AuthoredRow
from unify.common.stale_reason import StaleReason, coerce_stale_reasons

from .verification import (
    ClassSource,
    Fixture,
    FunctionContract,
    SideEffectClass,
    StaticReviewRecord,
    VerificationPolicy,
    VerificationSummary,
)


class Function(AuthoredRow):
    """
    Represents a function stored in the FunctionManager.

    Functions can be written in multiple languages (Python, Bash, Zsh, Sh, PowerShell)
    and can be either user-defined (with implementation source code) or primitives
    (action methods from state managers with no stored implementation).
    """

    function_id: Optional[int] = Field(
        None,
        description=(
            "Unique identifier for the function. "
            "Auto-assigned for user functions, explicit stable IDs for primitives."
        ),
    )
    language: Literal["python", "bash", "zsh", "sh", "powershell"] = Field(
        "python",
        description=(
            "The language/interpreter for this function. "
            "Defaults to 'python' for backward compatibility."
        ),
        json_schema_extra={"ui_editable": True},
    )
    name: str = Field(
        ...,
        description="The name of the function.",
        json_schema_extra={"ui_editable": True},
    )
    argspec: str = Field(
        ...,
        description=(
            "The function's signature. Format varies by language: "
            "Python: '(x: int, y: int) -> int'. "
            "Shell: '(input_file output_file --verbose)' or positional description."
        ),
        json_schema_extra={"ui_editable": True},
    )
    docstring: str = Field(
        "",
        description="The docstring of the function.",
        json_schema_extra={"ui_editable": True},
    )
    implementation: Optional[str] = Field(
        None,
        description=(
            "The full source code of the function. "
            "None for primitives (implementation lives in Python class)."
        ),
        json_schema_extra={"ui_editable": True},
    )
    depends_on: List[str] = Field(
        default_factory=list,
        description=(
            "Functions this function depends on, auto-detected from the AST "
            "at storage time. Bare names (e.g. 'helper') are compositional "
            "functions. Dotted names (e.g. 'primitives.contacts.ask') are "
            "environment namespaces; root segments resolve to fresh instances."
        ),
    )
    stale_reasons: List[StaleReason] = Field(
        default_factory=list,
        description="Structured records for declared dependencies that no longer resolve.",
    )
    embedding_text: str = Field(
        ...,
        description="The text used to generate the function's embedding.",
    )
    precondition: Optional[Dict[str, Any]] = Field(
        None,
        description="A dictionary representing the state required before the function can be run, e.g., {'url': '...'}.",
        json_schema_extra={"ui_editable": True},
    )

    guidance_ids: List[int] = Field(
        default_factory=list,
        description=(
            "List of Guidance.guidance_id values that reference this function; "
            "represents the inverse many-to-many relationship."
        ),
        json_schema_extra={"ui_editable": True},
    )

    side_effect_class: SideEffectClass = Field(
        SideEffectClass.unsafe_effectful,
        description=(
            "Effective effect class used by verification policy: safe_noop, "
            "read_only, idempotent_effectful or unsafe_effectful. Equals the "
            "detected lower bound unless a librarian confirmed a class."
        ),
    )
    side_effect_class_detected: SideEffectClass = Field(
        SideEffectClass.unsafe_effectful,
        description=(
            "Lower bound on the effect class derived deterministically from the "
            "AST: the maximum over primitives called, dependencies' classes and "
            "third-party imports."
        ),
    )
    class_source: ClassSource = Field(
        "inferred_third_party",
        description=(
            "Where the effective class came from: 'pure' (no I/O in the source), "
            "'primitives' (from the primitives called), 'inferred_third_party' "
            "(third-party imports raised the bound; treated as unsafe_effectful "
            "until confirmed) or 'librarian' (a librarian confirmed it)."
        ),
    )
    class_rationale: Optional[str] = Field(
        None,
        description="Rationale recorded when a librarian confirmed, raised or lowered the class.",
    )
    verification_policy: VerificationPolicy = Field(
        default_factory=VerificationPolicy,
        description="Librarian-set overrides that can only raise the trust bar.",
    )
    verified_hash: Optional[str] = Field(
        None,
        description=(
            "Trust hash the ledger summary applies to (source, dependencies, venv "
            "and language). None when the function has never been verified under "
            "its current content."
        ),
    )
    static_review: Optional[StaticReviewRecord] = Field(
        None,
        description="Cached static-review verdict for verified_hash.",
    )
    ledger: VerificationSummary = Field(
        default_factory=VerificationSummary,
        description="Fold of Functions/Verifications rows for verified_hash.",
    )
    contract: FunctionContract = Field(
        default_factory=FunctionContract,
        description="Tier-0 contract: input/output JSON schemas and postconditions.",
    )
    fixtures: List[Fixture] = Field(
        default_factory=list,
        description=(
            "Recorded (args, result) pairs a safe_noop function must reproduce; "
            "replayed whenever the trust hash changes."
        ),
    )

    verify: bool = Field(
        True,
        description=(
            "Whether calls to this compositional function still run under "
            "verification. Derived by the trust policy from the ledger — never "
            "set by a tool, a prompt or the model that wrote the code. True "
            "while the function is on the ramp; False once accumulated verdicts "
            "for verified_hash meet the policy for its effect class. Any change "
            "to source, dependencies, venv or linked guidance sets it back to "
            "True. On primitive rows this field carries an unrelated meaning: "
            "for integration primitives it is the confirmation-required flag "
            "(confirmation_required, or a write/destructive/bulk_export action "
            "class), and the ledger does not apply to primitives."
        ),
    )

    # Primitive-specific fields
    is_primitive: bool = Field(
        False,
        description=(
            "Whether this is an action primitive (state manager method) rather than "
            "a user-defined function. Primitives have no stored implementation."
        ),
    )

    primitive_class: Optional[str] = Field(
        None,
        description="Fully-qualified class path for primitive execution routing.",
    )

    primitive_method: Optional[str] = Field(
        None,
        description="Method name on the primitive class.",
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Source-specific metadata for function subtypes. Provider-backed "
            "integration rows store catalogue-level tool metadata here; "
            "per-user connection state is resolved live and never stored on "
            "Function rows."
        ),
    )

    venv_id: Optional[int] = Field(
        None,
        description=(
            "VirtualEnv.venv_id for the Python virtual environment to use when "
            "executing this function. Only applies when language='python'. "
            "If None, uses the project's default environment."
        ),
        json_schema_extra={"ui_editable": True},
    )

    windows_os_required: bool = Field(
        False,
        description=(
            "Whether this function requires Windows OS execution. When True "
            "and desktop_mode='windows', routes to the remote Windows VM. "
            "Used for functions depending on Windows-only libraries like xlwings."
        ),
        json_schema_extra={"ui_editable": True},
    )

    # ── Usage trace: the memory strength behind activation-weighted
    # retrieval (see function_manager/activation.py). Written fire-and-
    # forget at the execution choke points; read at query time by
    # search_functions to compute standing. Never consulted by execution
    # itself — a dormant function still runs when addressed directly.
    created_at: Optional[str] = Field(
        None,
        description=(
            "ISO timestamp of the row's creation. Creation counts as the "
            "function's first activation event (the newborn grace), so a "
            "fresh function surfaces long enough to earn its first call."
        ),
    )
    usage_calls: int = Field(
        0,
        description="Total recorded invocations across all execution paths.",
    )
    usage_last_called_at: Optional[str] = Field(
        None,
        description="ISO timestamp of the most recent recorded invocation.",
    )
    usage_recent_calls: List[str] = Field(
        default_factory=list,
        description=(
            "Bounded window of recent invocation timestamps (ISO), kept for "
            "rhythm estimation — decay runs against the function's own "
            "median inter-use interval, not a global half-life."
        ),
    )
    usage_search_hits: int = Field(
        0,
        description=(
            "How often search surfaced this function. Retrieved-but-never-"
            "called is a near-miss relevance signal, distinct from unused."
        ),
    )

    # Source-defined custom function tracking
    custom_key: Optional[str] = Field(
        None,
        description=(
            "Stable sync identity of a source-defined custom function. "
            "Defined to equal the function name (the call-site contract). "
            "None for user-added functions or primitives."
        ),
    )
    custom_hash: Optional[str] = Field(
        None,
        description=(
            "Hash of source-defined custom function for sync detection. "
            "None for user-added functions or primitives. "
            "Present for functions defined in the custom/ folder."
        ),
    )

    @field_validator("stale_reasons", mode="before")
    @classmethod
    def _validate_stale_reasons(cls, v):
        return coerce_stale_reasons(v)
