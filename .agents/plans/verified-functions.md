# Plan: verified functions — earning the compiled steady state

**Status:** Phases 0–9 landed. In `unify`: schema, classification, ledger,
policy, settings, backfill; tier-0 contracts and fixtures; verifier passes and
prompt builders; run supervisor with barrier/memo/rewind/held; trust flips,
invalidation, spot checks and offline promotion; librarian tools and prompt;
purpose-tagged token accounting; docs. In `colleague` (branch
`verified-functions`, commit `1e3e9ac`): the fire-series engine, the
`silent_drift` / `edge_week` / `repair_locality` /
`change_without_regression` experiments, the six-week `teaching` extension,
tokens by purpose in the unify arm's ledger, `plot_distillation_curve.py`,
README/DESIGN updates including the "held scores below correct" rule — all
self-testing. First live runs recorded (colleague `614b7b4`): the hermes arm
on all four experiments. The unify arm's pre-fix round is measured and
committed (colleague `2c8f84b`, `f1764e5`, `faff798` and the standing
results sweep): edge_week 2/10 on all four variants, cwr 8/12, rl 4/20,
sd_units 2/20, sd_page 8/20. Three defects account for every
verifier-attributable loss, all fixed: verifier verdicts returned as
schema-shaped tool calls parsed as UNSURE and silently held every fire
after the first (unillm `284b459`); a held run was invisible on the task
handle so holds scored wrong instead of held (`held_outcome`/`run_stats`
now public); and the precondition probe vetoed scheduler-dispatched runs
by reasoning about the wall clock ("not due yet") — its prompt now names
due-ness as the scheduler's authority alone. The tier-0 contracts caught
injected drift exactly as designed (rl fires 5–6). Post-fix re-measurement is deliberately
deferred while the verification implementation evolves (a rerun round was
started 2026-08-21 and cancelled); when it resumes, its open questions
are: does repair engage once verdicts parse (rl pre-fix shows
`repair_tokens=0` through six post-drift fires), and does sd_page's
silent wrong-batch delivery — the one true capability loss — persist.
Full pre-fix evidence: colleague `2c8f84b`, `f1764e5`, `faff798`,
`a8f0479`.
Orchestra treats `held` as terminal (`07641c6b`). Written against
`staging` @ `5ec185ada` (2026-08-17). Line numbers below are approximate
as of that commit; symbols are exact. Re-verify both before editing.

**Conversational arm (2026-08-21).** The refinement track showed the
compiled steady state existed only on the cron-fired path: recurring work
driven turn-by-turn through one persist=True `act` session never engaged
distillation, because the post-run review only fires when a session
completes and a persist loop never self-completes — six structurally
identical weekly requests replanned live every week (9–13 calls,
368k–718k prompt tokens/week, rising ~75k/week; unifyai/colleague GH runs
32509789685 / 32508329275 / 32512181620). Fixed in `f1b21a1b5`: the
persist-mode turn boundary is now a review boundary. `_StorageCheckHandle`
runs the librarian after each completed turn that ran tools (serialized,
coalescing), records each summary as a proactive pass for the final
on-stop review, and delivers it into the live loop as a `_transcript_note`
— a loop-authored user notice appended at the safe drain points without
granting an LLM turn. The review prompt gains a live-session framing plus
a "Recurring Deliverables Without A Task" section (the conversational
analogue of the entrypoint review: stated-but-dormant rules belong in the
stored function, refinements are `overwrite=True` edits, summaries name
function ids so the next request executes the stored function), and the
persist-mode actor prompt now states the per-turn schedule and the
convergence contract. Three follow-up commits flattened the session
curve: `d8173f138` (reviewed turns shed their tool payloads in place via
a `_compact_transcript` sentinel + the no-op review concludes in one
sentence + repeat requests are one execution and a report),
`ec1c4ce25` (the reviewed-span compactor also strips provider reasoning
payloads — encrypted blobs and summaries were the dominant transcript
bulk, re-billed every dispatch and unreadable in the librarian prompt),
and `ec767a6c0` (a persist loop sheds completed-turn reasoning at every
park, without waiting for the review to cover the span). Four live
refinement rounds from these commits (results in
`colleague/tracks/refinement/results/2026-08-21T*-unify-*`): every round
6/6 weeks pass + control UNSUPPORTED-as-designed — the week-4 paraphrase
and week-5 dormant-rule guards held in all four, and the stored skeleton
fired its scoped `query_llm` joints (~0.5k tokens each) every round.
Cost: run totals 7.93M → 5.29M prompt tokens; week 1 1.02M → 259k;
replay weeks ~420–580k including the librarian; the session context is
now sawtooth-bounded (~45–90k) instead of rising without limit. What
remains above the cron path's few-thousand-token steady state is the
conversational surface's floor — each turn still pays the ~16k system
prompt across ~6–10 framing calls plus the turn review — and trap/
amendment weeks where the model chooses to re-verify semantics
(1.4–2.8M on one such week per round, which week varying with
sampling).

**Thesis.** A recurring task's steady state is a stored function firing with
no model in the loop. Today the harness reaches that state by one librarian
judgement after one successful run, with no independent check that the
function does what its docstring says, no evidence that it keeps doing so,
no protection against silent drift, and no way to know which leaf to repair.
This plan makes the compiled state *earned*: every stored function carries a
verification ledger; verifier passes (deterministic first, LLM second) run
before and after each call while the function is untrusted; trust is granted
by a deterministic policy from accumulated evidence, never by the LLM that
wrote the code; trust is lost automatically on any change; and all of the
machinery is transient — once a function has earned `verify=False`, calls
cost nothing and block nothing.

The design lineage is `unity/actor/hierarchical_actor.py` @ `7e80ce201^`
(purged 2026-02-07): `@verify` at every function boundary, async
`pending_verifications`, cancel-later-and-replay on failure, verified
functions skipped on replay, storage gated on verification. Reuse the ideas,
not the monolith.

Read first: `unify/function_manager/types/function.py`,
`unify/function_manager/function_manager.py` (`add_functions`,
`execute_function`, `filter_functions(_return_callable=True)`),
`unify/actor/code_act_actor.py` (`_CodeActEntrypointHandle`, the
`if entrypoint is not None:` branch of `act()`, `_repair_symbolic_entrypoint`,
`_StorageCheckHandle._run_lifecycle`, the `_STORAGE_*` prompts,
`attach_entrypoint_to_recurring_task`, `submit_offline_certification_evidence`),
`unify/actor/single_function_actor.py` (`_verify_execution`),
`unify/task_scheduler/task_scheduler.py` (`_build_task_entrypoint_review`,
`_attach_entrypoint_to_definition`, `_offline_promotion_rejection_reasons`,
`_promote_definition_to_offline`, `execute`),
`unify/task_scheduler/types/{task,execution}.py`,
`unify/function_manager/steering.py` (precedent for AST instrumentation),
`.agents/rules/product-vocabulary.md`,
`.agents/rules/test-philosophy-symbolic-vs-eval.md`, `tests/README.md`.

---

## 1. Goals, non-goals, and what is deliberately left out

**Goals**

1. Every compositional `Function` row carries a verification ledger and an
   effect class; `Function.verify` becomes a *derived* flag.
2. Four verification passes exist — static review (per source hash),
   argument review, precondition probe, post-execution probe — plus tier-0
   deterministic checks (contracts, fixtures). Which passes apply is decided
   by effect class.
3. Verification never serialises task code except at one place: an
   irreversible effect does not execute while a verdict it depends on is
   pending. Everything else is asynchronous.
4. A failed verdict rewinds and repairs *the leaf that failed*, without
   re-executing effects that already ran, bounded by a rewind budget.
5. Trust flips by deterministic policy from ledger evidence; the librarian
   can raise the bar, never lower it below the detected effect class.
6. Trust is invalidated automatically on source change, dependency change,
   venv change, linked-guidance change, verdict failure, or spot-check failure.
7. Offline promotion is derived from the ledger (verified closure), replacing
   the self-attested certification evidence.
8. Token accounting distinguishes planning, verification and repair so the
   distillation curve can be plotted.
9. The `colleague` benchmark gains the scenarios that exercise all of the
   above and the plot that shows it.

**Non-goals**

- Verifying primitives (`is_primitive=True`). Their `verify` field today
  means "confirmation required" for integration primitives
  (`function_manager.py` ~L2399: `confirmation_required or action_class in
  {"write","destructive","bulk_export"}`); that overload is left as is and
  the ledger applies to compositional functions only. Record the overload in
  the field docstring so nobody conflates the two.
- Rewinding GUI/desktop state. The old course-correction sub-agent is not
  revived. Compositional functions that drive a desktop are classified
  `unsafe_effectful` and get the blocking pre-pass; nothing more.
- Human approval flows. Tier-2 (a person confirms the first real firing of an
  irreversible effect) is the existing integrations approval mechanism
  (`unify/integrations/approval.py`, `confirmation_required`). No new
  approval UI is built here.

**Explicitly out of scope, tracked elsewhere** (named so they are not
silently dropped): a Prime Agent arm for `colleague`; Console UI for the
verification ledger (Functions rail); training-data export of verdicts.

---

## 2. Vocabulary (per `.agents/rules/product-vocabulary.md`)

- **task** — the durable recurring unit (`Task` row). Not "workflow"; a
  workflow is an installable package.
- **function** — a `Function` row in `Functions/Compositional`. Never "skill"
  for one function.
- **entrypoint** — the function bound to a task (`Task.entrypoint`), also
  called the **root** of the run.
- **leaf** — any compositional function invoked (transitively) during a run,
  including the root.
- **verdict** — the outcome of one verification pass on one call:
  `PASS | FAIL | UNSURE`.
- **ledger** — the per-function evidence summary plus the append-only
  `Functions/Verifications` rows.
- **trusted** — `verify=False`. **untrusted / on the ramp** — `verify=True`.
- **symbolic run** — an entrypoint execution that bypasses the CodeAct LLM
  loop. **agentic run** — a CodeAct loop.

---

## 3. Current state (what exists, what it does)

- `Function.verify: bool` (`types/function.py` ~L91) exists, default `True`,
  honoured only by `SingleFunctionActor` (`single_function_actor.py` ~L904-961,
  `_verify_execution` ~L602: a one-shot post-hoc LLM check with
  `success/reason`). `SingleFunctionActor` has no production call sites
  outside `unify/actor/__init__.py`, `manager_registry.py`,
  `comms/primitives.py` imports — confirm before deleting anything.
- `Function.precondition: Optional[Dict]` exists and is never checked at run
  time (the old `_ensure_precondition` was already commented out).
- `Function.custom_hash` / `custom_key` are the source-sync identity for
  `custom/` functions; `stale_reasons: List[StaleReason]` records unresolved
  dependencies; `depends_on` is AST-derived
  (`dependency_analysis.collect_dependencies_from_source`).
- `add_functions` (`function_manager.py` ~L4157-4407) parses, collects deps,
  rejects third-party imports without `venv_id`, blocks dangerous builtins,
  `exec`s the source and takes `inspect.signature`. It never calls the
  function.
- Entrypoint execution: `CodeActActor.act(entrypoint=…)`
  (`code_act_actor.py` ~L5433-5525) — `_run_entrypoint_once` injects the
  function and its `depends_on` closure into `sandbox.global_state` via
  `fm.filter_functions(_return_callable=True, _namespace=…)`, calls it,
  returns the raw result; `_run_entrypoint` retries once after
  `_repair_symbolic_entrypoint` (~L5066) when
  `entrypoint_repair_attempts > 0`; the result is wrapped in
  `_CodeActEntrypointHandle` (~L429) and **returned before the storage phase**
  — symbolic runs never reach the librarian.
- Repair: `_repair_symbolic_entrypoint` runs an LLM tool loop with FM CRUD and
  `run_diagnostic_probe` (read-only), overwrites in place keeping
  `function_id`; refuses deployment-owned (`custom_hash`) functions.
- Graduation: `TaskScheduler._build_task_entrypoint_review` (~L393) hands the
  librarian a review context only when `task.entrypoint is None` and the task
  repeats/triggers; the librarian tool `attach_entrypoint_to_recurring_task`
  writes `{"entrypoint": function_id}` after **one** run.
- Offline: `submit_offline_certification_evidence` →
  `_offline_promotion_rejection_reasons` (~L564) checks a self-attested
  evidence dict (`OFFLINE_CERTIFICATION_REQUIRED_EVIDENCE_FIELDS`,
  `…_ALLOWED_RISK_CLASSIFICATIONS = {safe_noop, read_only,
  idempotent_effectful, unsafe_effectful}`, `…_REQUIRED_ATTESTATIONS`) and
  "must not execute the entrypoint".
- Storage loop: `_StorageCheckHandle._run_lifecycle` runs the librarian after
  every non-failed agentic `act()` (`can_store` default `True`); prompts
  `_STORAGE_WHAT_CAN_BE_STORED / _THREE_STORES / _SUB_AGENT_PATTERNS /
  _BASE_INSTRUCTIONS`.
- Execution rows: `Tasks/Executions`, `ExecutionState` in
  `task_scheduler/types/execution.py` = `scheduled | triggerable | running |
  completed | failed | cancelled`.
- Integration primitives carry `action_class` in their metadata
  (`unify/integrations/primitives.py`); built-in state-manager primitives
  carry nothing equivalent.
- README.md L269 claims graduation happens "after enough successful runs";
  the code graduates after one.

---

## 4. Design

### 4.1 Effect classes

Reuse the existing offline-certification taxonomy as the single effect
vocabulary — do not introduce `pure/read/write`:

| `side_effect_class` | Meaning | Examples |
|---|---|---|
| `safe_noop` | Pure computation; no I/O beyond its arguments | parse, aggregate, format |
| `read_only` | Reads external state; no mutation | fetch orders, search web, query a table |
| `idempotent_effectful` | Mutates, but re-running with the same inputs converges to the same state | upsert row by key, write file at path, set a field |
| `unsafe_effectful` | Mutates non-idempotently or irreversibly | send message/email, delete, pay, post, run desktop actions |

**Detection is deterministic and yields a lower bound.** New module
`unify/function_manager/verification/classify.py`:

1. Walk the AST (reuse `dependency_analysis`) and collect every primitive
   call (dotted `primitives.*` / `computer_primitives.*` names) and every
   compositional dependency; class = max over primitives called and over
   dependencies' classes (transitive, cycles broken at the visited set).
2. Primitive classes come from a committed table in the same module,
   `PRIMITIVE_EFFECT_CLASSES: dict[str, SideEffectClass]`, keyed by the
   dotted primitive name as it appears in `Functions/Primitives`. The
   implementer must enumerate **every** primitive row in `Functions/Primitives`
   (`fm.list_functions(include_primitives=True)`) and classify each one
   explicitly. Integration primitives map from their `action_class`:
   `read → read_only`, `write → idempotent_effectful`, `destructive |
   bulk_export → unsafe_effectful`, missing → `unsafe_effectful`. Every
   `primitives.comms.send_*` is `unsafe_effectful`. Any primitive absent from
   the table classifies as `unsafe_effectful` **and** emits a `logger.error`
   at classification time so the gap is loud.
3. Third-party imports outside a small allowlist of pure libraries (`json`,
   `re`, `math`, `datetime`, `pandas`, `numpy`, `pydantic`, `dataclasses`,
   `typing`, `collections`, `itertools`, `functools`, `decimal`, `statistics`)
   raise the lower bound to `read_only`; `requests`, `httpx`, `aiohttp`,
   `smtplib`, `subprocess`, `os.system`, `shutil.rmtree`, `open(..., 'w')`
   raise it to `unsafe_effectful`. Keep the lists in `classify.py`; extending
   them is a normal code change.
4. Result: `side_effect_class_detected` (lower bound) and
   `side_effect_class` (effective). Effective = detected unless the librarian
   has confirmed a class; the librarian may **raise** the class freely and may
   lower it only down to the detected lower bound, with a recorded rationale.
   Until a librarian confirmation exists, functions whose detection relied on
   third-party imports (`class_source == "inferred_third_party"`) are treated
   as `unsafe_effectful` for policy purposes (safe default); functions whose
   detection came purely from primitives (`class_source == "primitives"`) or
   pure Python (`"pure"`) use the detected class directly.

### 4.2 Schema

**`Function` (`unify/function_manager/types/function.py`) — new fields, all
persisted in `Functions/Compositional`:**

```python
side_effect_class: SideEffectClass            # effective, see 4.1
side_effect_class_detected: SideEffectClass   # lower bound from AST
class_source: Literal["pure", "primitives", "inferred_third_party", "librarian"]
class_rationale: Optional[str]                # set when the librarian confirms/raises/lowers
verification_policy: VerificationPolicy       # librarian overrides, see 4.8
verified_hash: Optional[str]                  # hash the ledger summary applies to; None = never
static_review: Optional[StaticReviewRecord]   # verdict for verified_hash, see 4.5
ledger: VerificationSummary                   # counters for verified_hash, see 4.8
contract: FunctionContract                    # see 4.3
fixtures: List[Fixture]                       # see 4.4, capped
verify: bool                                  # DERIVED by policy; keep name, rewrite docstring
```

`verified_hash` = SHA-256 over: normalised `implementation`, the sorted
resolved `depends_on` names with *their* `verified_hash`es, `venv_id` and
the venv's `pyproject` hash, and `language`. Compute in
`unify/function_manager/verification/ledger.py::function_trust_hash(fm, fn)`.
Any component changing changes the hash. `custom_hash` stays what it is
(source-sync identity of `custom/` functions); do not overload it.

Types live in `unify/function_manager/types/verification.py`:
`SideEffectClass` (StrEnum, the four values), `VerificationPolicy`,
`VerificationSummary`, `StaticReviewRecord`, `FunctionContract`, `Fixture`,
`Verdict`, `VerdictKind` (`static | args | precondition | post | tier0 |
spot_check`), `VerificationRow`.

**New Orchestra context `Functions/Verifications`** — append-only rows, one
per verdict, defined next to the other table constants in
`function_manager.py` (`FUNCTIONS_VERIFICATIONS_TABLE`):

```
function_id, function_hash, kind (VerdictKind), verdict (PASS|FAIL|UNSURE),
reason (str, ≤ 2k chars), fault (leaf|caller|None), call_site (str: parent
function name or "root"), args_signature (sha256 of canonical kwargs),
run_key (Tasks/Executions.run_key or None), task_id (or None),
prompt_tokens, completion_tokens, cost, wall_ms, created_at
```

`VerificationSummary` on the function row is a fold over these for
`verified_hash`: `passes: dict[VerdictKind, int]`, `fails: int`,
`unsure: int`, `distinct_args_signatures: list[str]` (cap 32; store hashes
only), `last_verdict_at`, `spot_checks: int`. The fold is recomputed from rows
on write (single writer per run; last-writer-wins is acceptable — a lost
increment only delays trust, never grants it).

**`Task`** — no new fields. **`ExecutionState`** — add `held = "held"`:
the run finished without performing an irreversible effect because a verdict
it depended on was `FAIL`/`UNSURE`/timed out and the rewind budget was
exhausted or the class forbids proceeding; the owner is notified (4.7).
Everything that switches on `ExecutionState` must handle `held` (grep every
use).

### 4.3 Contracts (tier 0)

`FunctionContract`:

```python
input_schema: Optional[dict]        # JSON Schema for call kwargs
output_schema: Optional[dict]       # JSON Schema for the return value
postconditions: List[str]           # Python boolean expressions over `result`, `kwargs`
source: Literal["type_hints", "librarian", "none"]
```

- `input_schema` / `output_schema` are generated at store time from type
  hints where present (pydantic `TypeAdapter(...).json_schema()` on the
  signature and return annotation); the librarian is instructed to type-hint
  everything it stores (4.10). Where hints are absent the schema is `None`
  and the contract records `source="none"`.
- `postconditions` are authored by the librarian; each is compiled with
  `ast.parse(mode="eval")` at store time and rejected if it references
  anything but `result`, `kwargs`, builtins from a small allowlist
  (`len`, `all`, `any`, `isinstance`, `min`, `max`, `sum`, `abs`, `round`,
  `sorted`, `set`, `list`, `dict`, `str`, `int`, `float`, `bool`) and the
  standard comparison/boolean operators. Evaluate them in a fresh namespace
  with those names only.
- Tier-0 runs on **every** call while `verify=True` **and** on every call
  after trust for `read_only`/effectful classes when
  `FunctionSettings.verification.tier0_always=True` (default `True`; it is
  microseconds). Input validation runs before the call and its failure is a
  `FAIL` with `fault=caller`; output validation and postconditions run after
  and their failure is a `FAIL` with `fault=leaf`.
- Tier-0 verdicts are recorded as `kind=tier0`. They **count toward trust**
  only for `safe_noop` (see 4.8); for other classes they gate but do not
  substitute for LLM passes.

### 4.4 Fixtures

`Fixture = {args: dict, result: Any, args_signature: str, captured_at,
run_key}` — JSON-serialisable only; anything that fails
`json.dumps(default=str)` size-capped at `max_fixture_bytes` is not captured.

- Captured only for `safe_noop` functions, on calls whose post verdict was
  `PASS` (or tier-0 `PASS` when the class needs no LLM post pass), up to
  `max_fixtures_per_function` distinct `args_signature`s (default 5, oldest
  evicted).
- The librarian may hand-author fixtures for `safe_noop` functions at store
  time (`add_functions(..., fixtures={name: [...]})`).
- **Used at hash change.** When a `safe_noop` function's `verified_hash`
  changes (repair or edit), replay every fixture through the new
  implementation *before* the function is trusted again; all-pass ⇒
  `passes[tier0]` is seeded with `len(fixtures)` and, if that meets policy,
  the function is trusted immediately without waiting for live runs. Any
  fail ⇒ the repair/edit is rejected back to its author with the failing
  fixture (repair loop: `FAIL` reason includes the fixture diff; librarian
  path: `add_functions` raises `FixtureRegressionError`).
- Fixtures are never replayed for other classes (the world moved).

### 4.5 Verification passes

All four passes are implemented in `unify/actor/verification_runtime.py`
(runtime + supervisor) and `unify/actor/prompt_builders.py`
(`build_static_review_prompt`, `build_args_review_prompt`,
`build_precondition_probe_prompt`, `build_post_probe_prompt`, each returning
`(static_prefix, stable_block, volatile_block)` — see 4.6). Verdict model:

```python
class Verdict(BaseModel):
    verdict: Literal["PASS", "FAIL", "UNSURE"]
    reason: str
    fault: Optional[Literal["leaf", "caller"]]   # required when FAIL
```

| Pass | When | Blocking? | Input | Recorded as |
|---|---|---|---|---|
| **static review** | first call under a new `verified_hash` (lazy), then cached in `static_review` | for `unsafe_effectful`/`idempotent_effectful`: yes, before the first call; for others: raced with execution | function source, contract, docstring, its dependencies' names+docstrings; **no** call context (deliberately hash-pure) | `kind=static`, once per hash |
| **argument review** | every call while untrusted | effectful classes: yes; `safe_noop`/`read_only`: raced | contract + concrete kwargs + intent chain (4.6) | `kind=args` |
| **precondition probe** | every call while untrusted, for effectful classes, or any class with a non-null `precondition` | effectful: yes; else raced | `precondition` dict + intent chain + a read-only probe budget (reuse `_run_repair_diagnostic_probe` — extract it to `verification_runtime.run_probe`) | `kind=precondition` |
| **post probe** | every call while untrusted, all classes except `safe_noop` (which uses tier-0 + fixtures) | never — async | result, tier-0 outcome, interactions log captured by the wrapper (which primitives were called with which args and what they returned), pre/post probe if effectful, intent chain | `kind=post` |

**Blocking** here means: for effectful leaves the leaf does not execute until
the pass returns `PASS`. **Raced** means: launched concurrently with
execution; a later `FAIL` triggers the same rewind as a failed post verdict.

Static review's `FAIL` blocks the *first* execution of that hash for
effectful classes and triggers repair targeting that function; for other
classes it triggers rewind+repair on landing. Static review's `UNSURE` is
recorded and does not block.

Every pass uses `new_llm_client(...)` (`unify/common/llm_client.py`) tagged
with `purpose="verification"` (4.12) and the model configured by
`FunctionSettings.verification.model` (default: the actor's default model;
make it a setting so a cheaper model can be swapped in).

**Delete** `SingleFunctionActor._verify_execution` and its
`VerificationResult`; `SingleFunctionActor` uses the shared post pass. If
`SingleFunctionActor` turns out to have no callers, delete it (confirm the
imports listed in §3 first; `manager_registry.py` and `comms/primitives.py`
import it — check whether those are live paths).

### 4.6 Prompt structure

Every per-call pass prompt is assembled as three blocks in this order, so the
prefix cache holds across firings of the same task:

1. **static prefix** — decision framework, output schema, examples per pass
   kind (constant).
2. **stable block** — top-level goal (task name + description, or the agentic
   request), the linked guidance content for the root and for this leaf
   (`Function.guidance_ids` → `GuidanceManager` bodies, capped at
   `max_guidance_chars`, default 6k), the **intent chain**: for every frame
   from root to this leaf, `name`, `docstring`, effect class, and the exact
   call-site source line with argument *expressions* (not values); the
   immediate parent's source (window of ±20 lines around the call site) and
   this leaf's full source; children by name + docstring; this leaf's
   contract.
3. **volatile block** — the concrete kwargs (values), tier-0 outcome,
   sibling results already produced in this run (repr, capped 2k chars each),
   pre/post probe output, interactions log, result repr (capped 4k chars).

The intent chain is maintained by the wrappers on a `contextvars.ContextVar`
stack `current_verification_frames` (list of `Frame(function_id, name,
docstring, effect_class, call_site_line, args_repr)`), pushed on entry and
popped on exit — the same shape as the old `plan.call_stack` /
`scoped_context_snapshot`. The static review pass omits blocks 2's chain and
block 3 entirely (hash-pure).

**Never** include the conversation transcript or the CodeAct trajectory in a
verifier prompt. The verifier's context is per-frame and bounded.

### 4.7 Runtime: wrappers, pending verdicts, barrier, memo, rewind

Implemented in `unify/actor/verification_runtime.py`; wired into the
`if entrypoint is not None:` branch of `CodeActActor.act()`.

**Wrapping.** After `_run_entrypoint_once` injects the closure into
`sandbox.global_state`, replace each compositional callable `f` in the
namespace with `VerifiedCall(f, meta, supervisor)` **iff** `meta.verify is
True`; trusted functions are left untouched (zero overhead). The root is
wrapped the same way. Wrapping must preserve `inspect.iscoroutinefunction`
(wrap sync and async separately) and `functools.wraps`. Steering probes from
`steering.py` continue to apply to the *inner* source; the wrapper is not
instrumented.

**Per call, the wrapper:**
1. pushes its `Frame`; computes `args_signature`;
2. runs tier-0 input validation (sync, microseconds); `FAIL` ⇒
   `supervisor.fail(verdict)` (see below) — do not call the function;
3. if this leaf is effectful (`idempotent_effectful`/`unsafe_effectful`):
   `await supervisor.barrier()` — waits until **every** pending verdict in the
   run has landed as `PASS` (total order; no dependency graph); if any pending
   verdict lands `FAIL`, `barrier()` raises `RewindRequested`; if any lands
   `UNSURE` or times out (`pending_verdict_timeout_s`), the effectful leaf is
   **not** executed and `barrier()` raises `HoldRequested`; then runs static
   review (if not cached for this hash), args review and precondition probe
   **to completion, in that order**, each `FAIL` ⇒ `supervisor.fail`, each
   `UNSURE` ⇒ `HoldRequested`; only then calls the function;
   for `safe_noop`/`read_only`: launches static review (if uncached), args
   review and precondition probe as background verdict tasks and calls the
   function immediately;
4. checks the **memo** first: if `(function_id, verified_hash,
   args_signature)` is in `supervisor.memo` (a result from a previous attempt
   in this run whose verdict was `PASS`, or from a trusted function), return it
   without executing — this is what makes rewind safe;
5. executes; captures the interactions log (a run-scoped list appended by the
   primitives layer — reuse the existing interaction/notification queues where
   present; add a lightweight `record_interaction(name, args, result)` hook in
   `FunctionManager.execute_function`/primitive dispatch if none exists);
6. runs tier-0 output validation + postconditions (sync); `FAIL` ⇒
   `supervisor.fail`;
7. launches the post probe as a background verdict task (never awaited by
   the caller); returns the result; pops its `Frame`.
8. Every verdict, when it lands, is written to `Functions/Verifications`
   immediately (append-only; the row write is fire-and-forget with a bounded
   retry, never blocks the run).

**Supervisor** (`RunVerificationSupervisor`, one per entrypoint run):
- `pending: OrderedDict[ordinal, asyncio.Task[Verdict]]` in call order.
- `fail(verdict, frame)`: cancel every pending verdict with a higher ordinal;
  cancel the entrypoint task; record `RewindRequested(frame, verdict)`.
- `barrier()`: `await` all pending in order; semantics above.
- `memo`: results keyed as in step 4, populated only when the call's own
  verdicts (tier-0 + post) landed `PASS`, or immediately for trusted
  functions.
- **Rewind loop** (replaces `_run_entrypoint`'s `attempts_remaining` loop):
  ```
  for attempt in range(1 + max_rewinds_per_run):
      try:   result = await run_root()        # wrappers consult memo
             await supervisor.drain()         # wait for all remaining verdicts
             if any FAIL among them: raise RewindRequested(...)
             return result
      except RewindRequested as rw:
             await repair(rw)                 # 4.7 "Repair targeting"
             continue
      except HoldRequested as h:
             return Held(h)                   # ExecutionState.held
  return Held(exhausted)
  ```
  `max_rewinds_per_run` (default 2) replaces `entrypoint_repair_attempts`
  everywhere (`TaskScheduler.execute` ~L1316 passes it; delete the old kwarg
  and its plumbing).
- **Repair targeting.** Generalise `_repair_symbolic_entrypoint(entrypoint_id,
  …)` to `_repair_function(function_id, *, failure: Exception | Verdict,
  frames: list[Frame], …)`. Target selection: an exception ⇒ the innermost
  compositional function in the traceback (map filenames/qualnames injected by
  `filter_functions(_return_callable=True)` back to `function_id`s — the
  sandbox already labels injected sources; if it does not, label them);
  a `FAIL` verdict with `fault="leaf"` ⇒ that leaf; `fault="caller"` ⇒ the
  parent frame; a second `RewindRequested` on the same target within one run
  ⇒ escalate one frame up (the old `ReplanFromParentException` semantics).
  Deployment-owned functions (`custom_hash is not None`) still refuse repair
  and produce `Held` with reason `deployment_owned_function_failed`.
  The repair prompt keeps the current shape (FM CRUD + `run_diagnostic_probe`,
  overwrite in place, keep `function_id`), plus: the failing verdict, the
  frame chain, and the instruction that fixture replay will run for
  `safe_noop` targets. A successful repair changes `verified_hash` ⇒ the
  ledger resets for that function (4.8) and its wrapper is re-armed for the
  next attempt.
- **Delivery.** The task's result is delivered (notification / `_result_str`)
  only after `drain()` completes with no `FAIL`
  (`FunctionSettings.verification.deliver_before_root_verdict=False`,
  default). If set `True`, deliver immediately and, on a later `FAIL` whose
  rewind succeeds, post a follow-up owner notification: "Correction to
  <task name> delivered at <time>: <verifier reason>. Corrected result:
  <result>". Owner notification uses the same path the scheduler already uses
  for task results.
- **Held.** `Held` writes `ExecutionState.held`, stores the verdict/reason on
  the execution row, and notifies the owner: "Holding <task name>: could not
  verify <leaf> (<reason>). Nothing was sent/changed. Payload retained on the
  execution row." No effect executes; the run does not retry until the next
  scheduled fire.
- **Timeouts.** `pending_verdict_timeout_s` (default 120) applies per verdict
  task; on timeout the verdict is `UNSURE` with reason `timeout`.
- **Cancellation/steering.** A steer/stop on the task handle cancels pending
  verdicts and the entrypoint task exactly as today (`_CodeActEntrypointHandle`
  cancellation semantics unchanged); pending verdict rows are written as
  `UNSURE reason=cancelled`.
- **Steady state.** With every function in the closure trusted: no wrappers,
  no supervisor tasks, no barrier, no memo — `_run_entrypoint_once` as today.
  Assert this in a symbolic test by counting created tasks.

### 4.8 Ledger update, trust policy, invalidation, spot checks

`unify/function_manager/verification/policy.py`:

```python
class VerificationPolicy(BaseModel):          # librarian-settable overrides
    always_verify: bool = False               # never trust; keep passes forever
    required_passes: Optional[int] = None     # may only RAISE the class default
    min_distinct_inputs: Optional[int] = None # may only RAISE
    fixture_only: bool = False                # safe_noop only: tier0+fixtures suffice
    spot_check_rate: Optional[float] = None   # may only RAISE
```

Class defaults live in `FunctionSettings.verification` (4.13):

| class | required LLM passes (`args` **and** `post` each ≥ N) | min distinct inputs | tier-0 counts toward N? | static review required |
|---|---|---|---|---|
| `safe_noop` | 0 (LLM passes not run) | 1 | yes (`tier0` ≥ 1, fixtures seed it) | yes |
| `read_only` | 3 | 2 | no | yes |
| `idempotent_effectful` | 3 | 2 | no | yes |
| `unsafe_effectful` | 5 | 3 | no | yes |

`derive_verify(fn) -> bool` (pure function of the row; called after every
ledger write and on load):

```
trusted = (fn.verified_hash == function_trust_hash(fn)
           and fn.static_review.verdict == PASS
           and passes meet the table (after policy raises)
           and len(distinct_args_signatures) >= min_distinct_inputs
           and fn.ledger.fails == 0                 # for this hash
           and not policy.always_verify)
fn.verify = not trusted
```

**Invalidation** — all of these set `verified_hash=None`, clear
`static_review`, zero the summary, and set `verify=True`; the previous rows
in `Functions/Verifications` are kept (history), the summary just points at
the new hash:
- `implementation` change (repair, librarian overwrite, sync from `custom/`);
- any dependency's `verified_hash` change (transitive, via
  `function_trust_hash`);
- `venv_id` change or the venv's pyproject hash change;
- a linked guidance entry (`guidance_ids`) is updated or deleted —
  `GuidanceManager` update/delete calls
  `FunctionManager.invalidate_trust_for_guidance(guidance_id)`; also append
  `StaleReason(kind="guidance_changed", …)`;
- any `FAIL` verdict (including tier-0 and spot checks) — `fails += 1`
  already fails the predicate; on the *next* successful repair the hash
  changes and counters reset naturally; on a `FAIL` with no repair (e.g.
  `Held`), the function stays untrusted with `fails > 0` until an edit
  changes the hash.

**Spot checks.** For `idempotent_effectful`/`unsafe_effectful` functions
that are trusted **and** have `contract.output_schema is None`, each call is
sampled at `spot_check_rate` (default 0.1) to run the post probe
asynchronously (never blocking; a `FAIL` invalidates trust and, because the
call already happened, produces an owner notification of the same shape as a
correction). Trusted functions with an output contract are not spot-checked
(rate 0) unless the policy raises it. `safe_noop`/`read_only`: never.

**UNSURE** never counts, never invalidates, never rewinds; it is recorded.
Three consecutive `UNSURE` on the same `(function, kind)` emit a
`logger.warning` and an owner notification once ("<name> cannot be verified
automatically; it will keep being checked") — no state change.

**Where verdicts count.** Only verdicts produced in **symbolic runs** (and
`SingleFunctionActor`, if retained) count toward `passes`. Calls made from
inside an agentic CodeAct loop (`execute_function` or plan namespace) run
tier-0, record `tier0` rows and capture fixtures, but do **not** run the LLM
passes and do not accrue `args`/`post` passes — the model that is in the
loop is not an independent verifier. (This keeps the ramp meaningful and
avoids paying twice inside agentic runs.)

### 4.9 Storage loop changes (librarian)

Prompts in `code_act_actor.py` (`_STORAGE_*`) gain, and the librarian is
held to:

- **Thin effects.** A function that performs an irreversible effect must do
  only that: compute in one function, send/delete/pay in another, and let the
  root compose them. State the rule with a two-function example. This is what
  makes the barrier and memo cheap and the blame precise.
- **Type hints on every parameter and the return**, and a docstring whose
  first sentence is a checkable postcondition. Where a postcondition is
  expressible as an expression, author it (`postconditions=[...]` via
  `add_functions(..., contracts={name: FunctionContract})`).
- **Confirm the effect class.** New librarian tool
  `confirm_side_effect_class(function_id, side_effect_class, rationale)`
  (bounded per 4.1). New tool `set_verification_policy(function_id,
  VerificationPolicy)` (raise-only per 4.8). Both are also available in the
  repair loop's tool set.
- **Fixtures for pure functions**: `add_functions(..., fixtures=...)` when
  the trajectory contains concrete inputs/outputs the function reproduces.
- **Remove** `submit_offline_certification_evidence` and every reference to
  certification evidence (4.11). `attach_entrypoint_to_recurring_task` stays,
  minus its `certification_metadata` parameter.
- The librarian does **not** touch the ledger. Ledger writes are runtime-only.

`add_functions` gains `contracts`, `fixtures`, `side_effect_class_confirmation`
kwargs mirroring the tools; it computes `side_effect_class_detected`,
`contract` from hints, `verified_hash`, and sets `verify=True`,
`ledger=VerificationSummary()`.

### 4.10 Offline promotion rework

Delete: `OFFLINE_CERTIFICATION_REQUIRED_EVIDENCE_FIELDS`,
`OFFLINE_CERTIFICATION_ALLOWED_RISK_CLASSIFICATIONS` (the values move to
`SideEffectClass`), `OFFLINE_CERTIFICATION_REQUIRED_ATTESTATIONS`,
`MAX_OFFLINE_CERTIFICATION_REVISION_ATTEMPTS`,
`_missing_certification_value`, `_offline_promotion_rejection_reasons`, the
`submit_offline_certification_evidence` tool, and every read of
`certification_evidence`/`certification_metadata` on task rows.

Add `TaskScheduler.offline_eligible(task) -> tuple[bool, list[str]]`:
eligible iff `task.entrypoint is not None` and every function in the root's
transitive compositional closure has `verify is False`, and none of them is
`unsafe_effectful` with `class_source == "inferred_third_party"` (i.e. an
unconfirmed dangerous class). Reasons list names the offending
`function_id`s.

`_promote_definition_to_offline(task_id)` becomes: check `offline_eligible`,
write `{"offline": True}`. Promotion is attempted automatically at the end of
every symbolic run whose ledger write flips the last untrusted function in
the closure to trusted (`FunctionSettings.verification.auto_promote_offline`,
default `True`), and is exposed as the librarian tool
`promote_task_offline(task_id)` which only checks eligibility. Loss of trust
in any closure member does **not** demote a task (offline is delivery, and
the headless lane can run verifier passes) — it just means the offline run
verifies again.

### 4.11 Where verification runs

- **Symbolic runs** (scheduler entrypoint, `TaskScheduler.execute`, offline
  headless runner, REST offline triggers): full runtime (4.7). The headless
  runner must have LLM access for passes and repair — confirm it does (repair
  already runs there per `19b4ae6db`).
- **Agentic runs**: tier-0 + fixtures + rows only (4.8, last paragraph). No
  wrappers beyond tier-0; no supervisor.
- **`SingleFunctionActor`**: if retained, uses the shared post pass and
  writes rows exactly like a symbolic run of a single-leaf closure.
- **`primitives.actor.act(...)` sub-agents** given `prompt_functions`: agentic
  ⇒ tier-0 only.

### 4.12 Telemetry

- Verifier and repair LLM clients are created with a `purpose` tag
  (`"verification"` / `"repair"`); planning (CodeAct loop) calls carry
  `"planning"`. Implement the tag on the `new_llm_client` factory (or the
  `llm_broker` seam — pick the one every actor LLM call already passes
  through and use only that one) so unillm hooks can read it.
- Each `Tasks/Executions` row records `tokens: {planning: {prompt,
  completion}, verification: {...}, repair: {...}}` and `verdicts: {PASS: n,
  FAIL: n, UNSURE: n}` for the run, plus `rewinds: int` and `held_reason`.
- The `colleague` unify arm's in-process unillm meter reads the tag and
  emits the split into its per-phase ledger (Phase 9).

### 4.13 Settings

`unify/function_manager/settings.py::FunctionSettings.verification:
VerificationSettings` (new pydantic model in the same file):

```
model: Optional[str] = None                       # None ⇒ actor default
tier0_always: bool = True
required_passes: dict[SideEffectClass, int] = {safe_noop: 0, read_only: 3, idempotent_effectful: 3, unsafe_effectful: 5}
min_distinct_inputs: dict[SideEffectClass, int] = {safe_noop: 1, read_only: 2, idempotent_effectful: 2, unsafe_effectful: 3}
spot_check_rate: dict[SideEffectClass, float] = {idempotent_effectful: 0.1, unsafe_effectful: 0.1}
max_rewinds_per_run: int = 2
pending_verdict_timeout_s: int = 120
deliver_before_root_verdict: bool = False
auto_promote_offline: bool = True
max_fixtures_per_function: int = 5
max_fixture_bytes: int = 8192
max_guidance_chars: int = 6000
unsure_warning_threshold: int = 3
```

Every number above is a decision, not a placeholder; change them via
settings, not by editing call sites.

### 4.14 Migration / backfill

Orchestra contexts are schemaless logs, so no migration script is needed for
new fields; behaviour on old rows:

- On first load of a compositional row lacking `side_effect_class`, run
  `classify` and persist `side_effect_class(_detected)`, `class_source`,
  `contract` (from hints), `verified_hash=None`, `verify=True`,
  `ledger=VerificationSummary()`. Do this in `FunctionManager`'s row hydration
  path once, idempotently.
- Existing tasks with `entrypoint` set keep it and start on the ramp.
- Existing `offline=True` tasks keep `offline=True` (see 4.10, last
  sentence).
- Old `certification_*` keys on task rows are ignored and removed on the next
  write of that row.
- No compatibility shims: `entrypoint_repair_attempts` and the certification
  tool are removed outright; callers updated in the same change.

---

## 5. Rollout phases

Each phase is independently mergeable to `staging`, has its own tests, and
leaves behaviour intact except where stated. Commit with explicit paths.
Run `tests/parallel_run.sh <touched dirs>` per phase; symbolic tests must
pass with `UNILLM_CACHE=read-only` in CI, so record cache entries for every
new LLM prompt (see `tests/README.md`, "LLM Cache Refresh").

**Phase 0 — spec in tree.** Copy this file to `.agents/plans/verified-functions.md`
in the worktree; keep its Status line updated per phase.

**Phase 1 — schema, classification, ledger (no behaviour change).**
`types/verification.py`; new `Function` fields; `Functions/Verifications`
table constant + writer; `classify.py` with the complete
`PRIMITIVE_EFFECT_CLASSES` table (every primitive row classified — add a
test that lists `Functions/Primitives` and asserts none is missing);
`ledger.py::function_trust_hash`, `policy.py::derive_verify`;
`VerificationSettings`; backfill on hydration; `verify` docstring rewritten
(mention the primitive overload).
Tests (`tests/function_manager/storage/`): hash changes on each component;
`derive_verify` truth table per class; classification of a corpus of sample
sources (pure / primitives / third-party / comms send / desktop) incl. the
`unsafe_effectful` default and the loud log; backfill idempotency.
Acceptance: all existing tests green; every compositional row hydrates with
a class; `verify` still `True` everywhere.

**Phase 2 — tier 0 and fixtures.** `contracts.py` (schema from hints,
postcondition compile/eval with the allowlist), `add_functions(contracts=,
fixtures=)`, tier-0 checks + fixture capture in a minimal wrapper applied on
**agentic** `execute_function` and symbolic runs (rows written, no LLM, no
rewind yet — tier-0 `FAIL` raises `ContractViolation` for now); fixture
replay on hash change in `add_functions(overwrite=True)` and in repair.
Tests: schema generation; postcondition allowlist rejects `__import__`,
attribute access on modules, calls outside the allowlist; capture caps;
replay pass seeds `tier0`; replay fail ⇒ `FixtureRegressionError`.

**Phase 3 — verifier passes.** `verification_runtime.py` (passes only, no
supervisor), prompt builders with the three-block split and the intent-chain
`ContextVar`, `Verdict` model, `purpose="verification"` tag; static review
cached per hash; `SingleFunctionActor._verify_execution` deleted in favour of
the shared post pass; probe extracted from repair.
Tests: symbolic — prompt assembly (stable block byte-identical across two
calls of the same call site with different args; static review prompt has no
call context); verdict parsing/`fault` required on `FAIL`; eval (cached) —
args review flags a wrong recipient; post pass flags a units change against
a fixture; static review flags contract mismatch.

**Phase 4 — runtime supervisor.** Wrappers on the entrypoint closure,
pending verdicts, barrier semantics, memo, rewind loop with
`max_rewinds_per_run`, `_repair_function` with target selection and
escalation, `ExecutionState.held` + owner notification, delivery-after-drain,
correction follow-up when `deliver_before_root_verdict=True`, ledger rows on
symbolic runs, timeouts, cancellation.
Tests (`tests/actor/code_act/`, `tests/task_scheduler/`): a three-leaf
closure where a `read_only` leaf's post verdict fails after a downstream
`safe_noop` leaf ran ⇒ later verdicts cancelled, root re-invoked, memo
prevents re-executing the trusted leaf, failed leaf repaired (stubbed
repair) and re-executed, downstream re-executed; an `unsafe_effectful` leaf
never executes while an upstream verdict is pending (assert ordering with
`tests/async_helpers.py`); `UNSURE` on the barrier ⇒ `held`, effect not
executed, owner notified, execution row `held`; exhausted rewinds ⇒ `held`;
`fault="caller"` targets the parent; second failure on same target escalates;
trusted closure creates zero verifier tasks; steer/stop cancels pending
verdicts and rows read `UNSURE cancelled`; `custom_hash` function `FAIL` ⇒
`held deployment_owned_function_failed`.

**Phase 5 — trust flips, invalidation, spot checks, offline rework.**
`derive_verify` applied after each ledger write; invalidation hooks
(implementation/deps/venv/guidance/FAIL); guidance-side call into
`FunctionManager.invalidate_trust_for_guidance`; spot checks; deletion of
the certification path; `offline_eligible`, auto-promotion, librarian
`promote_task_offline`.
Tests: ramp to trust for each class with the exact counts from the table
(and one short: policy raise blocks it); guidance edit invalidates; repair
invalidates then fixture replay re-trusts a `safe_noop` leaf immediately;
spot-check `FAIL` invalidates and notifies; offline eligibility truth table;
auto-promotion fires exactly when the last leaf flips; loss of trust does not
demote.

**Phase 6 — librarian changes.** Prompt additions (thin effects, type
hints, postconditions, class confirmation), new tools, removal of the
certification tool, `attach_entrypoint_to_recurring_task` signature.
Tests: eval (cached) — the librarian, given a trajectory that computes and
sends, stores two functions and a root; confirms a class within bounds;
attempts to lower below detected ⇒ tool returns rejection.

**Phase 7 — telemetry.** `purpose` tag on all actor LLM calls; per-execution
token split, verdict counts, rewinds, held reason; unillm hook exposure.
Tests: a symbolic run records `verification` tokens > 0 and `planning` == 0;
an agentic run records `planning` > 0.

**Phase 8 — docs.** README L269 (and L28, L188) rewritten to state the real
lifecycle: bind after one review, trust earned by verification, offline when
the closure is trusted; `unify/actor/README.md`,
`unify/function_manager/README.md`, `ARCHITECTURE.md` (a "Verification"
subsection under the Actor/FunctionManager sections); CHANGELOG `[Unreleased]
### Added / ### Changed`; `Function.verify` docstring; this plan's Status.

**Phase 9 — benchmark (`~/colleague`, separate repo; use its own worktree).**
See §7.

---

## 6. Docs and vocabulary checks

- Grep the diff for `workflow` used as the recurring-unit noun and for
  `skill` used for a single function; both are vocabulary regressions.
- No temporal comments ("new", "updated", "now") in code or docstrings.
- No `try/except` around verifier calls except the specific, expected
  failures (LLM transport errors ⇒ `UNSURE reason=llm_error`; JSON parse ⇒ one
  retry then `UNSURE`).
- Generic error surfaces to users (owner notifications say what was held and
  why in the verifier's words, never a traceback).

---

## 7. Benchmark additions (`~/colleague`)

Rules from `DESIGN.md` apply unchanged: outcome scoring only, no LLM judges,
deterministic fixtures, never score whether an arm has an abstraction. Each
scenario is a new experiment directory mirroring
`colleague/tracks/standing/drift_recovery/` (`README.md`, `fixture.py`,
`protocol.py`, per-arm drivers, `run_*.sh`, `plot.py`, `results/`).

1. **`standing/silent_drift`** — same shape as `drift_recovery`, but the API
   keeps field names and changes meaning mid-series (`amount` minor→major
   units at fire 5; a second variant: default page size 100→50). Score per
   fire: `correct` (2) > `held with reason` (1: an owner message containing
   the fixture's hold marker and no report) > `wrong` (0). Ground truth
   recomputed. Report tokens per fire.
2. **`standing/edge_week`** — four regular weeks, then week five: empty
   result set / duplicated row / unexpected currency / contact without an
   email (four sub-scenarios). Truth defines the correct week-5 output
   ("report says zero, with reason" etc.). Score week 5 as in (1); report
   tokens week 5 vs weeks 1–4.
3. **`standing/repair_locality`** — three independent leaves (orders,
   refunds, tickets → one report); drift only refunds at fire 5. Score:
   recovery (correct after drift), repair tokens, and **byte-identity of the
   orders and tickets sections before vs after repair**.
4. **`teaching` extension** — after the taught week, request weeks 2–6
   unattended; at week 5 amend one rule in one sentence. Score exact
   recipient sets per week (existing scorer), tokens per week, and that the
   amendment lands in weeks 5–6 with no regression in the untouched rule. If
   the taught procedure ends in an external send, add: owner saw a preview
   before the first real send (message-present outcome) and never after.
5. **`standing/change_without_regression`** — after a correct steady state,
   request "add a column for refunds"; score the new column and byte-identity
   of every old column over the next three fires.
6. **Telemetry + plot.** The unify arm's meter (`colleague/arms/` in-process
   unillm hook) splits `planning` / `verification` / `repair` from the
   `purpose` tag; every experiment's ledger carries the split; a shared
   `plot_distillation_curve.py` renders tokens-per-fire by purpose per arm
   across the `standing` track. Non-unify arms report all tokens as
   `planning`. Add the figure to `README.md` "Results so far".
7. README/DESIGN: state the lifecycle thesis (distil → verify → bind →
   repair) and describe scenarios 1–5 in DESIGN's track table; add the
   "held scores below correct" rule to §"Non-negotiable rules".

Expected honest outcome on the current build (before Phases 1–7 land):
unify loses (1) and (2). Publish that, then the fix, then the new result.

---

## 8. Invariants (check every one before calling a phase done)

1. `verify` is never written directly by any tool or prompt; only
   `derive_verify`.
2. The LLM that produced or repaired a function never grants trust; only
   ledger evidence from independent passes does.
3. No effectful leaf executes while any earlier verdict in the run is
   pending, `FAIL`, or `UNSURE`.
4. No effect executes twice within a run: memo covers every `PASS`ed and
   every trusted call.
5. A trusted closure runs with zero verifier tasks, zero wrappers, zero
   extra awaits.
6. Any change to source, deps, venv, or linked guidance sets `verify=True`
   before the next call.
7. A `FAIL` never silently proceeds; it rewinds, holds, or notifies.
8. Verifier prompts contain no transcript and no trajectory.
9. Static review is hash-pure (no call context) so its cache is valid across
   call sites.
10. Every primitive has an explicit effect class in `classify.py`; an
    unclassified one is `unsafe_effectful` and logged as an error.
11. `held` is a first-class execution state handled wherever
    `ExecutionState` is switched on.
12. Owner-facing messages are generic and specific at once: what was held or
    corrected, and the verifier's reason — never internals.

---

## 9. Decisions already made (do not relitigate during implementation)

- Effect taxonomy = the four existing `risk_classification` values.
- Trust counts only from symbolic-run verdicts (and `SingleFunctionActor`).
- Barrier is total-order over pending verdicts, not dependency-tracked.
- Delivery waits for the root verdict by default.
- Fixtures only for `safe_noop`; replayed only on hash change.
- Static review is lazy (first call under a new hash), not at store time.
- Guidance edits invalidate trust of linked functions.
- Loss of trust never demotes an offline task.
- `entrypoint_repair_attempts` is replaced, not kept alongside
  `max_rewinds_per_run`.
- Primitive `verify` overload is documented, not changed.
