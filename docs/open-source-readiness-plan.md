# Open-Source Readiness Plan

> Extracted from shared Cursor chat: [Commit history preservation in open source](https://cursor.com/dashboard/shared-chats?shareId=commit-history-preservation-in-open-source-Z8jMBKmOROJQ)

This plan covers every identified blocker and required change across all six repositories (Unity, Unify, UniLLM, Orchestra, Communication, Console), organized into workstreams that can be parallelized.

---

## Context & Key Decisions

### Can we open-source with full commit history?

**No — not as-is.** There are several categories of problems baked into the git history:

1. **Leaked GCP Service Account Private Key (Critical)** — A full GCP service account private key was committed in `6de00528` and removed in `0769d020`, but the key material is permanently embedded in the git history. Project: `saas-368716`. Private key ID: `adbaf4e209bbe86ef9ebe2ce3bfdaed1ce523ba4`. If this key hasn't been rotated already, it should be immediately.

2. **LLM Cache Files (up to 51MB)** — `.cache.json` and `.cache.ndjson` were committed into history. These contain full LLM request/response payloads including system prompts, tool definitions, and user-provided content (e.g., email addresses, personal messages). The largest blob is 51MB.

3. **Client/Customer Data** — The `intranet/data/` directory was committed with organization-specific policy documents (housing association policies — ASB, safeguarding, complaints, rent, etc.) and `intranet/repairs/MDH` (~9MB). This looks like real customer data.

4. **Personal Email Addresses in Commit Metadata** — Author emails in the commit history include personal addresses (`*@gmail.com`, `*@outlook.com`) for multiple contributors. These can't be removed without rewriting history.

5. **Binary/Media Files** — A `.mov` file (6MB), various test images, and the large cache blobs inflate the history. The `.git` directory is **206MB** across ~70K objects and **9,858 commits**.

### Does Unity have genuine novelty?

**Yes, unambiguously.** The architectural innovations are real, deep, and not present in either competitor (OpenClaw, Hermes). But the novelty is in the *plumbing*, not in user-visible features:

1. **Steerable handles as a universal return type** — Every manager method (`ask`, `update`, `execute`) returns a `SteerableToolHandle` with `pause`/`resume`/`interject`/`stop`/`ask`.
2. **Nested async tool loops with propagated steering** — Inner LLM tool loops return steerable handles that the outer loop can control.
3. **CodeAct (code-first plans over typed primitives)** — The Actor generates executable Python that calls typed primitives, not JSON tool calls.
4. **The "distributed OS" shape** — Unity is a network of specialized state managers, each with their own LLM-powered tool loop, storage, and domain expertise.

### The Orchestra coupling question

Orchestra is a hosted API at `api.unify.ai/v0`. Users just need a `UNIFY_KEY`. This is the Redis Labs / Supabase model: open-source client, hosted backend. The coupling is the *business model*, not a liability.

### UniLLM billing transparency

UniLLM's `costs.py` has a 20% default margin (`_DEFAULT_COST_MARGIN = 1.2`) and `deduct_credits` calls are wired into every LLM call path. **Recommendation:** Don't maintain public/private forks. Instead, make billing a pluggable hook — production deployment registers the hook at init, open-source users get a clean LLM client with no billing. One codebase, no forks.

### Strategic positioning

Don't try to compete with OpenClaw on stars or developer adoption. Own the positioning: *"OpenClaw is a great personal assistant. We built an enterprise orchestration system."*

---

## Workstream 1: Git History Sanitization (Unity, Unify, UniLLM)

All three repos to be open-sourced need history cleaned before going public. Use `git-filter-repo` for all of these.

### Unity

- **GCP private key**: `application_default_credentials.json` committed in `6de00528`, removed in `0769d020`. Full RSA key for service account is in history. Rotate this key immediately if not already done.
- **LLM cache files**: `.cache.json` (up to 51MB) and `.cache.ndjson` committed historically. Contain full LLM prompts, tool definitions, user-provided content.
- **Customer data**: `intranet/data/` (housing association policy documents) and `intranet/repairs/MDH` (~9MB).
- **Binary bloat**: `browser_demo/server/avatar.mov` (6MB), various test images.
- **Personal emails in commit metadata**: Multiple contributors' personal `@gmail.com` / `@outlook.com` addresses. Rewrite via mailmap if needed.

### Unify and UniLLM

- Audit both histories for any committed `.env`, cache files, or credentials. Quick search found none, but run a full `git-filter-repo --analyze` pass on each.

---

## Workstream 2: Private Submodule and Dependency Decoupling

### Unity

- **Private git submodule**: `.cursor/rules/global-rules` points to a private SSH repo (`global-cursor-rules.git`). Options:
  - Inline the rules directly into the Unity repo (remove submodule, commit contents)
  - Make the `global-cursor-rules` repo public
  - Remove the submodule entirely and keep only Unity-specific rules
- **Sibling path dependencies in `pyproject.toml`**: `[tool.uv.sources]` pins `unify = { path = "../unify" }` and `unillm = { path = "../unillm" }`. These are local dev overrides; CI and Dockerfiles clone the public repos directly.
- **Dockerfile clones repos**: Lines clone `unifyai/unify`, `unifyai/unillm`, `unifyai/magnitude` using `GITHUB_TOKEN`. Now that unify and unillm are public, these can use unauthenticated HTTPS clones.
- **`agent-service/package.json`**: References `file:../magnitude/packages/...`. Same pattern — publish or make public.

### UniLLM

- Same `[tool.uv.sources]` pattern: `unify = { path = "../unify", editable = true }`. Local dev override; CI clones the public repo.

---

## Workstream 3: Hardcoded URLs, GCP Project IDs, and Internal References

### Unity — Production defaults that point to Unify infrastructure

- **`unity/settings.py`**: `ORCHESTRA_URL` defaults to `https://api.unify.ai/v0`; `GCP_PROJECT_ID` defaults to a specific project ID. Change defaults to empty/None with clear error messages requiring explicit configuration.
- **`tests/parallel_run.sh`**: Auto-sets `UNITY_COMMS_URL` to Unify Cloud Run URLs based on git branch. Remove or gate behind an env flag.
- **`.env.example`**: Contains staging Cloud Run URLs. Replace with placeholder values.
- **Scripts** (`scripts/dev/`, `scripts/stress_test/`, `scripts/kubernetes/`): Many hardcode staging/production URLs, GCP project IDs, and Unify-specific bucket names. Either remove internal scripts or move to a private overlay.
- **Guides** (`guides/INFRA.md`, `guides/PREVIEW_ENVIRONMENT.md`): Reference internal GCP infrastructure, VM DNS patterns (`*.vm.unify.ai`), cluster names. Remove or redact.
- **Prompt builders**: References to `unify.ai`, `console.unify.ai` in system prompts (product branding — probably fine to keep, but review).
- **`README.md`**: References to internal infrastructure and deployment patterns. Rewrite for open-source audience.

### Unify

- `unify/__init__.py`: `BASE_URL` defaults to `https://api.unify.ai/v0`. This is intentional (users will use the hosted API), but document the override clearly.

### UniLLM

- `logs/README.md` references `https://api.unify.ai/v0/logs`. Minor — update documentation.

---

## Workstream 4: Admin Endpoint Migration (Unity + Orchestra)

Unity production code calls Orchestra admin endpoints that regular users cannot access. Each needs a user-facing equivalent.

### Orchestra — New user-scoped endpoints needed

*(Spend queries, message completion, profile sync — each needs a user-scoped equivalent of the current admin endpoint)*

### Unity — Switch to user-facing endpoints

- **`unity/spending_limits.py`**: ~~Replace `AsyncAdminClient` (which uses `ORCHESTRA_ADMIN_KEY`) with regular Unify SDK calls using `UNIFY_KEY`.~~ **Done** — now uses `AsyncSpendClient` with `UNIFY_KEY` against user-authenticated Orchestra endpoints.
- **`unity/contact_manager/backend_sync.py`**: Replace admin `POST /admin/assistant/update-user` and `PATCH /admin/assistant/{id}` with user-scoped equivalents.
- **`unity/conversation_manager/domains/comms_utils.py`**: Replace `PUT /admin/messages/{id}/complete` with user-scoped equivalent.
- **`unity/file_manager/sync/manager.py`**: `GET /admin/assistant?agent_id=` for SSH key. This is infrastructure-specific (managed VMs). Make graceful no-op when infrastructure unavailable.

### Unity — Infrastructure endpoints (graceful degradation)

The K8s job and VM pool endpoints in Communication (`/infra/job/*`, `/infra/vm/pool/*`) should not be user-facing. Files that depend on them:

- `unity/conversation_manager/assistant_jobs_api.py`
- `unity/conversation_manager/assistant_jobs.py`
- `unity/conversation_manager/comms_manager.py`

These need to degrade gracefully (or be abstracted behind an interface) when `ORCHESTRA_ADMIN_KEY` and managed infrastructure are unavailable.

---

## Workstream 5: Communication Service — User-Scoped Endpoints + Billing

### User-key auth on outbound action endpoints

Communication already has the `authenticate_user_api_key` pattern in `communication/dependencies.py` (used by tunnels and UniLLM proxy). Extend to outbound actions:

- `POST /phone/send-text` — add user auth + verify `From` number ownership via Orchestra
- `POST /phone/send-call` — same pattern
- `POST /phone/dispatch-livekit-agent` — add user auth + verify `assistant_id` ownership
- `POST /gmail/send` — add user auth + verify `from` address ownership
- `GET /gmail/attachment` — add user auth + verify `receiver_email` ownership

### Per-action cost deduction (greenfield)

No billing code exists in Communication today. Need to add:

- Credit balance check before executing Twilio/Gmail action
- Cost computation per action type (SMS segment, call minute, email, WhatsApp message)
- `deduct_credits` call on success (same `unify.deduct_credits` pattern as UniLLM)
- Failure handling: don't charge if Twilio/Gmail call fails

### Rate limiting on Communication API

Communication's main app has no per-user rate limiting (only adapters have IP-based 120/60s). Add:

- Per-user rate limit categories (`comms_sms`, `comms_call`, `comms_email`, `comms_whatsapp`)
- Tiered limits (default / established / verified) matching Orchestra's pattern
- DB-backed counters (reuse or replicate Orchestra's `RateLimitCounterDAO` pattern)

### Abuse prevention

- Destination uniqueness limits (max N unique recipients per hour for new accounts)
- Minimum credit balance requirements for outbound comms
- Consider requiring phone verification on the user's own account

---

## Workstream 6: UniLLM Billing Decoupling

### Make billing opt-in (not hardcoded)

- **`unillm/costs.py`**: Change `_DEFAULT_COST_MARGIN` from `1.2` to `1.0` for open-source, or make the entire margin/deduction path conditional on a feature flag.
- **`unillm/clients/uni_llm.py`**: The `_safe_deduct_credits` calls after every LLM completion (sync, async, streaming — ~6 call sites) should be gated behind a billing hook or feature flag. When no billing is configured, skip silently.
- **`unillm/clients/base.py`**: `_validate_api_key` requires `UNIFY_KEY` at construction time. For open-source users who just want a multi-provider LLM client, this should be optional (only required when billing is active).
- **`unillm/logger.py`**: `log_usage` computes `billed_cost` and calls `deduct_credits`. Same conditional treatment.

**Recommended approach**: Extract billing into a pluggable hook. Production deployment registers the hook at init. Open-source users get a clean LLM client with no billing. One codebase, no forks.

---

## Workstream 7: CI/CD for Public Repositories

### Current CI dependencies (all three repos)

- `secrets.CLONE_TOKEN` for cloning `unifyai/orchestra`, `unifyai/unify`, `unifyai/unillm`
- `secrets.GCP_SERVICE_ACCOUNT_JSON` + `vars.GCP_PROJECT_ID` + bucket vars
- `secrets.UNIFY_KEY` for test orchestra seeding
- UniLLM also needs `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`
- Tests run against live Orchestra with real LLM calls (cached)

### What to do

- Create a public CI workflow that runs a subset of tests (unit tests, symbolic tests that don't need live Orchestra)
- Keep the full integration test suite in a private CI pipeline (triggered on PRs from maintainers)
- Document how contributors can run tests locally with a self-hosted Orchestra or against the hosted API
- Ensure `.cache.ndjson` files (LLM response caches) do not contain sensitive content before committing to public repos

---

## Workstream 8: Documentation and Developer Experience

### Unity

- Rewrite `README.md` for open-source audience: what it is, how to set up, how to get API keys, how to configure
- Create a `CONTRIBUTING.md` with development setup (clone sibling repos, install deps, configure env)
- Clean up or remove internal scripts (`scripts/dev/`, `scripts/kubernetes/`, `scripts/stress_test/`) that reference internal infrastructure
- Remove or redact internal guides (`guides/INFRA.md`, `guides/PREVIEW_ENVIRONMENT.md`)

### Orchestra

- Ensure all user-facing endpoints (including new ones from Workstream 4) are included in the OpenAPI schema
- Publish complete API reference docs at `docs.unify.ai`

### Communication

- Document the ~5-6 user-facing outbound action endpoints (send SMS, send email, make call, dispatch agent, download attachment)
- No need to document adapters/webhooks/scheduled tasks (internal infrastructure)

---

## Workstream 9: License and Legal

- Choose a license (MIT, Apache 2.0, or similar) for Unity, Unify, and UniLLM
- Add `LICENSE` file to all three repos
- Audit for any third-party code that has incompatible licenses
- Review `.cursor/rules/` content — these contain detailed architecture descriptions of the private repos (Orchestra, Communication, Console). Decide whether to keep, redact, or remove.
- Review test fixtures and sandbox scenarios for any real user data or PII

---

## Workstream 10: Architecture Blog / Narrative (Non-Code)

This is the "street cred" component:

- Write a technical architecture post explaining nested steerable async tool loops, CodeAct, the distributed manager pattern
- Position clearly vs OpenClaw/Hermes: "personal assistant" vs "enterprise orchestration system"
- Time publication to coincide with repo going public
- Consider extracting a standalone demo or example that shows the novel patterns without requiring full infrastructure

---

## Dependency Graph

Workstreams **1, 2, 6, 7, 9, 10** can start immediately in parallel.

Workstreams **4 and 5** (Orchestra and Communication changes) are prerequisites for Workstream **3** (cleaning Unity's admin key usage).

Workstream **8** (docs) should come last, after the code changes stabilize.

## Estimated Effort

| Workstream | Effort |
|---|---|
| W1: Git history sanitization | 1–2 days |
| W2: Dependency decoupling | 2–3 days |
| W3: Hardcoded URLs/references | 2–3 days |
| W4: Admin endpoint migration | 4–6 days |
| W5: Communication user-scoping + billing | 4–6 days |
| W6: UniLLM billing decoupling | 1–2 days |
| W7: CI/CD for public repos | 2–3 days |
| W8: Documentation | 3–4 days |
| W9: License and legal | 1 day |
| W10: Architecture blog | 3–5 days |

**Critical path**: W4 + W5 (8–12 days) then W3 + W8 (4–7 days). With parallelization, total calendar time is approximately **2–3 weeks** with dedicated effort.
