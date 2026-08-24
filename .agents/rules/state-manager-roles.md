---
description: State Manager Roles and Boundaries (Routing Guide)
---

Use this to decide which manager to call, what each owns, and where its jurisdiction ends. Keep manager docstrings implementation‑agnostic; this guide is only for high‑level routing and composition.

### ConversationManager
- **Role**: Live chat orchestrator. Routes user requests to `Actor` for code-first execution and wires steering (pause/resume/interject/stop) during conversations.
- **Scope**: Conversation‑level control and message flow; returns/relays steerable handles from inner tools.
- **Connections**:
  - **Steered by**: Top-level UI/controller (outside managers).
  - **Steers**: `Actor.act` (central intelligence); relays in‑flight handles from `Actor.act` and `TaskScheduler.execute`.

### Actor
- **Role**: Central intelligence that orchestrates all state managers through code-first plans. Generates and executes Python plans that call primitives and top-level JSON tools.
- **Scope**: Code-first execution via `act()` method. Generates Python plans that orchestrate `primitives.contacts.*`, `primitives.tasks.*`, etc., plus top-level JSON tools such as `GuidanceManager_*` and `KnowledgeManager_*`. Wires in‑flight handles to `ConversationManager` for real‑time steering.
- **Connections**:
  - **Steered by**: `ConversationManager` (primary caller of `act()`).
  - **Steers**: State manager primitives (`primitives.contacts.*`, `primitives.tasks.*`, etc.), `KnowledgeManager_*` / `GuidanceManager_*` JSON tools, `TaskScheduler`, and the `ConversationManager` handle (`ask`/`interject`/`get_full_transcript`). Uses `FunctionManager` for function discovery and execution.

### Actor routing playbook
- **Read‑only questions**
  - Tasks → `primitives.tasks.ask` or `TaskScheduler.ask`
  - Contacts → `primitives.contacts.ask`
  - Transcripts → `primitives.transcripts.ask` (may call `primitives.contacts.ask` for participants)
  - Knowledge → `KnowledgeManager_search` / `KnowledgeManager_filter` / `KnowledgeManager_get_knowledge` (top-level JSON tool calls, not primitives)
  - Secrets (metadata/placeholders only) → `primitives.secrets.ask`
  - Time‑sensitive/web ("today/latest/now") → `primitives.web.ask`
  - About a specific received file (filename known) → `primitives.files.ask`
- **Mutations (create/edit/delete/merge)**
  - Tasks → `primitives.tasks.update` or `TaskScheduler.update`
  - Contacts → `primitives.contacts.update`
  - Knowledge claims → `KnowledgeManager_add_knowledge` / `KnowledgeManager_update_knowledge` / `KnowledgeManager_invalidate_knowledge` / `KnowledgeManager_supersede_knowledge` / `KnowledgeManager_delete_knowledge` (top-level JSON tool calls)
  - Guidance → `GuidanceManager_add_guidance` / `GuidanceManager_update_guidance` / `GuidanceManager_delete_guidance` (top-level JSON tool calls)
  - Secrets → `primitives.secrets.update`
- **Execution vs. interaction**
  - Ephemeral live action (UI control/one‑off interaction) → `Actor.act` (called via ConversationManager)
  - Durable, tracked work → `TaskScheduler.execute` (via `primitives.tasks.execute`)
  - Never use `TaskScheduler.update` to start work; always use `execute`.
- **Storing new data or files (any source)**
  - Rows from an API / connected app / user input, specific files, whole folders, or a reshape of a stored table → `primitives.ingestion.submit(source, target)`. One verb for every source/target pairing; returns a run handle immediately.
  - Observe and recover with `primitives.ingestion.get_status` / `get_logs` / `wait` / `retry` / `cancel` / `pause` / `resume`. `status.next_step` states the one action that makes sense. `status.files` breaks a batch down per file; `retry(files=[...])` aims at one of them, and `only="stale"` takes over an attempt whose lease lapsed.
  - Close a run with `primitives.ingestion.reconcile` before calling the data ready: it reports rows landed against rows expected **and** the columns that are blank in every row sampled. A row count alone once agreed with a run that committed 449,287 rows holding no values.
  - There is no `primitives.data.ingest` — a direct ingest blocked the plan for the length of the write and left nothing to inspect if it died part-way.
- **Generative UI (canvases)**
  - Author, revise and inspect interactive views → `primitives.canvas.*` (`create_view`, `update_view`, `refresh_props`, `preview`, `list_invocations`, …). Bind live data to stored tables (including ones an ingestion run just produced via `status.contexts`), never to a provider call.
- **File → knowledge distillation**
  - Parse with `primitives.files.parse`, then distill durable statements into typed claims via `KnowledgeManager_add_knowledge` (attach `source_refs` pointing at the file / transcript / user statement). There is no `primitives.knowledge` pipeline and no NL `ask`/`update`/`refactor` loop on KnowledgeManager.
- **Images**
  - Images are referenced **by filesystem path** across the entire stack. Screenshot directories (`Screenshots/User/`, `Screenshots/Assistant/`) and other workspace paths serve as the universal cross‑manager pointer for visual content. Managers and plans reference images via their relative filepath — no special `images` parameter or structured ref types are needed at the public API boundary.

### ImageManager
- **Role**: Persistent image store and metadata registry. Provides durable `image_id`‑keyed storage in the `Images` context, backing filesystem images with cloud persistence and queryable metadata.
- **Data model & identity**:
  - Every stored image has a unique numeric `image_id` (stable within the active assistant context).
  - Image rows store base64 bytes or a cloud object URL, plus metadata (caption, timestamp, mime/type).
  - Each image row carries an optional `filepath` field that records the local filesystem path the image was saved to. This is the bridge between the filesystem‑based reference convention and the persistent `Images` context.
- **ImageHandle wrapper**:
  - Internal code operates on an `ImageHandle` abstraction that wraps an image row and exposes:
    - `image_id`, optional `caption`/metadata, `filepath`
    - `raw()` → returns image bytes (resolves base64 vs signed URL transparently)
    - `ask(question)` → sends the image to a vision‑capable model and returns a text answer
- **Cross‑manager image convention**:
  - **Filesystem paths are the universal image reference.** When images need to flow between managers (e.g., from `Actor` plans into `GuidanceManager.update`), they are referenced by their workspace‑relative filepath (e.g., `Screenshots/User/2026-02-16T14-30-45.jpg`). The receiving manager or its internal tools can resolve filepaths to `image_id`s via `ImageManager.filter_images(filter="filepath == '...'")` when persistent storage linkage is needed.
  - Some managers accept an `images` parameter using `ImageRefs`/`AnnotatedImageRef` types for structured image attachment (e.g., `GuidanceManager.add_guidance` and `update_guidance` accept annotated image references to link visual content directly to guidance entries). This is the appropriate mechanism when a manager's data model has a first-class `images` field. The filesystem-path convention above applies at the *orchestration* boundary (Actor plans, cross-manager references); within a manager's own CRUD interface, structured image refs are the native format.
- **Connections**:
  - **Steered by**: managers that persist or query images (e.g., `GuidanceManager`, `TranscriptManager`).
  - **Steers**: — (exposes image records/handles; managers decide when/how to query or persist images).

### TaskScheduler
- **Role**: Owner of durable tasks and their execution.
- **Scope**: ask (read‑only about tasks), update (create/edit/delete tasks), execute (start tasks; returns live handle). Tasks are independent — there is no queue chaining or ordering between them.
- **Edge**: Never use `update` to "start" work; always start via `execute` (after any needed updates).
- **Connections**:
  - **Steered by**: `Actor` (via `primitives.tasks.*`); `ConversationManager` (for direct task management).
  - **Steers**: `Actor` (to run tasks via `execute`); exposes a live `ActiveTask` handle that is wired back through `ConversationManager` for steering.

### KnowledgeManager
- **Role**: Passive typed claim ledger for durable domain knowledge (facts, policies, definitions, decisions, constraints, insights, preferences) with provenance (`source_refs`) and lifecycle status (active / superseded / invalidated).
- **Scope**: CRUD and lifecycle operations (`search`, `filter`, `get_knowledge`, `add_knowledge`, `update_knowledge`, `delete_knowledge`, `invalidate_knowledge`, `supersede_knowledge`, `reconcile_sources`, `clear`) exposed as first-class JSON tool calls on the CodeActActor (`KnowledgeManager_*`) — **not** as `primitives.knowledge.*`. No natural-language `ask` / `update` / `refactor` tool loops.
- **Negative scope**: Does **not** own people/contacts (ContactManager), procedural how-tos/SOPs (GuidanceManager), user Python functions (FunctionManager), received file bytes/parsing (FileManager), or secrets/credentials (SecretManager).
- **Writers**: KnowledgeManager is a passive store. Live writers are the Actor / ConversationManager (user-requested claim storage), StorageCheck (trajectory distillation into claims), and optionally MemoryManager (offline consolidation). Distill file/transcript content into claims with `source_refs`; do not treat KnowledgeManager as a schema-refactoring table store.
- **Connections**:
  - **Steered by**: `Actor` (via top-level `KnowledgeManager_*` JSON tool calls, not primitives); optional offline writes from `MemoryManager`.
  - **Steers**: — (exposes typed claim records; callers attach provenance and decide when to write).

### ContactManager
- **Role**: Source of truth for people/contact records.
- **Scope**: ask (read‑only), update (create/edit/delete/merge contacts).
- **Connections**:
  - **Steered by**: `Actor` (via `primitives.contacts.*`); read‑only usage by `TranscriptManager.ask` (to resolve/compare contacts during transcript queries).
  - **Steers**: —

### TranscriptManager
- **Role**: Store and retrieval surface for message transcripts.
- **Scope**: ask (read‑only retrieval, filtering, analysis); may expose summarization in implementations.
- **Edge**: Summarize conversations here; write long‑term distilled facts to `KnowledgeManager` if needed.
- **Connections**:
  - **Steered by**: `Actor` (via `primitives.transcripts.*`).
  - **Steers**: `ContactManager.ask` (for participant lookup/attributes in transcript answers).

### FileManager
- **Role**: Read‑only registry and parsing for received/downloaded files.
- **Scope**: exists/list, parse, ask about a specific file (read‑only tool loop), describe (storage discovery).
- **Connections**:
  - **Steered by**: `Actor` (via `primitives.files.*`).
  - **Steers**: `DataManager` (internally delegates filter/search/reduce/join operations).

### DataManager
- **Role**: Low‑level data operations on any Unify context.
- **Scope**: Canonical implementation of filter, search, reduce, join, insert, update, delete, vectorize, plot. `ingest` exists as the low-level chunked write engine but is **not** exposed to the Actor — storing new data routes through `IngestionManager` so every write is recorded, checkpointed and recoverable.
- **Connections**:
  - **Steered by**: `FileManager` (delegates data ops), `IngestionManager` (row writes via the shared checkpointed engine), `Actor` (via `primitives.data.*`).
  - **Steers**: — (pure primitives module, no high‑level tool loops).

### IngestionManager
- **Role**: The one verb for storing data and files from anywhere — `submit(source, target)` — with a resumable, checkpointed engine behind it that in-process and worker-fleet execution share.
- **Scope**: `submit`, `get_status`, `get_logs`, `wait`, `list_runs`, `retry`, `cancel`, `pause`, `resume`, `reconcile` via `primitives.ingestion.*`. Sources: `RowsSource` (anything in hand), `FilesSource` / `FolderSource` (parse via the file pipeline), `TableSource` (reshape stored rows). Targets: `TableTarget` (one queryable context) or `CollectionTarget` (documents kept whole, inner tables extracted). Runs and their events are rows in `Ingestion/Runs` + `Ingestion/Events`.
- **Tiering**: never a caller's choice. Files parse off the assistant's process whenever a worker fleet is reachable; rows and tables run in process only under a measured row ceiling. Both tiers write the same artifacts, leases and checkpoints, so the tier affects latency and nothing else.
- **Negative scope**: does not query or reshape-in-place (DataManager), does not answer questions about file contents (FileManager `ask`), does not own provider fetches (integrations fetch, ingestion stores).
- **Connections**:
  - **Steered by**: `Actor` (via `primitives.ingestion.*`); `FileManager` (attachment ingestion submits here).
  - **Steers**: `DataManager.ingest` (row writes), the file parse pipeline, and the hosted pipeline control plane (`/infra/pipeline/*`) for dispatched runs.

### CanvasManager
- **Role**: Assistant-authored generative React UI. The actor writes real TSX against `@unity/canvas-kit`; it is linted, typechecked, bundled, rendered headlessly and critiqued before publish; Console displays it in a genuinely isolated frame.
- **Scope**: `create_view`, `update_view`, `refresh_props`, `get_view`, `list_views`, `delete_view`, `preview`, `run_invocation`, `list_invocations` via `primitives.canvas.*`. Rows live in `Canvas/Views` / `Canvas/Actions` / `Canvas/Invocations`; a routing token is registered with the backend on publish.
- **Data plane**: query bindings (context-backed tables, executed server-side per view) and materialised props (LLM-shaped reads frozen at author/refresh time). Connected-app data must be **stored first** (via `primitives.ingestion.submit`) and bound as an ordinary table.
- **Write plane**: declared, schema-validated actions; the viewer's input is validated server-side, recorded as an invocation row, then executed by the assistant in one of three lanes (stored function, task trigger, actor request).
- **Connections**:
  - **Steered by**: `Actor` (via `primitives.canvas.*`); `ConversationManager` (executes recorded invocations on `CanvasInvocationRequested`).
  - **Steers**: `DataManager` (binding dry-runs), `FunctionManager` (action target resolution and execution), `TaskScheduler` (task-lane triggers), the actor (assistant-lane requests).

### WebSearcher
- **Role**: Lightweight, text-based retrieval engine for quick one-off internet queries (headlines, weather, definitions, current events).
- **Scope**: ask only (search, extract, crawl, map against the public web); returns live handle. No gated-site access, no browser automation, no credentials. For authenticated or complex web procedures, use Tavily + SecretManager + ComputerPrimitives directly via code-first plans.
- **Connections**:
  - **Steered by**: `Actor` (via `primitives.web.*`).
  - **Steers**: — (results may subsequently be persisted as typed claims via `KnowledgeManager_add_knowledge` when requested).

### SecretManager
- **Role**: Owner of secrets.
- **Scope**: ask (metadata/placeholder answers only), update (create/edit/delete secrets).
- **Connections**:
  - **Steered by**: `Actor` (via `primitives.secrets.*`).
  - **Steers**: —

### BlacklistManager
- **Role**: Catalogue of blocked contact details per communication medium (email, SMS, phone).
- **Scope**: Primitive CRUD operations — filter/create/update/delete blacklist entries. No tool loops; purely programmatic.
- **Connections**:
  - **Steered by**: `Actor` (via `primitives.blacklist.*`); `ConversationManager` (for direct blacklist management).
  - **Steers**: —

### FunctionManager
- **Role**: Catalogue of user‑supplied Python functions and metadata.
- **Scope**: add/list/get_precondition/delete/filter/search over functions.
- **Connections**:
  - **Steered by**: `Actor` (loads/executes functions during actions). `GuidanceManager` reads function rows via the shared "Functions" context (no direct API calls).
  - **Steers**: —


### GuidanceManager
- **Role**: Owner of procedural how-to information: step-by-step instructions, standard operating procedures, software usage walkthroughs, and strategies for composing functions together.
- **Scope**: CRUD operations (search, filter, add_guidance, update_guidance, delete_guidance) exposed as first-class JSON tool calls on the CodeActActor — **not** as primitives. Read tools are gated by the discovery-first policy; write tools are available in the doing loop (for user-requested guidance storage) and the storage review loop (for compositional strategy extraction).
- **Builtins library**: reads also federate over a global, read-only guidance catalogue (`Guidance` context in the public-read `Builtins` project) holding entries imported from the Agent Skills ecosystem with stable hash-based ids and `is_builtin=True`. Default-on for every assistant; `update_guidance`/`delete_guidance` refuse builtin ids. Seeded from the committed snapshot `unity/guidance_manager/builtins_guidance.json`, generated by `scripts.skill_migration.builtins_import` from commit-pinned upstream repos (`builtins_skills.lock.json`); upstream drift is detected (`--check`) but never auto-applied.
- **Connections**:
  - **Steered by**: `Actor` (via top-level `GuidanceManager_*` JSON tool calls, not primitives).
  - **Steers**: reads functions from the shared "Functions" context to surface linked functions.

### WorkflowManager
- **Role**: Catalogue of installable **workflows** — hand-curated, versioned packages that set an assistant up for a recurring job in one install (procedures, claims, functions and a recurring task, planted together). Owner of install state and settings, nothing else.
- **Scope**: `list_workflows`, `get_workflow`, `install_workflow`, `uninstall_workflow`, `get_installation_params` as first-class `WorkflowManager_*` JSON tool calls — **not** primitives. Installing fans out each surface's existing `sync_custom`; there is no second content store and no second reconcile loop. `reconcile_installed` and `execute_requests` are upkeep, deliberately **not** tools (boot/ops, and a single-slug update is `install_workflow` re-run). Tools enter the actor's schema only when a catalogue is configured (`UNIFY_WORKFLOWS_DIR`), so deployments without a shelf keep byte-identical tool schemas and LLM caches.
- **Negative scope**: **has no runtime.** No executions, no steering handle, no run-now, no run history — that all belongs to the tasks it plants (`TaskScheduler`). "Run the workflow now" resolves to triggering its task. It also does not own connections (declared as requirements, resolved against the integrations layer), does not author bundles (git-only, curated in unify-deploy), and never pre-seeds contacts, transcripts or blacklist entries.
- **Requirements gate arming, never planting**: an install with an unmet requirement still plants everything and returns `connect_required`; the planted tasks stay disarmed until the connection lands, and a repeat install arms them.
- **Shelf vs installation**: the catalogue listing and each artifact's published copy are platform data in the public-read `Builtins` project (`Workflows/Catalog`, `Workflows/Content`), admin-seeded and hash-guarded. Everything per-assistant — installations, params, planted rows, requested changes (`Workflows/Requests`) — lives in the assistant's own contexts.
- **Connections**:
  - **Steered by**: `Actor` (via top-level `WorkflowManager_*` JSON tool calls, not primitives); boot (`bootstrap_workflow_catalog` → `reconcile_installed` + `execute_requests`); `ConversationManager` on a `WorkflowRequestRequested` event.
  - **Steers**: each surface's `sync_custom` — `GuidanceManager`, `KnowledgeManager`, `TaskScheduler`, `FunctionManager` — plus `TaskScheduler.set_custom_tasks_enabled` to arm or hold what it planted.

### MemoryManager
- **Role**: Offline memory maintenance (periodic, non‑interactive).
- **Scope**: One‑shot methods that return strings (no live handles), e.g., updating contacts/knowledge from transcripts.
- **Connections**:
  - **Steered by**: Offline scheduler/controller (outside live chat orchestration).
  - **Steers**: `ContactManager.update`, `KnowledgeManager` typed claim APIs (`add_knowledge` / `update_knowledge` / lifecycle methods) to persist distilled facts from transcripts.

### EventBus
- **Role**: Cross‑cutting, in‑process publish/subscribe backbone and searchable event log used by all managers for telemetry and coordination.
- **Scope**: Managers publish structured events (notably `ManagerMethod` for incoming/outgoing `ask`/`update`/`execute`) via a thin logging wrapper; the bus supports `publish`, `search` (filterable queries), `join_published`/`join_callbacks` for deterministic flushing, per‑type window sizing, auto‑pinning, and callback registration.
- **Connections**:
  - **Steered by**: All public manager methods (through the logging decorator) and other components that emit operational events.
  - **Steers**: `MemoryManager` (registers callbacks to react to message and `ManagerMethod` events for maintenance passes); tests and higher‑level orchestrators query the bus to observe recent activity.

### Precedence and source of truth
- **Code is canonical**: This guide is descriptive. If the implementation ever contradicts these descriptions or relationships, the current code takes precedence.
- **Keep in sync**: As managers evolve, update this document alongside changes to cross‑manager wiring, public surfaces, or prompt composition so it remains accurate.
- **Where to update**: Prefer updating manager base docstrings (public API contracts) and prompt builders for tool composition guidance, and reflect those changes here.
