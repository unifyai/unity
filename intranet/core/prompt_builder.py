"""
Centralized prompts for the Midland Heart RAG system.
"""

from __future__ import annotations
import json
import textwrap
from typing import List, Dict, Any, Optional
from unity.common.prompt_helpers import now_utc_str


def _now() -> str:  # UTC timestamp helper
    return now_utc_str()


# Enhanced metadata extraction prompt for new parser
def build_metadata_extraction_prompt() -> str:
    """Build prompt for LLM-based metadata extraction using Pydantic model validation."""
    from unity.file_manager.parser.types.document import DocumentMetadataExtraction

    # Get the Pydantic model schema
    schema = DocumentMetadataExtraction.model_json_schema()

    return f"""
DOCUMENT METADATA EXTRACTION FOR MIDLAND HEART RAG SYSTEM

You are an expert document analyzer for Midland Heart, a UK social housing provider.
Extract comprehensive metadata from the provided document text to enable effective RAG retrieval.

RESPONSE FORMAT:
Your response must be a valid JSON object that exactly matches this Pydantic model schema:

{json.dumps(schema, indent=2)}

FIELD GUIDELINES:

1. **document_type**: Choose from the exact literal values only
2. **category**: Choose from the exact literal values only
3. **summary**: 2-3 sentences focusing on what tenants/staff need to know
4. **key_topics**: Use snake_case format (e.g., "mobility_scooters", "fire_safety")
5. **named_entities**: Organize by type:
   - "organizations": ["Midland Heart", "DVLA", etc.]
   - "policies": Referenced policy names
   - "locations": Specific places mentioned
   - "numbers": Important numbers, limits, percentages
   - "dates": Key dates and deadlines
   - "legislation": Laws and regulations referenced
6. **content_tags**: Include synonyms and related search terms
7. **confidence_score**: Your confidence in extraction accuracy (0.0-1.0)

EXAMPLE OUTPUT:
{{
  "document_type": "policy",
  "category": "safety",
  "summary": "Policy governing the use and storage of mobility scooters in Midland Heart properties. Sets speed limits, weight restrictions, and storage requirements to ensure fire safety compliance.",
  "key_topics": ["mobility_scooters", "speed_limits", "weight_restrictions", "storage_rules", "fire_safety", "class_2_vehicles"],
  "named_entities": {{
    "organizations": ["Midland Heart", "DVLA", "Fire Service"],
    "policies": ["Fire Safety Policy", "ASB Policy"],
    "locations": ["communal_areas", "stairwells", "corridors"],
    "numbers": ["4_mph", "8_mph", "150_kg", "230_kg"],
    "dates": ["October_2024"],
    "legislation": ["Road_Traffic_Act", "Fire_Safety_Order"]
  }},
  "content_tags": ["mobility_aid", "electric_scooter", "disability_access", "fire_risk", "storage_guidelines", "speed_control"],
  "confidence_score": 0.95
}}

CRITICAL: Return ONLY the JSON object, no additional text or markdown formatting.

ANALYZE THE FOLLOWING DOCUMENT:
"""


# ────────────────────────────────────────────────────────────────────────────
# ask: tool-loop RAG system prompt
# ────────────────────────────────────────────────────────────────────────────


def build_intranet_ask_instructions(*, generate_follow_up: bool = False) -> str:
    text = """
   INTRANET RAG — Retrieval addendum (single-table)
   -----------------------------------------------
   Use this *in addition* to the general KnowledgeManager ask prompt. Keep it brief and non-redundant.

   Table choice
   - Work with the table that exposes **summary** (dense text) and **content_text** (longer body).
     If multiple tables exist, prefer the one that clearly stores unified content.

   Semantic retrieval (primary — search **summary only**)
   - Perform semantic search with `_search` using **only** the summary column as the reference:
       • `{"summary": "<user_query>"}`
   - Do **not** include `content_text` in semantic `references`.
   - If precise numbers/quotes/dates are required, first get the **row ids** from the summary search,
     then fetch those rows directly with `_filter` to read `content_text` (no semantic over `content_text`).
   - Choose a modest `k` to minimise noise.

   Tiered passes (fast → slow; expand only if needed)
   - If the initial pass is insufficient, escalate in a **single-table** hierarchy using `_search` with filters:
       1) Paragraphs (highest precision):    `references={"summary": "<user_query>"}, filter="content_type == 'paragraph'", k=10`
       2) Sections (broader context):        `references={"summary": "<user_query>"}, filter="content_type == 'section'",   k=6`
       3) Documents (coarsest fallback):     `references={"summary": "<user_query>"}, filter="content_type == 'document'",  k=3`
   - Move to the next tier **only** when evidence is insufficient or you need more context.
   - When using `_search`, always pass in the `references` parameter with the `summary` column as the reference.
   - If only filtering is needed without semantic search, use `_filter` instead of `_search`.

   Optional narrowing
   - When the request contains strict, unambiguous structured constraints (e.g., specific ids, types or ranges),
     you may first inspect with `_filter` to understand candidates; then still perform `_search` for ranking.
     (Do **not** attempt to emulate semantic matching with `_filter`.)

   Answer assembly
   - Synthesize a concise factual answer strictly from retrieved evidence.
   - Cite stable identifiers and enough context for traceability (e.g., IDs and titles if present).
   - If evidence is insufficient or conflicting, say so and (optionally) suggest one precise follow-up.
   """.strip()

    if not generate_follow_up:
        text += " Return an empty list for 'follow_up_questions' in the final JSON. You do NOT need to suggest follow-up questions."
    return text


# ────────────────────────────────────────────────────────────────────────────
# ask_llm: direct RAG system prompt
# ────────────────────────────────────────────────────────────────────────────


def build_intranet_ask_llm_prompt(
    *,
    include_activity: bool = True,
    case_specific_instructions: str | None = None,
    aggregated_fulltexts: List[str] | None = None,
    aggregated_docs: Optional[List[Dict[str, Any]]] = None,
    logger: Any | None = None,
    generate_follow_up: bool = False,
) -> str:
    """
    Build the system message for a vanilla RAG query over Midland Heart policy documents.

    This prompt is used by KnowledgeManager.ask_llm, which bypasses tool-use and
    sends all available full-texts directly to the model for a single-shot answer.
    """

    activity_block = "{broader_context}" if include_activity else ""

    # Core instructions for Midland Heart policy QA
    core = textwrap.dedent(
        """
        You are a retrieval-augmented generation assistant helping Midland Heart employees
        query their company's policy documentation. Your job is to read the provided policy
        texts and answer user questions accurately, citing the relevant parts when useful.

        Guidelines
        ---------
        • Only rely on information contained in the provided context. If the answer is not in the context, say you do not have enough information.
        • Lead with a direct, concise answer, then add 1–2 sentences of practical context (implications, conditions, caveats). Include short supporting quotes when helpful.
        • Resolve ambiguities conservatively; do not invent policy details.
        • When policies impose conditions, list the key conditions and exceptions.
        • If the question asks for steps or eligibility, present a clear, ordered list.
        • If multiple documents conflict, point out the discrepancy succinctly.
        • Tailor tone for internal staff (clear, practical, compliance-aware).
        • If dates/revisions are relevant, mention the most recent policy revision you see.

        Citations and Grounding (Mandatory)
        -----------------------
        • Ground the final answer in the provided documents and include a Sources section (mandatory).
        • In Sources, list:
          - The exact document title(s) as given (do not invent new titles).
          - When possible, the specific section heading/title where the answer was found.
        • If the referenced information came from an extracted table, parse the table and present its findings in a human‑friendly way. In the citation, mention that it is from a table and which document it was found in. Tables may appear under a proxy heading like "Tables (html)" and may not be linked to a specific section.
        • Never use internal IDs, doctags, or synthetic labels in citations—only the visible textual titles/headings present in the provided text.
        • If asked which section contains certain information, infer the best matching section heading based on the raw text. If no clear heading exists, state that it comes from an unheaded passage of the document.
        • Any short supporting quotes must be clearly attributed to the cited document/section and kept minimal.
        • IMPORTANT: Return a complete list of citations as a JSON array in the `sources` field. Inline citations in the `answer` are allowed when helpful.

        Summarization (Raw Text) Mode
        ----------------------------
        When the user asks to "summarize" a policy/document, produce a direct, document‑level summary from the raw text (do not perform staged paragraph→section→document flows here).
        • Preserve numeric precision:
          - Keep ALL numbers/thresholds/units exactly as written (percentages, dates, limits, measurements).
          - Never round or paraphrase numeric values.
        • Maintain exact terminology: keep technical terms, acronyms, proper nouns verbatim.
        • Executive overview (2–3 sentences): purpose, audience/stakeholders, key outcomes.
        • Structured content summary:
          - Major topics and relationships, processes/procedures, responsibilities and roles
          - All critical specifications/requirements/guidelines and exceptions/edge cases
          - Temporal elements (dates, deadlines, durations) and references to external standards/policies
        • Include metadata at the end:
          - Key Topics (3–8), Named Entities (orgs, systems, roles, locations), Critical Values (important numbers with units)
        • Tables: if present in the raw text, parse and present salient figures in human‑friendly bullets (retain exact values).
        • Concision: remove redundancy while preserving every distinct number, limit, or requirement.
        • Grounding: include a Sources section (as above) with document titles and, when inferable, section headings.

        Output (Strict)
        ------
        • Synthesize a concise factual answer strictly from retrieved evidence.
        • After the direct answer, add 1–2 sentences of context so an employee can apply the information without guesswork (e.g., scope, exceptions, escalation paths).
        • For multi‑part, cross‑policy, scenario‑based complex questions, expand with short paragraphs or bullets covering each sub‑part; aim to minimize follow‑up questions by being complete and human‑digestible.
        • Always include a Sources section (mandatory) with:
          - exact document title(s)
          - section heading/title when inferable; otherwise note it’s an unheaded passage
          - for table‑sourced facts, note it is from a table in the cited document
        • Short supporting quotes are permitted only when they directly support a claim; keep them minimal and attribute them to the cited source.
        • Do not ask the user follow‑up questions in your final response.
        """,
    ).strip()

    # Follow-up generation policy
    if not generate_follow_up:
        core += "\n\nReturn an empty list for 'follow_up_questions' in the final JSON. You do NOT need to suggest follow-up questions."

    # Aggregate documents into a bounded section with optional logging of trimming/skips
    MAX_TOTAL_CHARS = 1000000
    joined: List[str] = []
    total = 0

    if aggregated_docs is not None:
        for idx, d in enumerate(aggregated_docs):
            title = d.get("title")
            content = d.get("content")
            if not isinstance(content, str) or not content:
                continue
            remaining = MAX_TOTAL_CHARS - total
            if remaining <= 0:
                if logger:
                    logger.info(
                        f"ask_llm: capacity exhausted before DOC_{idx+1} (title={title!r}) – skipping",
                    )
                break
            prefix = (
                f"Title: {title}\n\n"
                if isinstance(title, str) and title.strip()
                else ""
            )
            payload = prefix + content
            if len(payload) > remaining:
                snippet = payload[:remaining]
                if logger:
                    logger.info(
                        f"ask_llm: trimming DOC_{idx+1} (title={title!r}) from {len(payload)} to {len(snippet)} chars",
                    )
            else:
                snippet = payload
            joined.append(f"<DOC_{idx+1}>\n{snippet}\n</DOC_{idx+1}>")
            total += len(snippet)
    else:
        texts = aggregated_fulltexts or []
        for idx, t in enumerate(texts):
            if not isinstance(t, str) or not t:
                continue
            remaining = MAX_TOTAL_CHARS - total
            if remaining <= 0:
                if logger:
                    logger.info(
                        f"ask_llm: capacity exhausted before DOC_{idx+1} – skipping",
                    )
                break
            snippet = t if len(t) <= remaining else t[:remaining]
            if logger and len(t) > remaining:
                logger.info(
                    f"ask_llm: trimming DOC_{idx+1} (no title) from {len(t)} to {len(snippet)} chars",
                )
            joined.append(f"<DOC_{idx+1}>\n{snippet}\n</DOC_{idx+1}>")
            total += len(snippet)

    documents_block = "\n".join(
        [
            "Policy Documents (Full Text)",
            "----------------------------",
            *(joined if joined else ["<no documents provided>"]),
            "",
        ],
    )

    extra = (case_specific_instructions or "").strip()
    if extra:
        extra = f"\nCase-specific instructions\n--------------------------\n{extra}\n"

    parts: list[str] = [
        activity_block,
        core,
        "",
        documents_block,
        extra,
        f"Current UTC time: {_now()}.",
    ]

    return "\n".join(parts)


def build_intranet_update_instructions() -> str:
    return """
   INTRANET RAG — Update addendum (single-table)
   --------------------------------------------
   Use this *in addition* to the general KnowledgeManager update prompt. Keep it brief and non-redundant.

   Discovery (read-only)
   - Call `ask` to locate the exact row ids to modify.
     • The retrieval step must perform semantic search over **summary only**.
     • Do **not** add `content_text` to semantic `references`.
     • When exact numbers, dates, codes or quotes are needed: after obtaining candidate **row ids**
       from the summary search, use `_filter` on those ids to read `content_text` directly.
   - You may include clear structured constraints (specific ids, types, ranges) in the ask text to
     guide retrieval, but do not replace semantic matching with `_filter`—use `_filter` only to fetch
     the already-identified rows or to apply strict non-semantic constraints.

   - When the `ask` flow runs semantic retrieval, prefer a **tiered escalation** with `_search` on the unified table:
        1) `filter="content_type == 'paragraph'", k=8`  → precise hits
        2) `filter="content_type == 'section'",   k=5`   → broaden if still unclear
        3) `filter="content_type == 'document'",  k=3`   → coarse fallback
     Proceed to the next tier **only** if the prior tier cannot confidently identify targets.

   Writes (minimal, precise)
   - Use `update_rows` when you know the target ids; use `add_rows` to create new records.
   - Only introduce new columns when strictly necessary (`create_empty_column`), and prefer `rename_column`
     over drop+create for naming fixes.
   - Do **not** manage embedding/vector columns; they are handled internally.

   File ingestion (when requested)
   - Use `ingest_documents` to parse and insert records into the Content table (set the `table` argument appropriately).
     When replacing existing content, set `replace_existing=True`.

   Coherence & verification
   - When materially changing long-form text, update `summary` only if the user supplied one; otherwise leave it
     (a separate summarisation pipeline may refresh it later).
   - Perform a light re-check via `ask` (brief, read-only) to confirm the change, then reply with concise counts
     and key identifiers.

   Constraints
   - Keep changes additive where possible; avoid destructive operations unless explicitly requested.
   - Do not hallucinate schema or values; only reference columns that actually exist.
   - Minimise tool calls: discover with `ask` → write (`update_rows`/`add_rows`) → quick verification with `ask`.
   """.strip()
