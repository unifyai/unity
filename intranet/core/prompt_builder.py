"""
Centralized prompts for the Midland Heart RAG system.
"""

from __future__ import annotations
import json


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


def build_intranet_ask_instructions(*, generate_follow_up: bool = False) -> str:
    text = """
   INTRANET RAG — Retrieval addendum (single-table)
   -----------------------------------------------
   Use this *in addition* to the general KnowledgeManager ask prompt. Keep it brief and non-redundant.

   Table choice
   - Work with the table that exposes **summary** (dense text) and **content_text** (longer body).
     If multiple tables exist, prefer the one that clearly stores unified content.

   Semantic retrieval (primary — search **summary only**)
   - Perform semantic search with `_search` using **only** the summary column:
       • `{"summary": "<user_query>"}`
   - Do **not** include `content_text` in semantic `references`.
   - If precise numbers/quotes/dates are required, first get the **row ids** from the summary search,
     then fetch those rows directly with `_filter` to read `content_text` (no semantic over `content_text`).
   - Choose a modest `k` to minimise noise.

   Tiered passes (fast → slow; expand only if needed)
   - If the initial pass is insufficient, escalate in a **single-table** hierarchy using `_search` with filters:
       1) Paragraphs (highest precision):    `filter="content_type == 'paragraph'", k=10`
       2) Sections (broader context):        `filter="content_type == 'section'",   k=6`
       3) Documents (coarsest fallback):     `filter="content_type == 'document'",  k=3`
   - Move to the next tier **only** when evidence is insufficient or you need more context.

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
   - Use `ingest_documents` to parse and insert records into the content table (set the `table` argument appropriately).
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
