from __future__ import annotations


def build_table_catalog_prompt(*, embedding_budget_tokens: int) -> str:
    emb_budget = int(embedding_budget_tokens) if embedding_budget_tokens else 8000
    return f"""
You are summarizing a data table for Retrieval Augmented Generation (RAG).

TASK:
- Produce a dense, retrieval-optimized bullet summary of the table meaning.
- Use ONLY the provided table profile (columns, column descriptions, sample rows, and rules).

CRITICAL RULES:
1) Preserve ALL numeric values exactly as shown in sample rows (no rounding).
2) Preserve exact column names; do not rename columns.
3) If column descriptions/rules are provided, use them to infer semantics (but do not hallucinate).
4) Note important units, date formats, identifiers, and categorical values.
5) Mention relationships between columns when supported by the sample.

OUTPUT FORMAT:
- Bullet points grouped into sections:
  - Overview
  - Key Columns (with meaning/units)
  - Important Patterns / Constraints
  - Example Values (include representative values verbatim)
  - Query Hints (synonyms and likely questions this table answers)

OUTPUT BUDGET:
- The entire summary must be ≤ {emb_budget} tokens.
- If needed, compress by removing redundancy, but NEVER drop numeric values or column names.

TABLE PROFILE:
""".lstrip()


def build_spreadsheet_summary_prompt(*, embedding_budget_tokens: int) -> str:
    """
    Prompt for summarizing a *spreadsheet profile* (CSV/XLSX) for RAG.

    Input to this prompt is expected to be a bounded textual profile containing:
    - sheet names
    - table labels
    - columns
    - bounded sample rows (JSON)

    The output is used as `FileParseResult.summary` and must be embedding-safe.
    """
    emb_budget = int(embedding_budget_tokens) if embedding_budget_tokens else 8000
    return f"""
You are summarizing a spreadsheet (CSV/XLSX) for Retrieval Augmented Generation (RAG).

TASK:
- Produce a dense, retrieval-optimized bullet summary of what this spreadsheet contains.
- Use ONLY the provided spreadsheet profile (sheet names, table labels, columns, sample rows).

CRITICAL RULES:
1) Do NOT dump the full data. Summarize structure and meaning.
2) Preserve ALL numeric values exactly as shown in sample rows (no rounding).
3) Preserve exact column names; do not rename columns.
4) Mention sheet names and table labels so users can navigate.
5) Include query hints (synonyms, likely questions this data answers).

OUTPUT FORMAT (bullets):
- Overview (what the spreadsheet is about)
- Sheets and Tables (list, with short descriptions)
- Key Columns (meaning/units/identifiers)
- Notable Values / Constraints (only what appears in samples)
- Query Hints

OUTPUT BUDGET:
- The entire summary must be ≤ {emb_budget} tokens.
- If needed, compress by removing redundancy, but NEVER drop column names or numeric values from samples.

SPREADSHEET PROFILE:
""".lstrip()
