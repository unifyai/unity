from __future__ import annotations


def build_paragraph_summary_prompt(*, embedding_budget_tokens: int) -> str:
    """Paragraph-level summarization prompt with strict preservation rules."""
    emb_budget = int(embedding_budget_tokens) if embedding_budget_tokens else 8000
    return f"""
You are a precision summarization assistant for diverse document types.

TASK: Create a comprehensive bullet-point summary that captures EVERY important detail from this paragraph.

CRITICAL RULES:
1. **PRESERVE ALL NUMERIC VALUES EXACTLY AS WRITTEN**
   - Keep original numbers, percentages, measurements, limits
   - Include units exactly as stated (any currency symbols, measurement units, percentages)
   - Never round, approximate, or paraphrase numbers

2. **MAINTAIN EXACT TERMINOLOGY**
   - Use the exact words/phrases from the source
   - Keep technical terms, acronyms, and proper nouns unchanged
   - Preserve domain-specific language verbatim

3. **CAPTURE ALL KEY INFORMATION**
   - Every requirement, rule, condition, or exception
   - All dates, deadlines, and timeframes
   - Each step in a process or procedure
   - All named entities (people, organizations, systems, locations)
   - Any referenced documents, standards, or external sources

4. **STRUCTURE FOR CLARITY**
   - Use bullet points for distinct facts
   - Group related information together
   - Maintain logical flow and relationships

5. **INCLUDE METADATA AT END**
   After the main summary, add:
   - Key Topics: [comma-separated list of 3-5 main concepts]
   - Named Entities: [notable names, organizations, systems]
   - Critical Values: [important numbers/measurements with units]

OUTPUT BUDGET:
- The ENTIRE summary must be ≤ {emb_budget} tokens (cl100k_base approximation).
- If needed, compress by removing redundancy and prose, but NEVER drop numeric values or technical terms.

TEMPLATE DIRECTIVE:
- If a COMPRESSION DIRECTIVE is appended at the end of this prompt, you MUST apply it while preserving all numbers and key terms.

PARAGRAPH TO SUMMARIZE:
""".lstrip()


def build_section_summary_prompt(*, embedding_budget_tokens: int) -> str:
    """Section-level summarization prompt (synthesize paragraph summaries)."""
    emb_budget = int(embedding_budget_tokens) if embedding_budget_tokens else 8000
    return f"""
You are synthesizing multiple summaries into a comprehensive higher-level summary.

TASK: Combine these paragraph summaries into a unified section summary that preserves all critical information.

RULES:
1. **PRESERVE NUMERIC PRECISION**
   - Carry forward ALL numbers, measurements, and units exactly
   - Never generalize specific values

2. **CONSOLIDATE WITHOUT LOSING DETAIL**
   - Merge related points while keeping specifics
   - Eliminate only true redundancy
   - Maintain all unique facts, requirements, conditions

3. **ORGANIZE HIERARCHICALLY**
   - Group related information logically
   - Use nested structure for complex relationships

4. **MAINTAIN RELATIONSHIPS**
   - Note connections between different parts
   - Highlight dependencies or prerequisites

5. **AGGREGATED METADATA**
   After the summary:
   - Combined Topics: [comprehensive topic list from all inputs]
   - All Named Entities: [complete list, no duplicates]
   - All Critical Values: [complete list with context]
   - Cross-References: [related documents/sections mentioned]

OUTPUT BUDGET:
- The ENTIRE summary must be ≤ {emb_budget} tokens (cl100k_base).
- Prioritize density and factual completeness over stylistic prose.

TEMPLATE DIRECTIVE:
- If a COMPRESSION DIRECTIVE is appended at the end of this prompt, you MUST apply it while preserving all numbers and key terms.

INPUT PARAGRAPH SUMMARIES TO SYNTHESIZE:
""".lstrip()


def build_document_summary_prompt(*, embedding_budget_tokens: int) -> str:
    """Document-level summarization prompt (synthesize section summaries)."""
    emb_budget = int(embedding_budget_tokens) if embedding_budget_tokens else 8000
    return f"""
You are creating a comprehensive document summary from multiple section summaries.

TASK: Synthesize these section summaries into a complete document overview that enables effective retrieval and understanding.

REQUIREMENTS:
1. **EXECUTIVE OVERVIEW** (2-3 sentences)
   - Document purpose and main subject matter
   - Target audience or stakeholders
   - Key outcomes or actions required

2. **STRUCTURED CONTENT SUMMARY**
   - Major topics and their relationships
   - All critical specifications, requirements, or guidelines
   - All numeric values/thresholds with context
   - Important temporal elements (dates, deadlines, durations)
   - Key processes, procedures, or methodologies

3. **PRESERVE ALL SPECIFICS**
   - Every unique number, measurement, or quantitative value
   - All named entities and references
   - Complete enumeration of requirements or specifications
   - All exceptions, edge cases, or special conditions

4. **RETRIEVAL OPTIMIZATION**
   - Include synonyms and alternative terms
   - Identify questions this document could answer
   - Extract searchable keywords and phrases

5. **COMPREHENSIVE METADATA**
   - Document Classification: [inferred from content]
   - Primary Topics: [comprehensive list]
   - All Entities: [organizations, people, systems, locations, products]
   - All Quantitative Data: [every number/measurement with context]
   - External References: [all mentioned documents/standards/sources]
   - Scope/Applicability: [domains, contexts, or conditions where this applies]

OUTPUT BUDGET:
- The ENTIRE document summary must be ≤ {emb_budget} tokens (cl100k_base).
- Use terse bullets when needed; retain ALL numbers and proper nouns verbatim.

TEMPLATE DIRECTIVE:
- If a COMPRESSION DIRECTIVE is appended at the end of this prompt, you MUST apply it while preserving all numbers and key terms.

SECTION SUMMARIES TO SYNTHESIZE:
""".lstrip()


def build_chunked_text_summary_prompt(
    chunk_number: int,
    total_chunks: int,
    *,
    embedding_budget_tokens: int,
) -> str:
    """Prompt for summarizing a chunk of text when a section/document is split."""
    emb_budget = int(embedding_budget_tokens) if embedding_budget_tokens else 8000
    return f"""
You are summarizing chunk {chunk_number} of {total_chunks} from a larger document.

TASK: Extract all important information from this text chunk while maintaining context awareness.

SPECIAL CONSIDERATIONS:
1. This is a partial document - note if information seems incomplete
2. Preserve any references to other parts of the document
3. Keep all specific details that might connect to other chunks
4. Flag any sentences that appear to be cut off

APPLY STANDARD RULES:
- Preserve ALL numbers, units, measurements exactly
- Keep technical terms and proper nouns unchanged
- Capture every requirement, condition, exception
- Note all entities and references

OUTPUT BUDGET:
- Keep the chunk summary ≤ {emb_budget} tokens (cl100k_base).
- If necessary, compress by removing redundancy; preserve numbers and key terms.

For incomplete elements, add markers:
- [CONTINUES FROM PREVIOUS] - if starting mid-sentence/thought
- [CONTINUES TO NEXT] - if ending mid-sentence/thought
- [PARTIAL INFORMATION] - if clearly missing context

TEXT CHUNK {chunk_number}/{total_chunks}:
""".lstrip()
