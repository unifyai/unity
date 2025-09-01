import json


def build_metadata_extraction_prompt() -> str:
    """Build prompt for LLM-based metadata extraction using Pydantic model validation."""
    from unity.file_manager.parser.types.document import DocumentMetadataExtraction

    # Get the Pydantic model schema
    schema = DocumentMetadataExtraction.model_json_schema()

    return f"""
DOCUMENT METADATA EXTRACTION FOR RETRIEVAL AUGMENTED GENERATION (RAG)

You are an expert document analyzer tasked with extracting comprehensive metadata.
Your goal is to enable effective semantic search and retrieval by capturing all relevant aspects of this document.

RESPONSE FORMAT:
Your response must be a valid JSON object that exactly matches this Pydantic model schema:

{json.dumps(schema, indent=2)}

FIELD GUIDELINES:

1. **document_type**: Choose from the exact literal values only
   - "policy": formal rules, regulations, or guidelines
   - "procedure": step-by-step instructions or processes
   - "guideline": recommendations or best practices
   - "handbook": comprehensive reference materials
   - "form": fillable documents or templates
   - "template": reusable document formats
   - "other": anything that doesn't fit above categories

2. **category**: Choose from the exact literal values only
   - "safety": health, safety, security, risk management
   - "hr": human resources, employment, staff matters
   - "finance": budgets, accounting, financial procedures
   - "operations": day-to-day operational procedures
   - "maintenance": repairs, upkeep, technical maintenance
   - "legal": contracts, compliance, regulatory matters
   - "tenancy": customer/client/tenant related matters
   - "general": cross-functional or uncategorized content

3. **key_topics**: 3-8 main topics in snake_case format
   - Use underscores between words (e.g., "fire_safety", "data_protection")
   - Focus on concepts that users might search for
   - Include both specific and general terms

4. **named_entities**: Organize discovered entities by type:
   - "organizations": Company names, departments, external bodies
   - "policies": Names of referenced policies or procedures
   - "locations": Specific places, buildings, or areas mentioned
   - "numbers": Important quantities, limits, percentages, measurements
   - "dates": Key dates, deadlines, time periods
   - "legislation": Laws, regulations, standards referenced
   - "people": Named individuals or roles (if applicable)
   - "systems": IT systems, software, tools mentioned

5. **content_tags**: 5-12 searchable keywords including:
   - Synonyms and alternative terms
   - Related concepts users might search for
   - Both formal and informal terminology
   - Singular and plural forms where relevant

6. **confidence_score**: Your confidence in extraction accuracy (0.0-1.0)
   - 0.8-1.0: High confidence, clear document with good structure
   - 0.6-0.8: Medium confidence, some ambiguity or missing context
   - Below 0.6: Low confidence, unclear or incomplete document

EXAMPLE OUTPUT:
{{
  "document_type": "policy",
  "category": "safety",
  "key_topics": ["mobility_equipment", "speed_limits", "weight_restrictions", "storage_rules", "fire_safety", "accessibility", "resident_safety"],
  "named_entities": {{
    "organizations": ["Property Management", "Health and Safety Department", "Fire Service"],
    "policies": ["Fire Safety Policy", "Accessibility Guidelines", "Equipment Storage Policy"],
    "locations": ["communal_areas", "storage_rooms", "evacuation_routes"],
    "numbers": ["4_mph", "8_mph", "150_kg", "300_kg", "2_meters"],
    "dates": ["annual_inspection", "monthly_checks"],
    "legislation": ["Fire_Safety_Act", "Disability_Discrimination_Act", "Health_Safety_Regulations"]
  }},
  "content_tags": ["mobility", "equipment", "scooter", "wheelchair", "storage", "safety", "fire_risk", "accessibility", "speed_limit", "weight_limit", "inspection"],
  "confidence_score": 0.95
}}

CRITICAL: Return ONLY the JSON object, no additional text or markdown formatting.

ANALYZE THE FOLLOWING DOCUMENT:
"""


def build_paragraph_summary_prompt() -> str:
    """Build prompt for paragraph-level summarization with strict preservation rules."""
    return """
You are a precision summarization assistant for diverse document types.

TASK: Create a comprehensive bullet-point summary that captures EVERY important detail from this paragraph.

CRITICAL RULES:
1. **PRESERVE ALL NUMERIC VALUES EXACTLY AS WRITTEN**
   - Keep original numbers, percentages, measurements, limits
   - Include units exactly as stated (any currency symbols, measurement units, percentages)
   - Never round, approximate, or paraphrase numbers
   - Examples: "4.5%" stays "4.5%", "$1,234.56" stays "$1,234.56", "100km/h" stays "100km/h"

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

PARAGRAPH TO SUMMARIZE:
"""


def build_section_summary_prompt() -> str:
    """Build prompt for section-level summarization from paragraph summaries."""
    return """
You are synthesizing multiple summaries into a comprehensive higher-level summary.

TASK: Combine these paragraph summaries into a unified section summary that preserves all critical information.

RULES:
1. **PRESERVE NUMERIC PRECISION**
   - Carry forward ALL numbers, measurements, and units exactly
   - Never generalize specific values
   - Keep all ranges, limits, and thresholds

2. **CONSOLIDATE WITHOUT LOSING DETAIL**
   - Merge related points while keeping specifics
   - Eliminate only true redundancy
   - Maintain all unique facts, requirements, conditions

3. **ORGANIZE HIERARCHICALLY**
   - Group related information logically
   - Use nested structure for complex relationships
   - Maintain clear information flow

4. **MAINTAIN RELATIONSHIPS**
   - Note connections between different parts
   - Highlight dependencies or prerequisites
   - Identify any conflicts or contradictions

5. **AGGREGATED METADATA**
   After the summary:
   - Combined Topics: [comprehensive topic list from all inputs]
   - All Named Entities: [complete list, no duplicates]
   - All Critical Values: [complete list with context]
   - Cross-References: [related documents/sections mentioned]

INPUT PARAGRAPH SUMMARIES TO SYNTHESIZE:
"""


def build_document_summary_prompt() -> str:
    """Build prompt for document-level summarization from section summaries."""
    return """
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

SECTION SUMMARIES TO SYNTHESIZE:
"""


def build_chunked_text_summary_prompt(chunk_number: int, total_chunks: int) -> str:
    """Build prompt for summarizing a chunk of text when document is split."""
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

For incomplete elements, add markers:
- [CONTINUES FROM PREVIOUS] - if starting mid-sentence/thought
- [CONTINUES TO NEXT] - if ending mid-sentence/thought
- [PARTIAL INFORMATION] - if clearly missing context

TEXT CHUNK {chunk_number}/{total_chunks}:
"""
