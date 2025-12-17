from __future__ import annotations

import json


def build_metadata_extraction_prompt(*, schema_json: dict) -> str:
    """Prompt for LLM-based metadata extraction using Pydantic model validation."""
    return f"""
DOCUMENT METADATA EXTRACTION FOR RETRIEVAL AUGMENTED GENERATION (RAG)

You are an expert document analyzer tasked with extracting comprehensive metadata.
Your goal is to enable effective semantic search and retrieval by capturing all relevant aspects of this document.

RESPONSE FORMAT:
Your response must be a valid JSON object that exactly matches this Pydantic model schema:

{json.dumps(schema_json, indent=2)}

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
""".lstrip()
