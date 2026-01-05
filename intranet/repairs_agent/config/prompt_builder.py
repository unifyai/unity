"""
Build business context for CodeActActor from FilePipelineConfig.

This module transforms the JSON configuration (used for ingestion and static demo)
into a structured prompt section for the dynamic CodeActActor agent.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default path to the FilePipelineConfig JSON
DEFAULT_CONFIG_PATH = (
    Path(__file__).parent.parent.parent
    / "repairs"
    / "repairs_file_pipeline_config_5m.json"
)


def build_repairs_business_context(
    config_path: Optional[Path] = None,
) -> str:
    """
    Build business context string for CodeActActor system prompt.

    Loads FilePipelineConfig and extracts:
    - Global business rules
    - File-level context
    - Table/column definitions

    Parameters
    ----------
    config_path : Path, optional
        Path to FilePipelineConfig JSON. Defaults to repairs_file_pipeline_config_5m.json

    Returns
    -------
    str
        Formatted business context for prompt injection
    """
    config_path = config_path or DEFAULT_CONFIG_PATH

    try:
        with open(config_path) as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.warning(f"Config not found at {config_path}, using fallback context")
        return _build_fallback_context()

    business_contexts = config.get("ingest", {}).get("business_contexts", {})

    sections = []

    # Domain overview
    sections.append("#### Domain Overview")
    sections.append(
        "You are analyzing repairs and telematics data for a housing association.",
    )
    sections.append("Data covers July-November 2025.")
    sections.append("")

    # Global rules
    global_rules = business_contexts.get("global_rules", [])
    if global_rules:
        sections.append("#### Global Business Rules")
        for rule in global_rules:
            sections.append(f"- {rule}")
        sections.append("")

    # File-level context
    file_contexts = business_contexts.get("file_contexts", [])
    if file_contexts:
        sections.append("#### Data Files")
        for fc in file_contexts:
            file_path = fc.get("file_path", "Unknown")
            file_rules = fc.get("file_rules", [])
            sections.append(f"**{Path(file_path).name}**:")
            for rule in file_rules:
                sections.append(f"  - {rule}")
        sections.append("")

    # Key filter patterns
    sections.append("#### Key Filter Patterns")
    sections.append(
        "- **Completed jobs**: `WorksOrderStatusDescription` in ['Complete', 'Closed']",
    )
    sections.append("- **First time fix**: `FirstTimeFix` == 'Yes'")
    sections.append("- **No access**: `NoAccess` != 'None' and `NoAccess` != ''")
    sections.append("- **Follow-on required**: `FollowOn` == 'Yes'")
    sections.append("")

    # Column mappings
    sections.append("#### Column Mappings for Grouping")
    sections.append("- `GroupBy.OPERATIVE` → `OperativeWhoCompletedJob`")
    sections.append("- `GroupBy.PATCH` → `RepairsPatch`")
    sections.append("- `GroupBy.REGION` → `RepairsRegion`")
    sections.append("- `GroupBy.TOTAL` → None (aggregate all)")
    sections.append("")

    return "\n".join(sections)


def _build_fallback_context() -> str:
    """Build minimal fallback context if config is not available."""
    return """
#### Domain Overview
You are analyzing repairs and telematics data.

#### Discovery Required
Use `primitives.files.tables_overview()` to discover available tables.
Use `primitives.files.list_columns(table=...)` to understand column structure.
"""


# ─────────────────────────────────────────────────────────────────────────────
# System Prompt Extension for CodeActActor
# ─────────────────────────────────────────────────────────────────────────────
_REPAIRS_SYSTEM_PROMPT_TEMPLATE = """
### Domain: Repairs & Telematics Analysis

You are analyzing repairs operations data for a housing association.

{business_context}

### Workflow

1. **Search** for existing metric functions via FunctionManager - study their source code
2. **Discover** available tables via `primitives.files.tables_overview()`
3. **Compose** a solution using discovered functions or primitives
4. **Visualize** when charts/plots are requested
"""


def build_repairs_system_prompt(config_path: Optional[Path] = None) -> str:
    """
    Build the complete system prompt extension for repairs CodeActActor.

    Combines the workflow template with business context from FilePipelineConfig.

    Parameters
    ----------
    config_path : Path, optional
        Path to FilePipelineConfig JSON. Defaults to repairs_file_pipeline_config_5m.json

    Returns
    -------
    str
        Complete system prompt extension ready for CodeActActor
    """
    business_context = build_repairs_business_context(config_path)
    return _REPAIRS_SYSTEM_PROMPT_TEMPLATE.format(business_context=business_context)
