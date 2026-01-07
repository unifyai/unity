"""
Prompt builders for the Repairs Agent.

This module provides:
1. Business context extraction from FilePipelineConfig
2. System prompt for CodeActActor (dynamic agent)
3. Analyst prompts for BespokeRepairsAgent (static agent) - generates qualitative
   insights from raw metric results using an LLM
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

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


# ─────────────────────────────────────────────────────────────────────────────
# Analyst Prompt for BespokeRepairsAgent (Static Agent)
# ─────────────────────────────────────────────────────────────────────────────

_ANALYST_SYSTEM_PROMPT = """You are a senior data analyst at Midland Heart, one of the UK's largest housing associations managing over 33,000 homes across the Midlands.

Your role is to translate raw metric data into actionable business insights for the Repairs & Maintenance leadership team. You have deep expertise in:
- Housing association operations and KPIs
- Repairs and maintenance industry benchmarks
- Telematics and fleet management
- Operative performance analysis

When analyzing data, you should:
1. Interpret numbers in business context (what do they mean for operations?)
2. Identify trends, patterns, and outliers worth investigating
3. Compare against industry standards where relevant
4. Highlight actionable insights and potential improvements
5. Present findings clearly with tables where appropriate

Write in a professional but accessible tone. Use markdown formatting for clarity."""


def build_analyst_system_prompt(config_path: Optional[Path] = None) -> str:
    """Build system prompt for the analyst LLM."""
    business_context = build_repairs_business_context(config_path)
    return f"{_ANALYST_SYSTEM_PROMPT}\n\n### Business Context\n\n{business_context}"


def build_analyst_user_prompt(
    metric_name: str,
    metric_description: str,
    metric_docstring: Optional[str],
    params: Dict[str, Any],
    results: Dict[str, Any],
    plots: List[Dict[str, Any]],
) -> str:
    """
    Build user prompt for analyst LLM with metric context and raw results.

    Parameters
    ----------
    metric_name : str
        Name of the metric (e.g., "first_time_fix_rate")
    metric_description : str
        Short description from the registry
    metric_docstring : str, optional
        Full docstring explaining how the metric is calculated
    params : dict
        Parameters used for this query
    results : dict
        Raw MetricResult as a dictionary
    plots : list
        List of plot results with URLs

    Returns
    -------
    str
        User prompt for the analyst LLM
    """
    sections = []

    # Header
    sections.append(f"## Metric Analysis Request: {metric_name}")
    sections.append("")

    # Description
    sections.append(f"**Description**: {metric_description}")
    sections.append("")

    # Parameters used
    if params:
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        sections.append(f"**Parameters**: {param_str}")
    else:
        sections.append("**Parameters**: (defaults)")
    sections.append("")

    # How metric was calculated (from docstring)
    if metric_docstring:
        # Extract the meaningful parts of the docstring
        sections.append("### How This Metric Is Calculated")
        sections.append("")
        sections.append(metric_docstring.strip())
        sections.append("")

    # Raw results
    sections.append("### Raw Results")
    sections.append("")
    sections.append("```json")
    sections.append(json.dumps(results, indent=2, default=str))
    sections.append("```")
    sections.append("")

    # Visualizations
    if plots:
        sections.append("### Visualizations Generated")
        sections.append("")
        for plot in plots:
            title = plot.get("title", "Untitled")
            url = plot.get("url")
            if url:
                sections.append(f"- **{title}**: {url}")
            else:
                error = plot.get("error", "Unknown error")
                sections.append(f"- **{title}**: ⚠️ Failed - {error}")
        sections.append("")

    # Request
    sections.append("### Your Task")
    sections.append("")
    sections.append(
        "Analyze the above results and provide a comprehensive business analysis including:",
    )
    sections.append("1. **Executive Summary** - Key findings in 2-3 sentences")
    sections.append("2. **Detailed Analysis** - What the numbers tell us")
    sections.append("3. **Performance Insights** - Trends, outliers, comparisons")
    sections.append("4. **Recommendations** - Actionable next steps if any")
    sections.append("")
    sections.append(
        "Format your response with clear headings and use tables where helpful.",
    )

    return "\n".join(sections)
