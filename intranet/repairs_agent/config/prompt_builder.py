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
            # Handle both dict and Pydantic model
            if hasattr(plot, "model_dump"):
                plot = plot.model_dump()
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


# ─────────────────────────────────────────────────────────────────────────────
# Consolidated Analysis Prompt (across all parameter combinations)
# ─────────────────────────────────────────────────────────────────────────────


def build_consolidated_analyst_prompt(
    metric_name: str,
    metric_description: str,
    metric_docstring: Optional[str],
    all_results: List[Dict[str, Any]],
) -> str:
    """
    Build user prompt for consolidated analysis across all parameter combinations.

    Generates a concise template targeting 800-1200 words (3-4 pages) output.

    Parameters
    ----------
    metric_name : str
        Name of the metric (e.g., "first_time_fix_rate")
    metric_description : str
        Short description from the registry
    metric_docstring : str, optional
        Full docstring explaining how the metric is calculated
    all_results : list
        List of result dicts, each with 'params' and 'raw_results' keys

    Returns
    -------
    str
        User prompt for consolidated analysis
    """

    # Helper functions for value extraction and formatting
    def get_value(r: dict) -> float:
        """Extract primary metric value for sorting."""
        val = (
            r.get("percentage")
            or r.get("rate")
            or r.get("distance_miles")
            or r.get("count")
            or r.get("value")
        )
        if val is None:
            return 0.0
        return float(val) if isinstance(val, (int, float)) else 0.0

    def format_val(r: dict) -> str:
        """Format value for display."""
        pct = r.get("percentage")
        distance = r.get("distance_miles")
        count = r.get("count")
        if pct is not None:
            return f"{pct}%"
        elif distance is not None:
            return f"{distance} mi"
        elif count is not None:
            return str(count)
        return str(r.get("value", "N/A"))

    sections = []

    # Collect all plots for placeholder generation
    all_plots = []

    # Title - use human-readable name
    readable_name = metric_name.replace("_", " ").title()
    sections.append(f"# {readable_name}")
    sections.append("")
    sections.append(f"**Definition**: {metric_description}")
    sections.append("")

    # Compact data section
    sections.append("---")
    sections.append("## Data Summary")
    sections.append("")

    for item in all_results:
        params = item.get("params", {})
        result = item.get("raw_results", {})
        group_by = params.get("group_by", "total")

        # Extract plots if available
        plots = result.get("plots", [])
        for plot in plots:
            if plot.get("url"):
                all_plots.append(
                    {
                        "group_by": group_by,
                        "title": plot.get("title", f"Chart by {group_by}"),
                    },
                )

        total = result.get("total", "N/A")
        results_list = result.get("results", [])

        sections.append(f"### By {group_by.title()}")
        sections.append("")

        if not results_list:
            sections.append(f"Overall: **{total}**")
            sections.append("")
            continue

        # Sort results
        sorted_results = sorted(results_list, key=get_value, reverse=True)
        num_groups = len(sorted_results)

        sections.append(f"Overall: **{total}** ({num_groups} groups)")
        sections.append("")

        # Compact table - top 5 and bottom 5 for large, all for small
        if num_groups <= 12:
            sections.append(f"| {group_by.title()} | Value |")
            sections.append("|---|---|")
            for r in sorted_results:
                group = r.get("group", "Unknown")
                sections.append(f"| {group} | {format_val(r)} |")
        else:
            sections.append("**Top 5:**")
            sections.append("")
            sections.append(f"| {group_by.title()} | Value |")
            sections.append("|---|---|")
            for r in sorted_results[:5]:
                group = r.get("group", "Unknown")
                sections.append(f"| {group} | {format_val(r)} |")
            sections.append("")
            sections.append("**Bottom 5:**")
            sections.append("")
            sections.append(f"| {group_by.title()} | Value |")
            sections.append("|---|---|")
            for r in sorted_results[-5:]:
                group = r.get("group", "Unknown")
                sections.append(f"| {group} | {format_val(r)} |")

        sections.append("")

    # Analysis template - structured and concise
    sections.append("---")
    sections.append("## Required Analysis Output")
    sections.append("")
    sections.append(
        "Generate a **concise analysis** (800-1200 words max) using this EXACT structure:",
    )
    sections.append("")
    sections.append("---")
    sections.append("")
    sections.append("### 1. Executive Summary")
    sections.append("_2-3 sentences: Key finding + overall assessment._")
    sections.append("")
    sections.append("### 2. Performance Overview")
    sections.append(
        "_1 paragraph: Overall rate/value, what it means, benchmark context if known._",
    )
    sections.append("")
    sections.append("### 3. Analysis by Grouping")
    sections.append("")
    sections.append(
        "For EACH grouping present in the data, write ONE focused paragraph:",
    )
    sections.append("")
    sections.append("**By [Grouping Name]**")
    sections.append("- Top/bottom performers (cite specific names and values)")
    sections.append("- Key pattern or insight")
    sections.append("- Any outliers or data quality notes")
    sections.append("")

    # Plot placeholders
    if all_plots:
        sections.append("### 4. Visualizations")
        sections.append("")
        for i, plot in enumerate(all_plots, 1):
            sections.append(f"**[PLOT {i}: {plot['title']}]**")
            sections.append("")
            sections.append("_[Screenshot to be inserted]_")
            sections.append("")

    sections.append("### 5. Recommendations")
    sections.append("_3-5 bullet points of actionable next steps._")
    sections.append("")
    sections.append("---")
    sections.append("")

    # Constraints
    sections.append("## CONSTRAINTS (must follow)")
    sections.append("")
    sections.append("- **MAX 1200 words** - be concise, no filler")
    sections.append("- **NO 'Data Sources' section** - omit internal references")
    sections.append("- **NO requests for more data** - work with what's given")
    sections.append("- **CITE SPECIFIC VALUES** - name operatives, patches, dates")
    sections.append("- **USE COMPACT TABLES** where comparing values")
    sections.append(
        "- **SKIP sections** if no relevant data (e.g., skip 'By Region' if not present)",
    )

    return "\n".join(sections)
