# Repairs Analysis Agent - Technical Specification

**Version**: 0.2.0
**Status**: Draft
**Last Updated**: 2026-01-05

---

## 1. Executive Summary

This specification defines the architecture for a **dual-mode Repairs Analysis Agent** that provides:

1. **Static Demo (BespokeRepairsAgent)**: LLM-free, direct execution of hardcoded metric query functions
2. **Dynamic Demo (CodeActActor)**: LLM-powered agent that discovers, retrieves, and orchestrates metric functions and/or FileManager primitives to answer complex analytical queries

Both modes share a **single source of truth**: the metric query functions in `metrics/definitions.py`, which are pure chains of FileManager tool calls (with optional minimal Python aggregation logic).

### Key Architecture Principles

- **Business Context Injection**: CodeActActor receives domain-specific business context via the same slot-filling pattern used by `FileManager.ask`
- **Discovery-First Pattern**: Metric functions should use `tables_overview()` for schema discovery rather than hardcoding table paths
- **Semantic Layer**: Business rules, table definitions, and column descriptions provide the "semantic layer" that makes data understandable to the LLM
- **Wrapper Pattern**: CodeActActor wraps FileManager primitives with business context, not raw primitives

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    REPAIRS ANALYSIS AGENT PACKAGE                           │
│                    intranet/repairs_agent/                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                    BUSINESS CONTEXT LAYER                              │ │
│  │                                                                        │ │
│  │   config/                                                              │ │
│  │   ├── repairs_file_pipeline_config.json  # Single source of truth     │ │
│  │   └── prompt_builder.py                  # Builds BusinessContextPayload│ │
│  │                                                                        │ │
│  │   Provides:                                                            │ │
│  │   - Global business rules (e.g., "Complete/Closed = completed job")    │ │
│  │   - File-level context (e.g., "This file contains repairs data")       │ │
│  │   - Table/column definitions (semantics stored at ingestion time)      │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                              │                                              │
│                              │ injects into                                 │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                   SINGLE SOURCE OF TRUTH                             │   │
│  │                                                                      │   │
│  │   metrics/                                                          │   │
│  │   ├── definitions.py   # Metric functions (discovery-first pattern) │   │
│  │   ├── helpers.py       # Reusable compositional helpers             │   │
│  │   ├── plots.py         # Plot configurations per metric             │   │
│  │   └── types.py         # GroupBy, TimePeriod, MetricResult          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              │ used by                                      │
│                              ▼                                              │
│  ┌──────────────────────┐    ┌────────────────────────────────────────┐    │
│  │   STATIC DEMO        │    │        DYNAMIC DEMO                    │    │
│  │                      │    │                                        │    │
│  │  static/             │    │  dynamic/                              │    │
│  │  ├── agent.py        │    │  ├── agent.py (CodeActActor wrapper)   │    │
│  │  │   (BespokeAgent)  │    │  │   + BusinessContextPayload          │    │
│  │  └── cli.py          │    │  ├── sync.py (sync to FunctionManager) │    │
│  └──────────────────────┘    │  └── cli.py                            │    │
│                              └────────────────────────────────────────┘    │
│                                                                             │
│  scripts/                    # Shell scripts for running both demos        │
│  data/                       # Excel files (repairs + telematics)          │
│  tests/                      # Unit tests                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Design Decisions

### 3.1 Metric Function Purity

Metric functions are designed as **pure chains of FileManager tool calls** with minimal Python logic:

- **Primary approach**: Pure FileManager tool call chains (`reduce`, `filter_files`, `visualize`)
- **Allowed**: Simple Python for aggregation (percentage calculation, result normalization)
- **NOT allowed**: External libraries beyond Python stdlib
- **Helpers**: Common reusable operations (normalization, percentage) extracted to `helpers.py`
- **LLM visibility**: Full source code + docstring available via FunctionManager

### 3.2 FunctionManager Integration

- **Sync strategy**: Separate `sync.py` script (not auto-sync on startup)
- **When to sync**: One-time offline sync; re-sync when implementations change
- **Sync command**: `python -m intranet.repairs_agent.dynamic.sync`

### 3.3 Implementation Visibility for CodeActActor

The CodeActActor receives:
- **Full implementation source code** via `search_functions_by_similarity()`
- **Complete docstrings** explaining the tool chain and parameters
- **Dependency chain** via `calls` field for understanding composition

### 3.4 On-the-fly Composition Freedom

CodeActActor is free to:
- Search and use existing metric functions directly
- Understand metric implementations and compose variations
- Use FileManager primitives (`primitives.files.reduce()`, etc.) directly
- Mix compositional functions with primitives as needed
- No restrictions on orchestration approach

### 3.5 Data Domain

- **Unified domain**: Repairs and telematics treated as single "repairs analysis" domain
- **Separate tables**: Data remains in separate FileManager tables
- **Joins**: Available via FileManager (though not yet exposed as primitive)
- **Clear naming**: Metric names indicate data source where relevant

### 3.6 Visualization

- **`include_plots` flag**: Part of metric functions, returns plots in `MetricResult`
- **CodeActActor choice**: LLM decides when to enable plots based on user query
- **Direct primitive**: `primitives.files.visualize()` also available for custom visualizations

---

## 4. Business Context Architecture

### 4.1 The Problem: CodeActActor Has No Domain Knowledge

Unlike `FileManager.ask()` which receives a `BusinessContextPayload` containing:
- Global rules ("Vehicle field contains operative name")
- File-level context ("This is repairs data from July-Nov 2025")
- Table/column definitions ("FirstTimeFix: Yes means fixed on first visit")

The vanilla `CodeActActor` has **no mechanism** for receiving business context. It only gets:
- `environments` (computer_primitives, primitives)
- `tools` (execute_python_code, FunctionManager tools)
- Access to `primitives.files` in the sandbox

**Result**: CodeActActor would blindly call FileManager primitives without understanding:
- What tables exist and what they mean
- What columns exist and their business semantics
- Business rules for filtering (e.g., what constitutes a "completed" job)
- How file paths on disk map to context paths in FileManager

### 4.2 What's Stored vs. Runtime Context

During file ingestion, FileManager stores:
- ✅ `column_descriptions` - Returned by `list_columns()`, `schema_explain()`
- ✅ `table_description` - Returned by `tables_overview()`

**NOT stored** (comes from config at runtime):
- ❌ Global rules (e.g., "Complete/Closed status = completed job")
- ❌ File-level rules (e.g., "This file covers July-Nov 2025")
- ❌ Table-level rules (e.g., "Use JobTicketReference as unique job ID")

### 4.3 Context Path Mapping Problem

**Critical Issue**: Files on disk don't map 1:1 to FileManager context paths.

```
Disk path:
  /home/.../MDH Repairs Data July - Nov 25 - DL V1.xlsx

Becomes FileManager context:
  {base}/Files/{alias}/{safe_path}/Tables/{table_name}

Example:
  Files/Repairs/MDH_Repairs_Data_July_-_Nov_25_-_DL_V1.xlsx/Tables/Raised_01-07-2025_to_30-11-2025
```

Transformations applied:
- Whitespace → underscore
- Path truncation beyond threshold
- Safe character mapping
- Alias prepending

**Implication**: Hardcoded table paths in metrics are brittle. The LLM (or the metric function) must **discover** actual context paths at runtime.

### 4.4 Solution: Business Context Injection for CodeActActor

Wire `BusinessContextPayload` into `CodeActActor` using the same slot-filling pattern as `FileManager.ask`:

```python
# In DynamicRepairsAgent.__init__():
from intranet.repairs_agent.config.prompt_builder import build_repairs_code_act_context

# Load business context from FilePipelineConfig
self._business_payload = build_repairs_code_act_context()

# Pass to CodeActActor via system prompt extension
self._actor = CodeActActor(
    function_manager=...,
    environments=...,
    business_payload=self._business_payload,  # NEW: wire in business context
)
```

The `build_repairs_code_act_context()` function:
1. Loads `FilePipelineConfig` from JSON
2. Extracts global rules, file rules, table rules
3. Formats into a structured prompt section
4. Returns `BusinessContextPayload` compatible with slot-filling

### 4.5 Slot-Filling Template for CodeActActor

Add to `CodeActActor` prompt:

```
## Domain Context

{business_context.domain_overview}

### Data Tables Available
{business_context.tables_summary}

### Business Rules
{business_context.global_rules}

### Column Definitions
Use `primitives.files.list_columns(table=...)` or `primitives.files.schema_explain(table=...)`
to get column definitions. Key columns include:
{business_context.key_columns}

### Discovery Tools
Before querying data, use these tools to understand the schema:
- `primitives.files.tables_overview()` - List all available tables with context paths
- `primitives.files.list_columns(table=...)` - Get column names and descriptions
- `primitives.files.schema_explain(table=...)` - Get detailed schema explanation
```

---

## 5. Metric Function Refactoring Strategy

### 5.1 Current Problem: Hardcoded Paths

Current metrics hardcode table paths as global constants:

```python
# Current (PROBLEMATIC for CodeActActor)
REPAIRS_FILE = "/home/hmahmood24/unity/intranet/repairs/MDH Repairs Data July - Nov 25 - DL V1.xlsx"
REPAIRS_TABLE = f"{REPAIRS_FILE}.Tables.Raised_01-07-2025_to_30-11-2025"

@register("first_time_fix_rate", "...")
async def first_time_fix_rate(tools: ToolsDict, ...):
    reduce_tool = tools.get("reduce")
    raw_ftf = reduce_tool(
        table=REPAIRS_TABLE,  # ← Hardcoded! LLM can't learn this pattern
        ...
    )
```

**Problems**:
1. LLM sees hardcoded string, not the discovery process
2. Paths may change between environments
3. No pattern for CodeActActor to learn and replicate for novel queries

### 5.2 Refactoring Options

#### Option A: Discovery-First Pattern (Recommended for Production)

Metrics discover table paths at runtime:

```python
async def first_time_fix_rate(
    files: "FileManager",
    *,
    group_by: GroupBy = GroupBy.OPERATIVE,
    ...
) -> MetricResult:
    """
    Calculate First Time Fix rate.

    Discovery Chain:
        1. files.tables_overview() - Find repairs table path
        2. files.list_columns(table=...) - Verify required columns exist
        3. files.reduce(...) - Aggregate FTF jobs
        4. files.reduce(...) - Aggregate total completed (for percentage)
        5. Python division for percentage
        6. files.visualize(...) - Generate plots if requested

    Example
    -------
    >>> result = await first_time_fix_rate(files, group_by=GroupBy.REGION)
    """
    # Step 1: Discover repairs table
    tables = files.tables_overview()
    repairs_table = _find_table_by_pattern(tables, pattern="Repairs")

    if not repairs_table:
        raise ValueError("Repairs table not found. Run tables_overview() to see available tables.")

    # Step 2: Verify required columns
    columns = files.list_columns(table=repairs_table)
    required = ["FirstTimeFix", "JobTicketReference", "WorksOrderStatusDescription"]
    missing = [c for c in required if c not in columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Step 3-6: Rest of implementation...
```

**Pros**:
- LLM sees complete discovery → query trajectory
- Works across environments (no hardcoded paths)
- Teaches the "right way" to compose queries

**Cons**:
- More verbose implementation
- Extra API calls for discovery (can be cached)
- Slower execution (negligible for demo)

#### Option B: Parameterized Paths (Recommended for Demo)

Accept table paths as parameters with defaults from constants:

```python
# Constants as defaults (can be overridden)
DEFAULT_REPAIRS_TABLE = "Files/Repairs/MDH_Repairs_Data.../Tables/Raised_..."

async def first_time_fix_rate(
    files: "FileManager",
    *,
    table: str = DEFAULT_REPAIRS_TABLE,  # Explicit, overridable
    group_by: GroupBy = GroupBy.OPERATIVE,
    ...
) -> MetricResult:
    """
    Calculate First Time Fix rate.

    Parameters
    ----------
    files : FileManager
        FileManager instance
    table : str, default DEFAULT_REPAIRS_TABLE
        Table context path. Use `files.tables_overview()` to discover available tables.
    ...

    Tool Chain:
        1. files.reduce(table=table, ...) - Count FTF jobs
        ...
    """
    raw_ftf = files.reduce(
        table=table,  # Parameterized, not hardcoded
        metric="count",
        ...
    )
```

**Pros**:
- Minimal changes to existing code
- LLM sees table path is a parameter, not magic constant
- Works for static demo (uses defaults)
- Works for CodeActActor (can discover and pass)

**Cons**:
- Still relies on constants for static demo
- Discovery not demonstrated in implementation

#### Option C: Hybrid - Discovery in Docstring, Constants in Code (Demo Shortcut)

Keep constants but document discovery pattern explicitly:

```python
# Global constants (for static demo efficiency)
REPAIRS_TABLE = "Files/Repairs/.../Tables/Raised_..."

async def first_time_fix_rate(
    files: "FileManager",
    ...
) -> MetricResult:
    """
    Calculate First Time Fix rate.

    IMPORTANT FOR CODEACT COMPOSITION:
    This metric uses REPAIRS_TABLE constant which maps to the repairs data table.
    For on-the-fly composition, discover tables with:
        tables = files.tables_overview()
        repairs_table = [t for t in tables if "Repairs" in t["name"]][0]["path"]

    Tool Chain:
        1. files.reduce(table=REPAIRS_TABLE, metric="count", keys="JobTicketReference",
                       filter="`FirstTimeFix` == 'Yes'", group_by=...)
        2. files.reduce(...) for total completed
        3. Python percentage calculation
        4. files.visualize(...) for plots

    Filter Patterns:
        - Completed jobs: `WorksOrderStatusDescription` in ['Complete', 'Closed']
        - First time fix: `FirstTimeFix` == 'Yes'
        - No access: `NoAccess` != 'None' and `NoAccess` != ''

    Column Mappings:
        - GroupBy.OPERATIVE → "OperativeWhoCompletedJob"
        - GroupBy.PATCH → "RepairsPatch"
        - GroupBy.REGION → "RepairsRegion"
    """
```

**Pros**:
- Zero code changes to existing metrics
- LLM gets explicit guidance on discovery and patterns
- Rich examples of filter syntax and column mappings

**Cons**:
- Docstring gets verbose
- Discovery not enforced, just documented

### 5.3 Recommendation for Demo (Tomorrow)

**Use Option C (Hybrid)** for the demo:

1. **Keep existing metric implementations as-is** (constants, hardcoded paths)
2. **Enrich docstrings** with:
   - Explicit discovery instructions for CodeActActor
   - Filter pattern examples
   - Column mapping documentation
   - Tool chain with exact arguments
3. **Add business context injection** to `DynamicRepairsAgent`
4. **Add discovery tools to system prompt** (`tables_overview`, `list_columns`, `schema_explain`)

For production (post-demo), refactor to **Option A (Discovery-First)**.

### 5.4 Gold Standard Metric Function Template

Here's the recommended docstring structure for CodeActActor consumption:

```python
@register("metric_name", "Short description")
async def metric_name(
    tools: ToolsDict,
    group_by: GroupBy = GroupBy.OPERATIVE,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_period: TimePeriod = TimePeriod.DAY,
    include_plots: bool = False,
) -> MetricResult:
    """
    [One-line summary of what this metric measures]

    [2-3 sentence business context explaining why this metric matters]

    Discovery Pattern (for CodeActActor composition):
    -------------------------------------------------
    To replicate or adapt this metric:

        # 1. Discover available tables
        tables = files.tables_overview()
        repairs_table = next(t["path"] for t in tables if "Repairs" in t.get("name", ""))

        # 2. Check columns
        columns = files.list_columns(table=repairs_table)
        # Required: [list required columns]

    Tool Chain:
    -----------
        1. reduce(table=REPAIRS_TABLE, metric="count", keys="JobTicketReference",
                 filter="[exact filter expression]", group_by="[column name]")
           → Returns: Dict[str, {"count": int}]

        2. [Additional tool calls...]

        3. Python: [aggregation logic]

        4. visualize(...) if include_plots=True

    Filter Expressions Used:
    ------------------------
        - Base filter: `[exact expression]`
        - Date filter: `[column] >= '[YYYY-MM-DD]' and [column] <= '[YYYY-MM-DD]'`

    Column Mappings:
    ----------------
        - GroupBy.OPERATIVE → "OperativeWhoCompletedJob"
        - GroupBy.PATCH → "RepairsPatch"
        - GroupBy.REGION → "RepairsRegion"
        - GroupBy.TOTAL → None (no grouping)

    Parameters
    ----------
    [standard parameter docs]

    Returns
    -------
    MetricResult
        [description of result structure]

    Example
    -------
    >>> result = await metric_name(tools, group_by=GroupBy.REGION, include_plots=True)
    >>> print(f"Overall: {result.total}%")
    """
```

---

## 6. Component Specifications

### 6.1 Metric Functions (Single Source of Truth)

**Location**: `intranet/repairs_agent/metrics/definitions.py`

**Design Principles**:
- Each metric function is a **pure chain of FileManager tool calls**
- Minimal Python logic allowed (aggregation, percentage calculation) - **no external libraries**
- All custom logic (normalization, grouping) extracted to **compositional helpers**
- Functions receive `tools: ToolsDict` (dict of FileManager tools) or `files: FileManager` as first parameter
- Full source code + docstring must be LLM-readable for composition
- **Docstrings include discovery patterns and exact tool call arguments for CodeActActor learning**

**Current Signature** (works for static demo):

```python
async def first_time_fix_rate(
    tools: ToolsDict,  # Dict containing reduce, filter_files, visualize, etc.
    *,
    group_by: GroupBy = GroupBy.OPERATIVE,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_period: TimePeriod = TimePeriod.DAY,
    return_absolute: bool = False,
    include_plots: bool = False,
) -> MetricResult:
```

**Target Signature** (for production, enables discovery):

```python
async def first_time_fix_rate(
    files: "FileManager",  # FileManager instance with all tools
    *,
    table: Optional[str] = None,  # None = discover via tables_overview()
    group_by: GroupBy = GroupBy.OPERATIVE,
    ...
) -> MetricResult:
```

### 6.2 Compositional Helpers

**Location**: `intranet/repairs_agent/metrics/helpers.py`

**Purpose**: Reusable pure Python functions for common operations across metrics

```python
"""
Compositional helper functions for metric calculations.

These helpers are stored in FunctionManager alongside metric functions,
enabling CodeActActor to discover and use them for custom compositions.
"""

from typing import Any, Dict, Optional


def normalize_grouped_result(
    result: Any,
    extract_fn: callable = None,
) -> Dict[str, int]:
    """
    Normalize FileManager reduce result to simple {group: value} format.

    The reduce tool can return values in different formats:
    - Direct int: 123
    - Dict with 'count' key: {'shared_value': None, 'count': 123}
    - Dict with nested structure

    Parameters
    ----------
    result : Any
        Raw result from files.reduce() call
    extract_fn : callable, optional
        Custom extraction function, defaults to extracting 'count'

    Returns
    -------
    dict
        Normalized mapping of group name to numeric value

    Example
    -------
    >>> raw = {"North": {"count": 50}, "South": {"count": 75}}
    >>> normalize_grouped_result(raw)
    {"North": 50, "South": 75}
    """
    ...


def compute_percentage(numerator: int, denominator: int) -> float:
    """
    Safely compute percentage with zero-division handling.

    Parameters
    ----------
    numerator : int
        The count for the subset
    denominator : int
        The total count

    Returns
    -------
    float
        Percentage rounded to 2 decimal places, or 0.0 if denominator is 0

    Example
    -------
    >>> compute_percentage(85, 100)
    85.0
    >>> compute_percentage(0, 0)
    0.0
    """
    if denominator == 0:
        return 0.0
    return round((numerator / denominator) * 100, 2)


def build_date_filter(
    base_filter: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
    date_column: str = "`VisitDate`",
) -> Optional[str]:
    """
    Build combined filter expression with date range constraints.

    Parameters
    ----------
    base_filter : str, optional
        Base filter expression (e.g., status conditions)
    start_date : str, optional
        Start date in YYYY-MM-DD format
    end_date : str, optional
        End date in YYYY-MM-DD format
    date_column : str, default "`VisitDate`"
        Column name for date filtering

    Returns
    -------
    str or None
        Combined filter expression, or None if no filters

    Example
    -------
    >>> build_date_filter("`Status` == 'Complete'", "2025-07-01", "2025-09-30")
    "(`Status` == 'Complete') and `VisitDate` >= '2025-07-01' and `VisitDate` <= '2025-09-30'"
    """
    filters = []
    if base_filter:
        filters.append(f"({base_filter})")
    if start_date:
        filters.append(f"{date_column} >= '{start_date}'")
    if end_date:
        filters.append(f"{date_column} <= '{end_date}'")
    return " and ".join(filters) if filters else None


def resolve_group_by(group_by: "GroupBy") -> Optional[str]:
    """
    Resolve GroupBy enum to actual column name for reduce queries.

    Parameters
    ----------
    group_by : GroupBy
        Grouping dimension enum value

    Returns
    -------
    str or None
        Column name for grouping, or None for total aggregation

    Mapping
    -------
    GroupBy.OPERATIVE → "OperativeWhoCompletedJob"
    GroupBy.PATCH → "RepairsPatch"
    GroupBy.REGION → "RepairsRegion"
    GroupBy.TOTAL → None
    """
    ...
```

### 6.3 Static Demo Agent

**Location**: `intranet/repairs_agent/static/agent.py`

No changes from original spec. Uses existing metric functions directly with `ToolsDict`.

### 6.4 Dynamic Demo Agent (Updated)

**Location**: `intranet/repairs_agent/dynamic/agent.py`

**Key Change**: Inject `BusinessContextPayload` into the system prompt.

```python
"""
DynamicRepairsAgent: LLM-powered agent for natural language repairs analysis.

This agent uses CodeActActor to interpret natural language queries, discover
relevant metric functions from FunctionManager, and compose Python code to
answer complex analytical questions about repairs and telematics data.

Key Architecture:
- Business context injection via slot-filling (same pattern as FileManager.ask)
- Discovery-first guidance in system prompt
- Access to pre-built metric functions AND raw FileManager primitives
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from unity.actor.code_act_actor import CodeActActor
from unity.actor.environments import StateManagerEnvironment
from unity.actor.handle import ActorHandle
from unity.function_manager.primitives import Primitives
from unity.image_manager.types.image_refs import ImageRefs
from unity.image_manager.types.raw_image_ref import RawImageRef
from unity.image_manager.types.annotated_image_ref import AnnotatedImageRef
from unity.manager_registry import ManagerRegistry

from ..config.prompt_builder import build_repairs_business_context

logger = logging.getLogger(__name__)


REPAIRS_SYSTEM_PROMPT_EXTENSION = """
### Domain: Repairs & Telematics Analysis

You are analyzing repairs operations data for a housing association.

{business_context}

### Available Tools

1. **Pre-built Metric Functions** (via FunctionManager):
   - Search with `search_functions_by_similarity("first time fix rate")` to find relevant metrics
   - Retrieved functions include FULL SOURCE CODE - study them to understand the approach
   - Functions use FileManager primitives (reduce, filter_files, visualize)

2. **FileManager Primitives** (via `primitives.files`):
   - `primitives.files.tables_overview()` - **START HERE** - List all available tables
   - `primitives.files.list_columns(table=...)` - Get column names and descriptions
   - `primitives.files.schema_explain(table=...)` - Get detailed schema explanation
   - `primitives.files.reduce(table, metric, keys, filter, group_by)` - Aggregate data
   - `primitives.files.filter_files(filter, tables, limit)` - Query raw records
   - `primitives.files.visualize(tables, plot_type, x_axis, y_axis, ...)` - Generate charts

### Discovery-First Workflow

**ALWAYS start by discovering what data is available:**

```python
# Step 1: See what tables exist
tables = primitives.files.tables_overview()
for t in tables:
    print(f"  {t['name']}: {t['path']}")

# Step 2: Find the right table
repairs_table = next(t["path"] for t in tables if "Repairs" in t.get("name", ""))

# Step 3: Check columns
columns = primitives.files.list_columns(table=repairs_table)
print(columns)

# Step 4: Now query with the discovered path
result = primitives.files.reduce(
    table=repairs_table,
    metric="count",
    keys="JobTicketReference",
    filter="`WorksOrderStatusDescription` in ['Complete', 'Closed']",
)
```

### Workflow for Answering Queries

1. **Search** for existing metric functions that match the user's query
2. **Read** the implementation source code to understand the approach
3. **Discover** tables with `tables_overview()` if composing new queries
4. **Compose** a solution using discovered functions or primitives
5. **Visualize** when the user requests charts/plots (use `include_plots=True` or call `visualize` directly)
"""


class DynamicRepairsAgent:
    """
    LLM-powered agent for natural language repairs/telematics analysis.

    Usage:
        agent = DynamicRepairsAgent()

        # Ask a natural language question
        handle = await agent.ask("What's the first time fix rate by region?")
        result = await handle.result()
        print(result)

        # Complex query with visualization
        handle = await agent.ask(
            "Compare no-access rates between North and South regions "
            "over the last 3 months, and show me a chart"
        )
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
    ) -> None:
        logger.info("Initializing DynamicRepairsAgent...")

        # Initialize primitives for sandbox execution
        self._primitives = Primitives()

        # Load business context from FilePipelineConfig
        self._business_context = build_repairs_business_context(config_path)

        # Build system prompt with injected business context
        self._system_prompt_extension = REPAIRS_SYSTEM_PROMPT_EXTENSION.format(
            business_context=self._business_context,
        )

        # Create CodeActActor with FunctionManager access
        self._actor = CodeActActor(
            function_manager=ManagerRegistry.get_function_manager(),
            environments=[StateManagerEnvironment(self._primitives)],
            system_prompt_extension=self._system_prompt_extension,
        )

        logger.info("DynamicRepairsAgent initialized successfully")

    async def ask(
        self,
        query: str,
        *,
        images: Optional[ImageRefs | list[RawImageRef | AnnotatedImageRef]] = None,
    ) -> ActorHandle:
        """
        Answer a natural language query about repairs/telematics data.

        The agent will:
        1. Search FunctionManager for relevant metric functions
        2. Retrieve implementations + docstrings to understand approach
        3. Discover tables using tables_overview() if needed
        4. Generate Python code using discovered functions or primitives
        5. Execute in sandbox with `primitives.files` access

        Parameters
        ----------
        query : str
            Natural language question about repairs or telematics data
        images : ImageRefs, optional
            Images to include for context (e.g., screenshots)

        Returns
        -------
        ActorHandle
            Steerable handle for the running analysis task

        Example
        -------
        >>> handle = await agent.ask("What is the first time fix rate?")
        >>> result = await handle.result()
        """
        return await self._actor.act(
            description=query,
            images=images,
        )

    async def close(self) -> None:
        """Clean up resources."""
        await self._actor.close()
```

### 6.5 Business Context Prompt Builder

**Location**: `intranet/repairs_agent/config/prompt_builder.py`

```python
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

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "repairs" / "repairs_file_pipeline_config_5m.json"


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
        logger.warning(f"Config not found at {config_path}, using empty context")
        return _build_fallback_context()

    business_contexts = config.get("ingest", {}).get("business_contexts", {})

    sections = []

    # Domain overview
    sections.append("#### Domain Overview")
    sections.append("You are analyzing repairs and telematics data for a housing association.")
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
    sections.append("- **Completed jobs**: `WorksOrderStatusDescription` in ['Complete', 'Closed']")
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
```

### 6.6 FunctionManager Sync Script

**Location**: `intranet/repairs_agent/dynamic/sync.py`

No changes from original spec.

---

## 7. Data Flow Diagrams

### 7.1 Static Demo Flow

```
User CLI Command
    │
    ▼
python -m intranet.repairs_agent.scripts.run_repairs_query \
    --query first_time_fix_rate --params '{"group_by": "patch"}'
    │
    ▼
BespokeRepairsAgent.ask("first_time_fix_rate", group_by="patch")
    │
    ▼
@register registry lookup → first_time_fix_rate() function
    │
    ▼
tools["reduce"]() → tools["reduce"]() → Python % calc → [tools["visualize"]()]
    │
    ▼
MetricResult(results=[...], total=85.2, plots=[...])

# Parallel execution (all metrics):
./intranet/repairs_agent/scripts/parallel_queries.sh
```

### 7.2 Dynamic Demo Flow (Updated)

```
User Natural Language Query
    │
    "What's the first time fix rate by region, and show me a chart?"
    │
    ▼
DynamicRepairsAgent.ask(query)
    │
    ├──► Business context from FilePipelineConfig injected into system prompt
    │
    ▼
CodeActActor.act(description=query)
    │
    ├──► FunctionManager.search_functions_by_similarity(
    │        query="first time fix rate by region with visualization"
    │    )
    │    │
    │    ▼
    │    Returns: [{
    │        "name": "first_time_fix_rate",
    │        "implementation": "async def first_time_fix_rate(...): ...",
    │        "docstring": "Calculate First Time Fix rate... [includes discovery pattern]",
    │        "calls": ["normalize_grouped_result", "compute_percentage"],
    │    }, ...]
    │
    ├──► LLM reads business context + function implementation
    │
    ├──► LLM generates Python code:
    │    ```python
    │    # Discovery (from business context + function docstring)
    │    tables = primitives.files.tables_overview()
    │    repairs_table = next(t["path"] for t in tables if "Repairs" in t.get("name", ""))
    │
    │    # Use discovered table path
    │    result = primitives.files.reduce(
    │        table=repairs_table,
    │        metric="count",
    │        keys="JobTicketReference",
    │        filter="`FirstTimeFix` == 'Yes' and `WorksOrderStatusDescription` in ['Complete', 'Closed']",
    │        group_by="RepairsRegion",
    │    )
    │
    │    # ... percentage calculation ...
    │
    │    # Visualization
    │    plot_url = primitives.files.visualize(
    │        tables=repairs_table,
    │        plot_type="bar",
    │        x_axis="RepairsRegion",
    │        y_axis="count",
    │    )
    │    print(f"Chart: {plot_url}")
    │    ```
    │
    └──► CodeExecutionSandbox.execute(code)
         │
         ▼
         primitives.files.reduce() → ... → Result
```

### 7.3 Discovery-First Composition Flow

```
User Query: "What percentage of jobs in the North region were completed on time?"
    │
    ▼
CodeActActor searches FunctionManager → finds job_completed_on_time_rate()
    │
    ▼
LLM reads implementation docstring, sees:
    - Filter pattern: `WorksOrderReportedCompletedDate` <= `WorksOrderTargetDate`
    - Column: "RepairsRegion" for region filtering
    │
    ▼
LLM composes custom code:
    ```python
    # Step 1: Discover tables (from system prompt guidance)
    tables = primitives.files.tables_overview()
    repairs_table = next(t["path"] for t in tables if "Repairs" in t.get("name", ""))

    # Step 2: Query with region filter
    on_time_filter = (
        "`WorksOrderStatusDescription` in ['Complete', 'Closed'] and "
        "`WorksOrderReportedCompletedDate` != 'None' and "
        "`WorksOrderTargetDate` != 'None' and "
        "`WorksOrderReportedCompletedDate` <= `WorksOrderTargetDate` and "
        "`RepairsRegion` == 'North'"
    )

    on_time_count = primitives.files.reduce(
        table=repairs_table,
        metric="count",
        keys="JobTicketReference",
        filter=on_time_filter,
    )

    total_filter = (
        "`WorksOrderStatusDescription` in ['Complete', 'Closed'] and "
        "`RepairsRegion` == 'North'"
    )

    total_count = primitives.files.reduce(
        table=repairs_table,
        metric="count",
        keys="JobTicketReference",
        filter=total_filter,
    )

    # Calculate percentage
    pct = (on_time_count / total_count * 100) if total_count > 0 else 0
    print(f"North region on-time completion rate: {pct:.1f}%")
    ```
```

---

## 8. Directory Structure

```
intranet/repairs_agent/
├── __init__.py
├── SPEC.md                         # This specification document
├── README.md                       # User-facing documentation
│
├── config/                         # Business context configuration
│   ├── __init__.py
│   ├── prompt_builder.py           # Builds BusinessContextPayload for CodeActActor
│   └── repairs_file_pipeline_config.json  # (symlink to existing config)
│
├── metrics/                        # Single Source of Truth
│   ├── __init__.py
│   ├── definitions.py              # All 16+ metric functions
│   ├── helpers.py                  # Compositional helper functions
│   ├── plots.py                    # Plot configurations per metric
│   ├── types.py                    # GroupBy, TimePeriod, MetricResult, PlotResult
│   └── constants.py                # Table names, filters, column mappings
│
├── static/                         # Static Demo (LLM-free)
│   ├── __init__.py
│   ├── agent.py                    # BespokeRepairsAgent
│   ├── registry.py                 # @register decorator and registry
│   └── cli.py                      # CLI for static demo
│
├── dynamic/                        # Dynamic Demo (CodeActActor)
│   ├── __init__.py
│   ├── agent.py                    # DynamicRepairsAgent + business context injection
│   ├── sync.py                     # Sync metrics to FunctionManager
│   └── cli.py                      # CLI for dynamic demo
│
├── scripts/                        # All scripts (Python + Shell)
│   ├── __init__.py
│   ├── README.md                   # Comprehensive script documentation
│   │
│   │   # Python Scripts
│   ├── run_repairs_query.py        # Main query runner CLI
│   ├── query_logger.py             # Query logging utilities
│   ├── repairs_query_logger.py     # Repairs-specific query logger
│   ├── _query_generator.py         # Query generation helpers (internal)
│   │
│   │   # Shell Scripts - Query Execution
│   ├── parallel_queries.sh         # Run queries in parallel (tmux-based)
│   ├── list_queries.sh             # List available queries
│   ├── watch_queries.sh            # Watch query execution in real-time
│   │
│   │   # Shell Scripts - Session Management
│   ├── kill_failed_queries.sh      # Kill failed tmux sessions
│   ├── kill_server_queries.sh      # Kill all query tmux sessions
│   ├── _repairs_common.sh          # Shared shell utilities (internal)
│   │
│   │   # Shell Scripts - Dynamic Demo
│   ├── run_dynamic.sh              # Run dynamic CodeActActor agent
│   └── sync_functions.sh           # Sync metrics to FunctionManager
│
├── data/                           # Data files (gitignored or symlinked)
│   └── README.md                   # Data file locations and setup instructions
│
└── tests/                          # Unit and integration tests
    ├── __init__.py
    ├── test_metrics.py             # Test individual metric functions
    ├── test_helpers.py             # Test helper functions
    ├── test_static_agent.py        # Test BespokeRepairsAgent
    └── test_dynamic_agent.py       # Test DynamicRepairsAgent
```

### 8.1 Scripts Overview

The `scripts/` directory consolidates all execution and management scripts from the original `intranet/scripts/repairs/` location:

| Script | Purpose |
|--------|---------|
| `run_repairs_query.py` | Main CLI for executing static queries via BespokeRepairsAgent |
| `query_logger.py` | Logging infrastructure for query results |
| `repairs_query_logger.py` | Repairs-specific logging extensions |
| `_query_generator.py` | Internal helper for generating query combinations |
| `parallel_queries.sh` | Run multiple queries in parallel using tmux sessions |
| `list_queries.sh` | List all registered metrics with descriptions |
| `watch_queries.sh` | Real-time monitoring of query execution |
| `kill_failed_queries.sh` | Clean up failed tmux query sessions |
| `kill_server_queries.sh` | Kill all query-related tmux sessions |
| `_repairs_common.sh` | Shared shell functions (sourced by other scripts) |
| `run_dynamic.sh` | Launch the dynamic CodeActActor agent (NEW) |
| `sync_functions.sh` | Sync metric functions to FunctionManager ✓ |

**Usage examples after migration:**

```bash
# Static demo - list queries
python -m intranet.repairs_agent.scripts.run_repairs_query --list

# Static demo - run a query
python -m intranet.repairs_agent.scripts.run_repairs_query \
    --query first_time_fix_rate \
    --params '{"group_by": "region", "include_plots": true}'

# Static demo - parallel execution
./intranet/repairs_agent/scripts/parallel_queries.sh

# Dynamic demo - interactive
python -m intranet.repairs_agent.dynamic.cli

# Sync metrics to FunctionManager
python -m intranet.repairs_agent.dynamic.sync
```

---

## 9. Metrics Inventory

### 9.1 Repairs Metrics (from repairs data)

| # | Metric ID | Description | Key Fields |
|---|-----------|-------------|------------|
| 1 | `jobs_completed_per_day` | Jobs completed per man per day | OperativeWhoCompletedJob |
| 2 | `no_access_rate` | No Access % / Absolute number | NoAccess |
| 3 | `first_time_fix_rate` | First Time Fix % / Absolute | FirstTimeFix |
| 4 | `follow_on_required_rate` | Follow on Required % | FollowOn |
| 5 | `follow_on_materials_rate` | Follow on for Materials % | FollowOnDescription |
| 6 | `job_completed_on_time_rate` | Completed on time % | CompletedDate vs TargetDate |
| 7 | `repairs_completed_per_day` | Total repairs per day | WorksOrderReportedCompletedDate |
| 8 | `jobs_issued_per_day` | Jobs issued per day | WorksOrderIssuedDate |
| 9 | `jobs_requiring_materials_rate` | Jobs needing materials % | FollowOnDescription |
| 10 | `avg_repairs_per_property` | Avg repairs per property | FullAddress |
| 11 | `complaints_rate` | Complaints % (if data available) | - |
| 12 | `appointment_adherence_rate` | On-time arrival % | ArrivedOnSite vs ScheduledAppointment |

### 9.2 Telematics Metrics (from telematics data)

| # | Metric ID | Description | Key Fields |
|---|-----------|-------------|------------|
| 13 | `merchant_stops_per_day` | Merchant stops per day | EndLocation |
| 14 | `avg_duration_at_merchant` | Avg duration at merchant | Trip travel time |
| 15 | `distance_travelled_per_day` | Distance per day | Business distance |
| 16 | `avg_time_travelling` | Avg travel time per day | Trip travel time |

### 9.3 Grouping Dimensions

All metrics support grouping by:
- `operative` - Individual worker
- `trade` - Skill type (plumber, electrician, etc.)
- `patch` - Geographic patch/area
- `region` - Broader region (North/South)
- `total` - No grouping (aggregate)

---

## 10. Implementation Phases

### Phase 1: Business Context Integration (Day 1 - Demo Priority)
- [x] Create `config/prompt_builder.py` to build business context from FilePipelineConfig ✓
- [x] Update `DynamicRepairsAgent` to inject business context into system prompt ✓
- [x] Add discovery tools guidance to system prompt (`tables_overview`, `list_columns`, `schema_explain`) ✓
- [x] Enrich ALL metric docstrings with discovery patterns and exact tool arguments ✓
    - Repairs: `jobs_completed_per_day`, `no_access_rate`, `first_time_fix_rate`,
      `follow_on_required_rate`, `follow_on_materials_rate`, `job_completed_on_time_rate`,
      `jobs_requiring_materials_rate`, `avg_repairs_per_property`, `complaints_rate`,
      `appointment_adherence_rate`
    - Service: `repairs_completed_per_day`, `jobs_issued_per_day`
    - Telematics: `merchant_stops_per_day`, `avg_duration_at_merchant`,
      `distance_travelled_per_day`, `avg_time_travelling`
- [ ] Test end-to-end with CodeActActor

### Phase 2: Restructure and Refactor (Post-Demo)
- [x] Create `intranet/repairs_agent/` package structure ✓
- [x] Migrate `static/agent.py` and `static/registry.py` (BespokeRepairsAgent) ✓
- [x] Move types from `intranet/repairs/queries/_types.py` → `metrics/types.py` ✓
- [x] Move plots from `intranet/repairs/queries/_plots.py` → `metrics/plots.py` ✓
- [x] Migrate all scripts from `intranet/scripts/repairs/` → `scripts/` ✓
- [ ] Move constants from `intranet/repairs/queries/metrics.py` (deferred - large file)
- [ ] Refactor metrics to use `files: FileManager` parameter (Phase 4)
- [ ] Extract helpers to separate module (Phase 4)
- [ ] Update imports and ensure static demo works

### Phase 3: FunctionManager Integration (Post-Demo)
- [x] Create `sync.py` script ✓
- [x] Implement sync logic with proper dependency tracking ✓
- [x] Add proper script initialization (activate_project) to sync.py ✓
- [x] Test that functions appear in FunctionManager search ✓
- [x] Verify implementations returned with full source code ✓

### Phase 4: Discovery-First Refactoring (Production)
- [ ] Refactor all metrics to use discovery pattern (Option A)
- [ ] Remove hardcoded table path constants from metric implementations
- [ ] Add caching for discovery calls to avoid redundant API calls
- [ ] Update all docstrings with gold standard template

---

## 11. Open Questions

### 11.1 Resolved
- ✓ Metric function purity: Pure tool chains + minimal Python (no external libs)
- ✓ Sync strategy: Separate script, not auto-sync
- ✓ Implementation visibility: Full source + docstring
- ✓ Composition freedom: No restrictions
- ✓ Domain scope: Unified repairs/telematics domain
- ✓ Visualization: `include_plots` flag + direct primitive access
- ✓ Business context injection: Use same slot-filling pattern as FileManager.ask
- ✓ Metric refactoring strategy: Hybrid (Option C) for demo, Discovery-First (Option A) for production

### 11.2 For Further Discussion

1. **FileManager joins**: Should we add `filter_join` / `search_join` to `PRIMITIVE_SOURCES` for cross-table analysis?

2. **Telematics table handling**: Current metrics iterate over monthly tables. Should we consolidate or keep multi-table pattern?

3. **Recursive implementation retrieval**: Should `search_functions_by_similarity` also return implementations of functions in the `calls` chain?

4. **Error handling in dynamic mode**: How should CodeActActor handle metric function failures? Retry? Fallback to primitives?

5. **Caching**: Should metric results be cached? At what level (metric function, reduce call)?

6. **CodeActActor `system_prompt_extension` parameter**: Does this exist or need to be added?

---

## 12. Appendix

### A. Migration Checklist

#### From `intranet/repairs/queries/`:
- [x] `_types.py` → `metrics/types.py` ✓
- [ ] `metrics.py` → `metrics/definitions.py` + `metrics/constants.py` (LARGE FILE - deferred)
- [x] `_plots.py` → `metrics/plots.py` ✓
- [ ] `plot_utils.py` → `metrics/plot_utils.py` (deferred - uses existing unity utilities)

#### From `intranet/core/`:
- [x] `bespoke_repairs_agent.py` → `static/agent.py` + `static/registry.py` ✓

#### From `intranet/scripts/repairs/` (full directory migration):

**Python Scripts:**
- [x] `run_repairs_query.py` → `scripts/run_repairs_query.py` ✓
- [x] `query_logger.py` → `scripts/query_logger.py` ✓
- [x] `repairs_query_logger.py` → `scripts/repairs_query_logger.py` ✓
- [x] `_query_generator.py` → `scripts/_query_generator.py` ✓

**Shell Scripts - Query Execution:**
- [x] `parallel_queries.sh` → `scripts/parallel_queries.sh` ✓
- [x] `list_queries.sh` → `scripts/list_queries.sh` ✓
- [x] `watch_queries.sh` → `scripts/watch_queries.sh` ✓

**Shell Scripts - Session Management:**
- [x] `kill_failed_queries.sh` → `scripts/kill_failed_queries.sh` ✓
- [x] `kill_server_queries.sh` → `scripts/kill_server_queries.sh` ✓
- [x] `_repairs_common.sh` → `scripts/_repairs_common.sh` ✓

**Documentation:**
- [x] `README.md` → `scripts/README.md` (paths updated) ✓

#### Post-Migration Tasks:
- [ ] Update all import paths in Python scripts (e.g., `from intranet.repairs_agent.metrics...`)
- [ ] Update all shell script paths (e.g., `SCRIPT_DIR` references)
- [ ] Update `_repairs_common.sh` with new directory structure
- [ ] Update `parallel_queries.sh` to call new script locations
- [ ] Create symlinks from old locations to new (temporary, for backwards compatibility)
- [ ] Update any CI/CD scripts that reference old paths
- [ ] Remove old `intranet/scripts/repairs/` directory after verification

### B. Related Unity Components

- `unity/actor/code_act_actor.py` - CodeActActor implementation
- `unity/actor/prompt_builders.py` - System prompt building (may need extension for business_payload)
- `unity/function_manager/function_manager.py` - FunctionManager
- `unity/function_manager/primitives.py` - Primitives class and PRIMITIVE_SOURCES
- `unity/file_manager/managers/file_manager.py` - FileManager (reduce, visualize, etc.)
- `unity/file_manager/prompt_builders.py` - FileManager slot-filling (reference for business context injection)
- `unity/common/business_context.py` - BusinessContextPayload model

### C. Key Insight: Why Discovery-First Matters for CodeActActor

The fundamental problem with hardcoded table paths:

```python
# Hardcoded (CURRENT):
REPAIRS_TABLE = "/path/to/file.xlsx.Tables.Raised_01-07-2025"

async def first_time_fix_rate(tools, ...):
    result = reduce_tool(table=REPAIRS_TABLE, ...)  # LLM sees magic string
```

When CodeActActor reads this function's source, it sees:
1. A magic constant `REPAIRS_TABLE`
2. No indication of how to find other tables
3. No pattern to replicate for novel queries

**With Discovery-First:**

```python
async def first_time_fix_rate(files, ...):
    """
    Discovery Pattern:
        tables = files.tables_overview()
        repairs_table = next(t["path"] for t in tables if "Repairs" in t.get("name", ""))

    Tool Chain:
        1. files.reduce(table=repairs_table, ...)  # Uses discovered path
    """
    tables = files.tables_overview()
    repairs_table = _find_table(tables, "Repairs")
    result = files.reduce(table=repairs_table, ...)
```

Now CodeActActor learns:
1. **How to discover** tables (`tables_overview()`)
2. **How to select** the right table (pattern matching on name)
3. **How to use** the discovered path in queries
4. **Pattern is transferable** to any novel query

### D. Demo Shortcut: Docstring Enrichment

For tomorrow's demo, instead of refactoring all metrics, enrich docstrings:

```python
@register("first_time_fix_rate", "...")
async def first_time_fix_rate(tools: ToolsDict, ...):
    """
    Calculate First Time Fix rate.

    FOR CODEACT COMPOSITION - Discovery Pattern:
    ---------------------------------------------
    # If you need to adapt this metric or compose a new one:

    tables = primitives.files.tables_overview()
    repairs_table = next(t["path"] for t in tables if "Repairs" in t.get("name", ""))

    # This metric uses the constant REPAIRS_TABLE which equals:
    # "Files/Repairs/MDH_Repairs_Data.../Tables/Raised_01-07-2025_to_30-11-2025"

    Tool Chain (exact arguments):
    -----------------------------
    1. reduce(table=REPAIRS_TABLE, metric="count", keys="JobTicketReference",
              filter="`FirstTimeFix` == 'Yes'", group_by="RepairsRegion")
       → Returns: {"North": {"count": 50}, "South": {"count": 75}, ...}

    2. reduce(table=REPAIRS_TABLE, metric="count", keys="JobTicketReference",
              filter="`WorksOrderStatusDescription` in ['Complete', 'Closed']",
              group_by="RepairsRegion")
       → Returns: {"North": {"count": 100}, "South": {"count": 150}, ...}

    3. Python: percentage = (ftf_count / total_count) * 100

    4. visualize(tables=REPAIRS_TABLE, plot_type="bar", x_axis="RepairsRegion",
                 y_axis="FirstTimeFix") if include_plots=True

    Filter Expressions:
    -------------------
    - Completed: `WorksOrderStatusDescription` in ['Complete', 'Closed']
    - First Time Fix: `FirstTimeFix` == 'Yes'
    - With date range: ... and `VisitDate` >= '2025-07-01' and `VisitDate` <= '2025-09-30'

    Column Mappings:
    ----------------
    GroupBy.OPERATIVE → "OperativeWhoCompletedJob"
    GroupBy.PATCH → "RepairsPatch"
    GroupBy.REGION → "RepairsRegion"
    GroupBy.TOTAL → None

    [Standard Parameters section...]
    """
```

This gives CodeActActor everything it needs without code changes.
