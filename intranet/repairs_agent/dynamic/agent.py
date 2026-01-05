"""
DynamicRepairsAgent: LLM-powered agent for natural language repairs analysis.

This agent uses CodeActActor to interpret natural language queries, discover
relevant metric functions from FunctionManager, and compose Python code to
answer complex analytical questions about repairs and telematics data.

Key Architecture:
- Business context injection via task description prepending
- Discovery-first guidance in system prompt extension
- Access to pre-built metric functions AND raw FileManager primitives
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from unity.actor.handle import ActorHandle
    from unity.image_manager.types.annotated_image_ref import AnnotatedImageRef
    from unity.image_manager.types.image_refs import ImageRefs
    from unity.image_manager.types.raw_image_ref import RawImageRef

from intranet.repairs_agent.config.prompt_builder import build_repairs_business_context

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
    print(f"  {{t['name']}}: {{t['path']}}")

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
        """
        Initialize the DynamicRepairsAgent.

        Parameters
        ----------
        config_path : Path, optional
            Path to FilePipelineConfig JSON for business context.
            Defaults to repairs_file_pipeline_config_5m.json
        """
        from unity.actor.code_act_actor import CodeActActor
        from unity.actor.environments import StateManagerEnvironment
        from unity.function_manager.primitives import Primitives
        from unity.manager_registry import ManagerRegistry

        logger.info("Initializing DynamicRepairsAgent...")

        # Initialize primitives for sandbox execution
        self._primitives = Primitives()

        # Load business context from FilePipelineConfig
        self._business_context = build_repairs_business_context(config_path)

        # Build system prompt extension with injected business context
        self._system_prompt_extension = REPAIRS_SYSTEM_PROMPT_EXTENSION.format(
            business_context=self._business_context,
        )

        # Create CodeActActor with FunctionManager access and only StateManager environment
        # (no browser environment needed for data analysis)
        self._actor = CodeActActor(
            function_manager=ManagerRegistry.get_function_manager(),
            environments=[StateManagerEnvironment(self._primitives)],
        )

        logger.info("DynamicRepairsAgent initialized successfully")

    async def ask(
        self,
        query: str,
        *,
        images: Optional["ImageRefs | list[RawImageRef | AnnotatedImageRef]"] = None,
    ) -> "ActorHandle":
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
        # Prepend domain context to the query
        # This injects business context into the task description
        full_description = f"{self._system_prompt_extension}\n\n### User Query\n{query}"

        return await self._actor.act(
            description=full_description,
            images=images,
        )

    async def close(self) -> None:
        """Clean up resources."""
        if hasattr(self._actor, "close"):
            await self._actor.close()
