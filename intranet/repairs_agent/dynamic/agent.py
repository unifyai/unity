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

# ─────────────────────────────────────────────────────────────────────────────
# PATH AND ENVIRONMENT SETUP - MUST HAPPEN BEFORE ANY intranet/unity IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import sys
from pathlib import Path

# Add project root to sys.path for module resolution
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Initialize script environment (loads .env, sets up logging, etc.)
# This must happen at module level before any unity/intranet imports
from intranet.scripts.utils import initialize_script_environment

if not initialize_script_environment():
    print("ERROR: Failed to initialize script environment", file=sys.stderr)
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Standard imports (now safe after environment init)
# ─────────────────────────────────────────────────────────────────────────────
import logging
import warnings
from typing import TYPE_CHECKING, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Suppress only the noisy "Task exception was never retrieved" warnings
# from unillm's background telemetry task (which fails with 405 on staging).
# We keep other logging intact so CodeActActor output remains visible.
# ─────────────────────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=ResourceWarning)


def _install_asyncio_exception_filter() -> None:
    """
    Install a custom exception handler that silences only the unillm_log_query
    background task errors while letting other exceptions through.
    """
    import asyncio
    import sys

    _original_handler = sys.excepthook

    def _filtered_excepthook(exc_type, exc_value, exc_tb):
        # Check if this is from the unillm telemetry task
        tb_str = "".join(__import__("traceback").format_tb(exc_tb or []))
        if "unillm_log_query" in tb_str or "/v0/queries" in str(exc_value):
            return  # Silently ignore
        _original_handler(exc_type, exc_value, exc_tb)

    sys.excepthook = _filtered_excepthook

    # Also handle unhandled exceptions in asyncio tasks
    def _task_exception_handler(loop, context):
        exception = context.get("exception")
        task_name = ""
        if "task" in context:
            task_name = getattr(context["task"], "get_name", lambda: "")()

        # Suppress unillm_log_query task errors
        if task_name == "unillm_log_query":
            return
        if exception and "/v0/queries" in str(exception):
            return

        # Let other exceptions through to default handler
        loop.default_exception_handler(context)

    try:
        loop = asyncio.get_event_loop()
        loop.set_exception_handler(_task_exception_handler)
    except RuntimeError:
        pass  # No event loop yet, will be set when one is created


_install_asyncio_exception_filter()


if TYPE_CHECKING:
    from unity.actor.handle import ActorHandle
    from unity.image_manager.types.annotated_image_ref import AnnotatedImageRef
    from unity.image_manager.types.image_refs import ImageRefs
    from unity.image_manager.types.raw_image_ref import RawImageRef

from intranet.repairs_agent.config.prompt_builder import build_repairs_system_prompt

logger = logging.getLogger(__name__)


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

        # Build system prompt extension from FilePipelineConfig
        self._system_prompt_extension = build_repairs_system_prompt(config_path)

        # Create CodeActActor with FunctionManager access and only StateManager environment
        # (no computer environment needed for data analysis)
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


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────
async def _run_query(query: str, project: str) -> None:
    """Run a single query and print the result."""
    print("Creating DynamicRepairsAgent...", flush=True)
    agent = DynamicRepairsAgent()
    print("Agent created!\n", flush=True)

    print(f"Query: {query}\n", flush=True)
    print("=" * 60, flush=True)

    handle = await agent.ask(query)
    result = await handle.result()

    print("=" * 60, flush=True)
    print(f"\nResult:\n{result}", flush=True)


def main() -> None:
    """CLI entry point for DynamicRepairsAgent."""
    import argparse
    import asyncio

    from intranet.scripts.utils import activate_project

    parser = argparse.ArgumentParser(
        description="DynamicRepairsAgent - Natural language repairs analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agent.py "What is the first time fix rate?"
  python agent.py "Show me jobs completed per day by region" --project RepairsAgent5M
        """,
    )
    parser.add_argument(
        "query",
        help="Natural language query about repairs/telematics data",
    )
    parser.add_argument(
        "--project",
        default="RepairsAgent5M",
        help="Unity project name (default: RepairsAgent5M)",
    )

    args = parser.parse_args()

    # Activate the Unity project
    activate_project(args.project, overwrite=False)

    # Run the query
    asyncio.run(_run_query(args.query, args.project))


if __name__ == "__main__":
    main()
