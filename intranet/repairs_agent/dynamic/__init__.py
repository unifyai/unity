"""
Dynamic Demo module - LLM-powered DynamicRepairsAgent.

This module provides natural language query processing using CodeActActor
to discover, retrieve, and orchestrate metric functions and/or FileManager
primitives to answer complex analytical queries.

Components:
    - agent: DynamicRepairsAgent (CodeActActor wrapper with business context)
    - sync: Script to sync metric functions to FunctionManager

Usage:
    from intranet.repairs_agent.dynamic import DynamicRepairsAgent

    agent = DynamicRepairsAgent()
    handle = await agent.ask("What's the first time fix rate by region?")
    result = await handle.result()

    # With visualization
    handle = await agent.ask(
        "Compare no-access rates between North and South, show me a chart"
    )
"""

# Re-exports will be available after implementation
# from .agent import DynamicRepairsAgent

__all__ = [
    # "DynamicRepairsAgent",
]
