"""
Static Demo module - LLM-free BespokeRepairsAgent.

This module provides deterministic, direct execution of pre-defined metric
functions without any LLM orchestration. Ideal for dashboards, automated
reports, and performance-critical use cases.

Components:
    - agent: BespokeRepairsAgent facade
    - registry: @register decorator and query registry

Usage:
    from intranet.repairs_agent.static import BespokeRepairsAgent

    agent = BespokeRepairsAgent()
    queries = agent.list_queries()
    result = await agent.ask("first_time_fix_rate", group_by="region")
"""

# Re-exports will be available after migration
# from .agent import BespokeRepairsAgent
# from .registry import register

__all__ = [
    # "BespokeRepairsAgent",
    # "register",
]
