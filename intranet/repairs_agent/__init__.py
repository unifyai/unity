"""
Repairs Analysis Agent Package.

This package provides dual-mode repairs and telematics data analysis:

1. **Static Demo (BespokeRepairsAgent)**: LLM-free, direct execution of
   hardcoded metric query functions for dashboards and automated reports.

2. **Dynamic Demo (DynamicRepairsAgent)**: LLM-powered CodeActActor that
   discovers, retrieves, and orchestrates metric functions and/or FileManager
   primitives to answer complex analytical queries via natural language.

Package Structure:
    - config/     : Business context configuration and prompt builders
    - metrics/    : Single source of truth for metric definitions
    - static/     : LLM-free BespokeRepairsAgent
    - dynamic/    : LLM-powered DynamicRepairsAgent (CodeActActor)
    - scripts/    : CLI tools and shell scripts
    - tests/      : Unit and integration tests

Usage:
    # Static demo
    from intranet.repairs_agent.static import BespokeRepairsAgent
    agent = BespokeRepairsAgent()
    result = await agent.ask("first_time_fix_rate", group_by="region")

    # Dynamic demo
    from intranet.repairs_agent.dynamic import DynamicRepairsAgent
    agent = DynamicRepairsAgent()
    handle = await agent.ask("What's the first time fix rate by region?")
    result = await handle.result()
"""

__version__ = "0.2.0"
