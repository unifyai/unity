"""
Scripts module - CLI tools and shell scripts for repairs analysis.

This module consolidates all execution and management scripts for both
the static and dynamic demos.

Python Scripts:
    - run_repairs_query.py: Main CLI for executing static queries
    - query_logger.py: Logging infrastructure for query results
    - repairs_query_logger.py: Repairs-specific logging extensions
    - _query_generator.py: Internal helper for generating query combinations

Shell Scripts:
    - parallel_queries.sh: Run multiple queries in parallel (tmux-based)
    - list_queries.sh: List all registered metrics
    - watch_queries.sh: Real-time monitoring
    - kill_failed_queries.sh: Clean up failed tmux sessions
    - kill_server_queries.sh: Kill all query sessions
    - run_dynamic.sh: Launch dynamic CodeActActor agent
    - sync_functions.sh: Sync metrics to FunctionManager
"""
