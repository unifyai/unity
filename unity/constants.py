"""
unity/constants.py
==================

Runtime constants that cannot be configured via environment.

For configurable settings, import SETTINGS from unity.settings.
For logging configuration, import from unity.logger.
"""

from datetime import datetime, timezone

# Re-export logging utilities from logger module for backwards compatibility
from unity.logger import LOGGER, configure_log_dir, get_log_dir

# ─────────────────────────────────────────────────────────────────────────────
# True Runtime Constants (not configurable via environment)
# ─────────────────────────────────────────────────────────────────────────────

SESSION_ID = datetime.now(timezone.utc).isoformat()

__all__ = ["SESSION_ID", "LOGGER", "configure_log_dir", "get_log_dir"]
