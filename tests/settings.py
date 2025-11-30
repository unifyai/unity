"""
tests/settings.py
==================

Test environment settings using pydantic-settings.

TestingSettings inherits all production settings from unity.settings.ProductionSettings
and adds test-only configuration. This mirrors the structure of unity/settings.py.
"""

from unity.settings import ProductionSettings


class TestingSettings(ProductionSettings):
    """Test environment settings - inherits all production settings.

    Production settings (UNIFY_MODEL, UNIFY_CACHE, LLM_IO_DEBUG, etc.) are
    inherited from ProductionSettings. This class adds test-only settings.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Test Infrastructure Settings
    # ─────────────────────────────────────────────────────────────────────────
    UNIFY_DELETE_CONTEXT_ON_EXIT: bool = False
    UNIFY_OVERWRITE_PROJECT: bool = False
    UNIFY_REGISTER_SUMMARY_CALLBACKS: bool = False
    UNIFY_REGISTER_UPDATE_CALLBACKS: bool = False
    UNIFY_TESTS_RAND_PROJ: bool = False
    UNIFY_TESTS_DELETE_PROJ_ON_EXIT: bool = False
    UNIFY_CACHE_BENCHMARK: bool = False
    UNIFY_PRETEST_CONTEXT_CREATE: bool = False
    UNIFY_TEST_TAGS: str = ""  # Comma-separated list of tags for duration logging
    UNIFY_SKIP_SESSION_SETUP: bool = False  # Skip project/context creation (pre-done)


# Singleton instance for test code
SETTINGS = TestingSettings()
