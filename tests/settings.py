"""
tests/settings.py
==================

Test environment settings using pydantic-settings.

TestingSettings inherits all production settings from unity.settings.ProductionSettings
and adds test-only configuration. This mirrors the structure of unity/settings.py.
"""

import random
import string

from pydantic import Field
from pydantic.fields import computed_field

from unity.memory_manager.settings import MemorySettings
from unity.settings import ProductionSettings


class TestMemorySettings(MemorySettings):
    """Test overrides for MemorySettings - disables callbacks by default."""

    REGISTER_UPDATE_CALLBACKS: bool = False


class TestingSettings(ProductionSettings):
    """Test environment settings - inherits all production settings.

    Production settings (UNIFY_MODEL, UNIFY_CACHE, LLM_IO_DEBUG, etc.) are
    inherited from ProductionSettings. This class adds test-only settings.
    """

    # Override composed manager settings with test defaults
    memory: MemorySettings = Field(default_factory=TestMemorySettings)

    # ─────────────────────────────────────────────────────────────────────────
    # Test Infrastructure Settings
    # ─────────────────────────────────────────────────────────────────────────
    UNIFY_DELETE_CONTEXT_ON_EXIT: bool = False
    UNIFY_OVERWRITE_PROJECT: bool = False
    UNIFY_TESTS_RAND_PROJ: bool = False
    UNIFY_TESTS_DELETE_PROJ_ON_START: bool = True
    UNIFY_TESTS_DELETE_PROJ_ON_EXIT: bool = False
    UNIFY_CACHE_BENCHMARK: bool = False
    UNIFY_PRETEST_CONTEXT_CREATE: bool = False
    UNIFY_TEST_TAGS: str = ""  # Comma-separated list of tags for duration logging
    UNIFY_SKIP_SESSION_SETUP: bool = False  # Skip project/context creation (pre-done)
    UNITY_TEST_PROJECT_NAME: str = "UnityTests"

    # ─────────────────────────────────────────────────────────────────────────
    # Local Orchestra Settings
    # ─────────────────────────────────────────────────────────────────────────
    LOCAL_ORCHESTRA_BRANCH: str = (
        ""  # Git branch for local orchestra (default: auto-detect from unity branch)
    )

    # ─────────────────────────────────────────────────────────────────────────
    # File Lock Settings (for parallel test coordination)
    # ─────────────────────────────────────────────────────────────────────────
    UNITY_FILE_LOCK_TIMEOUT: float = 3600.0  # 1 hour - handles slow tests under load

    @computed_field
    @property
    def test_project_name(self) -> str:
        """Return the test project name based on settings.

        If UNIFY_TESTS_RAND_PROJ is True, returns a random project name.
        Otherwise, returns UNITY_TEST_PROJECT_NAME (defaults to 'UnityTests').
        """
        if self.UNIFY_TESTS_RAND_PROJ:
            suffix = "".join(
                random.choices(string.ascii_letters + string.digits, k=8),
            )
            return f"UnityTests_{suffix}"
        return self.UNITY_TEST_PROJECT_NAME


# Singleton instance for test code
SETTINGS = TestingSettings()
