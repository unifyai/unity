"""
tests/settings.py
==================

Test environment settings using pydantic-settings.

TestingSettings inherits all production settings from unity.settings.ProductionSettings
and adds test-only configuration. This mirrors the structure of unity/settings.py.

IMPORTANT: SETTINGS is a lazy proxy to avoid import-order issues. When test
subdirectories set environment variables in pytest_configure(), those run AFTER
the root conftest.py imports this module. The lazy proxy defers instantiation
until first actual use, allowing env vars to be set first.
"""

import os
import random
import string
from pathlib import Path
from typing import TYPE_CHECKING

# Set UNILLM_CACHE_DIR to repo root so cache location is consistent regardless of cwd
_REPO_ROOT = Path(__file__).parent.parent.resolve()
os.environ.setdefault("UNILLM_CACHE_DIR", str(_REPO_ROOT))

from pydantic import Field
from pydantic.fields import computed_field

from unity.memory_manager.settings import MemorySettings
from unity.settings import ProductionSettings

if TYPE_CHECKING:
    from typing import Any


class TestMemorySettings(MemorySettings):
    """Test overrides for MemorySettings - disables callbacks by default."""

    REGISTER_UPDATE_CALLBACKS: bool = False


class TestingSettings(ProductionSettings):
    """Test environment settings - inherits all production settings.

    Production settings (UNIFY_MODEL, UNIFY_CACHE, etc.) are inherited from
    ProductionSettings. This class adds test-only settings.
    """

    # Override composed manager settings with test defaults
    memory: MemorySettings = Field(default_factory=TestMemorySettings)

    # ─────────────────────────────────────────────────────────────────────────
    # Test Infrastructure Settings
    # ─────────────────────────────────────────────────────────────────────────
    UNITY_INCREMENTING_TIMESTAMPS: bool = (
        False  # Auto-increment timestamps for NEW markers
    )
    EVENTBUS_PUBLISHING_ENABLED: bool = False  # Disabled by default in tests
    UNIFY_DELETE_CONTEXT_ON_EXIT: bool = False
    UNIFY_OVERWRITE_PROJECT: bool = False
    UNIFY_TESTS_RAND_PROJ: bool = False
    UNIFY_TESTS_DELETE_PROJ_ON_START: bool = True
    UNIFY_TESTS_DELETE_PROJ_ON_EXIT: bool = False
    UNIFY_CACHE_STATS: bool = False
    UNIFY_PRETEST_CONTEXT_CREATE: bool = False
    UNIFY_TEST_TAGS: str = ""  # Comma-separated list of tags for duration logging
    UNIFY_SKIP_SESSION_SETUP: bool = False  # Skip project/context creation (pre-done)
    UNITY_TEST_PROJECT_NAME: str = "UnityTests"

    # ─────────────────────────────────────────────────────────────────────────
    # Local Orchestra Settings
    # ─────────────────────────────────────────────────────────────────────────
    LOCAL_ORCHESTRA_BRANCH: str = (
        # Git branch for local orchestra (default: auto-detect from unity branch)
        "staging"  # IMPORTANT: Currently hard-coded to staging, but should DELETE once staging becomes the true unity dev branch!
    )

    # ─────────────────────────────────────────────────────────────────────────
    # File Lock Settings (for parallel test coordination)
    # ─────────────────────────────────────────────────────────────────────────
    UNITY_FILE_LOCK_TIMEOUT: float = 3600.0  # 1 hour - handles slow tests under load

    # ─────────────────────────────────────────────────────────────────────────
    # Trace Upload Settings (for uploading OTEL traces to test context)
    # ─────────────────────────────────────────────────────────────────────────
    UNITY_TRACE_UPLOAD: bool = (
        False  # Enable/disable trace upload to {TestContext}/Trace
    )
    UNITY_TRACE_SERVICES: str = (
        "all"  # Services to include: "all" or comma-separated list
    )
    UNITY_TRACE_EXCLUDE_PATTERNS: str = (
        ""  # Comma-separated span name patterns to exclude
    )

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


class _SettingsProxy:
    """Lazy proxy that defers TestingSettings instantiation until first access.

    This solves import-order issues where test subdirectories set environment
    variables in pytest_configure(), which runs AFTER root conftest.py imports
    this module. By deferring instantiation, env vars can be set first.

    The proxy forwards all attribute access to the underlying TestingSettings
    instance, creating it on first use.
    """

    _instance: TestingSettings | None = None

    def _get_instance(self) -> TestingSettings:
        if self._instance is None:
            self._instance = TestingSettings()
        return self._instance

    def __getattr__(self, name: str) -> "Any":
        return getattr(self._get_instance(), name)

    def model_dump(self, **kwargs) -> dict:
        """Forward model_dump() to the underlying instance."""
        return self._get_instance().model_dump(**kwargs)


# Lazy singleton - instantiated on first use, not at import time
SETTINGS: TestingSettings = _SettingsProxy()  # type: ignore[assignment]
