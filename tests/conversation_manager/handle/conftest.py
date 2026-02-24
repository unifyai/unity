from __future__ import annotations

import os


def pytest_configure(config) -> None:
    """Configure manager implementations for handle integration tests."""
    os.environ["UNITY_CONTACT_IMPL"] = "real"
    os.environ["UNITY_TRANSCRIPT_IMPL"] = "real"
