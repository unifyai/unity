from __future__ import annotations

import re

import pytest
import unify


@pytest.fixture(scope="function")
def secret_manager_context(request):
    """Provide an isolated Unify context for each secret-manager test."""
    context_suffix = re.sub(r"[^A-Za-z0-9_/-]+", "_", request.node.name).strip("_")
    ctx = f"tests/secret_manager/{context_suffix}"
    # Create a fresh, test-specific context and make it active
    try:
        unify.set_context(ctx, relative=False)
    except Exception:
        pass
    yield ctx
    unify.delete_context(ctx)
    unify.unset_context()
