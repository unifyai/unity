from __future__ import annotations

import pytest
import unify


@pytest.fixture(scope="function")
def secret_manager_context(request):
    """Provide an isolated Unify context for each secret-manager test."""
    ctx = f"tests/test_secret_manager/{request.node.name}"
    # Create a fresh, test-specific context and make it active
    try:
        unify.set_context(ctx, relative=False)
    except Exception:
        pass
    yield ctx
    # Best-effort cleanup
    try:
        unify.delete_context(ctx)
    except Exception:
        pass
    try:
        unify.unset_context()
    except Exception:
        pass
