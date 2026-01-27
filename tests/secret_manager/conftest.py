from __future__ import annotations

import pytest
import unify


@pytest.fixture(scope="function")
def secret_manager_context(request):
    """Provide an isolated Unify context for each secret-manager test."""
    ctx = f"tests/secret_manager/{request.node.name}"
    # Create a fresh, test-specific context and make it active
    try:
        unify.set_context(ctx, relative=False)
    except Exception:
        pass
    yield ctx
    unify.delete_context(ctx)
    unify.unset_context()
