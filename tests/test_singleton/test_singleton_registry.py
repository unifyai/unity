import pytest

from unity.singleton_registry import SingletonRegistry, SingletonABCMeta


class _DummySingleton(metaclass=SingletonABCMeta):
    """Tiny throw-away class to verify the **generic** singleton contract."""

    def __init__(self):
        # Store *id(self)* so we can later assert a new instance is created
        self.identity = id(self)


@pytest.mark.asyncio
async def test_singleton_same_instance():
    """Multiple constructor calls must return the *identical* object."""

    first = _DummySingleton()
    second = _DummySingleton()

    # Both variables must point to *exactly* the same object
    assert first is second
    assert first.identity == second.identity

    # Registry should return that very instance as well
    assert SingletonRegistry.get(_DummySingleton) is first


@pytest.mark.asyncio
async def test_singleton_clear_creates_fresh_instance():
    """`SingletonRegistry.clear` must drop the cached instance so that the
    next instantiation yields a *new* object.
    """

    original = _DummySingleton()

    # Purge the registry manually (the session-wide fixture only runs between
    # tests; we also check the behaviour *within* a single test).
    SingletonRegistry.clear()

    replacement = _DummySingleton()

    assert original is not replacement
    assert original.identity != replacement.identity
