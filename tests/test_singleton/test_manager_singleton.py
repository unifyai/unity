import pytest

from unity.singleton_registry import SingletonRegistry
from tests.helpers import _handle_project

# Import a representative subset of managers – covering each family that uses
# SingletonABCMeta under the hood.  The *Simulated* variants are intentionally
# avoided to exercise the *real* classes while keeping test runtime low.
from unity.contact_manager.contact_manager import ContactManager
from unity.knowledge_manager.knowledge_manager import KnowledgeManager
from unity.memory_manager.memory_manager import MemoryManager
from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.task_scheduler.task_scheduler import TaskScheduler


# ---------------------------------------------------------------------------
#  Helper – parameterisation over the concrete manager classes
# ---------------------------------------------------------------------------
MANAGER_CLASSES = [
    ContactManager,
    KnowledgeManager,
    MemoryManager,
    TranscriptManager,
    TaskScheduler,
]


@pytest.mark.asyncio
@_handle_project
@pytest.mark.parametrize("manager_cls", MANAGER_CLASSES)
async def test_manager_is_singleton(manager_cls):
    """Every concrete *Manager* class must behave as a *singleton*."""

    first = manager_cls()
    second = manager_cls()

    # Both instantiations must return the *exact* same object
    assert (
        first is second
    ), f"{manager_cls.__name__} did not return a singleton instance"

    # The central registry must return the same object too
    assert SingletonRegistry.get(manager_cls) is first


@pytest.mark.asyncio
@_handle_project
@pytest.mark.parametrize("manager_cls", MANAGER_CLASSES)
async def test_manager_singleton_after_clear(manager_cls):
    """After `SingletonRegistry.clear` a brand-new instance should be created."""

    original = manager_cls()
    SingletonRegistry.clear()
    replacement = manager_cls()

    assert (
        original is not replacement
    ), f"{manager_cls.__name__} produced the same instance even after clearing the registry"
