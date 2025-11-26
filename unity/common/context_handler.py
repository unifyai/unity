import unify
from typing import List, Dict, Optional, Any, Union, Type
from concurrent.futures import ThreadPoolExecutor, as_completed
from unity.common.state_managers import BaseStateManager
from unify import create_context, create_fields
from pydantic import BaseModel


class TableContext(BaseModel):
    # TODO: Ideally should exist in Unify itself
    name: str
    description: str
    fields: Optional[Any] = None
    unique_keys: Optional[Dict[str, str]] = None
    auto_counting: Optional[Dict[str, Optional[str]]] = None
    foreign_keys: Optional[List[Dict[str, Any]]] = None


class ContextHandler:
    _setup_complete = False
    _registry = {}

    @staticmethod
    def _get_active_context() -> str:
        active_context = unify.get_active_context()
        assert (
            active_context["read"] == active_context["write"]
        ), "Read and write contexts must be the same"
        return active_context["read"]

    @staticmethod
    def _get_available_contexts() -> List[str]:
        return list(unify.get_contexts().keys())

    @staticmethod
    def _get_manager_name(
        manager: Union[BaseStateManager, Type[BaseStateManager]],
    ) -> str:
        try:
            return manager.__name__
        except:
            return type(manager).__name__

    @classmethod
    def _get_contexts_for_manager(
        cls,
        manager: Union[BaseStateManager, Type[BaseStateManager]],
        current_context: str,
    ) -> Dict[str, Dict]:
        """Extract the contexts for a manager, resolving context names to fully qualified names."""
        assert hasattr(
            manager,
            "Config",
        ), f"Manager {manager.__name__} must have a Config class attribute"
        assert hasattr(
            manager.Config,
            "required_contexts",
        ), "Config must have a required_contexts class attribute"

        out = {}

        for context in manager.Config.required_contexts:
            if context.foreign_keys:
                for foreign_key in context.foreign_keys:
                    foreign_key["references"] = (
                        f"{current_context}/{foreign_key['references']}"
                    )
            data = {
                "resolved_name": f"{current_context}/{context.name}",
                "table_context": context,
            }
            out[context.name] = data
        return out

    @classmethod
    def _get_managers(cls) -> List[Union[BaseStateManager, Type[BaseStateManager]]]:
        """Get the list of managers that have required contexts."""
        # TODO: Use dynamic discovery of managers, dynamic discover is slow atm
        # which defeats the purpose of having a context handler

        from unity.contact_manager.contact_manager import ContactManager
        from unity.knowledge_manager.knowledge_manager import KnowledgeManager
        from unity.transcript_manager.transcript_manager import TranscriptManager
        from unity.task_scheduler.task_scheduler import TaskScheduler
        from unity.guidance_manager.guidance_manager import GuidanceManager
        from unity.secret_manager.secret_manager import SecretManager
        from unity.web_searcher.web_searcher import WebSearcher
        from unity.image_manager.image_manager import ImageManager
        from unity.function_manager.function_manager import FunctionManager
        from unity.blacklist_manager.blacklist_manager import BlackListManager

        return [
            ContactManager,
            KnowledgeManager,
            TranscriptManager,
            TaskScheduler,
            ImageManager,
            GuidanceManager,
            SecretManager,
            WebSearcher,
            FunctionManager,
            BlackListManager,
        ]

    @classmethod
    def _create_context_wrapper(
        cls,
        manager_name: str,
        entry: Dict,
        remote_contexts: List[str],
    ):
        """Create unify context and ensure fields are created, store in registry."""
        table = entry["table_context"]
        target_name = entry["resolved_name"]
        if target_name not in remote_contexts:
            create_context(
                target_name,
                description=table.description,
                unique_keys=table.unique_keys,
                auto_counting=table.auto_counting,
                foreign_keys=table.foreign_keys,
            )
        # TODO: No need to check current fields, this has no effect if fields are already created
        # possibly can be eliminated if get_fields returns the context for the fields
        if table.fields:
            create_fields(table.fields, context=target_name)

        cls._registry[(manager_name, table.name)] = target_name

        return target_name

    @classmethod
    def refresh(
        cls,
        manager: Union[BaseStateManager, Type[BaseStateManager]],
        ctx_name: str,
    ):
        """Refresh the context by forgetting it and then getting it again."""
        cls.forget(manager, ctx_name)
        return cls.get_context(manager, ctx_name)

    @classmethod
    def forget(
        cls,
        manager: Union[BaseStateManager, Type[BaseStateManager]],
        ctx_name: str,
    ):
        """Remove the context from the registry."""
        manager_name = cls._get_manager_name(manager)
        key = (manager_name, ctx_name)
        cls._registry.pop(key, None)

    @classmethod
    def clear(cls) -> None:
        """Remove all cached contexts from the registry, primarily for test isolation."""
        cls._registry.clear()

    @classmethod
    def get_context(
        cls,
        manager: Union[BaseStateManager, Type[BaseStateManager]],
        ctx_name: str,
    ) -> Optional[str]:
        """Get the context from the registry, creating it if it doesn't exist."""
        manager_name = cls._get_manager_name(manager)
        key = (manager_name, ctx_name)
        ret = cls._registry.get(key)
        if ret is None:
            active_context = cls._get_active_context()
            contexts = cls._get_contexts_for_manager(manager, active_context)
            available_contexts = cls._get_available_contexts()
            ret = cls._create_context_wrapper(
                manager_name,
                contexts[ctx_name],
                available_contexts,
            )

        return ret

    @classmethod
    def setup(cls):
        """Setup the context handler by creating the contexts for all managers."""
        if cls._setup_complete:
            return

        current_context = cls._get_active_context()
        available_contexts = cls._get_available_contexts()

        with ThreadPoolExecutor() as executor:
            futures = []
            for manager in cls._get_managers():
                manager_name = cls._get_manager_name(manager)
                for _, entry in cls._get_contexts_for_manager(
                    manager,
                    current_context,
                ).items():
                    futures.append(
                        executor.submit(
                            cls._create_context_wrapper,
                            manager_name,
                            entry,
                            available_contexts,
                        ),
                    )

            for future in as_completed(futures):
                future.result()

        cls._setup_complete = True
