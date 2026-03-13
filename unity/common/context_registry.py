import unify
from unify import create_fields
from unity.common.state_managers import BaseStateManager
from unity.common.context_store import _PRIVATE_FIELDS, _create_context_with_retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Any, Union, Type
from pydantic import BaseModel


class TableContext(BaseModel):
    # TODO: Ideally should exist in Unify itself
    name: str
    description: str
    fields: Optional[Any] = None
    unique_keys: Optional[Dict[str, str]] = None
    auto_counting: Optional[Dict[str, Optional[str]]] = None
    foreign_keys: Optional[List[Dict[str, Any]]] = None


class ContextRegistry:
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
            # Create copies of foreign_keys to avoid mutating class-level config.
            # Without copying, the references get double-prefixed on subsequent
            # calls (e.g., across test runs), corrupting FK resolution.
            resolved_foreign_keys = None
            if context.foreign_keys:
                resolved_foreign_keys = []
                for foreign_key in context.foreign_keys:
                    fk_copy = foreign_key.copy()
                    fk_copy["references"] = (
                        f"{current_context}/{foreign_key['references']}"
                    )
                    resolved_foreign_keys.append(fk_copy)
            data = {
                "resolved_name": f"{current_context}/{context.name}",
                "table_context": context,
                "resolved_foreign_keys": resolved_foreign_keys,
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
        from unity.data_manager.data_manager import DataManager
        from unity.file_manager.managers.file_manager import FileManager

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
            DataManager,
            FileManager,
        ]

    @classmethod
    def _create_context_wrapper(
        cls,
        manager_name: str,
        entry: Dict,
    ):
        """Create unify context and ensure fields are created, store in registry.

        Idempotent: tolerates pre-existing contexts and concurrent creation.
        """
        table = entry["table_context"]
        target_name = entry["resolved_name"]
        # Use resolved_foreign_keys (with prefixed references) instead of
        # table.foreign_keys to avoid using mutated class-level config.
        resolved_foreign_keys = entry.get("resolved_foreign_keys")
        _create_context_with_retry(
            target_name,
            unique_keys=table.unique_keys,
            auto_counting=table.auto_counting,
            description=table.description,
            foreign_keys=resolved_foreign_keys,
        )
        # Idempotent field creation
        if table.fields:
            try:
                create_fields(table.fields, context=target_name)
            except Exception:
                pass  # Fields already exist or transient failure

        # Also create aggregation contexts for cross-assistant and cross-user queries
        cls._ensure_all_contexts(target_name, table)

        cls._registry[(manager_name, table.name)] = target_name

        return target_name

    @classmethod
    def _ensure_all_contexts(cls, target_name: str, table: TableContext) -> None:
        """
        Ensure aggregation contexts exist for cross-assistant and cross-user queries.

        Creates two contexts:
          - {user_id}/All/{suffix} - all assistants for this user
          - All/{suffix}           - all users, all assistants

        For test contexts (starting with "tests/"), the aggregation contexts are
        scoped to the test root for proper isolation:
          - {test_root}/{user_id}/All/{suffix}
          - {test_root}/All/{suffix}

        These contexts:
        - Have the same fields as the source context (for consistent querying)
        - Include private fields (_user, _user_id, _assistant, _assistant_id, _org, _org_id)
        - Have NO unique_keys or auto_counting (logs are added by reference)
        """
        parts = target_name.split("/")
        if len(parts) < 3:
            return

        # Handle test contexts: tests/.../{default_user_id}/{default_assistant_id}/Suffix
        # Find the user position by looking for the UNASSIGNED_USER_CONTEXT marker
        if parts[0] == "tests":
            from unity.session_details import UNASSIGNED_USER_CONTEXT

            try:
                user_idx = parts.index(UNASSIGNED_USER_CONTEXT)
            except ValueError:
                # Can't determine structure without the UNASSIGNED_USER_CONTEXT marker
                return

            # Need at least User/Assistant/Suffix after the test root
            if user_idx + 2 >= len(parts):
                return

            test_root = "/".join(parts[:user_idx])
            user_ctx = parts[user_idx]
            suffix = "/".join(parts[user_idx + 2 :])

            all_ctxs = [
                (
                    f"{test_root}/{user_ctx}/All/{suffix}",
                    f"Aggregation of {table.name} across all assistants for this user",
                ),
                (
                    f"{test_root}/All/{suffix}",
                    f"Global aggregation of {table.name} across all users and assistants",
                ),
            ]
        else:
            # Production path: User/Assistant/Suffix
            user_ctx = parts[0]
            suffix = "/".join(parts[2:])

            all_ctxs = [
                (
                    f"{user_ctx}/All/{suffix}",
                    f"Aggregation of {table.name} across all assistants for this user",
                ),
                (
                    f"All/{suffix}",
                    f"Global aggregation of {table.name} across all users and assistants",
                ),
            ]

        for all_ctx, description in all_ctxs:
            _create_context_with_retry(all_ctx, description=description)

            # Mirror fields from source context + add private fields
            if table.fields:
                fields_with_private = dict(table.fields)
                fields_with_private.update(_PRIVATE_FIELDS)
                try:
                    create_fields(fields_with_private, context=all_ctx)
                except Exception:
                    pass  # Fields already exist or transient failure

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
            ret = cls._create_context_wrapper(
                manager_name,
                contexts[ctx_name],
            )

        return ret

    @classmethod
    def setup(cls):
        """Setup the context handler by creating the contexts for all managers."""
        if cls._setup_complete:
            return

        current_context = cls._get_active_context()

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
                        ),
                    )

            for future in as_completed(futures):
                future.result()

        cls._setup_complete = True

    @classmethod
    def get_known_base_contexts(cls) -> List[str]:
        """
        Return all registered base context names across all managers.

        This returns the unresolved context names (e.g., "Contacts", "Knowledge",
        "Tasks") from each manager's Config.required_contexts, not the fully
        qualified paths.

        Returns
        -------
        list[str]
            Sorted list of unique base context names.

        Usage Examples
        --------------
        >>> base_contexts = ContextRegistry.get_known_base_contexts()
        >>> print(base_contexts)
        ['Blacklist', 'Contacts', 'Data', 'Functions', 'Guidance', ...]
        """
        base_contexts = set()
        for manager in cls._get_managers():
            if hasattr(manager, "Config") and hasattr(
                manager.Config,
                "required_contexts",
            ):
                for table_ctx in manager.Config.required_contexts:
                    base_contexts.add(table_ctx.name)
        return sorted(base_contexts)
