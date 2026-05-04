import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Final, List, Optional, Type, Union

import unify
from pydantic import BaseModel
from unify import create_fields

from unity.common.context_store import _PRIVATE_FIELDS, _create_context_with_retry
from unity.common.state_managers import BaseStateManager
from unity.common.tool_outcome import ToolError, ToolErrorException
from unity.session_details import SESSION_DETAILS

_log = logging.getLogger(__name__)

SPACE_CONTEXT_PREFIX: Final[str] = "Spaces/"
PERSONAL_ROOT_IDENTITY: Final[str] = "Personal"
PERSONAL_DESTINATION: Final[str] = "personal"
SPACE_DESTINATION_PREFIX: Final[str] = "space:"
INVALID_DESTINATION_ERROR: Final[str] = "invalid_destination"

_SHARED_SCOPED_TABLES: Final[frozenset[str]] = frozenset(
    {
        "Tasks",
        "Contacts",
        "Secrets",
        "Knowledge",
        "Guidance",
        "Functions/Compositional",
        "Functions/Meta",
        "Functions/Primitives",
        "Functions/VirtualEnvs",
        "FileRecords",
        "Files",
        "Data",
        "BlackList",
    },
)


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
    _registry: Dict[tuple[str, str, str], str] = {}
    _base_context: Optional[str] = None

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
        except AttributeError:
            return type(manager).__name__

    @staticmethod
    def _space_root_identity(space_id: int) -> str:
        return f"{SPACE_CONTEXT_PREFIX}{space_id}"

    @classmethod
    def _is_shared_scoped(cls, table_name: str) -> bool:
        """Return whether a table participates in shared-space routing."""
        if table_name in _SHARED_SCOPED_TABLES:
            return True
        parent = table_name
        while "/" in parent:
            parent = parent.rsplit("/", 1)[0]
            if parent in _SHARED_SCOPED_TABLES:
                return True
        return False

    @classmethod
    def _personal_root(cls, manager_name: str, table_name: str) -> str:
        base = cls._base_context or cls._get_active_context()
        if not base:
            raise RuntimeError(
                f"Cannot resolve context for {manager_name}.{table_name}: "
                "no base context available (ContextRegistry.setup() has not "
                "run or the active Unify context is empty)",
            )
        return base

    @classmethod
    def _invalid_destination(
        cls,
        table_name: str,
        destination: str | None,
        message: str,
    ) -> ToolErrorException:
        payload: ToolError = {
            "error_kind": INVALID_DESTINATION_ERROR,
            "message": message,
            "details": {
                "destination": destination,
                "space_ids": sorted(SESSION_DETAILS.space_ids),
                "table_name": table_name,
            },
        }
        return ToolErrorException(payload)

    @classmethod
    def _parse_destination(
        cls,
        manager_name: str,
        table_name: str,
        destination: str | None,
    ) -> tuple[str, str]:
        """Resolve a public destination string to a cache identity and root path."""
        if destination is None or destination == PERSONAL_DESTINATION:
            return PERSONAL_ROOT_IDENTITY, cls._personal_root(manager_name, table_name)

        if not isinstance(destination, str) or not destination.startswith(
            SPACE_DESTINATION_PREFIX,
        ):
            raise cls._invalid_destination(
                table_name,
                destination,
                "Destination must be 'personal' or 'space:<id>'.",
            )

        if not cls._is_shared_scoped(table_name):
            raise cls._invalid_destination(
                table_name,
                destination,
                f"Table {table_name!r} does not support space destinations.",
            )

        raw_space_id = destination[len(SPACE_DESTINATION_PREFIX) :]
        try:
            space_id = int(raw_space_id)
        except (TypeError, ValueError):
            raise cls._invalid_destination(
                table_name,
                destination,
                "Space destination must include an integer space id.",
            )
        if space_id not in SESSION_DETAILS.space_ids:
            raise cls._invalid_destination(
                table_name,
                destination,
                f"Assistant is not a member of space {space_id}.",
            )

        root_identity = cls._space_root_identity(space_id)
        return root_identity, root_identity

    @classmethod
    def _get_contexts_for_manager(
        cls,
        manager: Union[BaseStateManager, Type[BaseStateManager]],
        current_context: str,
        root_identity: str,
    ) -> Dict[str, Dict]:
        """Extract the contexts for a manager, resolving context names to fully qualified names."""
        assert hasattr(
            manager,
            "Config",
        ), f"Manager {cls._get_manager_name(manager)} must have a Config class attribute"
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
                "root_identity": root_identity,
                "root_context": current_context,
            }
            out[context.name] = data
        return out

    @classmethod
    def _get_managers(cls) -> List[Union[BaseStateManager, Type[BaseStateManager]]]:
        """Get the list of managers that have required contexts."""
        # TODO: Use dynamic discovery of managers, dynamic discover is slow atm
        # which defeats the purpose of having a context handler

        from unity.contact_manager.contact_manager import ContactManager
        from unity.dashboard_manager.dashboard_manager import DashboardManager
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
            DashboardManager,
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
    ) -> str:
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

        cls._registry[(manager_name, table.name, entry["root_identity"])] = target_name

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
        if target_name.startswith(SPACE_CONTEXT_PREFIX):
            return

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
    ) -> Optional[str]:
        """Refresh the context by forgetting it and then getting it again."""
        cls.forget(manager, ctx_name)
        return cls.get_context(manager, ctx_name)

    @classmethod
    def forget(
        cls,
        manager: Union[BaseStateManager, Type[BaseStateManager]],
        ctx_name: str,
    ) -> None:
        """Remove the context from the registry."""
        manager_name = cls._get_manager_name(manager)
        for key in list(cls._registry):
            if key[0] == manager_name and key[1] == ctx_name:
                cls._registry.pop(key, None)

    @classmethod
    def forget_departed_space_roots(cls, space_ids: list[int]) -> None:
        """Drop cached entries for shared roots the assistant can no longer reach."""
        reachable_roots = {cls._space_root_identity(space_id) for space_id in space_ids}
        for key in list(cls._registry):
            root_identity = key[2]
            if root_identity.startswith(SPACE_CONTEXT_PREFIX) and (
                root_identity not in reachable_roots
            ):
                cls._registry.pop(key, None)

    @classmethod
    def clear(cls) -> None:
        """Remove all cached contexts from the registry, primarily for test isolation."""
        cls._registry.clear()
        cls._setup_complete = False
        cls._base_context = None

    @classmethod
    def _ensure_context(
        cls,
        manager: Union[BaseStateManager, Type[BaseStateManager]],
        table_name: str,
        root_identity: str,
        root_context: str,
    ) -> str:
        manager_name = cls._get_manager_name(manager)
        key = (manager_name, table_name, root_identity)
        target_name = cls._registry.get(key)
        if target_name is not None:
            return target_name

        contexts = cls._get_contexts_for_manager(manager, root_context, root_identity)
        return cls._create_context_wrapper(manager_name, contexts[table_name])

    @classmethod
    def write_root(
        cls,
        manager: Union[BaseStateManager, Type[BaseStateManager]],
        table_name: str,
        *,
        destination: str | None,
    ) -> str:
        """Resolve and provision the root a write should target."""
        manager_name = cls._get_manager_name(manager)
        root_identity, root_context = cls._parse_destination(
            manager_name,
            table_name,
            destination,
        )
        cls._ensure_context(manager, table_name, root_identity, root_context)
        return root_context

    @classmethod
    def read_roots(
        cls,
        manager: Union[BaseStateManager, Type[BaseStateManager]],
        table_name: str,
    ) -> list[str]:
        """Resolve and provision the ordered roots a read should fan out across."""
        manager_name = cls._get_manager_name(manager)
        personal_root = cls._personal_root(manager_name, table_name)
        roots = [(PERSONAL_ROOT_IDENTITY, personal_root)]
        if cls._is_shared_scoped(table_name):
            roots.extend(
                (cls._space_root_identity(space_id), cls._space_root_identity(space_id))
                for space_id in sorted(set(SESSION_DETAILS.space_ids))
            )

        for root_identity, root_context in roots:
            cls._ensure_context(manager, table_name, root_identity, root_context)
        return [root_context for _, root_context in roots]

    @classmethod
    def get_context(
        cls,
        manager: Union[BaseStateManager, Type[BaseStateManager]],
        ctx_name: str,
    ) -> Optional[str]:
        """Get the context from the registry, creating it if it doesn't exist."""
        manager_name = cls._get_manager_name(manager)
        root_context = cls._personal_root(manager_name, ctx_name)
        return cls._ensure_context(
            manager,
            ctx_name,
            PERSONAL_ROOT_IDENTITY,
            root_context,
        )

    @classmethod
    def _provision_managers(
        cls,
        managers: List[Union[Type[BaseStateManager], BaseStateManager]],
        base: str,
    ) -> None:
        """Provision contexts for the given managers against *base*.

        Shared implementation behind :meth:`setup` and
        :meth:`setup_for_managers`.  Sets ``_base_context`` and
        concurrently creates every required context (+ aggregation
        contexts) via :meth:`_create_context_wrapper`.
        """
        cls._base_context = base

        with ThreadPoolExecutor() as executor:
            futures = []
            for manager in managers:
                manager_name = cls._get_manager_name(manager)
                for _, entry in cls._get_contexts_for_manager(
                    manager,
                    base,
                    PERSONAL_ROOT_IDENTITY,
                ).items():
                    futures.append(
                        executor.submit(
                            cls._create_context_wrapper,
                            manager_name,
                            entry,
                        ),
                    )

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    _log.warning("Context creation failed (will retry lazily): %s", e)

    @classmethod
    def setup(cls):
        """Setup the context handler by creating the contexts for all managers."""
        if cls._setup_complete:
            return

        cls._provision_managers(cls._get_managers(), cls._get_active_context())
        cls._setup_complete = True

    @classmethod
    def setup_for_managers(
        cls,
        managers: List[Union[Type[BaseStateManager], BaseStateManager]],
        *,
        base_context: Optional[str] = None,
    ) -> None:
        """Provision contexts for a specific subset of managers.

        Unlike :meth:`setup` which provisions **all** registered managers
        and sets ``_setup_complete``, this is designed for worker processes
        that only need a few managers (e.g. ``FileManager`` +
        ``DataManager`` for the ingest worker).

        It does **not** set ``_setup_complete`` so that a later full
        ``setup()`` call (if ever needed) still runs normally.

        Parameters
        ----------
        managers :
            Manager classes whose ``Config.required_contexts`` should be
            provisioned.
        base_context :
            Explicit base context string.  When *None* (the default),
            reads the current Unify active context via the SDK.
        """
        cls._provision_managers(
            managers,
            base_context or cls._get_active_context(),
        )

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
