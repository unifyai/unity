import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Final, FrozenSet, List, Optional, Type, Union

import unify
from pydantic import BaseModel
from unify import create_fields

from unity.common.context_store import _PRIVATE_FIELDS, _create_context_with_retry
from unity.common.state_managers import BaseStateManager
from unity.session_details import SESSION_DETAILS

_log = logging.getLogger(__name__)


HIVE_CONTEXT_PREFIX: Final[str] = "Hives/"
"""Shared prefix for every Hive-scoped context path.

A body that belongs to a Hive writes shared data under
``Hives/{hive_id}/<Table>/...`` so every member of the Hive reads the same
rows. This module owns the single definition on the Unity side — every
call site that needs to detect, build, or bypass a Hive path imports this
constant instead of hard-coding the literal.
"""


_HIVE_SCOPED_TABLES: Final[FrozenSet[str]] = frozenset(
    {"Tasks", "Contacts", "Secrets", "SecretDefault"},
)
"""Table names whose storage is shared across every body in a Hive.

:meth:`ContextRegistry.base_for` consults this set to decide whether a
manager resolves to the Hive root (``Hives/{hive_id}``) or to the per-body
root (``{user_id}/{assistant_id}``). Add a table name here to make every
body in a Hive read and write the same rows under
``Hives/{hive_id}/<Table>``.
"""


def _is_absolute_reference(ref: str, base: str) -> bool:
    """Return ``True`` when a foreign-key ``references`` value is already absolute.

    :meth:`ContextRegistry._get_contexts_for_manager` resolves relative
    references by routing them to the **target** table's base so
    ``Contacts.contact_id`` on a per-body overlay whose target lives in a
    Hive root resolves to ``Hives/{hive_id}/Contacts.contact_id``. Cross-root
    references that arrive already fully qualified (for example a hand-written
    ``Hives/{hive_id}/Contacts.contact_id`` or a reference that already
    includes the caller's user segment) must pass through untouched;
    double-prefixing them corrupts the FK target.

    A reference is absolute when it starts with :data:`HIVE_CONTEXT_PREFIX`
    or with the first path segment of ``base`` (typically the user id). The
    second case keeps legitimate same-scope references like
    ``{user_id}/Contacts.contact_id`` from being rewritten into
    ``{user_id}/{assistant_id}/{user_id}/...``.
    """
    if ref.startswith(HIVE_CONTEXT_PREFIX):
        return True
    if not base:
        return False
    user_segment = base.split("/", 1)[0]
    if user_segment and ref.startswith(f"{user_segment}/"):
        return True
    return False


def _target_table_from_reference(ref: str) -> Optional[str]:
    """Extract the target table path from a relative FK reference.

    ``"Contacts.contact_id"`` returns ``"Contacts"``;
    ``"Functions/Compositional.function_id"`` returns
    ``"Functions/Compositional"``. Returns ``None`` when the reference has
    no ``.<column>`` suffix so callers can fall back to the declaring
    table's base.
    """
    if "." not in ref:
        return None
    target_path, _column = ref.rsplit(".", 1)
    return target_path or None


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
    _registry: Dict[Any, str] = {}
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

    @classmethod
    def base_for(cls, table_name: str) -> str:
        """Return the context base for a manager's table.

        Tables in :data:`_HIVE_SCOPED_TABLES` resolve to
        ``Hives/{SESSION_DETAILS.hive_id}`` when the active body belongs to
        a Hive, giving every member of the Hive a shared storage root.
        Every other table resolves to the per-body base
        (``{user_id}/{assistant_id}``) regardless of Hive membership, so
        body-scoped surfaces (events, per-body task activations, per-body
        overlays) keep their private path even inside a Hive.

        To Hive-scope a manager, add its table name to
        :data:`_HIVE_SCOPED_TABLES` and declare any cross-root FK
        references (``Hives/{hive_id}/<Table>.col``) as absolute strings;
        :func:`_is_absolute_reference` keeps them untouched during FK
        rewriting.

        Raises
        ------
        RuntimeError
            When no base is available — ``SESSION_DETAILS`` is unpopulated
            and no active Unify context exists. A silent
            ``unknown/unknown`` fallback would hide a genuine configuration
            bug from the caller.
        """
        if table_name in _HIVE_SCOPED_TABLES and SESSION_DETAILS.hive_id is not None:
            return f"{HIVE_CONTEXT_PREFIX}{int(SESSION_DETAILS.hive_id)}"
        base = cls._base_context or cls._get_active_context()
        if not base:
            raise RuntimeError(
                f"Cannot resolve base context for table {table_name!r}: "
                "no base context available (ContextRegistry.setup() has not "
                "run or the active Unify context is empty)",
            )
        return base

    @classmethod
    def _get_contexts_for_manager(
        cls,
        manager: Union[BaseStateManager, Type[BaseStateManager]],
    ) -> Dict[str, Dict]:
        """Resolve each of a manager's required contexts to its full path.

        Each :class:`TableContext` is anchored by :meth:`base_for` so a
        manager that owns both Hive-shared and per-body tables resolves each
        one to the correct root. Foreign-key references are rewritten to
        the **target** table's base (via :meth:`base_for` on the referenced
        table name) so a per-body overlay pointing at a Hive-shared parent
        lands correctly on the shared root — and a Hive-shared table with
        an FK to a per-body parent stays in the per-body root. References
        that :func:`_is_absolute_reference` recognises as already qualified
        pass through untouched.
        """
        assert hasattr(
            manager,
            "Config",
        ), f"Manager {cls._get_manager_name(manager)} must have a Config class attribute"
        assert hasattr(
            manager.Config,
            "required_contexts",
        ), "Config must have a required_contexts class attribute"

        out: Dict[str, Dict] = {}
        for context in manager.Config.required_contexts:
            table_base = cls.base_for(context.name)
            resolved_foreign_keys: Optional[List[Dict[str, Any]]] = None
            if context.foreign_keys:
                resolved_foreign_keys = []
                for foreign_key in context.foreign_keys:
                    fk_copy = foreign_key.copy()
                    ref = foreign_key["references"]
                    if not _is_absolute_reference(ref, table_base):
                        target_table = _target_table_from_reference(ref)
                        target_base = (
                            cls.base_for(target_table) if target_table else table_base
                        )
                        fk_copy["references"] = f"{target_base}/{ref}"
                    resolved_foreign_keys.append(fk_copy)
            out[context.name] = {
                "resolved_name": f"{table_base}/{context.name}",
                "table_context": context,
                "resolved_foreign_keys": resolved_foreign_keys,
            }
        return out

    @classmethod
    def _get_managers(cls) -> List[Union[BaseStateManager, Type[BaseStateManager]]]:
        """Get the list of managers that have required contexts."""
        # TODO: Use dynamic discovery of managers, dynamic discover is slow atm
        # which defeats the purpose of having a context handler

        from unity.blacklist_manager.blacklist_manager import BlackListManager
        from unity.contact_manager.contact_manager import ContactManager
        from unity.dashboard_manager.dashboard_manager import DashboardManager
        from unity.data_manager.data_manager import DataManager
        from unity.file_manager.managers.file_manager import FileManager
        from unity.function_manager.function_manager import FunctionManager
        from unity.guidance_manager.guidance_manager import GuidanceManager
        from unity.image_manager.image_manager import ImageManager
        from unity.knowledge_manager.knowledge_manager import KnowledgeManager
        from unity.secret_manager.secret_manager import SecretManager
        from unity.task_scheduler.task_scheduler import TaskScheduler
        from unity.transcript_manager.transcript_manager import TranscriptManager
        from unity.web_searcher.web_searcher import WebSearcher

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
        """Create the unify context, install its fields, and cache the result.

        Idempotent: tolerates pre-existing contexts and concurrent creation.
        """
        table = entry["table_context"]
        target_name = entry["resolved_name"]
        resolved_foreign_keys = entry.get("resolved_foreign_keys")
        _create_context_with_retry(
            target_name,
            unique_keys=table.unique_keys,
            auto_counting=table.auto_counting,
            description=table.description,
            foreign_keys=resolved_foreign_keys,
        )
        if table.fields:
            try:
                create_fields(table.fields, context=target_name)
            except Exception:
                pass  # Fields already exist or transient failure

        cls._ensure_all_contexts(target_name, table)

        cls._registry[(manager_name, table.name)] = target_name

        return target_name

    @classmethod
    def _ensure_all_contexts(cls, target_name: str, table: TableContext) -> None:
        """Provision cross-assistant and cross-user aggregation shells.

        For a per-body path ``{user_id}/{assistant_id}/<suffix>``, two
        aggregation contexts are created so readers can query across every
        body owned by a user or across every user on the platform:

          - ``{user_id}/All/<suffix>`` — all assistants for this user
          - ``All/<suffix>``          — all users, all assistants

        For test contexts (``tests/.../{user}/{assistant}/<suffix>``), the
        aggregation shells are scoped to the test root so concurrent test
        runs stay isolated.

        Hive-scoped paths short-circuit: ``Hives/{hive_id}/...`` is already
        the shared-across-bodies surface, so minting
        ``Hives/{hive_id}/All/...`` shells would pollute the tree with
        unused siblings and force the Hive cascade delete to walk them.
        """
        if target_name.startswith(HIVE_CONTEXT_PREFIX):
            return

        parts = target_name.split("/")
        if len(parts) < 3:
            return

        if parts[0] == "tests":
            from unity.session_details import UNASSIGNED_USER_CONTEXT

            try:
                user_idx = parts.index(UNASSIGNED_USER_CONTEXT)
            except ValueError:
                return

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
        key = (manager_name, ctx_name)
        cls._registry.pop(key, None)

    @classmethod
    def clear(cls) -> None:
        """Remove all cached contexts from the registry, primarily for test isolation."""
        cls._registry.clear()
        cls._setup_complete = False
        cls._base_context = None

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
            contexts = cls._get_contexts_for_manager(manager)
            ret = cls._create_context_wrapper(manager_name, contexts[ctx_name])

        return ret

    @classmethod
    def _provision_managers(
        cls,
        managers: List[Union[Type[BaseStateManager], BaseStateManager]],
        base: str,
    ) -> None:
        """Provision contexts for the given managers against *base*.

        Shared implementation behind :meth:`setup` and
        :meth:`setup_for_managers`. Sets ``_base_context`` — the per-body
        fallback :meth:`base_for` uses for non-Hive-scoped tables — and
        concurrently materializes every required context (plus aggregation
        shells) through :meth:`_create_context_wrapper`.
        """
        cls._base_context = base

        with ThreadPoolExecutor() as executor:
            futures = []
            for manager in managers:
                manager_name = cls._get_manager_name(manager)
                for _, entry in cls._get_contexts_for_manager(manager).items():
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
    def setup(cls) -> None:
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
        """Return all registered base context names across all managers.

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
