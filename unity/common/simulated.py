from __future__ import annotations

import ast
import inspect
from typing import Any, Dict, List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────


def _extract_owner_method_pairs(
    cls: Any,
    target_attr: str,
    self_external_map: Dict[str, Any] | None = None,
    extra_class_names: Dict[str, Any] | None = None,
) -> List[Tuple[Any, str]]:
    """Extract (owner_class, method_name) pairs from cls.__init__ for target_attr.

    Finds assignments to self.<target_attr> that are built via methods_to_tool_dict(...)
    and returns each referenced method as (owner_class, method_name).
    """
    try:
        src = inspect.getsource(cls.__init__)
    except Exception:
        return []

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []

    results: List[Tuple[Any, str]] = []

    def _process_value(value: ast.AST) -> None:
        for node in ast.walk(value):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (
                (isinstance(func, ast.Name) and func.id == "methods_to_tool_dict")
                or (
                    isinstance(func, ast.Attribute)
                    and func.attr == "methods_to_tool_dict"
                )
            ):
                continue
            for arg in node.args:
                if not isinstance(arg, ast.Attribute):
                    continue
                name = arg.attr
                owner: Any | None = None
                root = arg.value
                # self.<attr> → current class
                if isinstance(root, ast.Name) and root.id == "self":
                    owner = cls
                # ContactManager.<method> or other explicitly named class
                elif (
                    isinstance(root, ast.Name)
                    and extra_class_names
                    and root.id in extra_class_names
                ):
                    owner = extra_class_names[root.id]
                # self._contact_manager.<method> (or other mapped external refs)
                elif (
                    isinstance(root, ast.Attribute)
                    and isinstance(root.value, ast.Name)
                    and root.value.id == "self"
                    and self_external_map
                    and root.attr in self_external_map
                ):
                    owner = self_external_map[root.attr]
                if owner and name:
                    results.append((owner, name))

    class _Visitor(ast.NodeVisitor):
        def _matches_target(self, target: ast.AST) -> bool:
            return (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr == target_attr
            )

        def visit_Assign(self, node: ast.Assign) -> None:  # type: ignore[override]
            if not node.targets:
                return
            if self._matches_target(node.targets[0]):
                _process_value(node.value)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # type: ignore[override]
            if node.value is None:
                return
            if self._matches_target(node.target):
                _process_value(node.value)

    _Visitor().visit(tree)
    return results


def _build_tool_dict(
    pairs: List[Tuple[Any, str]],
    include_class_name_for: set[Any] | None = None,
) -> Dict[str, Any]:
    from unity.common.llm_helpers import methods_to_tool_dict

    include_class_name_for = include_class_name_for or set()
    by_owner: Dict[Any, List[Any]] = {}
    for owner_cls, method_name in pairs:
        try:
            by_owner.setdefault(owner_cls, []).append(getattr(owner_cls, method_name))
        except Exception:
            # Skip if attribute lookup fails; fallback logic will handle empties
            pass

    tools: Dict[str, Any] = {}
    for owner_cls, methods in by_owner.items():
        if methods:
            tools.update(
                methods_to_tool_dict(
                    *methods,
                    include_class_name=(owner_cls in include_class_name_for),
                ),
            )
    return tools


# ─────────────────────────────────────────────────────────────────────────────
# ContactManager mirroring
# ─────────────────────────────────────────────────────────────────────────────


def mirror_contact_manager_tools(kind: str) -> Dict[str, Any]:
    """Build a tool-dict mirroring the real ContactManager's tool lists.

    kind: "ask" or "update". Uses AST reflection with a static fallback.
    """
    from unity.contact_manager.contact_manager import ContactManager
    from unity.common.llm_helpers import methods_to_tool_dict

    target_attr = "_ask_tools" if kind == "ask" else "_update_tools"

    try:
        pairs = _extract_owner_method_pairs(
            ContactManager,
            target_attr,
            self_external_map=None,
            extra_class_names={"ContactManager": ContactManager},
        )
        if pairs:
            # All ContactManager-owned methods; never include class name
            tools = _build_tool_dict(pairs)
            if tools:
                return tools
    except Exception:
        pass

    # Fallback – current canonical tool sets
    if kind == "ask":
        return methods_to_tool_dict(
            ContactManager._list_columns,
            ContactManager._filter_contacts,
            ContactManager._search_contacts,
            include_class_name=False,
        )
    else:
        return methods_to_tool_dict(
            ContactManager.ask,
            ContactManager._create_contact,
            ContactManager._update_contact,
            ContactManager._delete_contact,
            ContactManager._create_custom_column,
            ContactManager._delete_custom_column,
            ContactManager._merge_contacts,
            include_class_name=False,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TranscriptManager mirroring
# ─────────────────────────────────────────────────────────────────────────────


def mirror_transcript_manager_tools() -> Dict[str, Any]:
    """Build a tool-dict mirroring the real TranscriptManager's tools.

    Uses AST reflection of TranscriptManager.__init__ with a static fallback.
    """
    from unity.common.llm_helpers import methods_to_tool_dict
    from unity.transcript_manager.transcript_manager import TranscriptManager
    from unity.contact_manager.contact_manager import ContactManager

    try:
        pairs = _extract_owner_method_pairs(
            TranscriptManager,
            "_tools",
            self_external_map={"_contact_manager": ContactManager},
        )
        if pairs:
            tools = _build_tool_dict(pairs)
            if tools:
                return tools
    except Exception:
        pass

    # Fallback – current canonical tool set
    return methods_to_tool_dict(
        ContactManager._filter_contacts,
        TranscriptManager._filter_messages,
        TranscriptManager._search_messages,
        include_class_name=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TaskScheduler mirroring
# ─────────────────────────────────────────────────────────────────────────────


def mirror_task_scheduler_tools(kind: str) -> Dict[str, Any]:
    """Build a tool-dict mirroring the real TaskScheduler's tool exposure.

    Uses AST reflection of TaskScheduler.__init__ with a static fallback. Ensures
    that external tools like ContactManager.ask retain their class-qualified naming
    by applying include_class_name=True for those only.
    """
    from unity.common.llm_helpers import methods_to_tool_dict
    from unity.task_scheduler.task_scheduler import TaskScheduler
    from unity.contact_manager.contact_manager import ContactManager

    target_attr = "_ask_tools" if kind == "ask" else "_update_tools"

    try:
        pairs = _extract_owner_method_pairs(
            TaskScheduler,
            target_attr,
            self_external_map={"_contact_manager": ContactManager},
        )
        if pairs:
            tools = _build_tool_dict(pairs, include_class_name_for={ContactManager})
            if tools:
                return tools
    except Exception:
        pass

    # Fallback – current canonical tool sets (kept in sync with TaskScheduler)
    if kind == "ask":
        tools: Dict[str, Any] = {}
        tools.update(
            methods_to_tool_dict(
                TaskScheduler._filter_tasks,
                TaskScheduler._search_tasks,
                TaskScheduler._get_task_queue,
                include_class_name=False,
            ),
        )
        tools.update(
            methods_to_tool_dict(
                ContactManager.ask,
                include_class_name=True,
            ),
        )
        return tools
    else:
        return methods_to_tool_dict(
            # Ask entry point is exposed on update side
            TaskScheduler.ask,
            # Creation / deletion / cancellation
            TaskScheduler._create_task,
            TaskScheduler._delete_task,
            TaskScheduler._cancel_tasks,
            # Queue manipulation
            TaskScheduler._update_task_queue,
            # Attribute mutations
            TaskScheduler._update_task_name,
            TaskScheduler._update_task_description,
            TaskScheduler._update_task_status,
            TaskScheduler._update_task_start_at,
            TaskScheduler._update_task_deadline,
            TaskScheduler._update_task_repetition,
            TaskScheduler._update_task_priority,
            TaskScheduler._update_task_trigger,
            include_class_name=False,
        )
