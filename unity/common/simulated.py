from __future__ import annotations

import ast
import inspect
from typing import Any, Dict, List, Tuple


def _parse_methods_to_tool_dict_calls_from_annassign(
    src: str,
    target_attr: str,
) -> List[str]:
    """Return method attribute names passed to methods_to_tool_dict for target attr.

    Scans annotated assignments like:
        self._ask_tools: Dict[...] = { **methods_to_tool_dict(self._foo, ...) }
    and extracts the right-most attribute names from positional args.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []

    collected: List[str] = []

    class _Visitor(ast.NodeVisitor):
        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # type: ignore[override]
            target = node.target
            if not (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr == target_attr
            ):
                return
            value = node.value
            if value is None:
                return
            calls = [
                n
                for n in ast.walk(value)
                if isinstance(n, ast.Call)
                and (
                    (
                        isinstance(n.func, ast.Name)
                        and n.func.id == "methods_to_tool_dict"
                    )
                    or (
                        isinstance(n.func, ast.Attribute)
                        and n.func.attr == "methods_to_tool_dict"
                    )
                )
            ]
            if not calls:
                return
            call = calls[0]
            for arg in call.args:
                if isinstance(arg, ast.Attribute) and isinstance(arg.value, ast.Name):
                    if arg.value.id in {"self", "ContactManager"}:
                        collected.append(arg.attr)

    _Visitor().visit(tree)
    return collected


def mirror_contact_manager_tools(kind: str) -> Dict[str, Any]:
    """Build a tool-dict mirroring the real ContactManager's tool lists.

    kind: "ask" or "update".
    Uses AST reflection on ContactManager.__init__ with a static fallback.
    """
    # Local imports to avoid import-time cycles
    from unity.contact_manager.contact_manager import ContactManager
    from unity.common.llm_helpers import methods_to_tool_dict

    try:
        src = inspect.getsource(ContactManager.__init__)
    except Exception:
        src = ""

    ask_attrs: List[str] = _parse_methods_to_tool_dict_calls_from_annassign(
        src,
        "_ask_tools",
    )
    upd_attrs: List[str] = _parse_methods_to_tool_dict_calls_from_annassign(
        src,
        "_update_tools",
    )

    try:
        if kind == "ask" and ask_attrs:
            methods = [getattr(ContactManager, name) for name in ask_attrs]
            return methods_to_tool_dict(*methods, include_class_name=False)
        if kind == "update" and upd_attrs:
            methods = [getattr(ContactManager, name) for name in upd_attrs]
            return methods_to_tool_dict(*methods, include_class_name=False)
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


def _extract_tm_tool_attrs_from_real() -> List[Tuple[str, str]]:
    """Return (owner, method_name) from TranscriptManager.__init__ tools.

    Owner is one of {"TranscriptManager", "ContactManager"}.
    """
    from unity.transcript_manager.transcript_manager import TranscriptManager

    try:
        src = inspect.getsource(TranscriptManager.__init__)
    except Exception:
        return []

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []

    results: List[Tuple[str, str]] = []

    class _InitVisitor(ast.NodeVisitor):
        def visit_Assign(self, node: ast.Assign) -> None:  # type: ignore[override]
            if not node.targets:
                return
            target = node.targets[0]
            if not (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr == "_tools"
            ):
                return
            value = node.value
            if not isinstance(value, ast.Call):
                return
            func = value.func
            if not (
                (isinstance(func, ast.Name) and func.id == "methods_to_tool_dict")
                or (
                    isinstance(func, ast.Attribute)
                    and func.attr == "methods_to_tool_dict"
                )
            ):
                return
            for arg in value.args:
                owner = None
                name = None
                cur = arg
                if isinstance(cur, ast.Attribute):
                    tail = cur.attr
                    root = cur.value
                    if isinstance(root, ast.Attribute) and isinstance(
                        root.value,
                        ast.Name,
                    ):
                        if root.value.id == "self" and root.attr == "_contact_manager":
                            owner = "ContactManager"
                            name = tail
                    elif isinstance(root, ast.Name) and root.id == "self":
                        owner = "TranscriptManager"
                        name = tail
                if owner and name:
                    results.append((owner, name))

    _InitVisitor().visit(tree)
    return results


def mirror_transcript_manager_tools() -> Dict[str, Any]:
    """Build a tool-dict mirroring the real TranscriptManager's tools.

    Uses AST reflection of TranscriptManager.__init__ with a static fallback.
    """
    from unity.common.llm_helpers import methods_to_tool_dict
    from unity.transcript_manager.transcript_manager import TranscriptManager
    from unity.contact_manager.contact_manager import ContactManager

    mapping = _extract_tm_tool_attrs_from_real()
    methods: List[Any] = []
    if mapping:
        try:
            for owner, name in mapping:
                if owner == "ContactManager":
                    methods.append(getattr(ContactManager, name))
                elif owner == "TranscriptManager":
                    methods.append(getattr(TranscriptManager, name))
        except Exception:
            methods = []  # trigger fallback

    if methods:
        return methods_to_tool_dict(*methods, include_class_name=False)

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


def _extract_ts_tool_attrs_from_real(kind: str) -> List[Tuple[str, str]]:
    """Return (owner, method_name) pairs from TaskScheduler.__init__ tool dicts.

    kind: "ask" or "update". Owner is one of {"TaskScheduler", "ContactManager"}.
    """
    from unity.task_scheduler.task_scheduler import TaskScheduler

    try:
        src = inspect.getsource(TaskScheduler.__init__)
    except Exception:
        return []

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []

    results: List[Tuple[str, str]] = []
    target_attr = "_ask_tools" if kind == "ask" else "_update_tools"

    class _InitVisitor(ast.NodeVisitor):
        def visit_Assign(self, node: ast.Assign) -> None:  # type: ignore[override]
            if not node.targets:
                return
            target = node.targets[0]
            if not (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr == target_attr
            ):
                return
            value = node.value
            # The value is typically a dict literal with unpacked calls
            for call in ast.walk(value):
                if not isinstance(call, ast.Call):
                    continue
                func = call.func
                if not (
                    (isinstance(func, ast.Name) and func.id == "methods_to_tool_dict")
                    or (
                        isinstance(func, ast.Attribute)
                        and func.attr == "methods_to_tool_dict"
                    )
                ):
                    continue
                for arg in call.args:
                    owner = None
                    name = None
                    cur = arg
                    if isinstance(cur, ast.Attribute):
                        tail = cur.attr
                        root = cur.value
                        # self._contact_manager.ask → (ContactManager, 'ask')
                        if isinstance(root, ast.Attribute) and isinstance(
                            root.value,
                            ast.Name,
                        ):
                            if (
                                root.value.id == "self"
                                and root.attr == "_contact_manager"
                            ):
                                owner = "ContactManager"
                                name = tail
                        # self._foo → (TaskScheduler, 'foo')
                        elif isinstance(root, ast.Name) and root.id == "self":
                            owner = "TaskScheduler"
                            name = tail
                    if owner and name:
                        results.append((owner, name))

    _InitVisitor().visit(tree)
    return results


def mirror_task_scheduler_tools(kind: str) -> Dict[str, Any]:
    """Build a tool-dict mirroring the real TaskScheduler's tool exposure.

    Uses AST reflection of TaskScheduler.__init__ with a static fallback. Ensures
    that external tools like ContactManager.ask retain their class-qualified
    naming by applying include_class_name=True for those only.
    """
    from unity.common.llm_helpers import methods_to_tool_dict
    from unity.task_scheduler.task_scheduler import TaskScheduler
    from unity.contact_manager.contact_manager import ContactManager

    mapping = _extract_ts_tool_attrs_from_real(kind)

    # When mapping is available, build two buckets so we can preserve
    # the ContactManager.ask class-name in tool keys.
    if mapping:
        ts_methods: List[Any] = []
        cm_methods: List[Any] = []
        try:
            for owner, name in mapping:
                if owner == "TaskScheduler":
                    ts_methods.append(getattr(TaskScheduler, name))
                elif owner == "ContactManager":
                    cm_methods.append(getattr(ContactManager, name))
        except Exception:
            ts_methods, cm_methods = [], []  # trigger fallback

        if ts_methods or cm_methods:
            tools: Dict[str, Any] = {}
            if ts_methods:
                tools.update(
                    methods_to_tool_dict(*ts_methods, include_class_name=False),
                )
            if cm_methods:
                tools.update(
                    methods_to_tool_dict(*cm_methods, include_class_name=True),
                )
            if tools:
                return tools

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
