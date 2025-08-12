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
