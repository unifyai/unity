from __future__ import annotations

from importlib import import_module as _import_module
from typing import Any, Callable, Dict


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        try:
            if isinstance(obj, dict):
                return obj.get(name, default)
        except Exception:
            pass
    return default


def capture_inline_tools_registry(raw_tools: Dict[str, Any]) -> list[dict]:
    """
    Build a serialisable registry of inline tools from a mapping of
    {name -> callable | ToolSpec}. Skips non-importable closures.
    Preserves read_only/manager_tool flags when present on the callable.
    """
    out: list[dict] = []
    for _name, _val in (raw_tools or {}).items():
        try:
            fn: Callable
            # Accept ToolSpec-like wrapper with `.fn` attribute
            if hasattr(_val, "fn") and callable(_get_attr(_val, "fn")):
                fn = _val.fn  # type: ignore[assignment]
            else:
                fn = _val  # type: ignore[assignment]

            mod = _get_attr(fn, "__module__")
            qn = _get_attr(fn, "__qualname__")
            if not isinstance(mod, str) or not isinstance(qn, str):
                continue
            # Skip closures/local defs – not importable by qualname
            if "<locals>" in qn:
                continue
            ro = _get_attr(fn, "_tool_spec_read_only")
            mt = _get_attr(fn, "_tool_spec_manager_tool")
            out.append(
                {
                    "name": _name,
                    "module": mod,
                    "qualname": qn,
                    "read_only": bool(ro) if ro is not None else None,
                    "manager_tool": bool(mt) if mt is not None else None,
                },
            )
        except Exception:
            continue
    return out


def resolve_inline_tools(entry_tools: list[Any]) -> Dict[str, Callable]:
    """
    Resolve a list of ToolRef-like entries (having attributes or keys:
    name, module, qualname, read_only, manager_tool) into a mapping
    {name -> callable}. Applies read_only/manager_tool flags on the
    resolved callables for downstream normalisation.
    """
    tools: Dict[str, Callable] = {}
    for t in entry_tools or []:
        mod_name = _get_attr(t, "module")
        qualname = _get_attr(t, "qualname")
        name = _get_attr(t, "name")
        if (
            not isinstance(mod_name, str)
            or not isinstance(qualname, str)
            or not isinstance(name, str)
        ):
            raise ValueError("Inline tool refs must include name, module, qualname")

        mod = _import_module(mod_name)
        obj: Any = mod
        try:
            for part in str(qualname).split("."):
                obj = getattr(obj, part)
        except Exception as exc:
            raise ValueError(
                f"Failed to resolve tool {name} at {mod_name}.{qualname}",
            ) from exc

        # Apply flags expected by normalise_tools()
        try:
            if _get_attr(t, "read_only") is True:
                setattr(obj, "_tool_spec_read_only", True)
            if _get_attr(t, "manager_tool") is True:
                setattr(obj, "_tool_spec_manager_tool", True)
        except Exception:
            pass

        tools[name] = obj  # type: ignore[assignment]
    return tools


__all__ = (
    "capture_inline_tools_registry",
    "resolve_inline_tools",
)
