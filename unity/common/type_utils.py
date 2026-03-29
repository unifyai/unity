"""Vendored subset of Orchestra's type inference, matching, and normalization.

Source: orchestra/web/api/log/utils/type_utils.py
Purpose: Mirror Orchestra's type semantics in Unity so that pre-scan type
         inference and coercion produce results identical to what Orchestra
         would compute.  Keeping a local copy avoids a cross-deployment import
         (Orchestra is a separate service).

Only the **stdlib-only** portions are vendored.  Pydantic / jsonschema helpers
are deliberately excluded — they are not needed for bulk-ingestion type
inference and matching.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from datetime import time as _time
from datetime import timedelta
from typing import Any, Callable, Iterable, List, Optional, Set, Tuple

# =========================================================================
# Constants
# =========================================================================

SUPPORTED_BASE_TYPES = [
    "bool",
    "int",
    "float",
    "str",
    "datetime",
    "time",
    "date",
    "timedelta",
    "dict",
    "list",
    "image",
    "audio",
    "vector",
]

SPECIAL_FIELD_TYPES = [
    "Any",
    "NoneType",
    "enum",
]

JSON_SCHEMA_TYPE_ALIASES = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "list",
    "object": "dict",
    "null": "NoneType",
}

DEFAULT_FIELD_TYPE = "Any"

# =========================================================================
# AST parser for type strings
# =========================================================================


@dataclass
class _TypeNode:
    name: str
    args: List["_TypeNode"]


_TOKEN_SPEC = [
    ("LBRACK", r"\["),
    ("RBRACK", r"\]"),
    ("COMMA", r","),
    ("ELLIPSIS", r"\.\.\."),
    ("IDENT", r"[A-Za-z_][A-Za-z0-9_]*"),
    ("WS", r"\s+"),
]
_TOKEN_RE = re.compile("|".join(f"(?P<{n}>{p})" for n, p in _TOKEN_SPEC))


class _Token:
    __slots__ = ("typ", "val")

    def __init__(self, typ: str, val: str):
        self.typ = typ
        self.val = val


def _lex(s: str) -> List[_Token]:
    s = s or ""
    toks: List[_Token] = []
    pos = 0
    n = len(s)
    while pos < n:
        m = re.match(r"\s+", s[pos:])
        if m:
            pos += m.end()
            continue
        m = _TOKEN_RE.match(s, pos)
        if not m:
            raise ValueError(f"Unexpected character at position {pos}: {s[pos]!r}")
        typ = m.lastgroup
        val = m.group(typ)
        if typ != "WS":
            toks.append(_Token(typ, val))
        pos = m.end()
    return toks


_COLLECTION_CANON = {
    "list": "List",
    "dict": "Dict",
    "tuple": "Tuple",
    "set": "Set",
    "union": "Union",
    "optional": "Optional",
    "literal": "Literal",
    "annotated": "Annotated",
    "sequence": "Sequence",
    "mapping": "Mapping",
}
_COLLECTION_CANON_REVERSE = {v: k for k, v in _COLLECTION_CANON.items()}


def _canon_ident(name: str) -> str:
    if not name:
        return name
    lower = name.lower()
    if lower == "any":
        return "Any"
    if lower in ("nonetype", "none"):
        return "NoneType"
    if lower == "enum":
        return "enum"
    if lower in _COLLECTION_CANON:
        return _COLLECTION_CANON[lower]
    if lower in JSON_SCHEMA_TYPE_ALIASES:
        return JSON_SCHEMA_TYPE_ALIASES[lower]
    if lower in SUPPORTED_BASE_TYPES:
        return lower
    return name if name and name[0].isupper() else lower


class _Parser:
    def __init__(self, toks: List[_Token]):
        self.toks = toks
        self.i = 0

    def _peek(self, *kinds: str) -> bool:
        return self.i < len(self.toks) and self.toks[self.i].typ in kinds

    def _eat(self, kind: str) -> _Token:
        if not self._peek(kind):
            got = self.toks[self.i].typ if self.i < len(self.toks) else "EOF"
            raise ValueError(f"Expected {kind}, got {got}")
        t = self.toks[self.i]
        self.i += 1
        return t

    def parse(self) -> _TypeNode:
        node = self._parse_simple()
        if self.i != len(self.toks):
            raise ValueError("Unexpected tokens at end of type string")
        return node

    def _parse_simple(self) -> _TypeNode:
        if not self._peek("IDENT"):
            raise ValueError("Type must start with an identifier")
        name_tok = self._eat("IDENT")
        name = _canon_ident(name_tok.val)

        if name == "Optional":
            self._eat("LBRACK")
            inner = self._parse_simple()
            self._eat("RBRACK")
            return _TypeNode("Union", [inner, _TypeNode("NoneType", [])])

        if self._peek("LBRACK"):
            self._eat("LBRACK")
            args = self._parse_args()
            self._eat("RBRACK")
            if name == "Tuple" and len(args) == 2 and args[1].name == "Ellipsis":
                return _TypeNode("Tuple", [args[0], _TypeNode("Ellipsis", [])])
            return _TypeNode(name, args)

        if name == "None":
            name = "NoneType"
        return _TypeNode(name, [])

    def _parse_args(self) -> List[_TypeNode]:
        args: List[_TypeNode] = []
        while True:
            if self._peek("ELLIPSIS"):
                self._eat("ELLIPSIS")
                args.append(_TypeNode("Ellipsis", []))
            else:
                args.append(self._parse_simple())
            if self._peek("COMMA"):
                self._eat("COMMA")
                continue
            break
        return args


def _render(node: _TypeNode) -> str:
    name = node.name
    if name.lower() in ("list", "dict", "tuple", "set", "union"):
        name = _COLLECTION_CANON[name.lower()]
    elif name == "None":
        name = "NoneType"

    if not node.args:
        if name in SPECIAL_FIELD_TYPES:
            return name
        if name in _COLLECTION_CANON_REVERSE:
            return _COLLECTION_CANON_REVERSE[name]
        return name if name and name[0].isupper() else name.lower()

    if name == "Union":
        return f"Union[{', '.join(_render(a) for a in node.args)}]"
    if name == "Tuple" and len(node.args) == 2 and node.args[1].name == "Ellipsis":
        return f"Tuple[{_render(node.args[0])}, ...]"
    return f"{name}[{', '.join(_render(a) for a in node.args)}]"


def _parse_to_ast(type_str: str) -> _TypeNode:
    return _Parser(_lex(type_str)).parse()


# =========================================================================
# Public normalization API
# =========================================================================


def normalize_type_string(type_str: str) -> str:
    """Normalize a type string to canonical form.

    Examples::

        "Int"                       -> "int"
        "ANY"                       -> "Any"
        "nonetype"                  -> "NoneType"
        "Optional[int]"             -> "Union[int, NoneType]"
        "LIST[INT]"                 -> "List[int]"
        "Dict[Str, Float]"         -> "Dict[str, float]"
        "List[Dict[str, List[int]]]" -> "List[Dict[str, List[int]]]"
    """
    if not type_str:
        return type_str
    try:
        tree = _parse_to_ast(type_str)
        return _render(tree)
    except Exception:
        return type_str.strip()


def parse_nested_type(type_str: str) -> Tuple[str, Optional[List[str]]]:
    """Parse a type string into ``(base_type, inner_types | None)``."""
    if not type_str:
        return (type_str, None)
    try:
        tree = _parse_to_ast(type_str)
        norm = _render(tree)
        if not tree.args:
            return (norm, None)
        return (tree.name, [_render(a) for a in tree.args])
    except Exception:
        match = re.match(r"^(\w+)\[(.*)\]$", type_str.strip())
        if not match:
            return (type_str, None)
        base_type = match.group(1)
        inner_str = match.group(2)
        if "," in inner_str:
            inner_types = [part.strip() for part in inner_str.split(",")]
        else:
            inner_types = [inner_str.strip()]
        return (base_type, inner_types)


# =========================================================================
# types_match — structural compatibility check
# =========================================================================


def types_match(field_type: Any, inferred_type: str) -> bool:
    """Check whether *inferred_type* is compatible with *field_type*.

    Mirrors Orchestra's ``types_match`` semantics exactly (minus Pydantic
    schema support which is not needed for ingestion pre-scan).

    Key rules:
    * ``NoneType`` is a *weak* type — it matches any field type.
    * ``Any`` matches everything in either direction.
    * ``enum`` matches ``str``.
    * Containers match by family + structural recursion.
    """
    if not isinstance(field_type, str):
        return False

    norm_field = normalize_type_string(field_type)
    norm_inferred = normalize_type_string(inferred_type)

    if norm_field == norm_inferred or norm_field.lower() == norm_inferred.lower():
        return True
    if norm_field.lower() == "enum" and norm_inferred.lower() == "str":
        return True
    if norm_inferred == "NoneType" or norm_field == "NoneType":
        return True

    # -- AST constraint graph (vendored from Orchestra) --

    @dataclass
    class _TC:
        kind: str
        name: Optional[str] = None
        elements: Optional[List["_TC"]] = None
        key: Optional["_TC"] = None
        value: Optional["_TC"] = None
        variadic: bool = False

    def _family(n: str) -> Optional[str]:
        lower = n.lower()
        return lower if lower in ("list", "dict", "set", "tuple") else None

    def _to_c(node: _TypeNode) -> _TC:
        if node.name == "Any":
            return _TC(kind="any")
        if node.name == "NoneType":
            return _TC(kind="none")
        if node.name.lower() == "enum":
            return _TC(kind="enum")
        if node.name == "Union" and node.args:
            return _TC(kind="union", elements=[_to_c(a) for a in node.args])

        fam = _family(node.name)
        if fam is None:
            return _TC(kind="primitive", name=node.name)

        if fam == "list":
            if not node.args:
                return _TC(kind="list", elements=None)
            return _TC(kind="list", elements=[_to_c(a) for a in node.args])

        if fam == "set":
            if not node.args:
                return _TC(kind="set", elements=None)
            return _TC(kind="set", elements=[_to_c(a) for a in node.args])

        if fam == "dict":
            if len(node.args) >= 2:
                return _TC(
                    kind="dict",
                    key=_to_c(node.args[0]),
                    value=_to_c(node.args[1]),
                )
            return _TC(kind="dict", key=None, value=None)

        if fam == "tuple":
            if len(node.args) == 2 and node.args[1].name == "Ellipsis":
                return _TC(kind="tuple", elements=[_to_c(node.args[0])], variadic=True)
            if not node.args:
                return _TC(kind="tuple", elements=None, variadic=False)
            return _TC(
                kind="tuple",
                elements=[_to_c(a) for a in node.args],
                variadic=False,
            )

        return _TC(kind="primitive", name=node.name)

    def _sat(exp: _TC, inf: _TC) -> bool:
        if (
            exp.kind == "any"
            or inf.kind == "any"
            or inf.kind == "none"
            or exp.kind == "none"
        ):
            return True
        if (
            exp.kind == "enum"
            and inf.kind == "primitive"
            and (inf.name or "").lower() == "str"
        ):
            return True

        # Union handling: strip NoneType members to avoid NoneType's "weak"
        # semantics making every Optional[T] match any type.
        if exp.kind == "union":
            non_none = [m for m in (exp.elements or []) if m.kind != "none"]
            if non_none:
                return any(_sat(m, inf) for m in non_none)
            return True
        if inf.kind == "union":
            non_none = [m for m in (inf.elements or []) if m.kind != "none"]
            if non_none:
                return all(_sat(exp, m) for m in non_none)
            return True

        if exp.kind == "primitive" and inf.kind == "primitive":
            return (exp.name or "").lower() == (inf.name or "").lower()

        if (
            exp.kind in ("list", "set", "dict", "tuple")
            and exp.elements is None
            and exp.key is None
            and exp.value is None
            and not exp.variadic
        ):
            return inf.kind == exp.kind

        if exp.kind in ("list", "set"):
            if inf.kind != exp.kind:
                return False
            if inf.elements is None or exp.elements is None:
                return True
            return all(
                any(_sat(ea, ia) for ea in (exp.elements or []))
                for ia in (inf.elements or [])
            )

        if exp.kind == "dict":
            if inf.kind != "dict":
                return False
            if (
                exp.key is None
                or exp.value is None
                or inf.key is None
                or inf.value is None
            ):
                return True
            return _sat(exp.key, inf.key) and _sat(exp.value, inf.value)

        if exp.kind == "tuple":
            if inf.kind != "tuple":
                return False
            if exp.elements is None:
                return True
            if exp.variadic:
                base = exp.elements[0]
                if inf.elements is None:
                    return True
                if len(inf.elements) == 1 and inf.variadic:
                    return _sat(base, inf.elements[0])
                return all(_sat(base, ie) for ie in (inf.elements or []))
            else:
                if inf.elements is None:
                    return True
                if inf.variadic and len(inf.elements) == 1:
                    base = inf.elements[0]
                    return all(_sat(e, base) for e in exp.elements)
                return all(
                    any(_sat(e, ie) for e in exp.elements) for ie in inf.elements
                )

        return False

    try:
        exp_node = _parse_to_ast(norm_field)
        inf_node = _parse_to_ast(norm_inferred)
        return _sat(_to_c(exp_node), _to_c(inf_node))
    except Exception:
        return (
            norm_field == norm_inferred or norm_field.lower() == norm_inferred.lower()
        )


# =========================================================================
# String detectors (temporal patterns)
# =========================================================================


def _is_date_string(value: str) -> bool:
    try:
        if isinstance(value, str):
            clean_value = value.strip("\"'")
            for fmt in (
                "%Y-%m-%d",
                "%m/%d/%Y",
                "%d/%m/%Y",
                "%d-%m-%Y",
                "%B %d, %Y",
                "%b %d, %Y",
            ):
                try:
                    parsed_date = datetime.strptime(clean_value, fmt).date()
                    if isinstance(parsed_date, date):
                        return True
                except ValueError:
                    continue
            if re.match(r"^\d{4}-\d{2}-\d{2}$", clean_value):
                try:
                    date.fromisoformat(clean_value)
                    return True
                except ValueError:
                    pass
        return False
    except Exception:
        return False


def _is_timedelta_string(value: str) -> bool:
    try:
        if isinstance(value, str):
            clean_value = value.strip("\"'")
            iso_duration = r"^P(?=\d|T\d)(?:\d+Y)?(?:\d+M)?(?:\d+D)?(?:T(?=\d)(?:\d+H)?(?:\d+M)?(?:\d+(?:\.\d+)?S)?)?$"
            if re.match(iso_duration, clean_value):
                return True
            pg_interval = r"^(\d+\s+(?:day|days|hour|hours|minute|minutes|second|seconds)(?:\s+|$))+$"
            if re.match(pg_interval, clean_value, re.IGNORECASE):
                return True
            if re.match(r"^\d+:\d{2}(:\d{2})?$", clean_value):
                if not _is_time_string(clean_value):
                    return True
        return False
    except Exception:
        return False


def _is_time_string(value: str) -> bool:
    try:
        if isinstance(value, str):
            clean_value = value.strip("\"'")
            for fmt in (
                "%H:%M:%S",
                "%H:%M:%S.%f",
                "%H:%M",
                "%I:%M %p",
                "%I:%M:%S %p",
                "%I:%M:%S.%f %p",
            ):
                try:
                    datetime.strptime(clean_value, fmt)
                    return True
                except ValueError:
                    continue
        return False
    except Exception:
        return False


# =========================================================================
# Value -> type inference
# =========================================================================


def _collect_types_from_iterable(
    it: Iterable,
    *,
    media_detector: Any = None,
) -> List[str]:
    types = [infer_type_from_value(v, media_detector=media_detector) for v in it]
    return _unique_normalized(types)


def _unique_normalized(types: List[str]) -> List[str]:
    normalized = [normalize_type_string(t) for t in types]

    def _key(t: str) -> Tuple[int, str]:
        _, inner = parse_nested_type(t)
        if inner is None:
            if t in SPECIAL_FIELD_TYPES:
                bucket = 1
            elif t in SUPPORTED_BASE_TYPES:
                bucket = 0
            else:
                bucket = 2
        else:
            bucket = 3
        return (bucket, t)

    uniq: List[str] = []
    seen: Set[str] = set()
    for t in normalized:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return sorted(uniq, key=_key)


def _render_hetero_container(name: str, inner_types: List[str]) -> str:
    if not inner_types:
        return f"{name}[Any]"
    return f"{name}[{', '.join(inner_types)}]"


def _coalesce_for_dict_slot(inner_types: List[str]) -> str:
    if not inner_types:
        return "Any"
    uniq = _unique_normalized(inner_types)
    if len(uniq) == 1:
        return uniq[0]
    non_none = [t for t in uniq if t != "NoneType"]
    if len(non_none) == 1 and len(uniq) == 2 and "NoneType" in uniq:
        return non_none[0]
    return "Any"


def infer_type_from_value(
    value: Any,
    *,
    media_detector: Optional[Callable[[str], Optional[str]]] = None,
) -> str:
    """Infer a normalized type string from an arbitrary Python value.

    Mirrors Orchestra's ``infer_type_from_value`` exactly so that Unity's
    pre-scan produces the same type strings that Orchestra would infer.
    """
    if value is None:
        return "NoneType"

    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"

    if isinstance(value, datetime):
        return "datetime"
    if isinstance(value, date) and not isinstance(value, datetime):
        return "date"
    if isinstance(value, _time):
        return "time"
    if isinstance(value, timedelta):
        return "timedelta"

    if isinstance(value, str):
        if not value:
            return "str"
        if _is_time_string(value):
            return "time"
        if _is_date_string(value):
            return "date"
        if _is_timedelta_string(value):
            return "timedelta"
        try:
            datetime.fromisoformat(value)
            return "datetime"
        except Exception:
            pass
        if media_detector is not None:
            try:
                media = media_detector(value)
                if media:
                    return media
            except Exception:
                pass
        return "str"

    if isinstance(value, list):
        inner_types = _collect_types_from_iterable(value, media_detector=media_detector)
        return _render_hetero_container("List", inner_types)

    if isinstance(value, set):
        inner_types = _collect_types_from_iterable(value, media_detector=media_detector)
        return _render_hetero_container("Set", inner_types)

    if isinstance(value, tuple):
        elems = [infer_type_from_value(v, media_detector=media_detector) for v in value]
        elems_norm = [_unique_normalized([t])[0] for t in elems]
        if len(elems_norm) == 0:
            return "Tuple[Any]"
        if len(set(elems_norm)) == 1 and len(elems_norm) > 1:
            return f"Tuple[{elems_norm[0]}, ...]"
        return "Tuple[" + ", ".join(elems_norm) + "]"

    if isinstance(value, dict):
        key_types = _collect_types_from_iterable(
            value.keys(),
            media_detector=media_detector,
        )
        val_types = _collect_types_from_iterable(
            value.values(),
            media_detector=media_detector,
        )
        key_t = _coalesce_for_dict_slot(key_types)
        val_t = _coalesce_for_dict_slot(val_types)
        return f"Dict[{key_t}, {val_t}]"

    return "str"
