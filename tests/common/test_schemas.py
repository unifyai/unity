"""
pytest tests for the helper utilities:

* model_to_fields                – Pydantic model → unify field schema (unit tests)
* annotation_to_schema           – all supported annotation kinds
* method_to_schema               – schema structure & enum handling
"""

from __future__ import annotations

import json
from datetime import date, datetime, time, UTC
from enum import Enum

import pytest
import unify
from pydantic import BaseModel, Field

from tests.helpers import _handle_project
from unity.common.context_store import TableStore
from unity.common.model_to_fields import model_to_fields
from unity.transcript_manager.types.message import Message

import unity.common.llm_helpers as llmh

# --------------------------------------------------------------------------- #
#  UNIT TESTS: model_to_fields                                                #
#  These run without Orchestra and catch schema structure issues immediately  #
# --------------------------------------------------------------------------- #


class _NestedChild(BaseModel):
    child_id: int
    label: str


class _NestedParent(BaseModel):
    parent_id: int
    child: _NestedChild
    children: list[_NestedChild]
    optional_child: _NestedChild | None = None


class _SimpleModel(BaseModel):
    name: str
    age: int
    score: float
    active: bool
    created_at: datetime
    birth_date: date
    check_time: time
    tags: list[str]
    metadata: dict[str, str]
    maybe_count: int | None = None


class _OverrideModel(BaseModel):
    normal_field: str
    image_field: str = Field(json_schema_extra={"unify_type": "image"})


class _LongDescriptionModel(BaseModel):
    short_desc: str = Field(description="Short description")
    long_desc: str = Field(
        description="A" * 300,  # Exceeds Orchestra's 256 char limit
    )


def test_model_to_fields_no_ref_pointers():
    """$ref pointers must be resolved inline - no dangling references."""
    fields = model_to_fields(Message)
    serialized = json.dumps(fields)
    assert "$ref" not in serialized, "Found unresolved $ref in model_to_fields output"


def test_model_to_fields_nested_keys_present():
    """Nested Pydantic models must have their property names in the serialized schema."""
    fields = model_to_fields(Message)
    images_type = fields["images"]["type"]
    # images contains AnnotatedImageRef with raw_image_ref.image_id and annotation
    assert "raw_image_ref" in images_type
    assert "annotation" in images_type
    assert "image_id" in images_type


def test_model_to_fields_simple_type_mapping():
    """Simple types map to ColumnType strings, not JSON schemas."""
    fields = model_to_fields(_SimpleModel)

    # String types
    assert fields["name"]["type"] == "str"

    # Numeric types
    assert fields["age"]["type"] == "int"
    assert fields["score"]["type"] == "float"

    # Boolean
    assert fields["active"]["type"] == "bool"

    # Date/time types
    assert fields["created_at"]["type"] == "datetime"
    assert fields["birth_date"]["type"] == "date"
    assert fields["check_time"]["type"] == "time"

    # Containers (simple - not nested objects)
    assert fields["tags"]["type"] == "list"
    assert fields["metadata"]["type"] == "dict"

    # Optional unwraps to underlying type
    assert fields["maybe_count"]["type"] == "int"


def test_model_to_fields_nested_serialized_as_json():
    """Nested object schemas are serialized as JSON strings."""
    fields = model_to_fields(_NestedParent)

    # Single nested object
    child_type = fields["child"]["type"]
    assert child_type.startswith("{"), "Nested object should be JSON string"
    child_schema = json.loads(child_type)
    assert "child_id" in str(child_schema)
    assert "label" in str(child_schema)

    # List of nested objects
    children_type = fields["children"]["type"]
    assert children_type.startswith("{"), "List of objects should be JSON string"
    children_schema = json.loads(children_type)
    assert children_schema.get("type") == "array"
    assert "child_id" in str(children_schema)

    # Optional nested object
    optional_type = fields["optional_child"]["type"]
    assert optional_type.startswith("{"), "Optional nested should be JSON string"
    assert "child_id" in optional_type


def test_model_to_fields_unify_type_override():
    """json_schema_extra={'unify_type': ...} takes precedence."""
    fields = model_to_fields(_OverrideModel)

    assert fields["normal_field"]["type"] == "str"
    assert fields["image_field"]["type"] == "image"


def test_model_to_fields_uses_field_description():
    """Description comes from Field(..., description=...), not JSON Schema."""
    fields = model_to_fields(_LongDescriptionModel)

    # Short description is preserved
    assert fields["short_desc"].get("description") == "Short description"

    # Long description is also preserved (truncation happens in Orchestra, not here)
    # The key point is we use Field.description, not the JSON Schema description
    assert fields["long_desc"].get("description") == "A" * 300


def test_model_to_fields_mutable_flag():
    """All fields should have mutable=True."""
    fields = model_to_fields(_SimpleModel)
    for name, field_spec in fields.items():
        assert field_spec.get("mutable") is True, f"Field {name} missing mutable=True"


def test_model_to_fields_message_schema_complete():
    """Message model produces expected field structure."""
    fields = model_to_fields(Message)

    # All expected fields present
    expected_fields = {
        "message_id",
        "medium",
        "sender_id",
        "receiver_ids",
        "timestamp",
        "content",
        "exchange_id",
        "images",
        "metadata",
    }
    assert expected_fields <= set(fields.keys())

    # Simple fields have ColumnType strings
    assert fields["message_id"]["type"] == "int"
    assert fields["medium"]["type"] == "str"
    assert fields["timestamp"]["type"] == "datetime"
    assert fields["content"]["type"] == "str"

    # Nested field (images) is serialized JSON
    assert fields["images"]["type"].startswith("{")


# --------------------------------------------------------------------------- #
#  TEST DATA TYPES FOR SCHEMA TESTS                                           #
# --------------------------------------------------------------------------- #
class ColumnType(str, Enum):
    str = "str"
    int = "int"


class Person(BaseModel):
    name: str
    age: int


# Helper function defined at module scope to stabilise type-hint resolution
def _tool_with_optional_mapping(
    references: dict[str, str] | None = None,
    k: int = 10,
) -> None:  # pragma: no cover - schema only
    return None


# --------------------------------------------------------------------------- #
#  annotation_to_schema                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "t, checker",
    [
        (str, lambda s: s == {"type": "string"}),
        (int, lambda s: s == {"type": "integer"}),
        (
            ColumnType,
            lambda s: s["type"] == "string" and set(s["enum"]) == {"str", "int"},
        ),
        (
            Person,
            lambda s: s["type"] == "object" and {"name", "age"} <= set(s["properties"]),
        ),
        (
            dict[str, int],
            lambda s: s["type"] == "object"
            and s["additionalProperties"]["type"] == "integer",
        ),
        (
            list[Person],
            lambda s: s["type"] == "array" and s["items"]["type"] == "object",
        ),
    ],
)
def test_annotation_schema_conversion(t, checker):
    """Every major annotation flavour is converted correctly."""
    assert checker(llmh.annotation_to_schema(t))


# --------------------------------------------------------------------------- #
#  method_to_schema – enum round-trip                                         #
# --------------------------------------------------------------------------- #
def _demo_func(a: str, col: ColumnType):
    """Docstring for unit test."""
    return None


def test_schema_includes_enum():
    schema = llmh.method_to_schema(_demo_func)
    params = schema["function"]["parameters"]["properties"]
    assert params["a"]["type"] == "string"
    # Enum must appear with *exact* allowed literals
    assert params["col"]["enum"] == ["str", "int"]


# --------------------------------------------------------------------------- #
#  PRIVATE OPTIONAL ARGUMENTS ARE NOT EXPOSED                                 #
# --------------------------------------------------------------------------- #
def test_schema_hides_private_optionals() -> None:
    """
    *Optional* parameters whose names begin with an underscore (``_``)
    must **not** appear in the schema that is presented to the LLM.
    Required private parameters, however, *must* stay visible or the
    tool would become impossible to call – and their docs should stay too.
    """

    # ── 1. optional private argument should be hidden ─────────────────────
    def sample_tool(a: int, b: int = 0, _secret: str = "x") -> int:
        """
        Sample calculator.

        Args:
            a: first addend.
            b: second addend, defaults to 0.
            _secret: **internal** flag, never shown to the LLM.
        """
        return a + b

    schema = llmh.method_to_schema(sample_tool)
    props = schema["function"]["parameters"]["properties"]
    required = schema["function"]["parameters"]["required"]
    desc = schema["function"]["description"]

    # public arguments are present …
    assert "a" in props and "b" in props
    # … while the optional private one is not
    assert "_secret" not in props
    # and its doc-line has been pruned
    assert "_secret" not in desc

    # required list unchanged
    assert "a" in required and "b" not in required

    # ── 2. required private argument should be kept ───────────────────────
    def tool_with_required_private(x: int, _hidden: str) -> str:
        """
        Echo tool.

        Parameters
        ----------
        x : int
            Multiplier.
        _hidden : str
            Mandatory private value (must stay visible).
        """
        return _hidden * x

    schema2 = llmh.method_to_schema(tool_with_required_private)
    props2 = schema2["function"]["parameters"]["properties"]
    required2 = schema2["function"]["parameters"]["required"]
    desc2 = schema2["function"]["description"]

    # the *required* private parameter is still exposed …
    assert "_hidden" in props2 and "_hidden" in required2
    # … and its doc-line is still present
    assert "_hidden" in desc2


# --------------------------------------------------------------------------- #
#  `_parent_chat_context` MUST NEVER BE EXPOSED                                #
# --------------------------------------------------------------------------- #
def test_schema_hides_context_param() -> None:
    """
    The special ``_parent_chat_context`` argument is injected automatically by
    the tool-loop.  It must be hidden from both the schema **and** the
    docstring that is sent to the LLM.
    """

    def tool_with_ctx(a: int, _parent_chat_context: list[dict]):
        """
        Dummy tool.

        Parameters
        ----------
        a : int
            Some value.
        _parent_chat_context : list[dict]
            Internal plumbing, never surfaced.
        """
        return a

    def tool_with_ctx_optional(
        a: int,
        _parent_chat_context: list[dict] | None = None,
    ):
        """
        Dummy tool (optional ctx).

        Args:
            a: Some value.
            _parent_chat_context: Internal plumbing, never surfaced.
        """
        return a

    for fn in (tool_with_ctx, tool_with_ctx_optional):
        schema = llmh.method_to_schema(fn)
        props = schema["function"]["parameters"]["properties"]
        required = schema["function"]["parameters"]["required"]
        desc = schema["function"]["description"]

        assert "_parent_chat_context" not in props
        assert "_parent_chat_context" not in required
        # docstring has been scrubbed
        assert "_parent_chat_context" not in desc


# --------------------------------------------------------------------------- #
#  OPTIONAL[Dict[str, str]] COLLAPSES TO OBJECT (NO STRING ALTERNATIVE)       #
# --------------------------------------------------------------------------- #
def test_optional_dict_schema_simplification() -> None:
    """
    Optional[Dict[str, str]] should collapse to a plain object schema.
    Prior to the fix, NoneType was treated as "string", producing
    anyOf [object, string]. This test ensures only the object form remains.
    """

    schema = llmh.method_to_schema(_tool_with_optional_mapping)
    params = schema["function"]["parameters"]["properties"]
    refs_schema = params["references"]

    # Must be a plain object with string values
    assert "anyOf" not in refs_schema
    assert refs_schema["type"] == "object"
    assert refs_schema["additionalProperties"]["type"] == "string"


# --------------------------------------------------------------------------- #
#  BUILTIN dict HANDLING (images: dict | None)                                #
# --------------------------------------------------------------------------- #
def _tool_with_optional_builtin_mapping(
    images: dict | None = None,
) -> None:  # pragma: no cover - schema only
    return None


def test_optional_builtin_dict_schema() -> None:
    """
    Optional[builtin dict] should surface as a plain object to the LLM.
    Prior to the fix, builtin dict could degrade to "string" in unions,
    leading the model to send serialized strings for images.
    """

    schema = llmh.method_to_schema(_tool_with_optional_builtin_mapping)
    params = schema["function"]["parameters"]["properties"]
    images_schema = params["images"]

    assert "anyOf" not in images_schema
    assert images_schema["type"] == "object"
    # Unknown value types → allow arbitrary properties
    assert images_schema.get("additionalProperties") is True


def test_dict_annotation_schema() -> None:
    s = llmh.annotation_to_schema(dict)
    assert s["type"] == "object"
    assert s.get("additionalProperties") is True


# --------------------------------------------------------------------------- #
#  method_to_schema – docstring MRO fallback                                  #
# --------------------------------------------------------------------------- #
def test_schema_inherits_base_docstring() -> None:
    class _Base:
        def action(self, x: int) -> None:
            """Base doc: perform action."""
            return None

    class _Child(_Base):
        def action(self, x: int) -> None:
            # no docstring → should inherit from base via MRO
            return None

    schema = llmh.method_to_schema(_Child().action)
    desc = schema["function"]["description"]
    assert "Base doc: perform action." in desc


def test_schema_prefers_child_docstring() -> None:
    class _Base:
        def go(self) -> None:
            """Base doc: go."""
            return None

    class _Child(_Base):
        def go(self) -> None:
            """Child doc: go fast."""
            return None

    schema = llmh.method_to_schema(_Child().go)
    desc = schema["function"]["description"]
    assert "Child doc: go fast." in desc
    assert "Base doc" not in desc


# --------------------------------------------------------------------------- #
#  INHERITED DOCSTRING + PRUNED WRAPPER SIGNATURE                              #
#  When a thin wrapper inherits a docstring (via __doc__) from a base class    #
#  that documents _-prefixed internal params, those params must be stripped     #
#  from the LLM-facing description even though they don't appear in the        #
#  wrapper's own signature.                                                    #
# --------------------------------------------------------------------------- #
def test_schema_strips_hidden_params_from_inherited_doc() -> None:
    """
    Regression: CodeActActor wraps FunctionManager methods with thin
    closures that have pruned signatures (only public params) but inherit
    the full base-class docstring via ``__doc__`` assignment.

    The ``_``-prefixed internal params documented in the base docstring
    must NOT leak through to the LLM-facing tool description.
    """

    class _BaseFM:
        def search(
            self,
            *,
            query: str,
            n: int = 5,
            _return_callable: bool = False,
            _namespace: dict | None = None,
        ) -> list:
            """
            Search for items by similarity.

            Parameters
            ----------
            query : str
                Natural-language search text.
            n : int, default ``5``
                Max results to return.
            _return_callable : bool, default ``False``
                When ``True``, return callables instead of metadata dicts.
            _namespace : dict | None, default ``None``
                Target namespace dict for callable injection when
                ``_return_callable=True``.

            Returns
            -------
            list
                Up to ``n`` results.
            """
            return []

    # Thin wrapper that includes _-prefixed params in the signature so the
    # stripping machinery can see them, even though the body ignores them.
    async def FunctionManager_search(
        query: str,
        n: int = 5,
        _return_callable: bool = False,
        _namespace: dict | None = None,
    ) -> list:
        return []

    FunctionManager_search.__doc__ = _BaseFM.search.__doc__

    schema = llmh.method_to_schema(FunctionManager_search)
    props = schema["function"]["parameters"]["properties"]
    desc = schema["function"]["description"]

    # Public params visible in the schema
    assert "query" in props
    assert "n" in props

    # Internal params must NOT appear in the schema
    assert "_return_callable" not in props
    assert "_namespace" not in props

    # Internal param documentation must NOT appear in the description
    assert "_return_callable" not in desc
    assert "_namespace" not in desc
    assert "callable injection" not in desc


def test_schema_strips_hidden_param_references_from_returns() -> None:
    """
    The Returns section may contain conditional branches like
    ``When ``_return_callable=False``: ...`` that reference hidden params.
    These branches must be stripped from the LLM-facing description since
    the LLM cannot control the param they depend on.
    """

    def search(
        query: str,
        n: int = 5,
        _return_callable: bool = False,
        _also_return_metadata: bool = False,
    ) -> list:
        """
        Search for items.

        Parameters
        ----------
        query : str
            Search text.
        n : int, default ``5``
            Max results.
        _return_callable : bool, default ``False``
            When ``True``, return callables instead of metadata.
        _also_return_metadata : bool, default ``False``
            When ``True``, return both callables and metadata.

        Returns
        -------
        list[dict] | list[Callable] | dict
            - When ``_return_callable=False``: list of metadata dicts.
            - When ``_return_callable=True``: list of callables.
            - When ``_also_return_metadata=True``: a dict with both.
        """
        return []

    schema = llmh.method_to_schema(search)
    desc = schema["function"]["description"]

    # The Returns section should not reference hidden params
    assert "_return_callable" not in desc
    assert "_also_return_metadata" not in desc


def test_schema_strips_hidden_param_references_from_raises() -> None:
    """
    The Raises section may document errors for hidden-param validation
    (e.g. ``If ``_return_callable=True`` but ``_namespace`` is ``None````).
    These entries must be stripped from the LLM-facing description since
    the LLM cannot trigger these errors.
    """

    def search(
        query: str,
        _return_callable: bool = False,
        _namespace: dict | None = None,
    ) -> list:
        """
        Search for items.

        Parameters
        ----------
        query : str
            Search text.
        _return_callable : bool, default ``False``
            When ``True``, return callables.
        _namespace : dict | None, default ``None``
            Target namespace for injection.

        Raises
        ------
        ValueError
            If ``_return_callable=True`` but ``_namespace`` is ``None``.
        ValueError
            If ``_return_callable`` is set without proper context.
        """
        return []

    schema = llmh.method_to_schema(search)
    desc = schema["function"]["description"]

    # The Raises section should not reference hidden params
    assert "_return_callable" not in desc
    assert "_namespace" not in desc


def test_schema_plain_function() -> None:
    def _plain(a: int) -> None:
        """Plain function doc."""
        return None

    schema = llmh.method_to_schema(_plain)
    desc = schema["function"]["description"]
    assert desc == "Plain function doc."


# --------------------------------------------------------------------------- #
#  NESTED Pydantic field typing for Message.images (Transcripts shape)        #
# --------------------------------------------------------------------------- #


@_handle_project
def test_nested_image_schema_enforcement() -> None:
    """Provision a context with the Message schema and assert:
    - the `images` field is created with a nested JSON Schema (contains expected keys)
    - logging with a valid `images` payload succeeds
    - logging with an invalid `images` payload is rejected by the backend
    """

    # Build a per-test context under the active write context (mirrors other tests)
    try:
        ctxs = unify.get_active_context()
        base_ctx = ctxs.get("write") if isinstance(ctxs, dict) else None
    except Exception:
        base_ctx = None
    ctx = f"{base_ctx}/SchemaNestedImages" if base_ctx else "SchemaNestedImages"

    # Provision the table using model_to_fields(Message) so `images` carries the nested schema
    store = TableStore(
        ctx,
        unique_keys={"message_id": "int"},
        auto_counting={"message_id": None, "exchange_id": None},
        description="Schema test for nested images field",
        fields=model_to_fields(Message),
    )
    store.ensure_context()

    # 1) The created field should include a nested schema – assert key substrings
    fields = unify.get_fields(context=ctx)
    assert "images" in fields
    dtype = str(fields["images"].get("data_type"))
    # Expect array/list with object items including raw_image_ref + annotation
    assert "raw_image_ref" in dtype and "annotation" in dtype and "image_id" in dtype

    # Common required fields for the Message row
    common = {
        "medium": "email",
        "sender_id": 1,
        "receiver_ids": [2],
        # Pass ISO-8601 string – unify.log's JSON body must be serializable
        "timestamp": datetime.now(UTC).isoformat(),
        "content": "hello",
    }

    # 2) Valid nested payload – should succeed
    valid_payload = {
        **common,
        "images": [
            {"raw_image_ref": {"image_id": 101}, "annotation": "blue square"},
        ],
    }
    _ = unify.log(context=ctx, **valid_payload, new=True, mutable=True)

    # 3) Invalid nested payload – wrong key name for image id → must be rejected
    invalid_payload_bad_key = {
        **common,
        "images": [
            {"raw_image_ref": {"image_idx": 999}, "annotation": "oops"},  # wrong key
        ],
    }
    with pytest.raises(Exception):
        unify.log(context=ctx, **invalid_payload_bad_key, new=True, mutable=True)

    # 4) Invalid nested payload – wrong type for annotation → must be rejected
    invalid_payload_bad_type = {
        **common,
        "images": [
            {"raw_image_ref": {"image_id": 202}, "annotation": 123},  # not a string
        ],
    }
    with pytest.raises(Exception):
        unify.log(context=ctx, **invalid_payload_bad_type, new=True, mutable=True)


# --------------------------------------------------------------------------- #
#  GENERAL: model_to_fields supports arbitrary nested Pydantic models          #
#           and Unify enforces the resulting JSON Schemas                      #
# --------------------------------------------------------------------------- #


class _Address(BaseModel):
    street: str
    zip_code: int


class _Pet(BaseModel):
    name: str
    kind: str
    age: int | None = None


class _Payload(BaseModel):
    owner: str
    address: _Address
    pets: list[_Pet]
    primary_pet: _Pet | None = None


class _Record(BaseModel):
    record_id: int
    payload: _Payload


@_handle_project
def test_nested_pydantic_schema_enforcement() -> None:
    """Provision a context from an arbitrary nested Pydantic model and assert:
    - the serialized data_type for the nested field includes child property names;
    - logging succeeds for valid nested payloads;
    - logging fails for invalid shapes and wrong types.
    """

    # Create a dedicated context for this test
    try:
        ctxs = unify.get_active_context()
        base_ctx = ctxs.get("write") if isinstance(ctxs, dict) else None
    except Exception:
        base_ctx = None
    ctx = f"{base_ctx}/SchemaNestedPydantic" if base_ctx else "SchemaNestedPydantic"

    # Provision using the generalized model (not tied to Transcripts)
    store = TableStore(
        ctx,
        unique_keys={"record_id": "int"},
        auto_counting={"record_id": None},
        description="Schema test for arbitrary nested Pydantic models",
        fields=model_to_fields(_Record),
    )
    store.ensure_context()

    # Field typing should contain nested property names for the payload schema
    fields = unify.get_fields(context=ctx)
    assert "payload" in fields
    dtype = str(fields["payload"].get("data_type"))
    # Assert several nested keys appear in the serialized schema
    for needle in (
        "owner",
        "address",
        "pets",
        "primary_pet",
        "zip_code",
        "name",
        "kind",
    ):
        assert needle in dtype

    # Valid nested payload
    valid = {
        "record_id": 1,
        "payload": {
            "owner": "Alice",
            "address": {"street": "Main St", "zip_code": 90210},
            "pets": [
                {"name": "Rex", "kind": "dog", "age": 5},
                {"name": "Mittens", "kind": "cat"},
            ],
            "primary_pet": {"name": "Rex", "kind": "dog", "age": 5},
        },
    }
    _ = unify.log(context=ctx, **valid, new=True, mutable=True)

    # Invalid 1: wrong nested key (zip instead of zip_code) → reject
    invalid_bad_key = {
        "record_id": 2,
        "payload": {
            "owner": "Bob",
            "address": {"street": "Second", "zip": 10001},  # wrong key
            "pets": [{"name": "Fido", "kind": "dog"}],
        },
    }
    with pytest.raises(Exception):
        unify.log(context=ctx, **invalid_bad_key, new=True, mutable=True)

    # Invalid 2: wrong type in list (pets elements must be objects) → reject
    invalid_bad_list = {
        "record_id": 3,
        "payload": {
            "owner": "Charlie",
            "address": {"street": "Third", "zip_code": 11111},
            "pets": ["not-an-object"],  # wrong type
        },
    }
    with pytest.raises(Exception):
        unify.log(context=ctx, **invalid_bad_list, new=True, mutable=True)


# --------------------------------------------------------------------------- #
#  make_request_clarification_tool — docstring & schema                       #
# --------------------------------------------------------------------------- #


def test_request_clarification_tool_has_docstring_and_schema_description():
    """The programmatically generated request_clarification tool must expose a
    non-empty docstring so that method_to_schema produces a non-empty
    description for the LLM."""
    import asyncio

    fn = llmh.make_request_clarification_tool(asyncio.Queue(), asyncio.Queue())

    # The inner function should have a docstring.
    assert (
        fn.__doc__ and fn.__doc__.strip()
    ), "make_request_clarification_tool returned a function without a docstring"

    # The schema sent to the LLM should have a non-empty description.
    schema = llmh.method_to_schema(fn, "request_clarification")
    desc = schema["function"]["description"]
    assert (
        isinstance(desc, str) and desc.strip()
    ), f"method_to_schema produced an empty description: {desc!r}"
