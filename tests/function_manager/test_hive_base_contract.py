"""FunctionManager Hive-base contract.

The four Functions sub-tables — ``Functions/Compositional``,
``Functions/Primitives``, ``Functions/Meta``, ``Functions/VirtualEnvs`` —
are all in ``_HIVE_SCOPED_TABLES``. Each shared-row model declares an
``authoring_assistant_id`` column so reviewers can attribute shared rows
back to their authoring body. The ``VirtualEnvs`` catalog row is
Hive-shared even though the *on-disk* venv materialization stays per-body:
the materialization path derives from ``unify.get_active_context()``,
which remains body-scoped by design.
"""

from __future__ import annotations

import pytest

from unity.common.context_registry import ContextRegistry
from unity.function_manager.types.function import Function
from unity.function_manager.types.meta import FunctionsMeta
from unity.function_manager.types.venv import VirtualEnv
from unity.session_details import SESSION_DETAILS

pytestmark = pytest.mark.usefixtures("pinned_hive_body")

_HIVE_SHARED_SUBTABLES = (
    "Functions/Compositional",
    "Functions/Primitives",
    "Functions/Meta",
    "Functions/VirtualEnvs",
)


@pytest.mark.parametrize("table", _HIVE_SHARED_SUBTABLES)
def test_hive_member_resolves_function_subtable_to_hive_root(table):
    SESSION_DETAILS.hive_id = 42
    assert ContextRegistry.base_for(table) == "Hives/42"


@pytest.mark.parametrize("table", _HIVE_SHARED_SUBTABLES)
def test_solo_body_resolves_function_subtable_to_per_body_root(table, pinned_hive_body):
    assert SESSION_DETAILS.hive_id is None
    assert ContextRegistry.base_for(table) == pinned_hive_body


@pytest.mark.parametrize(
    "model",
    [Function, VirtualEnv, FunctionsMeta],
)
def test_shared_row_models_declare_authoring_assistant_id(model):
    field = model.model_fields.get("authoring_assistant_id")
    assert field is not None
    assert field.default is None
