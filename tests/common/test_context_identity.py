"""Identity verification for provisioned contexts.

A row write that reaches the backend before provisioning auto-creates the
target context bare — no unique keys, no auto-counting — and the backend
cannot retrofit that configuration later. Rows inserted into a bare context
receive no identity column, which is silent corruption. These tests pin the
backend shape and prove `_create_context_with_retry` refuses to treat such a
context as provisioned.
"""

from __future__ import annotations

import pytest
import unisdk

from tests.helpers import _handle_project
from unify.common.context_store import (
    ContextIdentityError,
    _create_context_with_retry,
)
from unify.common.log_utils import log as unity_log


@_handle_project
def test_bare_context_write_assigns_no_identity():
    base = unisdk.get_active_context()["write"]
    name = f"{base}/BareTable"

    # The write itself creates the context, bare.
    unisdk.log(context=name, new=True, payload="row")

    live = unisdk.get_context(name)
    assert live.get("unique_keys") in (None, [])
    assert live.get("auto_counting") in (None, {})

    rows = unisdk.get_logs(context=name)
    assert rows and "row_id" not in rows[0].entries

    with pytest.raises(ContextIdentityError):
        _create_context_with_retry(
            name,
            unique_keys={"row_id": "int"},
            auto_counting={"row_id": None},
        )


@_handle_project
def test_configured_context_passes_verification_and_assigns_ids():
    base = unisdk.get_active_context()["write"]
    name = f"{base}/ConfiguredTable"

    _create_context_with_retry(
        name,
        unique_keys={"row_id": "int"},
        auto_counting={"row_id": None},
    )
    # Re-ensuring an existing, correctly configured context stays idempotent.
    _create_context_with_retry(
        name,
        unique_keys={"row_id": "int"},
        auto_counting={"row_id": None},
    )

    log = unity_log(context=name, new=True, payload="row")
    assert log.entries["row_id"] == 0
