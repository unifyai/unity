"""Tests for post-ingest derived columns and field description passthrough.

Covers:
- ``ExplicitDerivedColumn`` / ``AutoDerivedColumn`` / ``PostIngestConfig``: model validation
- ``_derive_target_name``: separator-aware target name construction
- ``_run_post_ingest_rules``: config-driven rule execution with mocks
- ``run_ingest`` post_ingest integration
- Field description passthrough: widened ``fields`` type flows through to ``unify.create_fields``
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from unity.data_manager.ops.ingest_ops import (
    _derive_target_name,
    _run_post_ingest_rules,
)
from unity.data_manager.types.ingest import (
    AutoDerivedColumn,
    ExplicitDerivedColumn,
    PostIngestConfig,
)

# =============================================================================
# Model validation
# =============================================================================


class TestDerivedColumnModels:

    def test_explicit_all_fields_required(self):
        rule = ExplicitDerivedColumn(
            source_field="Trip travel time",
            target_name="Trip travel time duration seconds",
            equation="duration_seconds({lg:{field}})",
        )
        assert rule.kind == "explicit"
        assert rule.source_field == "Trip travel time"
        assert rule.target_name == "Trip travel time duration seconds"

    def test_explicit_missing_field_raises(self):
        with pytest.raises(Exception):
            ExplicitDerivedColumn(
                source_field="foo",
                equation="f({lg:{field}})",
            )

    def test_auto_all_fields_required(self):
        rule = AutoDerivedColumn(
            source_type="datetime",
            target_suffix="Date",
            equation="date({lg:{field}})",
        )
        assert rule.kind == "auto"
        assert rule.source_type == "datetime"
        assert rule.target_suffix == "Date"

    def test_auto_missing_field_raises(self):
        with pytest.raises(Exception):
            AutoDerivedColumn(
                source_type="datetime",
                equation="date({lg:{field}})",
            )

    def test_post_ingest_config_empty(self):
        cfg = PostIngestConfig()
        assert cfg.derived_columns == []

    def test_post_ingest_config_discriminated_from_dict(self):
        cfg = PostIngestConfig(
            **{
                "derived_columns": [
                    {
                        "kind": "auto",
                        "source_type": "datetime",
                        "target_suffix": "Date",
                        "equation": "date({lg:{field}})",
                    },
                    {
                        "kind": "explicit",
                        "source_field": "x",
                        "target_name": "y",
                        "equation": "f({lg:{field}})",
                    },
                ],
            },
        )
        assert len(cfg.derived_columns) == 2
        assert isinstance(cfg.derived_columns[0], AutoDerivedColumn)
        assert isinstance(cfg.derived_columns[1], ExplicitDerivedColumn)

    def test_invalid_kind_raises(self):
        with pytest.raises(Exception):
            PostIngestConfig(
                **{
                    "derived_columns": [
                        {
                            "kind": "unknown",
                            "source_type": "datetime",
                            "target_suffix": "Date",
                            "equation": "date({lg:{field}})",
                        },
                    ],
                },
            )


# =============================================================================
# _derive_target_name — separator-aware naming
# =============================================================================


class TestDeriveTargetName:

    def test_whitespace_separated(self):
        assert (
            _derive_target_name("Subtrip travel time", "duration seconds")
            == "Subtrip travel time duration seconds"
        )

    def test_whitespace_single_word_suffix(self):
        assert (
            _derive_target_name("Subtrip travel time", "Date")
            == "Subtrip travel time Date"
        )

    def test_underscore_separated(self):
        assert (
            _derive_target_name("subtrip_travel_time", "duration seconds")
            == "subtrip_travel_time_duration_seconds"
        )

    def test_underscore_single_word_suffix(self):
        assert _derive_target_name("visit_date", "Date") == "visit_date_date"

    def test_pascal_case(self):
        assert _derive_target_name("VisitDate", "Date") == "VisitDate_Date"

    def test_pascal_case_multi_word_suffix(self):
        assert (
            _derive_target_name("SubtripTravelTime", "duration seconds")
            == "SubtripTravelTime_DurationSeconds"
        )

    def test_single_word_uppercase_start(self):
        assert _derive_target_name("Departure", "Date") == "Departure_Date"

    def test_camel_case_lowercase_start(self):
        assert _derive_target_name("visitDate", "Date") == "visitDate_date"

    def test_lowercase_single_word(self):
        assert _derive_target_name("departure", "Date") == "departure_date"


# =============================================================================
# _run_post_ingest_rules — mocked integration
# =============================================================================


class TestRunPostIngestRules:

    @patch("unity.data_manager.ops.ingest_ops._ensure_derived_column")
    @patch("unity.data_manager.ops.ingest_ops._unify")
    def test_explicit_source_field(self, mock_unify, mock_ensure):
        config = PostIngestConfig(
            derived_columns=[
                ExplicitDerivedColumn(
                    source_field="Trip travel time",
                    target_name="Trip travel time duration seconds",
                    equation="duration_seconds({lg:{field}})",
                ),
            ],
        )
        result = _run_post_ingest_rules("test/ctx", config)

        assert result == ["Trip travel time duration seconds"]
        mock_ensure.assert_called_once_with(
            "test/ctx",
            key="Trip travel time duration seconds",
            equation="duration_seconds({lg:Trip travel time})",
            referenced_logs_context="test/ctx",
        )
        mock_unify.get_fields.assert_not_called()

    @patch("unity.data_manager.ops.ingest_ops._ensure_derived_column")
    @patch("unity.data_manager.ops.ingest_ops._unify")
    def test_auto_discovery_by_source_type(self, mock_unify, mock_ensure):
        mock_unify.get_fields.return_value = {
            "VisitDate": {"data_type": "datetime"},
            "Departure": {"data_type": "datetime"},
            "OperativeName": {"data_type": "str"},
            "_private": {"data_type": "datetime"},
        }

        config = PostIngestConfig(
            derived_columns=[
                AutoDerivedColumn(
                    source_type="datetime",
                    target_suffix="Date",
                    equation="date({lg:{field}})",
                ),
            ],
        )
        result = _run_post_ingest_rules("test/ctx", config)

        assert set(result) == {"VisitDate_Date", "Departure_Date"}
        assert mock_ensure.call_count == 2
        mock_ensure.assert_any_call(
            "test/ctx",
            key="VisitDate_Date",
            equation="date({lg:VisitDate})",
            referenced_logs_context="test/ctx",
        )
        mock_ensure.assert_any_call(
            "test/ctx",
            key="Departure_Date",
            equation="date({lg:Departure})",
            referenced_logs_context="test/ctx",
        )

    @patch("unity.data_manager.ops.ingest_ops._ensure_derived_column")
    @patch("unity.data_manager.ops.ingest_ops._unify")
    def test_mixed_rules(self, mock_unify, mock_ensure):
        mock_unify.get_fields.return_value = {
            "VisitDate": {"data_type": "datetime"},
            "OperativeName": {"data_type": "str"},
        }

        config = PostIngestConfig(
            derived_columns=[
                AutoDerivedColumn(
                    source_type="datetime",
                    target_suffix="Date",
                    equation="date({lg:{field}})",
                ),
                ExplicitDerivedColumn(
                    source_field="Idling time",
                    target_name="Idling time duration seconds",
                    equation="duration_seconds({lg:{field}})",
                ),
            ],
        )
        result = _run_post_ingest_rules("test/ctx", config)

        assert result == ["VisitDate_Date", "Idling time duration seconds"]
        assert mock_ensure.call_count == 2

    @patch("unity.data_manager.ops.ingest_ops._ensure_derived_column")
    def test_empty_config_no_calls(self, mock_ensure):
        config = PostIngestConfig(derived_columns=[])
        result = _run_post_ingest_rules("test/ctx", config)

        assert result == []
        mock_ensure.assert_not_called()


# =============================================================================
# run_ingest post_ingest integration
# =============================================================================


class TestRunIngestPostIngestIntegration:

    @patch("unity.data_manager.ops.ingest_ops._run_post_ingest_rules")
    @patch("unity.data_manager.ops.ingest_ops.PipelineExecutor")
    @patch("unity.data_manager.ops.ingest_ops.create_table_impl")
    def test_post_ingest_called_when_config_provided(
        self,
        mock_create,
        mock_executor_cls,
        mock_run_rules,
    ):
        from unity.data_manager.ops.ingest_ops import run_ingest

        mock_executor = mock_executor_cls.return_value
        mock_executor.execute.return_value = {}
        mock_run_rules.return_value = ["col_a"]

        config = PostIngestConfig(
            derived_columns=[
                AutoDerivedColumn(
                    source_type="datetime",
                    target_suffix="Date",
                    equation="date({lg:{field}})",
                ),
            ],
        )

        result = run_ingest(
            None,
            "test/ctx",
            [{"a": 1}],
            post_ingest=config,
        )

        mock_run_rules.assert_called_once_with("test/ctx", config)
        assert result.derived_columns_created == ["col_a"]

    @patch("unity.data_manager.ops.ingest_ops._run_post_ingest_rules")
    @patch("unity.data_manager.ops.ingest_ops.PipelineExecutor")
    @patch("unity.data_manager.ops.ingest_ops.create_table_impl")
    def test_no_post_ingest_when_none(
        self,
        mock_create,
        mock_executor_cls,
        mock_run_rules,
    ):
        from unity.data_manager.ops.ingest_ops import run_ingest

        mock_executor = mock_executor_cls.return_value
        mock_executor.execute.return_value = {}

        result = run_ingest(
            None,
            "test/ctx",
            [{"a": 1}],
            post_ingest=None,
        )

        mock_run_rules.assert_not_called()
        assert result.derived_columns_created == []

    @patch("unity.data_manager.ops.ingest_ops._run_post_ingest_rules")
    @patch("unity.data_manager.ops.ingest_ops.PipelineExecutor")
    @patch("unity.data_manager.ops.ingest_ops.create_table_impl")
    def test_empty_post_ingest_no_rules_called(
        self,
        mock_create,
        mock_executor_cls,
        mock_run_rules,
    ):
        from unity.data_manager.ops.ingest_ops import run_ingest

        mock_executor = mock_executor_cls.return_value
        mock_executor.execute.return_value = {}

        result = run_ingest(
            None,
            "test/ctx",
            [{"a": 1}],
            post_ingest=PostIngestConfig(derived_columns=[]),
        )

        mock_run_rules.assert_not_called()
        assert result.derived_columns_created == []


# =============================================================================
# Field description passthrough
# =============================================================================


class TestFieldsDescriptionPassthrough:

    @patch("unity.data_manager.ops.table_ops.unify")
    def test_rich_fields_payload_reaches_create_fields(self, mock_unify):
        """Verify that fields with descriptions pass through to unify.create_fields."""
        from unity.data_manager.ops.table_ops import create_table_impl

        rich_fields = {
            "WorksOrderRef": {
                "type": "str",
                "description": "Unique works order reference",
            },
            "VisitDate": {
                "type": "datetime",
                "description": "Timestamp of operative visit",
            },
        }

        create_table_impl(
            "test/ctx",
            description="Test table",
            fields=rich_fields,
        )

        mock_unify.create_fields.assert_called_once_with(
            rich_fields,
            context="test/ctx",
        )

    @patch("unity.data_manager.ops.table_ops.unify")
    def test_simple_fields_still_work(self, mock_unify):
        """Backward compatibility: plain {name: type_str} still works."""
        from unity.data_manager.ops.table_ops import create_table_impl

        simple_fields = {"name": "str", "age": "int"}

        create_table_impl("test/ctx", fields=simple_fields)

        mock_unify.create_fields.assert_called_once_with(
            simple_fields,
            context="test/ctx",
        )

    @patch("unity.data_manager.ops.table_ops.unify")
    def test_none_fields_skips_create_fields(self, mock_unify):
        """When fields is None, create_fields should not be called."""
        from unity.data_manager.ops.table_ops import create_table_impl

        create_table_impl("test/ctx", fields=None)

        mock_unify.create_fields.assert_not_called()


# =============================================================================
# Auto-derived rule type matching (TDD: failing tests first)
# =============================================================================


class TestAutoRuleTypeMatching:
    """Tests for auto-derived rule matching with normalized/Optional types.

    These test that ``_run_post_ingest_rules`` correctly matches
    auto-derived rules when field types use casing variants or Optional
    wrappers.  The naive ``dtype != rule.source_type`` comparison fails
    these; the ``types_match`` fix makes them pass.
    """

    @patch("unity.data_manager.ops.ingest_ops._ensure_derived_column")
    @patch("unity.data_manager.ops.ingest_ops._unify")
    def test_auto_rule_matches_normalized_type(self, mock_unify, mock_ensure):
        """Capital 'DateTime' should match source_type='datetime'."""
        mock_unify.get_fields.return_value = {
            "Departure": {"data_type": "DateTime"},
        }
        config = PostIngestConfig(
            derived_columns=[
                AutoDerivedColumn(
                    source_type="datetime",
                    target_suffix="Date",
                    equation="date({lg:{field}})",
                ),
            ],
        )
        result = _run_post_ingest_rules("test/ctx", config)
        assert "Departure_Date" in result
        mock_ensure.assert_called_once()

    @patch("unity.data_manager.ops.ingest_ops._ensure_derived_column")
    @patch("unity.data_manager.ops.ingest_ops._unify")
    def test_auto_rule_matches_optional_type(self, mock_unify, mock_ensure):
        """Union[datetime, NoneType] should match source_type='datetime'."""
        mock_unify.get_fields.return_value = {
            "Departure": {"data_type": "Union[datetime, NoneType]"},
        }
        config = PostIngestConfig(
            derived_columns=[
                AutoDerivedColumn(
                    source_type="datetime",
                    target_suffix="Date",
                    equation="date({lg:{field}})",
                ),
            ],
        )
        result = _run_post_ingest_rules("test/ctx", config)
        assert "Departure_Date" in result
        mock_ensure.assert_called_once()


# =============================================================================
# run_ingest coerce_types wiring
# =============================================================================


class TestRunIngestCoerceTypes:

    @patch("unity.data_manager.ops.ingest_ops._run_post_ingest_rules")
    @patch("unity.data_manager.ops.ingest_ops.PipelineExecutor")
    @patch("unity.data_manager.ops.ingest_ops.create_table_impl")
    def test_coerce_types_true_injects_explicit_types(
        self,
        mock_create,
        mock_executor_cls,
        mock_rules,
    ):
        from unity.data_manager.ops.ingest_ops import run_ingest

        mock_executor = mock_executor_cls.return_value
        mock_executor.execute.return_value = {}

        rows = [
            {"dt": "2025-01-01 12:00:00", "val": 42},
            {"dt": "2025-01-02 12:00:00", "val": 43},
        ]
        run_ingest(None, "test/ctx", rows, coerce_types=True)

        for row in rows:
            assert "explicit_types" in row
            assert "dt" in row["explicit_types"]
            assert "val" in row["explicit_types"]

    @patch("unity.data_manager.ops.ingest_ops._run_post_ingest_rules")
    @patch("unity.data_manager.ops.ingest_ops.PipelineExecutor")
    @patch("unity.data_manager.ops.ingest_ops.create_table_impl")
    def test_coerce_types_false_no_explicit_types(
        self,
        mock_create,
        mock_executor_cls,
        mock_rules,
    ):
        from unity.data_manager.ops.ingest_ops import run_ingest

        mock_executor = mock_executor_cls.return_value
        mock_executor.execute.return_value = {}

        rows = [{"dt": "2025-01-01 12:00:00", "val": 42}]
        run_ingest(None, "test/ctx", rows, coerce_types=False)

        for row in rows:
            assert "explicit_types" not in row

    @patch("unity.data_manager.ops.ingest_ops._run_post_ingest_rules")
    @patch("unity.data_manager.ops.ingest_ops.PipelineExecutor")
    @patch("unity.data_manager.ops.ingest_ops.create_table_impl")
    def test_coerce_types_true_coerces_empty_strings(
        self,
        mock_create,
        mock_executor_cls,
        mock_rules,
    ):
        from unity.data_manager.ops.ingest_ops import run_ingest

        mock_executor = mock_executor_cls.return_value
        mock_executor.execute.return_value = {}

        rows = [{"a": "", "b": "hello"}]
        result = run_ingest(None, "test/ctx", rows, coerce_types=True)

        assert rows[0]["a"] is None
        assert rows[0]["b"] == "hello"
        assert result.coercion_stats is not None
        assert result.coercion_stats["empty_strings_coerced"] == 1

    @patch("unity.data_manager.ops.ingest_ops._run_post_ingest_rules")
    @patch("unity.data_manager.ops.ingest_ops.PipelineExecutor")
    @patch("unity.data_manager.ops.ingest_ops.create_table_impl")
    def test_coerce_types_false_coerces_only_empty_strings(
        self,
        mock_create,
        mock_executor_cls,
        mock_rules,
    ):
        from unity.data_manager.ops.ingest_ops import run_ingest

        mock_executor = mock_executor_cls.return_value
        mock_executor.execute.return_value = {}

        rows = [{"a": "", "b": "garbage_not_a_datetime"}]
        run_ingest(None, "test/ctx", rows, coerce_types=False)

        assert rows[0]["a"] is None
        assert rows[0]["b"] == "garbage_not_a_datetime"

    @patch("unity.data_manager.ops.ingest_ops._run_post_ingest_rules")
    @patch("unity.data_manager.ops.ingest_ops.PipelineExecutor")
    @patch("unity.data_manager.ops.ingest_ops.create_table_impl")
    def test_coerce_types_true_coerces_type_mismatches(
        self,
        mock_create,
        mock_executor_cls,
        mock_rules,
    ):
        from unity.data_manager.ops.ingest_ops import run_ingest

        mock_executor = mock_executor_cls.return_value
        mock_executor.execute.return_value = {}

        rows = [
            {"dt": "2025-01-01 12:00:00"},
            {"dt": "2025-01-02 12:00:00"},
            {"dt": "garbage"},
        ]
        result = run_ingest(None, "test/ctx", rows, coerce_types=True)

        assert rows[0]["dt"] == "2025-01-01 12:00:00"
        assert rows[1]["dt"] == "2025-01-02 12:00:00"
        assert rows[2]["dt"] is None
        assert result.coercion_stats["type_coerced"] >= 1
