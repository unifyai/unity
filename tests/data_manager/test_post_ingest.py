"""Tests for post-ingest derived date columns and field description passthrough.

Covers:
- ``_derive_date_column_name``: pure naming logic (CamelCase / snake_case, with/without "date")
- ``_ensure_date_derived_columns``: mocked integration — verifies correct equations and column names
- Field description passthrough: widened ``fields`` type flows through to ``unify.create_fields``
"""

from __future__ import annotations

from unittest.mock import call, patch


from unity.data_manager.ops.ingest_ops import (
    _derive_date_column_name,
    _ensure_date_derived_columns,
)

# =============================================================================
# _derive_date_column_name — pure naming tests
# =============================================================================


class TestDeriveDateColumnName:
    """Verify suffix selection based on casing and presence of 'date'."""

    # -- Title/PascalCase, already contains "Date" ----------------------------

    def test_camel_trailing_date(self):
        assert _derive_date_column_name("VisitDate") == "VisitDate_Day"

    def test_camel_mid_date(self):
        assert (
            _derive_date_column_name("WorksOrderRaisedDate")
            == "WorksOrderRaisedDate_Day"
        )

    def test_camel_date_at_start(self):
        assert _derive_date_column_name("DateCreated") == "DateCreated_Day"

    # -- Title/PascalCase, no "date" -----------------------------------------

    def test_camel_no_date(self):
        assert _derive_date_column_name("Departure") == "Departure_Date"

    def test_camel_no_date_arrival(self):
        assert _derive_date_column_name("Arrival") == "Arrival_Date"

    def test_camel_no_date_multi_word(self):
        assert (
            _derive_date_column_name("ScheduledAppointmentStart")
            == "ScheduledAppointmentStart_Date"
        )

    # -- snake_case, already contains "date" ---------------------------------

    def test_snake_trailing_date(self):
        assert (
            _derive_date_column_name("last_modified_date") == "last_modified_date_day"
        )

    def test_snake_mid_date(self):
        assert _derive_date_column_name("update_date_utc") == "update_date_utc_day"

    # -- snake_case, no "date" -----------------------------------------------

    def test_snake_no_date(self):
        assert _derive_date_column_name("created_at") == "created_at_date"

    def test_snake_no_date_timestamp(self):
        assert _derive_date_column_name("event_timestamp") == "event_timestamp_date"

    # -- plain lowercase (no underscore, no title case) ----------------------

    def test_plain_lowercase_no_date(self):
        assert _derive_date_column_name("timestamp") == "timestamp_date"

    def test_plain_lowercase_with_date(self):
        assert _derive_date_column_name("dateonly") == "dateonly_day"

    # -- edge: single title-cased word ---------------------------------------

    def test_single_word_upper(self):
        assert _derive_date_column_name("Date") == "Date_Day"

    def test_single_word_upper_no_date(self):
        assert _derive_date_column_name("Time") == "Time_Date"


# =============================================================================
# _ensure_date_derived_columns — mocked integration
# =============================================================================


class TestEnsureDateDerivedColumns:

    @patch("unity.data_manager.ops.ingest_ops._ensure_derived_column")
    @patch("unity.data_manager.ops.ingest_ops._unify")
    def test_creates_columns_for_datetime_fields(self, mock_unify, mock_ensure):
        mock_unify.get_fields.return_value = {
            "VisitDate": {"data_type": "datetime"},
            "Departure": {"data_type": "datetime"},
            "OperativeName": {"data_type": "str"},
            "row_id": {"data_type": "int"},
            "_emb_col": {"data_type": "vector"},
        }

        result = _ensure_date_derived_columns("test/ctx")

        assert set(result) == {"VisitDate_Day", "Departure_Date"}
        assert mock_ensure.call_count == 2

        expected_calls = [
            call(
                "test/ctx",
                key="VisitDate_Day",
                equation="date({lg:VisitDate})",
                referenced_logs_context="test/ctx",
            ),
            call(
                "test/ctx",
                key="Departure_Date",
                equation="date({lg:Departure})",
                referenced_logs_context="test/ctx",
            ),
        ]
        mock_ensure.assert_has_calls(expected_calls, any_order=True)

    @patch("unity.data_manager.ops.ingest_ops._ensure_derived_column")
    @patch("unity.data_manager.ops.ingest_ops._unify")
    def test_skips_private_fields(self, mock_unify, mock_ensure):
        mock_unify.get_fields.return_value = {
            "_internal_date": {"data_type": "datetime"},
            "PublicDate": {"data_type": "datetime"},
        }

        result = _ensure_date_derived_columns("test/ctx")

        assert result == ["PublicDate_Day"]
        mock_ensure.assert_called_once()

    @patch("unity.data_manager.ops.ingest_ops._ensure_derived_column")
    @patch("unity.data_manager.ops.ingest_ops._unify")
    def test_no_datetime_fields_returns_empty(self, mock_unify, mock_ensure):
        mock_unify.get_fields.return_value = {
            "name": {"data_type": "str"},
            "count": {"data_type": "int"},
        }

        result = _ensure_date_derived_columns("test/ctx")

        assert result == []
        mock_ensure.assert_not_called()

    @patch("unity.data_manager.ops.ingest_ops._ensure_derived_column")
    @patch("unity.data_manager.ops.ingest_ops._unify")
    def test_empty_fields_returns_empty(self, mock_unify, mock_ensure):
        mock_unify.get_fields.return_value = {}

        result = _ensure_date_derived_columns("test/ctx")

        assert result == []
        mock_ensure.assert_not_called()


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
