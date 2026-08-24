"""A row count is not evidence that an ingestion worked.

A staging run reported 449,287 rows committed across fifteen tables. Every data
column in every row held the four characters ``None`` -- Python's ``str(None)``,
written where a value should have been. The count was accurate and the tables
were useless, and nothing in the run's own status disagreed with it: states were
terminal, checkpoints had advanced, rows_written matched.

So reconciliation asks the second question too -- does any column carry data --
because that is the one the count cannot answer.
"""

from __future__ import annotations

import pytest

from unify.ingestion_manager.ingestion_manager import _is_blank
from unify.ingestion_manager.types.run import TableReconciliation


class TestWhatCountsAsBlank:
    @pytest.mark.parametrize("value", [None, "", "   ", "None", "null", "NaN", "-"])
    def test_the_empty_renderings_are_blank(self, value):
        # "None" is the one that matters: it is what str(None) produces, it is
        # four printable characters, and every naive check reads it as data.
        assert _is_blank(value) is True, repr(value)

    @pytest.mark.parametrize("value", ["Repa0", 0, False, "0", "false"])
    def test_falsey_data_is_still_data(self, value):
        # Zero and False are legitimate values. Treating them as missing would
        # report a working numeric table as broken.
        assert _is_blank(value) is False, repr(value)


class TestTheFailureThisExistsFor:
    def test_a_full_row_count_with_blank_columns_is_not_complete(self):
        # The observed run, in miniature: every row present, no values in them.
        result = TableReconciliation(
            context="Data/Repairs/dimJobDetails",
            source_rows=13000,
            stored_rows=13000,
            empty_columns=["RepairStatusCode", "WorksOrderRef", "Trade"],
            sampled_rows=25,
        )

        assert result.complete is False
        assert "not evidence" in result.summary

    def test_the_summary_says_the_count_cannot_be_trusted(self):
        # A person reading only "13000 of 13000" would ship this table.
        result = TableReconciliation(
            context="Data/Repairs/dimJobDetails",
            source_rows=13000,
            stored_rows=13000,
            empty_columns=["Trade"],
        )

        assert "13000" in result.summary
        assert "blank in every row" in result.summary


class TestOrdinaryOutcomes:
    def test_a_populated_full_table_is_complete(self):
        result = TableReconciliation(
            context="Data/Repairs/DimDates",
            source_rows=54621,
            stored_rows=54621,
            sampled_rows=25,
        )

        assert result.complete is True
        assert "complete" in result.summary

    def test_more_rows_than_the_source_still_counts_as_complete(self):
        # A resumed run can re-commit a boundary chunk. Over-count is not loss.
        result = TableReconciliation(context="c", source_rows=100, stored_rows=104)

        assert result.complete is True

    def test_a_short_table_is_incomplete(self):
        result = TableReconciliation(context="c", source_rows=54621, stored_rows=4000)

        assert result.complete is False

    def test_an_unmeasured_source_falls_back_to_did_anything_land(self):
        # Source rows are not always knowable up front, and "some data arrived"
        # is the strongest claim available then -- so it is the one made.
        assert TableReconciliation(context="c", stored_rows=1).complete is True
        assert TableReconciliation(context="c", stored_rows=0).complete is False

    def test_zero_of_a_measured_zero_is_complete(self):
        # An export can legitimately be empty. Once the source has been measured
        # at zero there is nothing missing, so this is complete -- unlike the
        # unmeasured case above, where zero rows means nothing was learned.
        result = TableReconciliation(context="c", source_rows=0, stored_rows=0)

        assert result.complete is True
