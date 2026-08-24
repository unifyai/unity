"""A destination the backend would refuse is refused before it is dispatched.

The backend reports a bad context name by naming the rule it broke and never the
value that broke it: *"Invalid context name. Names can only contain alphanumeric
characters, underscores, dashes, and forward slashes. Consecutive slashes are not
allowed."*

Received synchronously that is a useful message. Received from a worker pod, four
retries into a dispatch, with fifteen destinations in flight and the attempted
name logged at DEBUG, it is close to useless -- and because the rejection is
deterministic while the message is never acked, one bad name had a fleet of
fifteen workers retrying it every twenty-one seconds indefinitely.

So the rule is stated here and checked before anything is published.
"""

from __future__ import annotations

import pytest

from unify.common.context_names import (
    InvalidContextName,
    assert_all_valid,
    check_all,
    is_valid,
    join_context,
    sanitise_segment,
    why_invalid,
)


class TestWhatTheBackendWouldRefuse:
    def test_consecutive_slashes_are_named_as_an_empty_segment(self):
        # The observed failure. Saying "empty path segment" points at the cause;
        # "invalid name" sends the reader back to the documentation.
        reason = why_invalid("Data/MHDataExtract11May//DimDates")
        assert reason is not None
        assert "empty path segment" in reason

    def test_a_space_is_reported_with_the_offending_character(self):
        # The likeliest real source: a folder called "MH data extract 11th May".
        reason = why_invalid("Data/MH data extract/DimDates")
        assert reason is not None
        assert "' '" in reason

    def test_a_dot_is_refused(self):
        # A file name carried into a path, e.g. "DimDates.csv".
        assert why_invalid("Data/Repairs/DimDates.csv") is not None

    @pytest.mark.parametrize("name", ["", "   ", None])
    def test_nothing_is_not_a_name(self, name):
        assert why_invalid(name) is not None

    @pytest.mark.parametrize("name", ["/Data/X", "Data/X/"])
    def test_a_leading_or_trailing_slash_leaves_an_empty_segment(self, name):
        assert why_invalid(name) is not None

    def test_untrimmed_whitespace_is_refused_rather_than_trimmed(self):
        # Trimming silently would write to a different place than was named.
        assert why_invalid(" Data/X") is not None

    @pytest.mark.parametrize(
        "name",
        [
            "Data/MHDataExtract11May/Repairs/DimDates",
            "Teams/4/Data/HomeIQ/WirralHousing",
            "Ingestion/Runs",
            "a-b_c/d-e_f",
        ],
    )
    def test_real_destinations_are_accepted(self, name):
        assert why_invalid(name) is None
        assert is_valid(name) is True


class TestReportingEveryProblemAtOnce:
    def test_all_bad_names_are_returned_not_just_the_first(self):
        # A caller about to create fifteen destinations wants one answer listing
        # what to fix, not fifteen round trips discovering them one at a time.
        problems = check_all(["Data/ok", "Data//bad", "Data/also bad", "Data/fine"])

        assert len(problems) == 2
        assert {name for name, _ in problems} == {"Data//bad", "Data/also bad"}

    def test_assert_all_valid_passes_silently_when_all_are_fine(self):
        assert_all_valid(["Data/a", "Data/b/c"])

    def test_assert_all_valid_names_every_offender(self):
        with pytest.raises(InvalidContextName) as excinfo:
            assert_all_valid(["Data//x", "Data/y z"], what="destination")

        message = str(excinfo.value)
        assert "2 destination(s)" in message
        assert "Data//x" in message
        assert "Data/y z" in message


class TestDerivingASafeName:
    def test_a_folder_name_with_spaces_becomes_usable(self):
        assert (
            sanitise_segment("MH data extract 11th May") == "MH_data_extract_11th_May"
        )

    def test_a_run_of_disallowed_characters_becomes_one_underscore(self):
        # Collapsing rather than deleting: two distinct names must not silently
        # become the same one. Dashes are legal, so they survive untouched --
        # only the disallowed run between them collapses.
        assert sanitise_segment("a  b") == "a_b"
        assert sanitise_segment("a  --  b") == "a_--_b"

    def test_a_slash_inside_a_segment_cannot_smuggle_a_level(self):
        assert "/" not in sanitise_segment("Repairs/2026")

    def test_a_segment_with_nothing_usable_returns_empty_not_a_default(self):
        # The caller must treat this as a missing name. Substituting a default
        # would put rows somewhere nobody named.
        assert sanitise_segment("...") == ""
        assert sanitise_segment("") == ""


class TestJoining:
    def test_segments_join_into_a_valid_path(self):
        assert join_context("Data", "Repairs", "DimDates") == "Data/Repairs/DimDates"

    def test_stray_slashes_between_segments_are_absorbed(self):
        assert join_context("Data/", "/Repairs/") == "Data/Repairs"

    def test_an_empty_segment_raises_and_says_which_position(self):
        # This is the defect that produces consecutive slashes. Dropping the
        # segment silently would move the table somewhere else.
        with pytest.raises(InvalidContextName) as excinfo:
            join_context("Data", "", "DimDates")

        assert "segment 1" in str(excinfo.value)

    def test_a_bad_character_surviving_a_join_still_raises(self):
        with pytest.raises(InvalidContextName):
            join_context("Data", "MH data extract")
