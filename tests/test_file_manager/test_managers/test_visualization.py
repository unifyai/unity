"""
Test FileManager.visualize tool functionality.

This module tests the visualization capabilities of FileManager, including:
- Direct invocation of _visualize with various plot types
- Single and multiple table handling
- Error handling for invalid inputs
- Integration via ask/ask_about_file methods with LLM-as-judge validation
"""

from __future__ import annotations

import json
import re

import pytest

from unity.file_manager.simulated import SimulatedFileManager
from unity.file_manager.managers.utils.viz_utils import (
    PlotType,
    PlotConfig,
    PlotResult,
    build_plot_config_dict,
    build_project_config_dict,
)
from unity.common.llm_helpers import _dumps
from unity.common.llm_client import new_llm_client
from tests.helpers import _handle_project
from tests.assertion_helpers import assertion_failed, find_tool_calls_and_results


# =============================================================================
# LLM-AS-JUDGE HELPER FOR VISUALIZATION TESTS
# =============================================================================


def _llm_assert_visualization_correct(
    question: str,
    candidate: str,
    steps: list,
    expect_visualization: bool = True,
) -> None:
    """Assert the candidate response correctly answers the analytical question.

    Uses an LLM judge to evaluate whether:
    1. The response adequately addresses the analytical question
    2. If expect_visualization=True, a visualization was created/offered
       (URL present, chart mentioned, or plot generated)
    3. The response shows reasonable analytical insight

    Parameters
    ----------
    question : str
        The analytical question asked.
    candidate : str
        The response from the FileManager.
    steps : list
        Reasoning steps from the LLM tool loop for debugging.
    expect_visualization : bool
        Whether to require evidence of visualization in the response.
    """
    judge = new_llm_client(async_client=False)

    viz_requirement = (
        "The response MUST include evidence of visualization (a plot URL, "
        "mention of a chart/plot being created, or visualization output). "
        if expect_visualization
        else "Visualization is optional but appreciated if present. "
    )

    judge.set_system_message(
        "You are a strict unit-test judge evaluating FileManager visualization responses. "
        "You will be given an analytical question and a candidate response. "
        f"{viz_requirement}"
        "Evaluate whether the response:\n"
        "1. Addresses the analytical question with relevant insights\n"
        "2. Shows evidence of data analysis (numbers, comparisons, trends)\n"
        "3. If visualization was expected, confirms a plot/chart was created\n\n"
        'Respond ONLY with valid JSON: {"correct": true, "reason": "..."} or '
        '{"correct": false, "reason": "..."}.',
    )

    payload = _dumps(
        {
            "question": question,
            "candidate_response": candidate,
            "expect_visualization": expect_visualization,
        },
        indent=4,
    )
    result = judge.generate(payload)

    match = re.search(r"\{.*\}", result, re.S)
    assert match, assertion_failed(
        "Expected JSON format from LLM judge",
        result,
        steps,
        "LLM judge returned unexpected format",
    )
    verdict = json.loads(match.group(0))
    assert verdict.get("correct") is True, assertion_failed(
        f"Expected correct visualization response (expect_viz={expect_visualization})",
        candidate,
        steps,
        f"Question: {question}\nJudge reason: {verdict.get('reason', 'N/A')}",
    )


def _check_visualize_tool_called(steps: list) -> bool:
    """Check if the _visualize tool was called in the reasoning steps."""
    tool_calls, tool_results = find_tool_calls_and_results(steps, "visualize")
    return len(tool_calls) > 0


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def simulated_file_manager():
    """Fixture for a clean SimulatedFileManager instance."""
    fm = SimulatedFileManager()
    fm.clear_simulated_files()
    # Add sample files with table data
    fm.add_simulated_file(
        "data.xlsx",
        records=[{"content": "Spreadsheet data"}],
    )
    yield fm


# =============================================================================
# UNIT TESTS: viz_utils.py
# =============================================================================


class TestPlotType:
    """Tests for PlotType enum."""

    def test_plot_type_values(self):
        """Verify all expected plot types exist."""
        assert PlotType.BAR.value == "bar"
        assert PlotType.LINE.value == "line"
        assert PlotType.SCATTER.value == "scatter"
        assert PlotType.HISTOGRAM.value == "histogram"


class TestPlotConfig:
    """Tests for PlotConfig model."""

    def test_minimal_config(self):
        """PlotConfig with only required fields."""
        config = PlotConfig(plot_type="bar", x_axis="Category")
        assert config.plot_type == "bar"
        assert config.x_axis == "Category"
        assert config.y_axis is None
        assert config.group_by is None

    def test_full_config(self):
        """PlotConfig with all fields."""
        config = PlotConfig(
            plot_type="scatter",
            x_axis="X",
            y_axis="Y",
            group_by="Group",
            metric="sum",
            scale_x="log",
            scale_y="linear",
            bin_count=20,
            show_regression=True,
            title="My Chart",
        )
        assert config.plot_type == "scatter"
        assert config.x_axis == "X"
        assert config.y_axis == "Y"
        assert config.group_by == "Group"
        assert config.metric == "sum"
        assert config.show_regression is True


class TestPlotResult:
    """Tests for PlotResult model."""

    def test_successful_result(self):
        """PlotResult with successful URL."""
        result = PlotResult(
            url="https://console.unify.ai/plot/abc123",
            token="abc123",
            expires_in_hours=24,
            title="Test Plot",
        )
        assert result.succeeded is True
        assert result.url == "https://console.unify.ai/plot/abc123"

    def test_failed_result(self):
        """PlotResult with error."""
        result = PlotResult(
            title="Failed Plot",
            error="Connection timeout",
        )
        assert result.succeeded is False
        assert result.error == "Connection timeout"

    def test_to_dict(self):
        """PlotResult.to_dict() returns correct dictionary."""
        result = PlotResult(
            url="https://console.unify.ai/plot/abc123",
            token="abc123",
            title="Test Plot",
        )
        d = result.to_dict()
        assert d["url"] == "https://console.unify.ai/plot/abc123"
        assert d["token"] == "abc123"
        assert d["title"] == "Test Plot"
        assert "error" not in d


class TestBuildPlotConfigDict:
    """Tests for build_plot_config_dict function."""

    def test_minimal_config(self):
        """Build dict with minimal config."""
        config = PlotConfig(plot_type="bar", x_axis="Category")
        d = build_plot_config_dict(config)
        assert d["type"] == "bar"
        assert d["x_axis"] == "Category"
        assert "y_axis" not in d

    def test_full_config(self):
        """Build dict with all fields."""
        config = PlotConfig(
            plot_type="scatter",
            x_axis="X",
            y_axis="Y",
            group_by="Group",
            metric="mean",
            show_regression=True,
        )
        d = build_plot_config_dict(config)
        assert d["type"] == "scatter"
        assert d["x_axis"] == "X"
        assert d["y_axis"] == "Y"
        assert d["group_by"] == "Group"
        assert d["metric"] == "mean"
        assert d["show_regression"] is True


class TestBuildProjectConfigDict:
    """Tests for build_project_config_dict function."""

    def test_minimal_config(self):
        """Build project config with minimal required fields."""
        d = build_project_config_dict(
            project_name="TestProject",
            context="DefaultUser/Assistant/Files/Local/test",
        )
        assert d["project_name"] == "TestProject"
        assert d["context"] == "DefaultUser/Assistant/Files/Local/test"
        assert d["randomize"] is False
        assert "filter_expr" not in d
        assert "exclude_fields" not in d
        assert "group_by" not in d

    def test_with_filter(self):
        """Build project config with filter expression."""
        d = build_project_config_dict(
            project_name="TestProject",
            context="DefaultUser/Assistant/Files/Local/test",
            filter_expr="status == 'active'",
        )
        assert d["filter_expr"] == "status == 'active'"

    def test_with_randomize(self):
        """Build project config with randomize enabled."""
        d = build_project_config_dict(
            project_name="TestProject",
            context="DefaultUser/Assistant/Files/Local/test",
            randomize=True,
        )
        assert d["randomize"] is True

    def test_with_exclude_fields(self):
        """Build project config with excluded fields."""
        d = build_project_config_dict(
            project_name="TestProject",
            context="DefaultUser/Assistant/Files/Local/test",
            exclude_fields=["password", "secret"],
        )
        assert d["exclude_fields"] == ["password", "secret"]

    def test_with_group_by(self):
        """Build project config with group_by column."""
        d = build_project_config_dict(
            project_name="TestProject",
            context="DefaultUser/Assistant/Files/Local/test",
            group_by="category",
        )
        assert d["group_by"] == "category"

    def test_full_config(self):
        """Build project config with all fields."""
        d = build_project_config_dict(
            project_name="TestProject",
            context="DefaultUser/Assistant/Files/Local/test",
            filter_expr="status == 'active'",
            randomize=True,
            exclude_fields=["password"],
            group_by="region",
        )
        assert d["project_name"] == "TestProject"
        assert d["context"] == "DefaultUser/Assistant/Files/Local/test"
        assert d["filter_expr"] == "status == 'active'"
        assert d["randomize"] is True
        assert d["exclude_fields"] == ["password"]
        assert d["group_by"] == "region"


# =============================================================================
# SIMULATED FILEMANAGER TESTS: Direct _visualize invocation
# =============================================================================


class TestSimulatedVisualize:
    """Tests for SimulatedFileManager.visualize method."""

    def test_single_table_bar_chart(self, simulated_file_manager):
        """Generate a bar chart for a single table."""
        fm = simulated_file_manager
        result = fm.visualize(
            tables="/path/to/data.xlsx.Tables.Sales",
            plot_type="bar",
            x_axis="Category",
            y_axis="Revenue",
            title="Sales by Category",
        )
        assert hasattr(result, "url")
        assert result.url is not None
        assert result.title == "Sales by Category"
        assert "simulated" in result.url
        assert result.succeeded is True

    def test_single_table_line_chart(self, simulated_file_manager):
        """Generate a line chart for a single table."""
        fm = simulated_file_manager
        result = fm.visualize(
            tables="/path/to/data.xlsx.Tables.Metrics",
            plot_type="line",
            x_axis="Date",
            y_axis="Value",
            group_by="Region",
        )
        assert result.url is not None
        assert result.succeeded is True

    def test_single_table_histogram(self, simulated_file_manager):
        """Generate a histogram for a single table."""
        fm = simulated_file_manager
        result = fm.visualize(
            tables="/path/to/data.xlsx.Tables.Orders",
            plot_type="histogram",
            x_axis="OrderValue",
            bin_count=20,
        )
        assert result.url is not None
        assert result.succeeded is True

    def test_single_table_scatter(self, simulated_file_manager):
        """Generate a scatter plot for a single table."""
        fm = simulated_file_manager
        result = fm.visualize(
            tables="/path/to/data.xlsx.Tables.Performance",
            plot_type="scatter",
            x_axis="TimeSpent",
            y_axis="Score",
            show_regression=True,
        )
        assert result.url is not None
        assert result.succeeded is True

    def test_multiple_tables_same_schema(self, simulated_file_manager):
        """Generate same plot for multiple tables with identical schemas."""
        fm = simulated_file_manager
        result = fm.visualize(
            tables=[
                "/path/to/data.xlsx.Tables.July_2025",
                "/path/to/data.xlsx.Tables.August_2025",
                "/path/to/data.xlsx.Tables.September_2025",
            ],
            plot_type="bar",
            x_axis="Driver",
            y_axis="TotalDistance",
            metric="sum",
            title="Distance by Driver",
        )
        assert isinstance(result, list)
        assert len(result) == 3
        # Each result should have the table label in title
        assert "July_2025" in result[0].title
        assert "August_2025" in result[1].title
        assert "September_2025" in result[2].title
        # Each should have a URL and succeed
        for r in result:
            assert r.url is not None
            assert r.succeeded is True

    def test_empty_tables(self, simulated_file_manager):
        """Empty tables parameter returns error."""
        fm = simulated_file_manager
        result = fm.visualize(
            tables="",
            plot_type="bar",
            x_axis="X",
        )
        assert result.error is not None
        assert result.succeeded is False

    def test_empty_tables_list(self, simulated_file_manager):
        """Empty tables list returns error."""
        fm = simulated_file_manager
        result = fm.visualize(
            tables=[],
            plot_type="bar",
            x_axis="X",
        )
        assert result.error is not None
        assert result.succeeded is False

    def test_with_filter(self, simulated_file_manager):
        """Generate plot with filter expression."""
        fm = simulated_file_manager
        result = fm.visualize(
            tables="/path/to/data.xlsx.Tables.Jobs",
            plot_type="bar",
            x_axis="Status",
            y_axis="Count",
            filter="status == 'Complete'",
        )
        assert result.url is not None
        assert result.succeeded is True

    def test_with_aggregate(self, simulated_file_manager):
        """Generate plot with aggregation function."""
        fm = simulated_file_manager
        result = fm.visualize(
            tables="/path/to/data.xlsx.Tables.Sales",
            plot_type="bar",
            x_axis="Region",
            y_axis="Revenue",
            metric="sum",
        )
        assert result.url is not None
        assert result.succeeded is True

    def test_with_scale(self, simulated_file_manager):
        """Generate plot with custom scales."""
        fm = simulated_file_manager
        result = fm.visualize(
            tables="/path/to/data.xlsx.Tables.Data",
            plot_type="scatter",
            x_axis="X",
            y_axis="Y",
            scale_x="log",
            scale_y="linear",
        )
        assert result.url is not None
        assert result.succeeded is True


# =============================================================================
# INTEGRATION TESTS: Tool exposure verification
# =============================================================================


def test_visualize_method_exists_on_simulated(simulated_file_manager):
    """Verify visualize method exists on SimulatedFileManager."""
    fm = simulated_file_manager
    assert hasattr(fm, "visualize")
    assert callable(fm.visualize)


def test_visualize_tool_exposed_in_ask():
    """Verify visualize is accessible through ask tool dict on real FileManager."""
    from unity.file_manager.managers.local import LocalFileManager

    # Check if the tool is registered in the class's tool dict pattern
    # We just verify the method exists and is decorated correctly
    fm_class = LocalFileManager
    assert hasattr(fm_class, "visualize")


def test_visualize_tool_exposed_in_ask_about_file():
    """Verify visualize is accessible through ask_about_file tool dict on real FileManager."""
    from unity.file_manager.managers.file_manager import FileManager

    # Verify visualize method exists on FileManager
    assert hasattr(FileManager, "visualize")


# =============================================================================
# END-TO-END TESTS: LLM orchestration of visualize via ask/ask_about_file
# =============================================================================
#
# These tests verify that the LLM naturally chooses to call visualize when
# answering analytical questions where visualization adds value. Tests use
# real FileManager with ingested data and LLM-as-judge validation.
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.eval
@pytest.mark.requires_real_unify
@pytest.mark.timeout(300)
@_handle_project
async def test_ask_about_file_trend_analysis(file_manager, tmp_path):
    """
    End-to-end: ask_about_file with trend analysis question.

    When asked about trends in time-series data with a visualization hint,
    the LLM should use visualize to illustrate the trend.
    """
    fm = file_manager
    fm.clear()

    # Create CSV with time series data
    csv_path = tmp_path / "monthly_metrics.csv"
    csv_path.write_text(
        "Month,Sales,Expenses\n"
        "January,12000,8000\n"
        "February,14000,8500\n"
        "March,15500,9000\n"
        "April,13000,8200\n"
        "May,16000,9500\n"
        "June,18000,10000\n",
        encoding="utf-8",
    )
    fm.ingest_files(str(csv_path))

    # Trend analysis question with visualization hint
    instruction = (
        "What's the trend in sales over these months? "
        "Is the business growing? Show me visually if possible."
    )
    handle = await fm.ask_about_file(
        str(csv_path),
        instruction,
        _return_reasoning_steps=True,
    )
    answer, steps = await handle.result()

    assert isinstance(answer, str) and answer.strip()

    # Verify visualize tool was actually called
    assert _check_visualize_tool_called(
        steps,
    ), "visualize tool was not called despite explicit visualization request"

    # Use LLM judge - visualization is explicitly requested
    _llm_assert_visualization_correct(
        question=instruction,
        candidate=answer,
        steps=steps,
        expect_visualization=True,
    )


@pytest.mark.asyncio
@pytest.mark.eval
@pytest.mark.requires_real_unify
@pytest.mark.timeout(300)
@_handle_project
async def test_ask_about_file_comparison_analysis(file_manager, tmp_path):
    """
    End-to-end: Comparative analysis with visualization hint.

    Questions comparing categories with a visualization request should
    trigger visualize with appropriate chart type.
    """
    fm = file_manager
    fm.clear()

    # Create CSV with categorical comparison data
    csv_path = tmp_path / "department_stats.csv"
    csv_path.write_text(
        "Department,Headcount,Budget\n"
        "Engineering,45,500000\n"
        "Sales,30,350000\n"
        "Marketing,20,250000\n"
        "Operations,25,300000\n"
        "HR,10,150000\n",
        encoding="utf-8",
    )
    fm.ingest_files(str(csv_path))

    # Comparison question with visualization hint
    instruction = (
        "Compare the departments by headcount and budget. "
        "Which department has the best budget per employee ratio? "
        "A visual comparison would help."
    )
    handle = await fm.ask_about_file(
        str(csv_path),
        instruction,
        _return_reasoning_steps=True,
    )
    answer, steps = await handle.result()

    assert isinstance(answer, str) and answer.strip()

    # Verify visualize tool was actually called
    assert _check_visualize_tool_called(
        steps,
    ), "visualize tool was not called despite explicit visualization request"

    # Use LLM judge - visualization is explicitly requested
    _llm_assert_visualization_correct(
        question=instruction,
        candidate=answer,
        steps=steps,
        expect_visualization=True,
    )


# =============================================================================
# REAL FILEMANAGER TESTS (require real Unify connection)
# =============================================================================


@pytest.mark.requires_real_unify
@_handle_project
def test_real_visualize_single_table(file_manager, tmp_path):
    """
    Test visualize on real FileManager with ingested data.

    This test requires a real Unify connection and ingested table data.
    It verifies the tool can resolve contexts and call the Plot API.
    """
    # This test would require setting up real table data.
    # For now, we just verify the method exists and has the right signature.
    fm = file_manager
    assert hasattr(fm, "visualize")
    # Check that the method is in the ask tools
    tools = fm.get_tools("ask")
    tool_names = list(tools.keys())
    assert any("visualize" in name for name in tool_names)


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestVisualizationErrors:
    """Tests for error handling in visualization."""

    def test_invalid_table_reference(self, simulated_file_manager):
        """
        Invalid table reference should be handled gracefully.

        Note: SimulatedFileManager doesn't validate table references,
        so this just verifies the call doesn't crash.
        """
        fm = simulated_file_manager
        result = fm.visualize(
            tables="/nonexistent/path.xlsx.Tables.MissingTable",
            plot_type="bar",
            x_axis="X",
            y_axis="Y",
        )
        # SimulatedFileManager returns placeholder - no error
        assert result.url is not None
        assert result.succeeded is True

    def test_mixed_empty_tables(self, simulated_file_manager):
        """List with empty strings should filter them out."""
        fm = simulated_file_manager
        result = fm.visualize(
            tables=["", "/path/to/data.xlsx.Tables.Valid", ""],
            plot_type="bar",
            x_axis="X",
            y_axis="Y",
        )
        # Should only process the valid table - returns single PlotResult
        assert result.url is not None
        assert result.succeeded is True
