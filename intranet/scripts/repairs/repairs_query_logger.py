"""
Bespoke Query Logger - File-based logging for repairs query results.

Creates a private `.repairs_queries/` directory structure with:
- Per-run subdirectories (timestamped for each script invocation)
- Per-query log files containing full results, timing, and metadata

Usage:
    from query_logger import QueryLogger

    with QueryLogger() as logger:
        result = await agent.ask("jobs_completed_per_day", group_by="patch")
        logger.log_query("jobs_completed_per_day", {"group_by": "patch"}, result, elapsed=1.23)

    # Or manual lifecycle:
    logger = QueryLogger()
    logger.start_run()
    logger.log_query(...)
    logger.finish_run()
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Directory name for query logs
QUERY_LOG_DIR = ".repairs_queries"


def _get_cwd_root() -> Path:
    """Get the current working directory as the root for logs."""
    return Path(os.getcwd())


def _sanitize_filename(name: str) -> str:
    """Sanitize a string for use in a filename."""
    # Replace problematic characters with underscores
    sanitized = re.sub(r"[^0-9A-Za-z._-]", "_", name)
    # Collapse multiple underscores
    sanitized = re.sub(r"_+", "_", sanitized)
    # Remove leading/trailing underscores
    return sanitized.strip("_")


def _format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"


def _params_to_suffix(params: Dict[str, Any]) -> str:
    """Convert params dict to a filename suffix."""
    if not params:
        return ""

    parts = []
    for key, value in sorted(params.items()):
        # Skip tools param as it's internal
        if key == "tools":
            continue
        # Convert value to string representation
        if isinstance(value, bool):
            val_str = "true" if value else "false"
        elif isinstance(value, (list, dict)):
            val_str = _sanitize_filename(json.dumps(value, sort_keys=True))
        else:
            val_str = _sanitize_filename(str(value))
        parts.append(f"{key}_{val_str}")

    return "__" + "_".join(parts) if parts else ""


class QueryLogEntry:
    """Represents a single query execution log entry."""

    def __init__(
        self,
        query_id: str,
        params: Dict[str, Any],
        result: Any,
        elapsed: float,
        success: bool = True,
        error: Optional[str] = None,
    ):
        self.query_id = query_id
        self.params = params
        self.result = result
        self.elapsed = elapsed
        self.success = success
        self.error = error
        self.timestamp = datetime.now(timezone.utc)

    def to_log_content(self) -> str:
        """Generate the full log file content."""
        lines = [
            "=" * 80,
            f"BESPOKE REPAIRS QUERY LOG",
            "=" * 80,
            "",
            f"Query ID:    {self.query_id}",
            f"Timestamp:   {self.timestamp.isoformat()}",
            f"Duration:    {_format_duration(self.elapsed)}",
            f"Status:      {'SUCCESS' if self.success else 'FAILED'}",
            "",
        ]

        # Parameters section
        lines.append("-" * 40)
        lines.append("PARAMETERS")
        lines.append("-" * 40)
        if self.params:
            for key, value in sorted(self.params.items()):
                if key != "tools":  # Skip internal tools param
                    lines.append(f"  {key}: {value}")
        else:
            lines.append("  (none)")
        lines.append("")

        # Error section (if failed)
        if not self.success and self.error:
            lines.append("-" * 40)
            lines.append("ERROR")
            lines.append("-" * 40)
            lines.append(self.error)
            lines.append("")

        # Result section
        lines.append("-" * 40)
        lines.append("RESULT")
        lines.append("-" * 40)
        if self.result is not None:
            try:
                result_json = json.dumps(self.result, indent=2, default=str)
                lines.append(result_json)
            except Exception:
                lines.append(str(self.result))
        else:
            lines.append("  (no result)")
        lines.append("")

        # Summary section
        if self.success and isinstance(self.result, dict):
            lines.append("-" * 40)
            lines.append("SUMMARY")
            lines.append("-" * 40)
            lines.append(f"  Metric:     {self.result.get('metric_name', 'N/A')}")
            lines.append(f"  Total:      {self.result.get('total', 'N/A')}")
            lines.append(f"  Groups:     {len(self.result.get('results', []))}")
            lines.append(f"  Group By:   {self.result.get('group_by', 'N/A')}")

            metadata = self.result.get("metadata") or {}
            if metadata.get("note"):
                lines.append(f"  Note:       {metadata['note']}")
            if metadata.get("status"):
                lines.append(f"  Status:     {metadata['status']}")
            lines.append("")

        lines.append("=" * 80)
        lines.append(f"END OF LOG - {self.query_id}")
        lines.append("=" * 80)

        return "\n".join(lines)


class QueryLogger:
    """
    Logger for bespoke repairs queries with file-based output.

    Creates a directory structure:
        .repairs_queries/
            {log_subdir}/                  # e.g., 2025-12-18T19-30-45_repairs_dev_pts_0
                {query_id}__{params}.log
                _run_summary.log
    """

    def __init__(
        self,
        root_dir: Optional[Path] = None,
        log_subdir: Optional[str] = None,
    ):
        """
        Initialize the logger.

        Args:
            root_dir: Root directory for logs. Defaults to current working directory.
            log_subdir: Subdirectory name for this run (e.g., "{datetime}_{socket}").
                       If not provided, generates a timestamped name.
        """
        self.root_dir = root_dir or _get_cwd_root()
        self.base_dir = self.root_dir / QUERY_LOG_DIR
        self.log_subdir = log_subdir
        self.run_dir: Optional[Path] = None
        self.run_start: Optional[float] = None
        self.run_timestamp: Optional[str] = None
        self.entries: List[QueryLogEntry] = []
        self._initialized = False

    def _ensure_base_dir(self) -> None:
        """Ensure the base log directory exists."""
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"⚠️  Warning: Could not create log directory {self.base_dir}: {e}")

    def start_run(self) -> Path:
        """
        Start a new logging run.

        Returns:
            Path to the run directory.
        """
        self._ensure_base_dir()

        self.run_start = time.perf_counter()
        now = datetime.now(timezone.utc)

        # Use provided log_subdir or generate timestamped name
        # Format: 2025-12-18T19-30-45_socket_name (if log_subdir provided)
        # Or: 2025-12-18T19-30-45.123456Z (legacy format)
        if self.log_subdir:
            self.run_timestamp = self.log_subdir
        else:
            self.run_timestamp = (
                now.strftime("%Y-%m-%dT%H-%M-%S") + f".{now.microsecond:06d}Z"
            )

        self.run_dir = self.base_dir / self.run_timestamp

        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._initialized = True
            print(f"📁 Log directory: {self.run_dir}")
        except Exception as e:
            print(f"⚠️  Warning: Could not create run directory: {e}")
            self._initialized = False

        self.entries = []
        return self.run_dir

    def log_query(
        self,
        query_id: str,
        params: Dict[str, Any],
        result: Any,
        elapsed: float,
        success: bool = True,
        error: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Log a query execution to a file.

        Args:
            query_id: The query identifier.
            params: Parameters passed to the query.
            result: Query result (or None if failed).
            elapsed: Execution time in seconds.
            success: Whether the query succeeded.
            error: Error message if failed.

        Returns:
            Path to the log file, or None if logging failed.
        """
        if not self._initialized or self.run_dir is None:
            return None

        # Create log entry
        entry = QueryLogEntry(
            query_id=query_id,
            params=params,
            result=result,
            elapsed=elapsed,
            success=success,
            error=error,
        )
        self.entries.append(entry)

        # Generate filename
        params_suffix = _params_to_suffix(params)
        filename = f"{_sanitize_filename(query_id)}{params_suffix}.log"
        filepath = self.run_dir / filename

        # Handle collision (same query+params run multiple times)
        if filepath.exists():
            i = 1
            while filepath.exists():
                filename = f"{_sanitize_filename(query_id)}{params_suffix}_{i}.log"
                filepath = self.run_dir / filename
                i += 1

        # Write log file
        try:
            content = entry.to_log_content()
            filepath.write_text(content, encoding="utf-8")
            return filepath
        except Exception as e:
            print(f"⚠️  Warning: Could not write log file {filepath}: {e}")
            return None

    def _parse_log_file(self, filepath: Path) -> Optional[Dict[str, Any]]:
        """
        Parse an existing log file to extract query information.

        Returns:
            Dictionary with query_id, status, duration, error (if any), or None if parsing fails.
        """
        try:
            content = filepath.read_text(encoding="utf-8")

            # Extract key fields using simple parsing
            info: Dict[str, Any] = {
                "filename": filepath.name,
                "query_id": None,
                "status": None,
                "duration": None,
                "timestamp": None,
                "params": {},
                "error": None,
                "total": None,
            }

            lines = content.split("\n")
            in_params = False
            in_error = False
            error_lines = []

            for line in lines:
                line_stripped = line.strip()

                if line_stripped.startswith("Query ID:"):
                    info["query_id"] = line_stripped.split(":", 1)[1].strip()
                elif line_stripped.startswith("Status:"):
                    status_str = line_stripped.split(":", 1)[1].strip()
                    info["status"] = status_str == "SUCCESS"
                elif line_stripped.startswith("Duration:"):
                    info["duration"] = line_stripped.split(":", 1)[1].strip()
                elif line_stripped.startswith("Timestamp:"):
                    info["timestamp"] = line_stripped.split(":", 1)[1].strip()
                elif line_stripped == "PARAMETERS":
                    in_params = True
                    in_error = False
                elif line_stripped == "ERROR":
                    in_error = True
                    in_params = False
                elif line_stripped.startswith("-" * 10) or line_stripped.startswith(
                    "=" * 10,
                ):
                    if in_error and error_lines:
                        info["error"] = "\n".join(error_lines).strip()
                    in_params = False
                    in_error = False
                    error_lines = []
                elif (
                    in_params
                    and ":" in line_stripped
                    and not line_stripped.startswith("(")
                ):
                    key, val = line_stripped.split(":", 1)
                    info["params"][key.strip()] = val.strip()
                elif in_error:
                    error_lines.append(line_stripped)
                elif line_stripped.startswith('"total":'):
                    # Try to extract total from JSON result
                    try:
                        total_str = line_stripped.split(":", 1)[1].strip().rstrip(",")
                        info["total"] = float(total_str)
                    except (ValueError, IndexError):
                        pass

            # Handle any remaining error lines
            if in_error and error_lines:
                info["error"] = "\n".join(error_lines).strip()

            return info if info["query_id"] else None

        except Exception:
            return None

    def _scan_directory_logs(self) -> List[Dict[str, Any]]:
        """
        Scan the run directory for all query log files and parse them.

        Returns:
            List of parsed log info dictionaries, sorted by filename.
        """
        if not self.run_dir or not self.run_dir.exists():
            return []

        logs = []
        for log_file in sorted(self.run_dir.glob("*.log")):
            # Skip the summary file itself
            if log_file.name.startswith("_"):
                continue

            info = self._parse_log_file(log_file)
            if info:
                logs.append(info)

        return logs

    def finish_run(self) -> Optional[Path]:
        """
        Finish the logging run and write a summary file.

        Scans ALL log files in the directory to create an aggregate summary,
        including global statistics across all queries.

        Returns:
            Path to the summary file, or None if logging failed.
        """
        if not self._initialized or self.run_dir is None:
            return None

        run_elapsed = time.perf_counter() - (self.run_start or 0)

        # Scan directory for ALL query logs (from all parallel processes)
        all_logs = self._scan_directory_logs()

        # Calculate global statistics
        total_queries = len(all_logs)
        successful = sum(1 for log in all_logs if log.get("status") is True)
        failed = sum(1 for log in all_logs if log.get("status") is False)
        success_rate = (successful / total_queries * 100) if total_queries > 0 else 0

        # Group by query type (base query_id without params)
        query_types: Dict[str, Dict[str, int]] = {}
        for log in all_logs:
            qid = log.get("query_id", "unknown")
            if qid not in query_types:
                query_types[qid] = {"success": 0, "fail": 0}
            if log.get("status"):
                query_types[qid]["success"] += 1
            else:
                query_types[qid]["fail"] += 1

        # Generate summary
        lines = [
            "=" * 80,
            "BESPOKE REPAIRS QUERY RUN SUMMARY",
            "=" * 80,
            "",
            f"Run Directory:  {self.run_dir}",
            f"Run Timestamp:  {self.run_timestamp}",
            f"Total Duration: {_format_duration(run_elapsed)}",
            "",
            "-" * 40,
            "GLOBAL STATISTICS",
            "-" * 40,
            f"  Total Queries:   {total_queries}",
            f"  Successful:      {successful}",
            f"  Failed:          {failed}",
            f"  Success Rate:    {success_rate:.1f}%",
            "",
        ]

        # Add per-query-type statistics
        if query_types:
            lines.append("-" * 40)
            lines.append("PER-METRIC STATISTICS")
            lines.append("-" * 40)
            for qid in sorted(query_types.keys()):
                stats = query_types[qid]
                total = stats["success"] + stats["fail"]
                rate = (stats["success"] / total * 100) if total > 0 else 0
                status_icon = (
                    "✓" if stats["fail"] == 0 else "✗" if stats["success"] == 0 else "◐"
                )
                lines.append(
                    f"  {status_icon} {qid}: {stats['success']}/{total} passed ({rate:.0f}%)",
                )
            lines.append("")

        # Add detailed query list
        lines.append("-" * 40)
        lines.append("QUERY DETAILS")
        lines.append("-" * 40)

        # First show successful queries
        successful_logs = [log for log in all_logs if log.get("status") is True]
        failed_logs = [log for log in all_logs if log.get("status") is False]

        if successful_logs:
            lines.append("")
            lines.append(f"  PASSED ({len(successful_logs)}):")
            for log in successful_logs:
                qid = log.get("query_id", "unknown")
                duration = log.get("duration", "N/A")
                params = log.get("params", {})
                params_str = ", ".join(
                    f"{k}={v}" for k, v in params.items() if k != "tools"
                )
                total = log.get("total")

                lines.append(f"    ✓ {qid}")
                if params_str:
                    lines.append(f"        Params: {params_str}")
                lines.append(f"        Duration: {duration}")
                if total is not None:
                    lines.append(f"        Total: {total}")

        if failed_logs:
            lines.append("")
            lines.append(f"  FAILED ({len(failed_logs)}):")
            for log in failed_logs:
                qid = log.get("query_id", "unknown")
                duration = log.get("duration", "N/A")
                params = log.get("params", {})
                params_str = ", ".join(
                    f"{k}={v}" for k, v in params.items() if k != "tools"
                )
                error = log.get("error", "Unknown error")

                lines.append(f"    ✗ {qid}")
                if params_str:
                    lines.append(f"        Params: {params_str}")
                lines.append(f"        Duration: {duration}")
                # Show first line of error in summary
                error_first_line = error.split("\n")[0] if error else "Unknown error"
                lines.append(f"        Error: {error_first_line}")
                lines.append(f"        Log: {log.get('filename', 'N/A')}")

        lines.append("")
        lines.append("=" * 80)
        lines.append("END OF RUN SUMMARY")
        lines.append("=" * 80)

        # Write summary file
        summary_path = self.run_dir / "_run_summary.log"
        try:
            summary_path.write_text("\n".join(lines), encoding="utf-8")
            return summary_path
        except Exception as e:
            print(f"⚠️  Warning: Could not write summary file: {e}")
            return None

    def get_terminal_summary(self) -> str:
        """
        Get a concise summary suitable for terminal output.

        Scans directory for ALL logs to provide accurate aggregate stats.

        Returns:
            Formatted summary string.
        """
        run_elapsed = time.perf_counter() - (self.run_start or 0)

        # Use directory scan if available for accurate totals
        all_logs = self._scan_directory_logs()

        if all_logs:
            # Use aggregated data from all log files
            successful = sum(1 for log in all_logs if log.get("status") is True)
            failed = sum(1 for log in all_logs if log.get("status") is False)
            total = len(all_logs)

            lines = [
                "",
                "─" * 60,
                f"📊 Run Summary: {successful}/{total} succeeded, {failed} failed",
                f"⏱️  Total time: {_format_duration(run_elapsed)}",
                "─" * 60,
            ]

            # Show failed queries prominently
            for log in all_logs:
                if log.get("status") is False:
                    qid = log.get("query_id", "unknown")
                    duration = log.get("duration", "N/A")
                    lines.append(f"  ✗ {qid}: {duration}")

            # Show passed count
            if successful > 0:
                lines.append(f"  ✓ {successful} queries passed")
        elif self.entries:
            # Fallback to in-memory entries
            successful = sum(1 for e in self.entries if e.success)
            failed = len(self.entries) - successful

            lines = [
                "",
                "─" * 60,
                f"📊 Run Summary: {successful} succeeded, {failed} failed",
                f"⏱️  Total time: {_format_duration(run_elapsed)}",
                "─" * 60,
            ]

            for entry in self.entries:
                status = "✓" if entry.success else "✗"
                lines.append(
                    f"  {status} {entry.query_id}: {_format_duration(entry.elapsed)}",
                )
        else:
            return "No queries executed."

        lines.append("─" * 60)
        if self.run_dir:
            lines.append(f"📁 Full logs: {self.run_dir}")

        return "\n".join(lines)

    def __enter__(self) -> "QueryLogger":
        """Context manager entry."""
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.finish_run()


def generate_aggregate_summary(run_dir: Path) -> Optional[Path]:
    """
    Generate an aggregate summary for all query logs in a directory.

    This is a standalone function that can be called after all parallel
    queries have completed to create a comprehensive summary.

    Args:
        run_dir: Path to the run directory containing query logs.

    Returns:
        Path to the summary file, or None if generation failed.
    """
    logger = QueryLogger()
    logger.run_dir = run_dir
    logger.run_timestamp = run_dir.name
    logger.run_start = time.perf_counter()  # Not accurate but needed
    logger._initialized = True

    return logger.finish_run()


def main():
    """CLI entry point for generating aggregate summaries."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate aggregate summary for query logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "run_dir",
        type=str,
        help="Path to the run directory containing query logs",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        dest="print_summary",
        help="Print summary to stdout instead of writing to file",
    )

    args = parser.parse_args()
    run_dir = Path(args.run_dir)

    if not run_dir.exists():
        print(f"Error: Directory not found: {run_dir}")
        return 1

    if not run_dir.is_dir():
        print(f"Error: Not a directory: {run_dir}")
        return 1

    # Generate summary
    logger = QueryLogger()
    logger.run_dir = run_dir
    logger.run_timestamp = run_dir.name
    logger.run_start = time.perf_counter()
    logger._initialized = True

    if args.print_summary:
        # Print to stdout
        all_logs = logger._scan_directory_logs()
        if not all_logs:
            print("No query logs found in directory.")
            return 1

        successful = sum(1 for log in all_logs if log.get("status") is True)
        failed = sum(1 for log in all_logs if log.get("status") is False)
        total = len(all_logs)

        print(f"\n{'=' * 60}")
        print(f"AGGREGATE SUMMARY: {run_dir.name}")
        print(f"{'=' * 60}")
        print(f"Total: {total} | Passed: {successful} | Failed: {failed}")
        print(f"Success Rate: {successful/total*100:.1f}%" if total > 0 else "N/A")
        print(f"{'=' * 60}\n")

        if failed > 0:
            print("FAILED QUERIES:")
            for log in all_logs:
                if log.get("status") is False:
                    print(f"  ✗ {log.get('query_id', 'unknown')}")
                    error = log.get("error", "")
                    if error:
                        first_line = error.split("\n")[0]
                        print(f"    Error: {first_line}")
            print()

        return 0 if failed == 0 else 1
    else:
        # Write to file
        summary_path = logger.finish_run()
        if summary_path:
            print(f"Summary written to: {summary_path}")
            return 0
        else:
            print("Failed to generate summary.")
            return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
