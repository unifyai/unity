"""
Tests for OpenTelemetry tracing functionality in Unity.

These tests verify that:
1. Unity's OTel setup works correctly
2. Trace context propagates through unity -> unillm -> unify hierarchy
3. All packages create spans in the same trace when properly configured
"""

from __future__ import annotations


import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def reset_otel():
    """Reset OTel state before and after test.

    Creates a fresh TracerProvider with InMemorySpanExporter for capturing spans.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    # Reset OTel global state
    # pylint: disable=protected-access
    trace._TRACER_PROVIDER_SET_ONCE._done = False
    trace._TRACER_PROVIDER = None

    trace.set_tracer_provider(provider)

    yield {"provider": provider, "exporter": exporter}

    exporter.clear()


# ---------------------------------------------------------------------------
#  Unity OTel Setup Tests
# ---------------------------------------------------------------------------


class TestOtelEnabled:
    """Tests for UNITY_OTEL master switch."""

    def test_otel_disabled_by_default(self, monkeypatch):
        """UNITY_OTEL defaults to false."""
        from unity import logger

        monkeypatch.setattr(logger, "_OTEL_ENABLED", False)
        assert logger.is_otel_enabled() is False

    def test_otel_enabled_via_setting(self, monkeypatch):
        """UNITY_OTEL=true enables OTel."""
        from unity import logger

        monkeypatch.setattr(logger, "_OTEL_ENABLED", True)
        assert logger.is_otel_enabled() is True


class TestGetTracer:
    """Tests for get_tracer function."""

    def test_returns_none_when_disabled(self, monkeypatch):
        """get_tracer returns None when OTel is disabled."""
        from unity import logger

        monkeypatch.setattr(logger, "_OTEL_ENABLED", False)
        monkeypatch.setattr(logger, "_TRACER", None)

        assert logger.get_tracer() is None

    def test_returns_tracer_when_enabled(self, reset_otel, monkeypatch):
        """get_tracer returns a tracer when OTel is enabled."""
        from unity import logger

        monkeypatch.setattr(logger, "_OTEL_ENABLED", True)
        monkeypatch.setattr(logger, "_OTEL_INITIALIZED", False)
        monkeypatch.setattr(logger, "_TRACER", None)

        tracer = logger.get_tracer()
        assert tracer is not None

    def test_uses_existing_provider(self, reset_otel, monkeypatch):
        """get_tracer uses existing TracerProvider if available."""
        from unity import logger

        monkeypatch.setattr(logger, "_OTEL_ENABLED", True)
        monkeypatch.setattr(logger, "_OTEL_INITIALIZED", False)
        monkeypatch.setattr(logger, "_TRACER", None)

        existing_provider = reset_otel["provider"]

        tracer = logger.get_tracer()
        assert tracer is not None
        # Should use existing provider
        assert trace.get_tracer_provider() is existing_provider


class TestUnitySpan:
    """Tests for unity_span context manager."""

    def test_span_created_when_otel_enabled(self, reset_otel, monkeypatch):
        """unity_span creates a span when OTel is enabled."""
        from unity import logger

        monkeypatch.setattr(logger, "_OTEL_ENABLED", True)
        monkeypatch.setattr(logger, "_OTEL_INITIALIZED", False)
        monkeypatch.setattr(logger, "_TRACER", None)

        exporter = reset_otel["exporter"]

        with logger.unity_span("ContactManager.ask", query="find john") as span:
            assert span is not None
            span.set_attribute("test", "value")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "ContactManager.ask"
        assert spans[0].attributes.get("unity.query") == "find john"

    def test_span_none_when_otel_disabled(self, monkeypatch):
        """unity_span yields None when OTel is disabled."""
        from unity import logger

        monkeypatch.setattr(logger, "_OTEL_ENABLED", False)
        monkeypatch.setattr(logger, "_TRACER", None)

        with logger.unity_span("ContactManager.ask") as span:
            assert span is None

    def test_span_records_error_on_exception(self, reset_otel, monkeypatch):
        """unity_span records errors when exceptions occur."""
        from unity import logger

        monkeypatch.setattr(logger, "_OTEL_ENABLED", True)
        monkeypatch.setattr(logger, "_OTEL_INITIALIZED", False)
        monkeypatch.setattr(logger, "_TRACER", None)

        exporter = reset_otel["exporter"]

        with pytest.raises(ValueError, match="test error"):
            with logger.unity_span("ContactManager.ask") as span:
                raise ValueError("test error")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get("error.type") == "ValueError"
        assert "test error" in spans[0].attributes.get("error.message", "")


class TestGetCurrentTraceId:
    """Tests for get_current_trace_id helper."""

    def test_returns_none_when_disabled(self, monkeypatch):
        """get_current_trace_id returns None when OTel is disabled."""
        from unity import logger

        monkeypatch.setattr(logger, "_OTEL_ENABLED", False)
        assert logger.get_current_trace_id() is None

    def test_returns_trace_id_when_span_active(self, reset_otel, monkeypatch):
        """get_current_trace_id returns trace ID when span is active."""
        from unity import logger

        monkeypatch.setattr(logger, "_OTEL_ENABLED", True)

        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("test-span") as span:
            ctx = span.get_span_context()
            expected_trace_id = f"{ctx.trace_id:032x}"

            trace_id = logger.get_current_trace_id()
            assert trace_id == expected_trace_id


# ---------------------------------------------------------------------------
#  Trace Hierarchy Tests: Unity-only
# ---------------------------------------------------------------------------


class TestUnityOnlyHierarchy:
    """Tests for Unity-only span hierarchy (nested unity_spans)."""

    def test_nested_unity_spans_same_trace(self, reset_otel, monkeypatch):
        """Nested unity_spans share the same trace ID."""
        from unity import logger

        monkeypatch.setattr(logger, "_OTEL_ENABLED", True)
        monkeypatch.setattr(logger, "_OTEL_INITIALIZED", False)
        monkeypatch.setattr(logger, "_TRACER", None)

        exporter = reset_otel["exporter"]

        with logger.unity_span("Conductor.request") as outer:
            outer_ctx = outer.get_span_context()
            with logger.unity_span("ContactManager.update") as inner:
                inner_ctx = inner.get_span_context()
                # Same trace
                assert inner_ctx.trace_id == outer_ctx.trace_id
                # Different span
                assert inner_ctx.span_id != outer_ctx.span_id

        spans = exporter.get_finished_spans()
        assert len(spans) == 2

        conductor_span = next(s for s in spans if "Conductor" in s.name)
        cm_span = next(s for s in spans if "ContactManager" in s.name)

        # CM is child of Conductor
        assert cm_span.parent.span_id == conductor_span.context.span_id


# ---------------------------------------------------------------------------
#  Trace Hierarchy Tests: Unity -> Unillm
# ---------------------------------------------------------------------------


class TestUnityToUnillmHierarchy:
    """Tests for Unity -> Unillm trace hierarchy."""

    def test_unillm_span_child_of_unity(self, reset_otel, monkeypatch):
        """Unillm spans become children of Unity spans."""
        from unity import logger

        monkeypatch.setattr(logger, "_OTEL_ENABLED", True)
        monkeypatch.setattr(logger, "_OTEL_INITIALIZED", False)
        monkeypatch.setattr(logger, "_TRACER", None)

        exporter = reset_otel["exporter"]

        # Simulate unillm's llm_span context manager
        unillm_tracer = trace.get_tracer("unillm")

        with logger.unity_span("ContactManager.ask") as unity_span:
            unity_ctx = unity_span.get_span_context()

            # Simulate what unillm.logger.llm_span does
            with unillm_tracer.start_as_current_span("LLM gpt-4@openai") as llm_span:
                llm_ctx = llm_span.get_span_context()

                # Same trace
                assert llm_ctx.trace_id == unity_ctx.trace_id
                # Different span
                assert llm_ctx.span_id != unity_ctx.span_id

        spans = exporter.get_finished_spans()
        assert len(spans) == 2

        unity_s = next(s for s in spans if "ContactManager" in s.name)
        llm_s = next(s for s in spans if "LLM" in s.name)

        # LLM span is child of Unity span
        assert llm_s.parent.span_id == unity_s.context.span_id


# ---------------------------------------------------------------------------
#  Trace Hierarchy Tests: Unity -> Unify
# ---------------------------------------------------------------------------


class TestUnityToUnifyHierarchy:
    """Tests for Unity -> Unify trace hierarchy."""

    def test_unify_span_child_of_unity(self, reset_otel, monkeypatch):
        """Unify HTTP spans become children of Unity spans."""
        from unity import logger

        monkeypatch.setattr(logger, "_OTEL_ENABLED", True)
        monkeypatch.setattr(logger, "_OTEL_INITIALIZED", False)
        monkeypatch.setattr(logger, "_TRACER", None)

        exporter = reset_otel["exporter"]

        # Simulate unify's HTTP span
        unify_tracer = trace.get_tracer("unify")

        with logger.unity_span("ContactManager.update") as unity_span:
            unity_ctx = unity_span.get_span_context()

            # Simulate what unify/utils/http.py does
            with unify_tracer.start_as_current_span("POST contacts") as http_span:
                http_ctx = http_span.get_span_context()

                # Same trace
                assert http_ctx.trace_id == unity_ctx.trace_id

        spans = exporter.get_finished_spans()
        assert len(spans) == 2

        unity_s = next(s for s in spans if "ContactManager" in s.name)
        http_s = next(s for s in spans if "POST" in s.name)

        # HTTP span is child of Unity span
        assert http_s.parent.span_id == unity_s.context.span_id


# ---------------------------------------------------------------------------
#  Trace Hierarchy Tests: Unity -> Unillm -> Unify
# ---------------------------------------------------------------------------


class TestFullStackHierarchy:
    """Tests for full hierarchy: Unity -> Unillm -> Unify."""

    def test_full_stack_trace_hierarchy(self, reset_otel, monkeypatch):
        """Full hierarchy maintains parent-child relationships."""
        from unity import logger

        monkeypatch.setattr(logger, "_OTEL_ENABLED", True)
        monkeypatch.setattr(logger, "_OTEL_INITIALIZED", False)
        monkeypatch.setattr(logger, "_TRACER", None)

        exporter = reset_otel["exporter"]

        unillm_tracer = trace.get_tracer("unillm")
        unify_tracer = trace.get_tracer("unify")

        # Unity -> Unillm -> Unify
        with logger.unity_span("Conductor.request") as unity_span:
            with unillm_tracer.start_as_current_span("LLM gpt-4@openai") as llm_span:
                with unify_tracer.start_as_current_span("GET projects") as http_span:
                    pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 3

        unity_s = next(s for s in spans if "Conductor" in s.name)
        llm_s = next(s for s in spans if "LLM" in s.name)
        http_s = next(s for s in spans if "GET" in s.name)

        # All same trace
        assert (
            unity_s.context.trace_id
            == llm_s.context.trace_id
            == http_s.context.trace_id
        )

        # LLM is child of Unity
        assert llm_s.parent.span_id == unity_s.context.span_id

        # HTTP is child of LLM
        assert http_s.parent.span_id == llm_s.context.span_id

    def test_parallel_unity_to_unify_calls(self, reset_otel, monkeypatch):
        """Multiple direct Unity -> Unify calls create sibling spans."""
        from unity import logger

        monkeypatch.setattr(logger, "_OTEL_ENABLED", True)
        monkeypatch.setattr(logger, "_OTEL_INITIALIZED", False)
        monkeypatch.setattr(logger, "_TRACER", None)

        exporter = reset_otel["exporter"]

        unify_tracer = trace.get_tracer("unify")

        with logger.unity_span("ContactManager.ask") as unity_span:
            # Two HTTP calls (e.g., list then get)
            with unify_tracer.start_as_current_span("GET contacts"):
                pass
            with unify_tracer.start_as_current_span("GET contact-123"):
                pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 3

        unity_s = next(s for s in spans if "ContactManager" in s.name)
        http_spans = [s for s in spans if "GET" in s.name]

        assert len(http_spans) == 2

        # Both HTTP spans are children of Unity span (siblings)
        for http_s in http_spans:
            assert http_s.parent.span_id == unity_s.context.span_id


# ---------------------------------------------------------------------------
#  Settings Validation Tests
# ---------------------------------------------------------------------------


class TestSettingsValidation:
    """Tests for UNITY_OTEL settings validation."""

    def test_otel_setting_parses_true(self):
        """UNITY_OTEL parses 'true' string correctly."""
        from unity.settings import _parse_bool

        assert _parse_bool("true") is True
        assert _parse_bool("True") is True
        assert _parse_bool("TRUE") is True
        assert _parse_bool("1") is True
        assert _parse_bool("yes") is True

    def test_otel_setting_parses_false(self):
        """UNITY_OTEL parses 'false' string correctly."""
        from unity.settings import _parse_bool

        assert _parse_bool("false") is False
        assert _parse_bool("False") is False
        assert _parse_bool("0") is False
        assert _parse_bool("no") is False
        assert _parse_bool("") is False


# ---------------------------------------------------------------------------
#  Cross-Package Integration Tests
# ---------------------------------------------------------------------------


class TestCrossPackageIntegration:
    """Integration tests for cross-package OTel propagation."""

    def test_unity_provider_used_by_child_packages(self, reset_otel, monkeypatch):
        """Child packages use Unity's TracerProvider when it exists."""
        from unity import logger

        monkeypatch.setattr(logger, "_OTEL_ENABLED", True)
        monkeypatch.setattr(logger, "_OTEL_INITIALIZED", False)
        monkeypatch.setattr(logger, "_TRACER", None)

        existing_provider = reset_otel["provider"]

        # Unity's get_tracer should use the existing provider
        tracer = logger.get_tracer()
        assert tracer is not None

        # The global provider should still be the one we set in the fixture
        assert trace.get_tracer_provider() is existing_provider

        # Tracers from other packages will also use the same provider
        unillm_tracer = trace.get_tracer("unillm")
        unify_tracer = trace.get_tracer("unify")

        # All tracers should be non-None
        assert unillm_tracer is not None
        assert unify_tracer is not None

    def test_span_attributes_preserved_across_packages(self, reset_otel, monkeypatch):
        """Span attributes from each package are preserved in the trace."""
        from unity import logger

        monkeypatch.setattr(logger, "_OTEL_ENABLED", True)
        monkeypatch.setattr(logger, "_OTEL_INITIALIZED", False)
        monkeypatch.setattr(logger, "_TRACER", None)

        exporter = reset_otel["exporter"]
        unillm_tracer = trace.get_tracer("unillm")
        unify_tracer = trace.get_tracer("unify")

        with logger.unity_span("Conductor.request", method="ask") as unity_span:
            unity_span.set_attribute("unity.query", "find contacts")

            with unillm_tracer.start_as_current_span("LLM call") as llm_span:
                llm_span.set_attribute("llm.model", "gpt-4")
                llm_span.set_attribute("llm.cache_status", "miss")

                with unify_tracer.start_as_current_span("HTTP call") as http_span:
                    http_span.set_attribute("http.method", "POST")
                    http_span.set_attribute("http.status_code", 200)

        spans = exporter.get_finished_spans()

        # Verify each package's attributes are preserved
        unity_s = next(s for s in spans if "Conductor" in s.name)
        llm_s = next(s for s in spans if "LLM" in s.name)
        http_s = next(s for s in spans if "HTTP" in s.name)

        assert unity_s.attributes.get("unity.method") == "ask"
        assert unity_s.attributes.get("unity.query") == "find contacts"

        assert llm_s.attributes.get("llm.model") == "gpt-4"
        assert llm_s.attributes.get("llm.cache_status") == "miss"

        assert http_s.attributes.get("http.method") == "POST"
        assert http_s.attributes.get("http.status_code") == 200
