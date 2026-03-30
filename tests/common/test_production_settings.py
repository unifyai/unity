"""
Tests for ProductionSettings LLM provider validation.

Verifies that unity.init() hard-fails when LLM provider credentials are missing
and UNITY_VALIDATE_LLM_PROVIDERS is enabled (the default).
"""

import pytest

from unity.settings import ProductionSettings


class TestLLMProviderValidation:
    """Tests for validate_llm_providers method."""

    def test_validation_fails_when_all_credentials_missing(self):
        """Validation raises RuntimeError when no credentials are set."""
        settings = ProductionSettings(
            UNITY_VALIDATE_LLM_PROVIDERS=True,
            OPENAI_API_KEY="",
            ANTHROPIC_API_KEY="",
        )
        with pytest.raises(RuntimeError) as exc_info:
            settings.validate_llm_providers()

        error_msg = str(exc_info.value)
        assert "At least one LLM provider credential is required" in error_msg

    def test_validation_passes_when_one_credential_provided(self):
        """Validation succeeds when at least one credential is set."""
        settings = ProductionSettings(
            UNITY_VALIDATE_LLM_PROVIDERS=True,
            OPENAI_API_KEY="sk-test",
            ANTHROPIC_API_KEY="",
        )
        settings.validate_llm_providers()

    def test_validation_passes_when_all_credentials_provided(self):
        """Validation succeeds when all credentials are set."""
        settings = ProductionSettings(
            UNITY_VALIDATE_LLM_PROVIDERS=True,
            OPENAI_API_KEY="sk-test-openai",
            ANTHROPIC_API_KEY="sk-ant-test",
        )
        # Should not raise
        settings.validate_llm_providers()

    def test_validation_skipped_when_disabled(self):
        """Validation is skipped when UNITY_VALIDATE_LLM_PROVIDERS=False."""
        settings = ProductionSettings(
            UNITY_VALIDATE_LLM_PROVIDERS=False,
            OPENAI_API_KEY="",
            ANTHROPIC_API_KEY="",
        )
        # Should not raise even with empty credentials
        settings.validate_llm_providers()

    def test_validation_enabled_by_default(self):
        """UNITY_VALIDATE_LLM_PROVIDERS defaults to True in code."""
        # Verify the class-level default is True (env vars may override at runtime)
        field_info = ProductionSettings.model_fields["UNITY_VALIDATE_LLM_PROVIDERS"]
        assert field_info.default is True
