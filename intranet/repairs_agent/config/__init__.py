"""
Configuration module for Repairs Analysis Agent.

This module provides business context configuration and prompt builders
for injecting domain knowledge into the CodeActActor.

Components:
    - prompt_builder: Builds BusinessContextPayload from FilePipelineConfig
"""

from .prompt_builder import build_repairs_business_context

__all__ = ["build_repairs_business_context"]
