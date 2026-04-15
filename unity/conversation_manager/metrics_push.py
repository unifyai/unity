"""Compatibility wrappers for deploy-specific metrics backends."""

from unity.deploy_runtime import init_metrics, shutdown_metrics

__all__ = ["init_metrics", "shutdown_metrics"]
