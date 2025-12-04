"""
Bespoke query definitions for Midland Heart repairs v0.

Import all query modules to trigger registration with @register decorator.
"""

from . import daily_repairs
from . import cross_reference

__all__ = [
    "daily_repairs",
    "cross_reference",
]
