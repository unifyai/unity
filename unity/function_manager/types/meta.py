"""
Pydantic model for the Functions/Meta context.

Stores metadata about the primitives, custom venvs, and custom functions sync state.
"""

from typing import Dict

from pydantic import BaseModel, Field


class FunctionsMeta(BaseModel):
    """
    Metadata record for the Functions context.

    Stores sync hashes to detect when primitives, custom venvs, or custom functions
    have changed and need re-synchronization.
    """

    meta_id: int = Field(
        1,
        description="Fixed ID for the single metadata row.",
    )
    primitives_hash_by_manager: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Per-manager hash of primitive signatures and docstrings. "
            "Keys are manager aliases (e.g., 'files', 'contacts'). "
            "Enables scoped primitive sync without global recomputation."
        ),
    )
    custom_venvs_hash: str = Field(
        "",
        description="Hash of all source-defined custom virtual environments.",
    )
    custom_functions_hash: str = Field(
        "",
        description="Hash of all source-defined custom function signatures.",
    )
