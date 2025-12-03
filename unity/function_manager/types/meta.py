"""
Pydantic model for the Functions/Meta context.

Stores metadata about the primitives sync state.
"""

from pydantic import BaseModel, Field


class FunctionsMeta(BaseModel):
    """
    Metadata record for the Functions context.

    Currently stores the primitives sync hash to detect when
    primitive docstrings/signatures have changed.
    """

    meta_id: int = Field(
        1,
        description="Fixed ID for the single metadata row.",
    )
    primitives_hash: str = Field(
        "",
        description="Hash of all primitive signatures and docstrings.",
    )
