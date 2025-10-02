from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class Function(BaseModel):
    """
    Represents a single Python function stored in the FunctionManager.
    """

    function_id: int = Field(..., description="Unique identifier for the function.")
    name: str = Field(..., description="The name of the function.")
    argspec: str = Field(
        ...,
        description="The function's signature, e.g., '(x: int, y: int) -> int'.",
    )
    docstring: str = Field("", description="The docstring of the function.")
    implementation: str = Field(
        ...,
        description="The full source code of the function.",
    )
    calls: List[str] = Field(
        [],
        description="A list of other functions called by this function.",
    )
    embedding_text: str = Field(
        ...,
        description="The text used to generate the function's embedding.",
    )
    precondition: Optional[Dict[str, Any]] = Field(
        None,
        description="A dictionary representing the state required before the function can be run, e.g., {'url': '...'}.",
    )

    guidance_ids: List[int] = Field(
        default_factory=list,
        description=(
            "List of Guidance.guidance_id values that reference this function; "
            "represents the inverse many-to-many relationship."
        ),
    )
