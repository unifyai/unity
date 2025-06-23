from pydantic import BaseModel, Field
from typing import List


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
