from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class Function(BaseModel):
    """
    Represents a single Python function stored in the FunctionManager.

    Functions can be either user-defined (with implementation source code) or
    primitives (action methods from state managers with no stored implementation).
    """

    function_id: Optional[int] = Field(
        None,
        description=(
            "Unique identifier for the function. "
            "Auto-assigned for user functions, explicit stable IDs for primitives."
        ),
    )
    name: str = Field(..., description="The name of the function.")
    argspec: str = Field(
        ...,
        description="The function's signature, e.g., '(x: int, y: int) -> int'.",
    )
    docstring: str = Field("", description="The docstring of the function.")
    implementation: Optional[str] = Field(
        None,
        description=(
            "The full source code of the function. "
            "None for primitives (implementation lives in Python class)."
        ),
    )
    depends_on: List[str] = Field(
        [],
        description="A list of other functions that this function depends on.",
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

    verify: bool = Field(
        True,
        description=(
            "Whether the function should be verified by the Actor upon completion. "
            "If True, the Actor may check initial/final states or logs to ensure success. "
            "If verification fails, the Actor may reimplement and overwrite the function in the 'Functions' store."
        ),
    )

    # Primitive-specific fields
    is_primitive: bool = Field(
        False,
        description=(
            "Whether this is an action primitive (state manager method) rather than "
            "a user-defined function. Primitives have no stored implementation."
        ),
    )

    primitive_class: Optional[str] = Field(
        None,
        description="Fully-qualified class path for primitive execution routing.",
    )

    primitive_method: Optional[str] = Field(
        None,
        description="Method name on the primitive class.",
    )

    venv_id: Optional[int] = Field(
        None,
        description=(
            "Optional reference to a VirtualEnv.venv_id specifying which virtual "
            "environment to use when executing this function. If None, the function "
            "runs in the project's default environment."
        ),
    )

    # Source-defined custom function tracking
    custom_hash: Optional[str] = Field(
        None,
        description=(
            "Hash of source-defined custom function for sync detection. "
            "None for user-added functions or primitives. "
            "Present for functions defined in the custom/ folder."
        ),
    )
