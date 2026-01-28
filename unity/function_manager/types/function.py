from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal


class Function(BaseModel):
    """
    Represents a function stored in the FunctionManager.

    Functions can be written in multiple languages (Python, Bash, Zsh, Sh, PowerShell)
    and can be either user-defined (with implementation source code) or primitives
    (action methods from state managers with no stored implementation).
    """

    function_id: Optional[int] = Field(
        None,
        description=(
            "Unique identifier for the function. "
            "Auto-assigned for user functions, explicit stable IDs for primitives."
        ),
    )
    language: Literal["python", "bash", "zsh", "sh", "powershell"] = Field(
        "python",
        description=(
            "The language/interpreter for this function. "
            "Defaults to 'python' for backward compatibility."
        ),
    )
    name: str = Field(..., description="The name of the function.")
    argspec: str = Field(
        ...,
        description=(
            "The function's signature. Format varies by language: "
            "Python: '(x: int, y: int) -> int'. "
            "Shell: '(input_file output_file --verbose)' or positional description."
        ),
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
            "Optional reference to a VirtualEnv.venv_id specifying which Python virtual "
            "environment to use when executing this function. Only applicable when "
            "language='python'. Ignored for other languages. If None, the function "
            "runs in the project's default Python environment."
        ),
    )

    windows_os_required: bool = Field(
        False,
        description=(
            "Whether this function requires execution on a Windows OS. "
            "When True and the assistant has desktop_mode='windows', "
            "execution routes to the remote Windows VM. "
            "Typically used for functions that depend on Windows-only libraries "
            "like xlwings or other COM automation tools."
        ),
    )

    # Remote execution data transfer
    data_required: List[str] = Field(
        default_factory=list,
        description=(
            "List of items specifying data to upload before remote execution. "
            "Each item is either: (1) an argument name whose runtime value is a local "
            "file/directory path, or (2) a static absolute path string (starts with '/'). "
            "At runtime, paths are uploaded to the remote VM preserving full directory "
            "structure under C:\\Unity, and argument values are rewritten to remote equivalents."
        ),
    )

    data_output: List[str] = Field(
        default_factory=list,
        description=(
            "List of argument names whose values are output file/directory paths "
            "that should be downloaded from the remote VM after execution completes. "
            "The remote paths are downloaded back to their corresponding local paths."
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
