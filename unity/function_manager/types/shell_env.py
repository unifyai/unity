from typing import Optional

from pydantic import BaseModel, Field


class ShellEnv(BaseModel):
    """
    Represents a shell environment containing CLI tool binaries.

    Each shell environment stores metadata about the tools it contains.
    The actual binary data is stored separately in ShellEnvBinary records
    to keep this table lightweight for listing operations.
    """

    shell_env_id: int = Field(
        ...,
        description="Unique auto-incrementing identifier for the shell environment.",
    )
    name: Optional[str] = Field(
        None,
        description="Human-readable name for the shell environment, e.g. 'data-tools'.",
    )
    platform: str = Field(
        ...,
        description=(
            "Platform identifier for the stored binaries, "
            "e.g. 'darwin-arm64', 'linux-x86_64'."
        ),
    )
    tools: str = Field(
        ...,
        description=(
            "JSON-encoded list of tool metadata objects. Each object has: "
            "'name' (str), 'size' (int, bytes), 'sha256' (str, hex digest)."
        ),
    )
    custom_hash: Optional[str] = Field(
        None,
        description=(
            "Hash of source-defined custom shell env for sync detection. "
            "None for user-added shell envs."
        ),
    )


class ShellEnvBinary(BaseModel):
    """
    Stores a single binary blob belonging to a ShellEnv.

    Separated from ShellEnv so that listing/querying shell envs
    does not pull multi-MB base64 blobs into memory.
    """

    binary_id: int = Field(
        ...,
        description="Unique auto-incrementing identifier for the binary record.",
    )
    shell_env_id: int = Field(
        ...,
        description="Foreign key referencing the parent ShellEnv.",
    )
    tool_name: str = Field(
        ...,
        description="Name of the tool binary, e.g. 'jq'.",
    )
    data: str = Field(
        ...,
        description="Base64-encoded binary content of the tool.",
    )
