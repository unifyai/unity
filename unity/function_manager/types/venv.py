from typing import Optional

from pydantic import BaseModel, Field


class VirtualEnv(BaseModel):
    """
    Represents a virtual environment configuration stored in the FunctionManager.

    Each virtual environment contains a pyproject.toml-style configuration that
    specifies the dependencies required to run functions that reference it.
    Functions without an explicit venv reference use the project's default environment.
    """

    venv_id: int = Field(
        ...,
        description="Unique auto-incrementing identifier for the virtual environment.",
    )
    name: Optional[str] = Field(
        None,
        description=(
            "Human-readable name for the venv. For custom venvs (from source), "
            "this is the filename without .toml extension."
        ),
    )
    venv: str = Field(
        ...,
        description=(
            "The raw pyproject.toml content defining the virtual environment's "
            "dependencies and configuration."
        ),
    )
    custom_hash: Optional[str] = Field(
        None,
        description=(
            "Hash of source-defined custom venv for sync detection. "
            "None for user-added venvs. "
            "Present for venvs defined in the custom/venvs/ folder."
        ),
    )
    authoring_assistant_id: Optional[int] = Field(
        None,
        description=(
            "Assistant id of the body that authored this catalog row. The "
            "on-disk venv materialization is per-body because the venv path "
            "derives from the active Unify context; this field only records "
            "which body first published the catalog entry inside the Hive."
        ),
    )
