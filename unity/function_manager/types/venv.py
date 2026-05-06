from typing import Optional

from pydantic import Field

from unity.common.authorship import AuthoredRow


class VirtualEnv(AuthoredRow):
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
