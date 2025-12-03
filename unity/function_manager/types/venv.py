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
    venv: str = Field(
        ...,
        description=(
            "The raw pyproject.toml content defining the virtual environment's "
            "dependencies and configuration."
        ),
    )
