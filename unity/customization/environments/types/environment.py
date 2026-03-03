from __future__ import annotations

from typing import Dict, List
from pydantic import BaseModel, Field, model_validator

UNASSIGNED = -1


class Environment(BaseModel):
    """Serialized environment definition.

    Stores everything needed to reconstruct a ``BaseEnvironment`` instance
    at actor construction time: source files, pip dependencies, and the
    ``module:attribute`` path that resolves to the live instance.
    """

    environment_id: int = Field(
        default=UNASSIGNED,
        description="Auto-incrementing unique identifier for the environment.",
        ge=UNASSIGNED,
    )
    name: str = Field(
        description="Human-readable name for the environment (e.g. 'colliers').",
        min_length=1,
        max_length=200,
    )
    env: str = Field(
        description=(
            "Module-attribute path to the BaseEnvironment instance, "
            "using the 'module:attribute' pattern (e.g. 'colliers_env:colliers_env')."
        ),
        min_length=3,
    )
    files: Dict[str, str] = Field(
        description=(
            "Mapping of filename to source code. "
            "Each key is a Python filename (e.g. 'service.py') and each value "
            "is the full source code for that file."
        ),
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description=(
            "List of pip dependency specifiers required by this environment "
            "(e.g. ['openpyxl>=3.1', 'numpy'])."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _inject_sentinel(cls, data: dict) -> dict:
        data.setdefault("environment_id", UNASSIGNED)
        return data

    def to_post_json(self) -> dict:
        exclude = {"environment_id"} if self.environment_id == UNASSIGNED else set()
        return self.model_dump(mode="json", exclude=exclude)
