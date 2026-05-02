"""Render shared-space membership guidance for runtime prompts."""

from __future__ import annotations

from unity.session_details import SpaceSummary

ACCESSIBLE_SPACES_MAX_DESCRIPTION_LENGTH = 1000
ACCESSIBLE_SPACES_HEADER = "Accessible shared spaces"
_NO_SHARED_SPACES_TEXT = (
    "No shared spaces are currently available. Use personal memory unless the "
    "user explicitly asks you to create or join a shared space."
)
_INTRO_TEXT = (
    "You are a member of the following shared spaces. Each space's description "
    "names the team, domain, or scope of work it exists for; use it to decide "
    "whether a write belongs in that space versus personal memory."
)
_PERSONAL_BULLET = (
    "- personal: your private memory. This is the privacy floor; content here "
    "is visible only to you."
)
_ROUTING_RULES = """Routing rules:
- Use personal memory for private notes, ambiguous ownership, or anything that should not be visible to collaborators.
- Use a shared space only when the content clearly belongs to that space's named team, domain, or scope of work.
- If the user asks you to write something more broadly than personal memory and the right space is unclear, ask a brief clarifying question before writing.
- Never invent a shared space destination. Use only personal memory or one of the listed space ids."""


def build_accessible_spaces_block(space_summaries: list[SpaceSummary]) -> str:
    """Render the shared-space guidance block for system prompts."""

    lines = [
        ACCESSIBLE_SPACES_HEADER,
        "",
        _INTRO_TEXT,
        "",
        _PERSONAL_BULLET,
    ]
    if space_summaries:
        for summary in sorted(space_summaries, key=lambda item: item.space_id):
            description = _bounded_description(summary.description)
            lines.append(
                f'- space:{summary.space_id} "{summary.name}" - {description}',
            )
    else:
        lines.append(f"- shared: {_NO_SHARED_SPACES_TEXT}")
    lines.extend(["", _ROUTING_RULES])
    return "\n".join(lines)


def _bounded_description(description: str) -> str:
    if len(description) <= ACCESSIBLE_SPACES_MAX_DESCRIPTION_LENGTH:
        return description
    return f"{description[:ACCESSIBLE_SPACES_MAX_DESCRIPTION_LENGTH - 3]}..."
