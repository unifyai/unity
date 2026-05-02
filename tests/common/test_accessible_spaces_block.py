"""Tests for the shared-space prompt guidance block."""

from unity.common.accessible_spaces_block import build_accessible_spaces_block
from unity.session_details import SpaceSummary


def test_renders_solo_assistant_block() -> None:
    """Solo assistants still receive explicit memory routing guidance."""

    block = build_accessible_spaces_block([])

    assert (
        block
        == """Accessible shared spaces

You are a member of the following shared spaces. Each space's description names the team, domain, or scope of work it exists for; use it to decide whether a write belongs in that space versus personal memory.

- personal: your private memory. This is the privacy floor; content here is visible only to you.
- shared: No shared spaces are currently available. Use personal memory unless the user explicitly asks you to create or join a shared space.

Routing rules:
- Use personal memory for private notes, ambiguous ownership, or anything that should not be visible to collaborators.
- Use a shared space only when the content clearly belongs to that space's named team, domain, or scope of work.
- If the user asks you to write something more broadly than personal memory and the right space is unclear, ask a brief clarifying question before writing.
- Never invent a shared space destination. Use only personal memory or one of the listed space ids."""
    )


def test_renders_multi_membership_in_order() -> None:
    """Shared memberships are rendered by ascending space id."""

    block = build_accessible_spaces_block(
        [
            SpaceSummary(
                space_id=9,
                name="Marketing",
                description="Marketing campaigns and analytics for the brand team.",
            ),
            SpaceSummary(
                space_id=3,
                name="Repairs",
                description="South-East repairs patch daily operations.",
            ),
        ],
    )

    assert '- space:3 "Repairs" - South-East repairs patch daily operations.' in block
    assert (
        '- space:9 "Marketing" - Marketing campaigns and analytics for the brand team.'
        in block
    )
    assert block.index("space:3") < block.index("space:9")
