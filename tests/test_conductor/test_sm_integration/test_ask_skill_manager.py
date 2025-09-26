from __future__ import annotations

import asyncio

import pytest

from unity.conductor.simulated import SimulatedConductor
from tests.helpers import _handle_project
from tests.test_conductor.utils import (
    tool_names_from_messages,
    assistant_requested_tool_names,
)


MANAGER = "SimulatedSkillManager"


# Each case couples a user question with explicit simulation guidance that
# instructs the SimulatedFunctionManager about which kinds of functions to
# hallucinate. This makes the simulation deterministic and domain-appropriate.
CASES: list[dict[str, str]] = [
    {
        "question": "What skills do you have for web browsing and extracting information?",
        "simulation": (
            "Catalogue focuses on web browsing and information extraction. Include functions such as: "
            "fetch_webpage(url: str) -> str; fetch_with_headers(url: str, headers: dict) -> str; "
            "parse_html(html: str) -> DOM; extract_links(html: str) -> list[str]; extract_text(html: str) -> str; "
            "crawl_site(url: str, max_pages: int = 10) -> list[dict]; summarize_text(text: str) -> str; "
            "is_allowed_by_robots(url: str) -> bool."
        ),
    },
    {
        "question": "Are you familiar with using PowerPoint?",
        "simulation": (
            "Catalogue emphasises PowerPoint/slide generation utilities. Include functions such as: "
            "create_presentation(title: str) -> str; add_slide(pres_id: str, layout: str = 'title') -> int; "
            "add_text_box(pres_id: str, slide_index: int, text: str) -> None; add_image(pres_id: str, slide_index: int, path: str) -> None; "
            "save_pptx(pres_id: str, path: str) -> str; convert_markdown_to_pptx(markdown: str) -> str; "
            "export_pdf(pres_id: str, path: str) -> str."
        ),
    },
    {
        "question": "Have you done much lead generation work before?",
        "simulation": (
            "Catalogue tailored to B2B lead generation. Include functions such as: "
            "find_company_domains(industry: str, region: str | None = None) -> list[str]; "
            "scrape_company_profiles(domains: list[str]) -> list[dict]; enrich_contacts(domain: str) -> list[dict]; "
            "score_leads(leads: list[dict]) -> list[dict]; dedupe_leads(leads: list[dict]) -> list[dict]; "
            "export_csv(rows: list[dict], path: str) -> str."
        ),
    },
]


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES)
@_handle_project
async def test_skill_questions_use_only_skill_manager_tool(case: dict[str, str]):
    cond = SimulatedConductor(
        description=(
            "Assistant that explains its skills via a skills catalogue; other managers exist but are not needed for these queries.\n\n"
            f"Simulation guidance (functions to hallucinate): {case['simulation']}"
        ),
    )

    # Nudge routing deterministically to SkillManager by explicitly stating the surface
    handle = await cond.ask(
        case["question"],
        _return_reasoning_steps=True,
    )

    answer, messages = await asyncio.wait_for(handle.result(), timeout=300)
    assert isinstance(answer, str) and answer.strip(), "Answer should be non-empty"

    # The only executed tool must be SimulatedSkillManager.ask and it should run exactly once
    executed_list = tool_names_from_messages(messages, MANAGER)
    executed = set(executed_list)
    assert executed, "Expected at least one tool call to occur"
    assert executed == {
        "SimulatedSkillManager_ask",
    }, f"Only SimulatedSkillManager_ask should run, saw: {sorted(executed)}"
    assert (
        executed_list.count("SimulatedSkillManager_ask") == 1
    ), f"Expected exactly one SimulatedSkillManager_ask call, saw order: {executed_list}"

    # Additionally confirm that any assistant tool selection(s) referenced only that tool
    requested = set(assistant_requested_tool_names(messages, MANAGER))
    assert requested, "Assistant should have requested at least one tool"
    assert requested <= {
        "SimulatedSkillManager_ask",
    }, f"Assistant should request only SimulatedSkillManager_ask, saw: {sorted(requested)}"
