import asyncio
from dotenv import load_dotenv
import logging
import unify

load_dotenv()
unify.activate("KnowledgeSandbox")
LG = logging.getLogger("knowledge_manager_integration")

from unity.knowledge_manager.knowledge_manager import KnowledgeManager
import unity.service
from scenario_builder import ScenarioBuilder
from scenario_store import ScenarioStore


async def build_scenario():
    # prepare Unify context
    unify.set_trace_context("Traces")
    for table in unify.get_contexts(prefix="Knowledge").keys():
        unify.delete_context(table)
    unify.create_context("Contacts")
    if "Traces" in unify.get_contexts():
        unify.delete_context("Traces")
    unify.create_context("Traces")

    # logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    LG.setLevel(logging.INFO)

    # manager & transcript vault
    km = KnowledgeManager()
    store = ScenarioStore()

    km = KnowledgeManager()
    scenario_text = (
        "Generate 20 diverse facts about electric-vehicle manufacturers. "
        "Cover launch years, battery capacities, warranty terms and sales "
        "figures in different regions.  Include numbers, dates and named "
        "entities so the schema has to evolve."
    )
    description = scenario_text + (
        "\nTry to batch actions – each `store` can add multiple rows/columns "
        "and `retrieve` can verify to avoid duplication."
    )

    builder = ScenarioBuilder(
        description=description,
        tools={
            "update": km.update,
            "ask": km.ask,
        },
    )

    try:
        await builder.create()
    except Exception as exc:
        raise RuntimeError(f"LLM seeding via ScenarioBuilder failed. {exc}")

    LG.info("[seed] done.")

    store.save_named("default", scenario_text)
    LG.info(f"[seed] transcript saved as {scenario_text}.")


async def main():
    # build scenario
    # uncomment if scenario is not built yet
    # await build_scenario()

    # run user manager
    await unity.service.start(manager_name="knowledge")


if __name__ == "__main__":
    asyncio.run(main())
