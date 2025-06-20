import asyncio
from dotenv import load_dotenv
import logging
import unify

load_dotenv()
unify.activate("ContactManagerIntegration")
LG = logging.getLogger("contact_manager_integration")

from unity.contact_manager.contact_manager import ContactManager
import unity.service
from scenario_builder import ScenarioBuilder
from scenario_store import ScenarioStore


async def build_scenario():
    # prepare Unify context
    unify.set_trace_context("Traces")
    ctxs = unify.get_contexts()
    if "Contacts" in ctxs:
        unify.delete_context("Contacts")
    unify.create_context("Contacts")
    if "Traces" in ctxs:
        unify.delete_context("Traces")
    unify.create_context("Traces")

    # logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    LG.setLevel(logging.INFO)

    # manager & transcript vault
    store = ScenarioStore()

    # Obtain the transcript that seeds the scenario
    scenario_text = (
        "Could you please add 20 superheroes to the contact list? \n"
        "Could you fill all the default columns, but could you also add a unique "
        "superpower for each superhero and the city that they are protecting?"
    )
    LG.info(f"[seed] loaded transcript {scenario_text}")

    LG.info("[seed] building synthetic contacts – this can take 20-40 s…")

    cm = ContactManager()
    description = (
        scenario_text.strip()
        if scenario_text
        else (
            "Generate 10 realistic business contacts across EMEA, APAC and AMER. "
            "Each contact needs first_name, surname, email_address and phone_number. "
            "Also create custom columns with varying industries and locations."
        )
    )
    description += (
        "\nTry to get as much done as you can with each `update` and `ask` call. "
        "They can deal with complex multi-step requests just fine."
    )

    builder = ScenarioBuilder(
        description=description,
        tools={  # expose only the public surface
            "update": cm.update,
            "ask": cm.ask,  # allows the LLM to check for duplicates if it wishes
        },
    )

    try:
        await builder.create()
    except Exception as exc:
        raise (f"LLM seeding via ScenarioBuilder failed. {exc}")

    LG.info("[seed] done.")

    store.save_named("default", scenario_text)
    LG.info(f"[seed] transcript saved as {scenario_text}.")


async def main():
    # build scenario
    # uncomment if scenario is not built yet
    # await build_scenario()

    # run user manager
    await unity.service.start(manager_name="contact")


if __name__ == "__main__":
    asyncio.run(main())
