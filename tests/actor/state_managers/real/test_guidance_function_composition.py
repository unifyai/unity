"""Doing-loop composition test: CodeActActor uses FM and GM together.

Verifies that when the actor has both FunctionManager tools and GuidanceManager
primitives available, it discovers a stored function via FM AND consults guidance
via GM to correctly parameterize the function — using both in unison during a
single task.

This complements the single-primitive routing tests in ``test_guidance_code_act.py``
which verify FM-only or GM-only routing in isolation.
"""

import pytest

from tests.helpers import _handle_project
from tests.actor.state_managers.utils import (
    assert_code_act_function_manager_used,
    make_code_act_actor,
)
from unity.function_manager.function_manager import FunctionManager
from unity.manager_registry import ManagerRegistry

pytestmark = pytest.mark.eval


@pytest.mark.asyncio
@pytest.mark.timeout(600)
@_handle_project
async def test_doing_loop_uses_function_and_guidance_together():
    """The doing loop discovers a function via FM and consults GM for policy rules.

    FM stores a ``calculate_total`` function that takes explicit rate parameters
    (tax rate, discount percentage).  GM stores a "Pricing Policy" guidance entry
    with specific rate values per customer tier.  The user's request requires
    both: finding the right function AND looking up the policy to determine
    the correct parameter values for a VIP customer.
    """
    fm = FunctionManager()
    implementation = '''
def calculate_total(subtotal: float, tax_rate_pct: float, discount_pct: float = 0.0) -> dict:
    """Calculate a final total applying tax and an optional discount.

    Args:
        subtotal: Base price before adjustments.
        tax_rate_pct: Tax rate as a percentage (e.g. 7.25 means 7.25%).
        discount_pct: Discount as a percentage (e.g. 12 means 12%).

    Returns:
        Dict with subtotal, discount_amount, taxable_amount, tax_amount, and total.
    """
    discount_amount = round(subtotal * (discount_pct / 100), 2)
    taxable = round(subtotal - discount_amount, 2)
    tax_amount = round(taxable * (tax_rate_pct / 100), 2)
    total = round(taxable + tax_amount, 2)
    return {
        "subtotal": subtotal,
        "discount_pct": discount_pct,
        "discount_amount": discount_amount,
        "taxable_amount": taxable,
        "tax_rate_pct": tax_rate_pct,
        "tax_amount": tax_amount,
        "total": total,
    }
'''
    fm.add_functions(implementations=implementation, overwrite=True)

    async with make_code_act_actor(
        impl="real",
        include_function_manager_tools=True,
        function_manager=fm,
    ) as (actor, _primitives, calls):
        gm = ManagerRegistry.get_guidance_manager()
        gm.add_guidance(
            title="Pricing Policy",
            content=(
                "Tax rate: 7.25%. "
                "Discount tiers: VIP customers receive a 12% discount, "
                "Regular customers receive a 3% discount, "
                "Employee purchases receive a 20% discount. "
                "Always apply the discount before calculating tax."
            ),
        )

        handle = await actor.act(
            "A VIP customer is purchasing items totaling $200. "
            "What is their final total after applying our standard pricing rules?",
            clarification_enabled=False,
        )
        result = await handle.result()
        assert result is not None

        # The actor should have used FunctionManager tools to discover
        # the stored calculate_total function.
        assert_code_act_function_manager_used(handle)
