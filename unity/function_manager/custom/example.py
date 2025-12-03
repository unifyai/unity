"""
Example custom functions for demonstration and testing.

These functions are automatically synced to Functions/Compositional
when the FunctionManager initializes.
"""

from unity.function_manager.custom import custom_function


@custom_function()
async def example_add(a: int, b: int) -> int:
    """
    Add two integers together.

    This is an example custom function to demonstrate the auto-sync feature.

    Args:
        a: First integer
        b: Second integer

    Returns:
        The sum of a and b
    """
    return a + b


@custom_function(verify=False)
async def example_uppercase(text: str) -> str:
    """
    Convert text to uppercase.

    Args:
        text: The text to convert

    Returns:
        The uppercase version of the text
    """
    return text.upper()


@custom_function(auto_sync=False)
async def draft_function_not_synced(x: int) -> int:
    """
    This function has auto_sync=False, so it will NOT be synced.

    Useful for work-in-progress functions.
    """
    return x * 2
