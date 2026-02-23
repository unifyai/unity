import asyncio
from pydantic import BaseModel, Field
from typing import List, Dict, Union
import statistics


async def login_to_website(
    url: str,
    username_field: str,
    password_field: str,
    username: str,
    password: str,
    submit_button: str,
    expectation: str,
):
    """
    Navigates to a URL and performs a login action.

    Args:
        url (str): The URL of the login page.
        username_field (str): A description of the username input field.
        password_field (str): A description of the password input field.
        username (str): The username to enter.
        password (str): The password to enter.
        submit_button (str): A description of the login/submit button.
        expectation (str): The expected state of the page after a successful login.
    """
    print(f"Navigating to {url} to log in.")
    await primitives.computer.navigate(url)  # type: ignore
    await asyncio.sleep(2)  # Wait for page to load

    print(f"Entering username into '{username_field}'.")
    await primitives.computer.act(  # type: ignore
        f"Type '{username}' into the {username_field}",
        expectation=f"The text '{username}' should be visible in the {username_field}.",
    )

    print(f"Entering password into '{password_field}'.")
    await primitives.computer.act(  # type: ignore
        f"Type '{password}' into the {password_field}",
        expectation="The password should be entered in the password field.",
    )

    print(f"Clicking the '{submit_button}'.")
    await primitives.computer.act(  # type: ignore
        f"Click the {submit_button}",
        expectation=expectation,
    )
    print("Login sequence completed.")


async def search_and_extract_links(
    search_engine_url: str,
    query: str,
    num_links: int = 5,
) -> List[str]:
    """
    Navigates to a search engine, performs a search, and extracts the top result links.

    Args:
        search_engine_url (str): The base URL of the search engine (e.g., 'https://www.google.com').
        query (str): The search query.
        num_links (int): The number of top links to extract.

    Returns:
        List[str]: A list of URLs from the search results.
    """
    print(f"Navigating to {search_engine_url} to search for '{query}'.")
    await primitives.computer.navigate(search_engine_url)  # type: ignore
    await asyncio.sleep(2)

    await primitives.computer.act(  # type: ignore
        f"Type '{query}' into the search bar and press Enter",
        expectation="A search results page should be displayed.",
    )
    await asyncio.sleep(2)

    class SearchResult(BaseModel):
        url: str = Field(description="The full URL of the search result link.")

    class SearchResults(BaseModel):
        links: List[SearchResult]

    SearchResults.model_rebuild()

    print(f"Extracting the top {num_links} links.")
    extracted_data = await primitives.computer.observe(  # type: ignore
        f"Extract the top {num_links} search result links.",
        response_format=SearchResults,
    )

    urls = [result.url for result in extracted_data.links]
    print(f"Found {len(urls)} links.")
    return urls


def calculate_list_statistics22(data: List[Union[int, float]]) -> Dict[str, float]:
    """
    Calculates basic statistics (mean, median, standard deviation) for a list of numbers.
    This is a pure Python function with no side effects.

    Args:
        data (List[Union[int, float]]): A list of numbers.

    Returns:
        Dict[str, float]: A dictionary containing the calculated statistics.
    """
    if not data:
        return {"mean": 0.0, "median": 0.0, "stdev": 0.0}

    mean = statistics.mean(data)
    median = statistics.median(data)
    # Standard deviation requires at least 2 data points
    stdev = statistics.stdev(data) if len(data) > 1 else 0.0

    stats = {
        "mean": round(mean, 2),
        "median": round(median, 2),
        "stdev": round(stdev, 2),
    }
    print(f"Calculated statistics: {stats}")
    return stats
