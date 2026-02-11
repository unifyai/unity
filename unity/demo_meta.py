"""
Demo assistant metadata fetching from Orchestra.

This module provides functionality to fetch prospect details from Orchestra's
demo assistant metadata endpoint. When a demo session starts with a demo_id,
Unity can use this to pre-populate the boss contact (contact_id=1) with
prospect information provided during demo creation.

The metadata is fetched once during initialization and cached in SETTINGS.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Timeout for demo metadata fetch (should be fast, non-critical)
DEMO_META_FETCH_TIMEOUT = 5.0


@dataclass
class DemoProspectDetails:
    """Prospect details from demo assistant metadata."""

    first_name: Optional[str] = None
    surname: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None

    def has_any_details(self) -> bool:
        """Check if any prospect details are available."""
        return any([self.first_name, self.surname, self.email, self.phone])


def _get_api_key() -> Optional[str]:
    """Get the admin API key for Orchestra calls."""
    return os.getenv("ORCHESTRA_ADMIN_KEY")


def _get_base_url() -> str:
    """Get the Orchestra API base URL."""
    return os.getenv("ORCHESTRA_URL", "https://api.unify.ai/v0")


async def fetch_demo_meta(demo_id: int) -> Optional[DemoProspectDetails]:
    """
    Fetch demo assistant metadata from Orchestra.

    Args:
        demo_id: The demo assistant metadata ID.

    Returns:
        DemoProspectDetails with prospect information, or None if fetch fails.
    """
    api_key = _get_api_key()
    if not api_key:
        logger.warning("ORCHESTRA_ADMIN_KEY not set, cannot fetch demo metadata")
        return None

    base_url = _get_base_url()
    url = f"{base_url}/demo/assistant/{demo_id}/meta"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=DEMO_META_FETCH_TIMEOUT,
            )

            if response.status_code == 404:
                logger.warning(f"Demo metadata not found for demo_id={demo_id}")
                return None

            if response.status_code != 200:
                logger.warning(
                    f"Failed to fetch demo metadata: status={response.status_code}"
                )
                return None

            data = response.json()
            return DemoProspectDetails(
                first_name=data.get("prospect_first_name"),
                surname=data.get("prospect_surname"),
                email=data.get("prospect_email"),
                phone=data.get("prospect_phone"),
            )

    except httpx.TimeoutException:
        logger.warning(f"Timeout fetching demo metadata for demo_id={demo_id}")
        return None
    except Exception as e:
        logger.warning(f"Error fetching demo metadata: {e}")
        return None


def apply_prospect_to_boss_contact(
    contact_manager,  # BaseContactManager
    prospect: DemoProspectDetails,
) -> bool:
    """
    Apply prospect details to the boss contact (contact_id=1).

    This updates the boss contact with any available prospect information
    from the demo metadata. Only non-None fields are applied.

    Args:
        contact_manager: The ContactManager instance.
        prospect: Prospect details from demo metadata.

    Returns:
        True if any updates were applied, False otherwise.
    """
    if not prospect.has_any_details():
        logger.info("No prospect details available to apply")
        return False

    # Build update kwargs with only non-None values
    update_kwargs = {"contact_id": 1}

    if prospect.first_name:
        update_kwargs["first_name"] = prospect.first_name
    if prospect.surname:
        update_kwargs["surname"] = prospect.surname
    if prospect.email:
        update_kwargs["email_address"] = prospect.email
    if prospect.phone:
        update_kwargs["phone_number"] = prospect.phone

    try:
        contact_manager.update_contact(**update_kwargs)
        logger.info(
            f"Applied prospect details to boss contact: "
            f"first_name={prospect.first_name}, surname={prospect.surname}, "
            f"email={prospect.email is not None}, phone={prospect.phone is not None}"
        )
        return True
    except Exception as e:
        logger.warning(f"Failed to apply prospect details to boss contact: {e}")
        return False
