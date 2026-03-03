"""
Colliers Healthcare — client customization.

Registers the Colliers-specific environment (financial data extraction
and web deal research tools) and behavioral guidelines.

When Colliers signs up, set ``_COLLIERS_ORG_ID`` to their real org_id
and the org-level registration will activate.  Until then, the same
customizations are registered at three levels for internal testing:

- **Team-level**: team_id=49 ("Colliers") in the Unify org — covers any
  assistant created with the Unify org API key.
- **User-level**: dan@unify.ai and yasser@unify.ai personal user IDs —
  covers assistants created with personal (org-less) API keys.

Secret *values* can be overridden per-environment via ``.secrets.json``
(gitignored).  The definitions here ensure the secret entries always
exist in the DB with their descriptions.
"""

from unity.customization.configs.types.actor_config import ActorConfig
from unity.customization.clients import register_org, register_team, register_user
from unity.customization.clients.colliers.colliers_env import ColliersEnvironment
from unity.customization.clients.colliers.guidelines import COLLIERS_GUIDELINES

_COLLIERS_CONFIG = ActorConfig(
    guidelines=COLLIERS_GUIDELINES,
)

_COLLIERS_ENVIRONMENTS = [ColliersEnvironment()]

_COLLIERS_GUIDANCE = [
    {
        "title": "Colliers Healthcare Workflows",
        "content": COLLIERS_GUIDELINES,
    },
]

_COLLIERS_SECRETS = [
    {
        "name": "COSTAR_USERNAME",
        "value": "adam.lenton@colliers.com",
        "description": "Costar.com login username / email",
    },
    {
        "name": "COSTAR_PASSWORD",
        "value": "Liverpool*2019",
        "description": "Costar.com login password",
    },
]

# ---------------------------------------------------------------------------
# Org-level registration (activates once Colliers signs up)
# ---------------------------------------------------------------------------

_COLLIERS_ORG_ID = -1

if _COLLIERS_ORG_ID != -1:
    register_org(
        _COLLIERS_ORG_ID,
        config=_COLLIERS_CONFIG,
        environments=_COLLIERS_ENVIRONMENTS,
        guidance=_COLLIERS_GUIDANCE,
        secrets=_COLLIERS_SECRETS,
    )

# ---------------------------------------------------------------------------
# Team-level registration (Unify-internal testing)
# ---------------------------------------------------------------------------

_UNIFY_COLLIERS_TEAM_ID = 49

register_team(
    _UNIFY_COLLIERS_TEAM_ID,
    config=_COLLIERS_CONFIG,
    environments=_COLLIERS_ENVIRONMENTS,
    guidance=_COLLIERS_GUIDANCE,
    secrets=_COLLIERS_SECRETS,
)

# ---------------------------------------------------------------------------
# User-level registration (personal / org-less API keys)
# ---------------------------------------------------------------------------

_DAN_USER_ID = "cli3t38uc0000s60k5zmgj8ez"  # dan@unify.ai on staging
_YASSER_USER_ID = "40144b2a-722f-4f41-8d9e-384c316ee19f"  # yasser@unify.ai on staging

for _uid in (_DAN_USER_ID, _YASSER_USER_ID):
    register_user(
        _uid,
        config=_COLLIERS_CONFIG,
        environments=_COLLIERS_ENVIRONMENTS,
        guidance=_COLLIERS_GUIDANCE,
        secrets=_COLLIERS_SECRETS,
    )
