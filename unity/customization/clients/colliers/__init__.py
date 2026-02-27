"""
Colliers Healthcare — client customization.

Registers the Colliers-specific environment (financial data extraction
and web deal research tools) and behavioral guidelines.

When Colliers signs up, set ``_COLLIERS_ORG_ID`` to their real org_id
and the org-level registration will activate.  Until then, the same
customizations are registered against the Unify-internal "Colliers"
team (team_id=1 in org_id=1) so dan@unify.ai and yasser@unify.ai
can test end-to-end.

Note: CoStar website credentials are managed via SecretManager and are
NOT stored in code.
"""

from unity.customization.configs.types.actor_config import ActorConfig
from unity.customization.clients import register_org, register_team
from unity.customization.clients.colliers.colliers_env import ColliersEnvironment
from unity.customization.clients.colliers.guidelines import COLLIERS_GUIDELINES

_COLLIERS_CONFIG = ActorConfig(
    guidelines=COLLIERS_GUIDELINES,
)

_COLLIERS_ENVIRONMENTS = [ColliersEnvironment()]

# ---------------------------------------------------------------------------
# Org-level registration (activates once Colliers signs up)
# ---------------------------------------------------------------------------

_COLLIERS_ORG_ID = -1

if _COLLIERS_ORG_ID != -1:
    register_org(
        _COLLIERS_ORG_ID,
        config=_COLLIERS_CONFIG,
        environments=_COLLIERS_ENVIRONMENTS,
    )

# ---------------------------------------------------------------------------
# Team-level registration (Unify-internal testing)
# ---------------------------------------------------------------------------

_UNIFY_COLLIERS_TEAM_ID = 49

register_team(
    _UNIFY_COLLIERS_TEAM_ID,
    config=_COLLIERS_CONFIG,
    environments=_COLLIERS_ENVIRONMENTS,
)
