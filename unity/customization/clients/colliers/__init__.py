"""
Colliers Healthcare — client customization.

Registers the Colliers-specific environment (financial data extraction
and web deal research tools) and behavioral guidelines at the org level.

Note: CoStar website credentials are managed via SecretManager and are
NOT stored in code.
"""

from unity.customization.configs.types.actor_config import ActorConfig
from unity.customization.clients import register_org
from unity.customization.clients.colliers.colliers_env import ColliersEnvironment
from unity.customization.clients.colliers.guidelines import COLLIERS_GUIDELINES

# Placeholder — replace with real org_id once Colliers signs up.
_COLLIERS_ORG_ID = -1

_ORG_CONFIG = ActorConfig(
    guidelines=COLLIERS_GUIDELINES,
)

_ORG_ENVIRONMENTS = [ColliersEnvironment()]

if _COLLIERS_ORG_ID != -1:
    register_org(
        _COLLIERS_ORG_ID,
        config=_ORG_CONFIG,
        environments=_ORG_ENVIRONMENTS,
    )
