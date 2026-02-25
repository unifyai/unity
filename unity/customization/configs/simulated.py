from __future__ import annotations

import functools
import logging
from typing import Any, Optional

from .base import BaseConfigManager
from .types.actor_config import ActorConfig

logger = logging.getLogger(__name__)


class SimulatedConfigManager(BaseConfigManager):
    """In-memory ConfigManager for testing and simulated sandboxes."""

    def __init__(
        self,
        description: str = "simulated config manager",
        *,
        simulation_guidance: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._description = description
        self._simulation_guidance = simulation_guidance
        self._config: Optional[ActorConfig] = None

    @functools.wraps(BaseConfigManager.save_config, updated=())
    def save_config(self, config: ActorConfig) -> None:
        self._config = config

    @functools.wraps(BaseConfigManager.load_config, updated=())
    def load_config(self) -> ActorConfig:
        if self._config is None:
            return ActorConfig()
        return self._config

    @functools.wraps(BaseConfigManager.clear, updated=())
    def clear(self) -> None:
        self._config = None
