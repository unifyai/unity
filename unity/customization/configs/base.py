from __future__ import annotations

from abc import abstractmethod

from ...manager_registry import SingletonABCMeta
from ...common.state_managers import BaseStateManager
from .types.actor_config import ActorConfig


class BaseConfigManager(BaseStateManager, metaclass=SingletonABCMeta):
    """Public contract for concrete config managers.

    A config manager stores a single ``ActorConfig`` per assistant context,
    allowing per-company actor configuration to be persisted in the DB
    rather than requiring dedicated deployments.
    """

    _as_caller_description: str = "the ConfigManager, managing actor configuration"

    @abstractmethod
    def save_config(self, config: ActorConfig) -> None:
        """Upsert the actor configuration.

        If a config already exists for the current context it is replaced.

        Parameters
        ----------
        config : ActorConfig
            The configuration to persist.
        """
        raise NotImplementedError

    @abstractmethod
    def load_config(self) -> ActorConfig:
        """Load the stored actor configuration.

        Returns
        -------
        ActorConfig
            The stored config, or an empty ``ActorConfig()`` (all-None
            fields) if nothing has been saved yet.
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Delete any stored configuration."""
        raise NotImplementedError
