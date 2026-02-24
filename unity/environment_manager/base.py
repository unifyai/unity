from __future__ import annotations

from abc import abstractmethod
from typing import Dict, List, Optional, TYPE_CHECKING

from ..manager_registry import SingletonABCMeta
from ..common.state_managers import BaseStateManager


class BaseEnvironmentManager(BaseStateManager, metaclass=SingletonABCMeta):
    """Public contract for concrete environment managers.

    An environment manager stores self-contained environment packages
    (source files + dependencies) and reconstructs live ``BaseEnvironment``
    instances at actor construction time.
    """

    _as_caller_description: str = (
        "the EnvironmentManager, managing custom execution environments"
    )

    @abstractmethod
    def upload_environment(
        self,
        *,
        name: str,
        files: Dict[str, str],
        dependencies: Optional[List[str]] = None,
        env: str,
    ) -> int:
        """Upload a new environment package.

        Parameters
        ----------
        name : str
            Human-readable name for the environment.
        files : Dict[str, str]
            Mapping of filename to source code.
        dependencies : List[str] | None
            Pip dependency specifiers required by this environment.
        env : str
            ``module:attribute`` path that resolves to a ``BaseEnvironment``
            instance once the files are loaded (e.g. ``"my_env:my_env"``).

        Returns
        -------
        int
            The auto-assigned ``environment_id``.
        """
        raise NotImplementedError

    @abstractmethod
    def load_environment(self, environment_id: int) -> "BaseEnvironment":
        """Reconstruct a live ``BaseEnvironment`` from a stored definition.

        Ensures dependencies are installed, writes source files to a temp
        package directory, imports the entry module, and resolves the
        ``module:attribute`` path to a ``BaseEnvironment`` instance.

        Parameters
        ----------
        environment_id : int
            Identifier of the stored environment.

        Returns
        -------
        BaseEnvironment
            A live, ready-to-inject environment instance.
        """
        raise NotImplementedError

    @abstractmethod
    def load_all_environments(self) -> List["BaseEnvironment"]:
        """Load all stored environments as live ``BaseEnvironment`` instances.

        Returns
        -------
        List[BaseEnvironment]
            All successfully loaded environments. Environments that fail
            to load are logged and skipped.
        """
        raise NotImplementedError

    @abstractmethod
    def list_environments(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[dict]:
        """List stored environment definitions.

        Parameters
        ----------
        filter : str | None
            Optional Python boolean expression for filtering.
        offset : int
            Pagination offset.
        limit : int
            Maximum number of results.

        Returns
        -------
        List[dict]
            Environment metadata dicts.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_environment(self, *, environment_id: int) -> dict:
        """Delete a stored environment by id.

        Parameters
        ----------
        environment_id : int
            Identifier of the environment to remove.

        Returns
        -------
        dict
            Outcome with the deleted ``environment_id``.
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError


if TYPE_CHECKING:
    from ..actor.environments.base import BaseEnvironment
