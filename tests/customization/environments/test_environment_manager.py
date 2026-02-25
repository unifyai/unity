"""Unit tests for EnvironmentManager using SimulatedEnvironmentManager.

These tests validate core functionality without requiring Unify backend
connectivity. They use a minimal inline BaseEnvironment subclass with no
third-party dependencies.
"""

from __future__ import annotations


import pytest

from unity.customization.environments.simulated import SimulatedEnvironmentManager
from unity.customization.environments.environment_manager import (
    _parse_env_path,
    _write_files_to_package,
    _import_and_resolve,
    _ensure_dependencies,
    _pkg_name_from_specifier,
)
from unity.actor.environments.base import BaseEnvironment

# ── Minimal test environment source code ──────────────────────────────────────

MINIMAL_ENV_SOURCE = '''\
from unity.actor.environments.base import BaseEnvironment, ToolMetadata


class GreeterService:
    """A trivial service for testing."""

    def greet(self, name: str) -> str:
        """Return a greeting."""
        return f"Hello, {name}!"


class GreeterEnvironment(BaseEnvironment):
    NAMESPACE = "greeter"

    def __init__(self):
        self._service = GreeterService()
        super().__init__(instance=self._service, namespace=self.NAMESPACE)

    def get_tools(self):
        return {
            f"{self.NAMESPACE}.greet": ToolMetadata(
                name=f"{self.NAMESPACE}.greet",
                is_impure=False,
            ),
        }

    def get_prompt_context(self):
        return f"### `{self.NAMESPACE}` — Greeter tools\\n\\n**`{self.NAMESPACE}.greet(name)`**"

    async def capture_state(self):
        return {"type": "greeter"}


greeter_env = GreeterEnvironment()
'''


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def manager():
    return SimulatedEnvironmentManager()


@pytest.fixture
def tmp_env_root(tmp_path):
    """Provide a temp directory for file extraction."""
    return tmp_path / "envs"


# ── Tests: parse_env_path ─────────────────────────────────────────────────────


class TestParseEnvPath:
    def test_valid_path(self):
        module, attr = _parse_env_path("my_module:my_attr")
        assert module == "my_module"
        assert attr == "my_attr"

    def test_no_colon(self):
        with pytest.raises(ValueError, match="must be 'module:attribute'"):
            _parse_env_path("no_colon_here")

    def test_empty_module(self):
        with pytest.raises(ValueError, match="non-empty"):
            _parse_env_path(":attr")

    def test_empty_attr(self):
        with pytest.raises(ValueError, match="non-empty"):
            _parse_env_path("module:")


# ── Tests: pkg_name_from_specifier ────────────────────────────────────────────


class TestPkgNameFromSpecifier:
    def test_plain_name(self):
        assert _pkg_name_from_specifier("numpy") == "numpy"

    def test_with_version(self):
        assert _pkg_name_from_specifier("openpyxl>=3.1") == "openpyxl"

    def test_with_extras(self):
        assert _pkg_name_from_specifier("package[extra]") == "package"

    def test_hyphenated(self):
        assert _pkg_name_from_specifier("my-package>=1.0") == "my_package"


# ── Tests: write_files_to_package ─────────────────────────────────────────────


class TestWriteFilesToPackage:
    def test_writes_files(self, tmp_env_root):
        files = {"hello.py": "x = 1\n", "sub/nested.py": "y = 2\n"}
        pkg_dir = _write_files_to_package(
            environment_id=42,
            files=files,
            root=tmp_env_root,
        )
        assert (pkg_dir / "hello.py").read_text() == "x = 1\n"
        assert (pkg_dir / "sub" / "nested.py").read_text() == "y = 2\n"

    def test_idempotent_no_rewrite(self, tmp_env_root):
        files = {"hello.py": "x = 1\n"}
        pkg_dir = _write_files_to_package(
            environment_id=1,
            files=files,
            root=tmp_env_root,
        )
        mtime1 = (pkg_dir / "hello.py").stat().st_mtime

        # Same content — should not rewrite
        _write_files_to_package(
            environment_id=1,
            files=files,
            root=tmp_env_root,
        )
        mtime2 = (pkg_dir / "hello.py").stat().st_mtime
        assert mtime1 == mtime2

    def test_rewrites_on_change(self, tmp_env_root):
        pkg_dir = _write_files_to_package(
            environment_id=1,
            files={"hello.py": "x = 1\n"},
            root=tmp_env_root,
        )
        original = (pkg_dir / "hello.py").read_text()

        _write_files_to_package(
            environment_id=1,
            files={"hello.py": "x = 2\n"},
            root=tmp_env_root,
        )
        updated = (pkg_dir / "hello.py").read_text()
        assert original != updated
        assert updated == "x = 2\n"


# ── Tests: import_and_resolve ─────────────────────────────────────────────────


class TestImportAndResolve:
    def test_resolves_attribute(self, tmp_env_root):
        pkg_dir = _write_files_to_package(
            environment_id=99,
            files={"simple_mod.py": "MY_VAR = 42\n"},
            root=tmp_env_root,
        )
        result = _import_and_resolve(
            pkg_dir=pkg_dir,
            module_name="simple_mod",
            attr_name="MY_VAR",
        )
        assert result == 42

    def test_missing_module(self, tmp_env_root):
        pkg_dir = _write_files_to_package(
            environment_id=100,
            files={"exists.py": "x = 1\n"},
            root=tmp_env_root,
        )
        with pytest.raises(ImportError, match="Could not import module"):
            _import_and_resolve(
                pkg_dir=pkg_dir,
                module_name="nonexistent",
                attr_name="x",
            )

    def test_missing_attribute(self, tmp_env_root):
        pkg_dir = _write_files_to_package(
            environment_id=101,
            files={"attr_mod.py": "x = 1\n"},
            root=tmp_env_root,
        )
        with pytest.raises(AttributeError, match="has no attribute"):
            _import_and_resolve(
                pkg_dir=pkg_dir,
                module_name="attr_mod",
                attr_name="missing",
            )


# ── Tests: ensure_dependencies ────────────────────────────────────────────────


class TestEnsureDependencies:
    def test_already_installed(self):
        # json is a stdlib module — should not trigger pip install
        _ensure_dependencies(["json"])

    def test_no_deps(self):
        _ensure_dependencies([])


# ── Tests: SimulatedEnvironmentManager CRUD ───────────────────────────────────


class TestSimulatedCRUD:
    def test_upload_returns_id(self, manager):
        eid = manager.upload_environment(
            name="test",
            files={"m.py": "x=1"},
            env="m:x",
        )
        assert isinstance(eid, int)
        assert eid >= 1

    def test_upload_increments_id(self, manager):
        eid1 = manager.upload_environment(
            name="a",
            files={"a.py": ""},
            env="a:x",
        )
        eid2 = manager.upload_environment(
            name="b",
            files={"b.py": ""},
            env="b:x",
        )
        assert eid2 == eid1 + 1

    def test_list_after_upload(self, manager):
        manager.upload_environment(
            name="test",
            files={"m.py": "x=1"},
            env="m:x",
        )
        envs = manager.list_environments()
        assert len(envs) == 1
        assert envs[0]["name"] == "test"

    def test_delete(self, manager):
        eid = manager.upload_environment(
            name="test",
            files={"m.py": "x=1"},
            env="m:x",
        )
        result = manager.delete_environment(environment_id=eid)
        assert result["outcome"] == "environment deleted"
        assert manager.list_environments() == []

    def test_delete_nonexistent(self, manager):
        with pytest.raises(ValueError, match="No environment found"):
            manager.delete_environment(environment_id=999)

    def test_upload_invalid_env_path(self, manager):
        with pytest.raises(ValueError, match="must be 'module:attribute'"):
            manager.upload_environment(
                name="bad",
                files={"m.py": ""},
                env="no_colon",
            )

    def test_clear(self, manager):
        manager.upload_environment(
            name="test",
            files={"m.py": "x=1"},
            env="m:x",
        )
        manager.clear()
        assert manager.list_environments() == []

    def test_list_with_filter(self, manager):
        manager.upload_environment(
            name="alpha",
            files={"a.py": ""},
            env="a:x",
        )
        manager.upload_environment(
            name="beta",
            files={"b.py": ""},
            env="b:x",
        )
        results = manager.list_environments(filter="name == 'alpha'")
        assert len(results) == 1
        assert results[0]["name"] == "alpha"


# ── Tests: load_environment end-to-end (SimulatedEnvironmentManager) ──────────


class TestSimulatedLoad:
    def test_load_returns_base_environment(self, manager):
        eid = manager.upload_environment(
            name="greeter",
            files={"greeter_env.py": MINIMAL_ENV_SOURCE},
            env="greeter_env:greeter_env",
        )
        env = manager.load_environment(eid)
        assert isinstance(env, BaseEnvironment)

    def test_loaded_namespace(self, manager):
        eid = manager.upload_environment(
            name="greeter",
            files={"greeter_env.py": MINIMAL_ENV_SOURCE},
            env="greeter_env:greeter_env",
        )
        env = manager.load_environment(eid)
        assert env.namespace == "greeter"

    def test_loaded_tools(self, manager):
        eid = manager.upload_environment(
            name="greeter",
            files={"greeter_env.py": MINIMAL_ENV_SOURCE},
            env="greeter_env:greeter_env",
        )
        env = manager.load_environment(eid)
        tools = env.get_tools()
        assert "greeter.greet" in tools

    def test_loaded_prompt_context(self, manager):
        eid = manager.upload_environment(
            name="greeter",
            files={"greeter_env.py": MINIMAL_ENV_SOURCE},
            env="greeter_env:greeter_env",
        )
        env = manager.load_environment(eid)
        ctx = env.get_prompt_context()
        assert "greeter" in ctx
        assert "greet" in ctx

    def test_loaded_service_callable(self, manager):
        eid = manager.upload_environment(
            name="greeter",
            files={"greeter_env.py": MINIMAL_ENV_SOURCE},
            env="greeter_env:greeter_env",
        )
        env = manager.load_environment(eid)
        instance = env.get_instance()
        assert instance.greet("World") == "Hello, World!"

    def test_load_all(self, manager):
        manager.upload_environment(
            name="greeter",
            files={"greeter_env.py": MINIMAL_ENV_SOURCE},
            env="greeter_env:greeter_env",
        )
        envs = manager.load_all_environments()
        assert len(envs) == 1
        assert isinstance(envs[0], BaseEnvironment)

    def test_load_nonexistent(self, manager):
        with pytest.raises(ValueError, match="No environment found"):
            manager.load_environment(999)

    def test_load_bad_env_path_attribute(self, manager):
        eid = manager.upload_environment(
            name="bad",
            files={"greeter_env.py": MINIMAL_ENV_SOURCE},
            env="greeter_env:nonexistent_attr",
        )
        with pytest.raises(AttributeError, match="has no attribute"):
            manager.load_environment(eid)

    def test_load_bad_env_path_module(self, manager):
        eid = manager.upload_environment(
            name="bad",
            files={"greeter_env.py": MINIMAL_ENV_SOURCE},
            env="nonexistent_module:greeter_env",
        )
        with pytest.raises(ImportError, match="Could not import module"):
            manager.load_environment(eid)


# ── Tests: multi-file environment ─────────────────────────────────────────────

SCHEMAS_SOURCE = """\
from pydantic import BaseModel


class Person(BaseModel):
    name: str
    age: int
"""

MULTI_FILE_ENV_SOURCE = """\
from unity.actor.environments.base import BaseEnvironment, ToolMetadata
from schemas import Person


class PersonService:
    def create_person(self, name: str, age: int) -> str:
        p = Person(name=name, age=age)
        return p.model_dump_json()


class PersonEnvironment(BaseEnvironment):
    NAMESPACE = "people"

    def __init__(self):
        self._service = PersonService()
        super().__init__(instance=self._service, namespace=self.NAMESPACE)

    def get_tools(self):
        return {
            f"{self.NAMESPACE}.create_person": ToolMetadata(
                name=f"{self.NAMESPACE}.create_person",
                is_impure=False,
            ),
        }

    def get_prompt_context(self):
        return "### people service"

    async def capture_state(self):
        return {"type": "people"}


person_env = PersonEnvironment()
"""


class TestMultiFileEnvironment:
    def test_multi_file_load(self, manager):
        eid = manager.upload_environment(
            name="people",
            files={
                "person_env.py": MULTI_FILE_ENV_SOURCE,
                "schemas.py": SCHEMAS_SOURCE,
            },
            env="person_env:person_env",
        )
        env = manager.load_environment(eid)
        assert isinstance(env, BaseEnvironment)
        assert env.namespace == "people"

    def test_multi_file_service_works(self, manager):
        eid = manager.upload_environment(
            name="people",
            files={
                "person_env.py": MULTI_FILE_ENV_SOURCE,
                "schemas.py": SCHEMAS_SOURCE,
            },
            env="person_env:person_env",
        )
        env = manager.load_environment(eid)
        result = env.get_instance().create_person("Alice", 30)
        assert "Alice" in result
        assert "30" in result
