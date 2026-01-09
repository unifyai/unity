import ast
import asyncio
import inspect
import functools
import json
import os
import signal
import sys
import logging
from pathlib import Path
from typing import Dict, List, Set, Union, Tuple, Any, Optional
import unify
from unify.utils.http import RequestError as _UnifyRequestError
from ..common.log_utils import create_logs as unity_create_logs
from ..common.embed_utils import list_private_fields
from ..common.search_utils import table_search_top_k
from .execution_env import create_base_globals
from .types.function import Function
from .types.meta import FunctionsMeta
from .types.venv import VirtualEnv
from .base import BaseFunctionManager
from ..common.model_to_fields import model_to_fields
from ..file_manager.managers.local import LocalFileManager
from ..image_manager.image_manager import ImageHandle
from ..manager_registry import ManagerRegistry
from ..common.filter_utils import normalize_filter_expr
from ..common.context_registry import ContextRegistry, TableContext
from .primitives import collect_primitives, compute_primitives_hash
from .custom_functions import (
    collect_custom_functions,
    compute_custom_functions_hash,
    collect_custom_venvs,
    compute_custom_venvs_hash,
)


logger = logging.getLogger(__name__)


class _DependencyVisitor(ast.NodeVisitor):
    """
    Statefully analyzes function AST to find direct calls and indirect calls
    via variables assigned function names, specifically looking for names
    known to the FunctionManager.
    """

    def __init__(self, known_function_names: Set[str]):
        self.known_function_names = known_function_names
        self.dependencies: Set[str] = set()
        self._assignment_map: Dict[str, str] = {}

    def visit_Assign(self, node: ast.Assign):
        # Only track simple assignments: target_var = potential_func_name
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target_var = node.targets[0].id
            if isinstance(node.value, ast.Name):
                assigned_name = node.value.id
                # Check if the assigned name is one of the functions we manage
                if assigned_name in self.known_function_names:
                    # Record the mapping for the current scope
                    self._assignment_map[target_var] = assigned_name
                # If variable is assigned something else, remove mapping
                elif target_var in self._assignment_map:
                    del self._assignment_map[target_var]
            # If variable is assigned non-Name, remove mapping
            elif target_var in self._assignment_map:
                del self._assignment_map[target_var]

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        func_node = node.func
        called_name: Optional[str] = None

        # Case 1: Direct call -> func_name()
        if isinstance(func_node, ast.Name):
            func_name = func_node.id
            # Check if it's a direct call to a known library function
            if func_name in self.known_function_names:
                called_name = func_name
            # Check if it's an indirect call via a mapped variable -> var()
            elif func_name in self._assignment_map:
                called_name = self._assignment_map[func_name]

        # Case 2: Method call -> obj.method() - generally ignore for dependency injection
        # (We assume obj like computer_primitives is globally available)

        if called_name:
            self.dependencies.add(called_name)

        self.generic_visit(node)  # Continue traversal

    def visit_Return(self, node: ast.Return):
        # Case 3: Return statement -> return func_name or return var
        if isinstance(node.value, ast.Name):
            returned_name = node.value.id
            # Check if returning a known function name directly
            if returned_name in self.known_function_names:
                self.dependencies.add(returned_name)
            # Also check if returning a variable that was assigned a function
            elif returned_name in self._assignment_map:
                self.dependencies.add(self._assignment_map[returned_name])
        self.generic_visit(node)


def _strip_custom_function_decorators(source: str) -> str:
    """
    Remove @custom_function decorators from a function source string.

    The @custom_function decorator is used for sync metadata only (it is effectively
    a no-op at runtime), but the symbol is not guaranteed to exist inside execution
    environments (e.g., Actor sandboxes or venv runner subprocesses).
    """
    try:
        lines = source.splitlines(keepends=True)
    except Exception:
        return source

    out: List[str] = []
    seen_def = False
    for line in lines:
        stripped = line.lstrip()
        if not seen_def and stripped.startswith("@custom_function"):
            continue
        if stripped.startswith("def ") or stripped.startswith("async def "):
            seen_def = True
        out.append(line)
    return "".join(out)


class _VenvFunctionProxy:
    """Proxy that wraps a venv-backed function as an awaitable callable."""

    def __init__(
        self,
        *,
        function_manager: "FunctionManager",
        func_data: Dict[str, Any],
        namespace: Dict[str, Any],
    ):
        self._function_manager = function_manager
        self._func_data = func_data
        self._namespace = namespace

        self.__name__ = str(func_data.get("name") or "unknown")
        self.__doc__ = str(func_data.get("docstring") or "")

    @staticmethod
    def _map_positional_args(
        *,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        implementation: str,
        func_name: str,
    ) -> dict[str, Any]:
        """
        Map positional args to kwargs using AST-extracted parameter names.

        Note: the venv runner currently executes with ``fn(**call_kwargs)``, so we can
        only support positional args by mapping them onto non-positional-only params.
        """
        if not args:
            return kwargs

        try:
            tree = ast.parse(implementation)
        except Exception as e:
            raise TypeError(
                f"Cannot map positional args for venv function '{func_name}': failed to parse implementation",
            ) from e

        if not tree.body or not isinstance(
            tree.body[0],
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            raise TypeError(
                f"Cannot map positional args for venv function '{func_name}': implementation must contain exactly one top-level function",
            )

        node: ast.FunctionDef | ast.AsyncFunctionDef = tree.body[0]
        if node.args.posonlyargs:
            raise TypeError(
                f"Cannot call venv function '{func_name}' with positional-only args; use keyword arguments",
            )
        if node.args.vararg is not None:
            raise TypeError(
                f"Cannot call venv function '{func_name}' with *args; use keyword arguments",
            )

        param_names = [a.arg for a in node.args.args]
        if len(args) > len(param_names):
            raise TypeError(
                f"Too many positional arguments for venv function '{func_name}'",
            )

        mapped: dict[str, Any] = dict(kwargs)
        for k, v in zip(param_names[: len(args)], args):
            if k in mapped:
                raise TypeError(
                    f"Multiple values for argument '{k}' in venv function '{func_name}'",
                )
            mapped[k] = v
        return mapped

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        venv_id = self._func_data.get("venv_id")
        if venv_id is None:
            raise ValueError(f"Venv proxy '{self.__name__}' missing venv_id")

        implementation = self._func_data.get("implementation")
        if not isinstance(implementation, str) or not implementation.strip():
            raise ValueError(f"Venv function '{self.__name__}' has no implementation")

        # Strip @custom_function decorators (not available in subprocess runner).
        implementation = _strip_custom_function_decorators(implementation)

        # Determine async-ness based on source (consistent with existing HierarchicalActor proxy).
        is_async = "async def" in implementation

        # Resolve RPC targets from the injected namespace (caller-controlled).
        primitives = self._namespace.get("primitives")
        computer_primitives = self._namespace.get("computer_primitives")

        call_kwargs = self._map_positional_args(
            args=args,
            kwargs=kwargs,
            implementation=implementation,
            func_name=self.__name__,
        )

        result = await self._function_manager.execute_in_venv(
            venv_id=int(venv_id),
            implementation=implementation,
            call_kwargs=call_kwargs,
            is_async=is_async,
            primitives=primitives,
            computer_primitives=computer_primitives,
        )

        if result.get("error"):
            raise RuntimeError(str(result.get("error")))
        return result.get("result")


class FunctionManager(BaseFunctionManager):
    """
    Keeps a catalogue of user-supplied Python functions and system primitives.

    User-defined functions are stored in `Functions/Compositional` with auto-incrementing
    IDs. System primitives (state manager methods) are stored in `Functions/Primitives`
    with explicit stable IDs that are consistent across all users.

    This separation ensures:
    - User function IDs are stable (adding/removing primitives doesn't affect them)
    - Primitive IDs are consistent across all users (based on PRIMITIVE_SOURCES order)
    - No ID collisions between the two namespaces
    """

    class Config:
        required_contexts = [
            TableContext(
                name="Functions/VirtualEnvs",
                description="Virtual environment configurations (pyproject.toml content).",
                fields=model_to_fields(VirtualEnv),
                unique_keys={"venv_id": "int"},
                auto_counting={"venv_id": None},
            ),
            TableContext(
                name="Functions/Compositional",
                description="User-defined functions with auto-incrementing IDs.",
                fields=model_to_fields(Function),
                unique_keys={"function_id": "int"},
                auto_counting={"function_id": None},
                foreign_keys=[
                    {
                        "name": "guidance_ids[*]",
                        "references": "Guidance.guidance_id",
                        "on_delete": "CASCADE",
                        "on_update": "CASCADE",
                    },
                    {
                        "name": "venv_id",
                        "references": "Functions/VirtualEnvs.venv_id",
                        "on_delete": "SET NULL",
                        "on_update": "CASCADE",
                    },
                ],
            ),
            TableContext(
                name="Functions/Primitives",
                description="System action primitives with stable explicit IDs.",
                fields=model_to_fields(Function),
                unique_keys={"function_id": "int"},
                # No auto_counting - primitives get explicit IDs from collect_primitives()
            ),
            TableContext(
                name="Functions/Meta",
                description="Metadata for primitives sync state.",
                fields=model_to_fields(FunctionsMeta),
                unique_keys={"meta_id": "int"},
            ),
        ]

    # ------------------------------------------------------------------ #
    #  Construction                                                      #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        *,
        daemon: bool = True,
        file_manager: Optional[LocalFileManager] = None,
    ) -> None:
        # No thread behavior; keep parameter for backward compatibility
        self._daemon = daemon
        # ToDo: expose tools to LLM once needed
        self._tools: Dict[str, callable] = {}
        self.include_in_multi_assistant_table = True

        # Internal monotonically-increasing function-id counter.  We keep it local
        # to the manager to avoid an expensive scan across *all* logs every
        # time we create a function.  Initialised lazily on first use.
        self._next_id: Optional[int] = None

        ctxs = unify.get_active_context()
        read_ctx, write_ctx = ctxs["read"], ctxs["write"]
        if not read_ctx:
            # Ensure the global assistant/context is selected before we derive our sub-context
            try:
                from .. import (
                    ensure_initialised as _ensure_initialised,
                )  # local to avoid cycles

                _ensure_initialised()
                ctxs = unify.get_active_context()
                read_ctx, write_ctx = ctxs["read"], ctxs["write"]
            except Exception:
                # If ensure fails (e.g. offline tests), proceed; downstream will fall back safely
                pass
        assert (
            read_ctx == write_ctx
        ), "read and write contexts must be the same when instantiating a FunctionManager."
        self._venvs_ctx = ContextRegistry.get_context(self, "Functions/VirtualEnvs")
        self._compositional_ctx = ContextRegistry.get_context(
            self,
            "Functions/Compositional",
        )
        self._primitives_ctx = ContextRegistry.get_context(self, "Functions/Primitives")
        self._meta_ctx = ContextRegistry.get_context(self, "Functions/Meta")

        # Track whether primitives, custom venvs, and custom functions have been synced
        self._primitives_synced = False
        self._custom_venvs_synced = False
        self._custom_functions_synced = False

        # ------------------------------------------------------------------ #
        #  File system mirroring (functions folder under FileManager root)    #
        # ------------------------------------------------------------------ #
        try:
            # Resolve a LocalFileManager instance (DI preferred, else via registry)
            self._fm: Optional[LocalFileManager] = (
                file_manager if file_manager is not None else LocalFileManager()
            )
        except Exception:
            self._fm = None

        self._functions_dir: Optional[Path] = None
        if self._fm is not None:
            try:
                # Access adapter root directly (LocalFileSystemAdapter._root)
                adapter = getattr(self._fm, "_adapter", None)
                root_dir = getattr(adapter, "_root", None) if adapter else None

                if root_dir is not None and isinstance(root_dir, Path):
                    functions_dir = root_dir / "functions"
                    functions_dir.mkdir(parents=True, exist_ok=True)
                    self._functions_dir = functions_dir
                    # Bootstrap: mirror existing functions from context to disk (idempotent)
                    self._bootstrap_functions_to_disk()
            except Exception:
                # Non-fatal – tests without FileManager still pass
                self._functions_dir = None

    @property
    def _dangerous_builtins(self) -> Set[str]:
        """
        A minimal set of truly dangerous built-ins that should never be allowed.
        These could compromise security or system integrity.
        """
        return {
            "eval",
            "exec",
            "compile",
            "__import__",
            "open",  # File system access should go through proper APIs
            "input",  # No interactive input in automated functions
            "breakpoint",  # No debugging breakpoints
            "exit",
            "quit",
        }

    def _parse_implementation(
        self,
        source: str,
    ) -> Tuple[str, ast.Module, ast.FunctionDef, str]:
        """
        Common syntactic checks (unchanged, but now returns the stripped
        source verbatim so we can persist it later).
        """
        stripped = source.lstrip("\n")
        first_line = stripped.splitlines()[0] if stripped else ""
        if first_line.startswith((" ", "\t")):
            raise ValueError(
                "Function definition must start at column 0 (no indentation).",
            )

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            raise ValueError(f"Syntax error:\n{e.text}") from e

        if len(tree.body) != 1 or not isinstance(
            tree.body[0],
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            raise ValueError(
                "Each implementation must contain exactly one top-level function.",
            )

        fn_node: Union[ast.FunctionDef, ast.AsyncFunctionDef] = tree.body[0]
        if fn_node.col_offset != 0:
            raise ValueError(
                f"Function {fn_node.name!r} must start at column 0 (no indentation).",
            )

        return fn_node.name, tree, fn_node, source

    def _collect_verified_dependencies(
        self,
        fn_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
        all_known_function_names: Set[str],
    ) -> Set[str]:
        """
        Uses the stateful _DependencyVisitor to find verified direct calls,
        indirect calls via variables, and returned function name references
        to other known library functions.
        """
        visitor = _DependencyVisitor(all_known_function_names)
        visitor.visit(fn_node)
        # Remove potential self-references if the visitor logic includes them
        visitor.dependencies.discard(fn_node.name)
        return visitor.dependencies

    def _collect_function_calls(
        self,
        fn_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    ) -> Set[str]:
        calls: Set[str] = set()
        for node in ast.walk(fn_node):
            if isinstance(node, ast.Call):
                name = self._format_callable_name(node.func)
                if name:
                    calls.add(name)
        return calls

    @staticmethod
    def _format_callable_name(callable_node: ast.AST) -> Optional[str]:
        """Return a best-effort fully qualified name for a callable.

        Handles both simple names (e.g., ``foo()``) and nested attributes
        (e.g., ``a.b.c()``). If the base of the attribute chain is not a simple
        ``ast.Name`` (e.g., ``get().b()``), this falls back to ``ast.unparse``
        when available.
        """
        # Simple function call: foo()
        if isinstance(callable_node, ast.Name):
            return callable_node.id

        # Attribute access: a.b.c()
        if isinstance(callable_node, ast.Attribute):
            parts: List[str] = []
            current: ast.AST = callable_node
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
                return ".".join(reversed(parts))
            # Fallback to unparse for complex bases like calls/subscripts
            try:
                return ast.unparse(callable_node)
            except Exception:
                pass
            return ".".join(reversed(parts)) if parts else None

        try:
            return ast.unparse(callable_node)
        except Exception:
            return None

    def _validate_function_calls(
        self,
        fn_name: str,
        calls: Set[str],
        provided_names: Set[str],
    ) -> None:
        """
        Validates function calls to prevent dangerous operations.

        Allows:
        - Built-in functions (except dangerous ones)
        - Any method calls on objects (e.g., computer_primitives.*, call_handle.*, call.*)
        - User-defined functions (tracked as dependencies)

        Disallows:
        - Dangerous built-in functions (eval, exec, etc.)
        """
        dangerous = self._dangerous_builtins

        for called in calls:
            # Allow all method calls (anything with a dot)
            # This includes computer_primitives.*, call_handle.*, obj.method(), etc.
            if "." in called:
                continue

            # Block only truly dangerous built-ins
            if called in dangerous:
                raise ValueError(
                    f"Dangerous built-in '{called}' is not permitted in {fn_name}(). "
                    f"Functions cannot use: {', '.join(sorted(dangerous))}",
                )

    # ------------------------------------------------------------------ #
    #  Private helpers for persistence                                    #
    # ------------------------------------------------------------------ #

    def _get_log_by_function_id(
        self,
        *,
        function_id: int,
        raise_if_missing: bool = True,
    ) -> Optional[unify.Log]:
        logs = unify.get_logs(
            context=self._compositional_ctx,
            filter=f"function_id == {function_id}",
            exclude_fields=list_private_fields(self._compositional_ctx),
        )
        if len(logs) == 0:
            if raise_if_missing:
                raise ValueError(f"No function with id {function_id!r} exists.")
            return None
        assert len(logs) == 1, f"Multiple functions found with id {function_id!r}."
        return logs[0]

    # ------------------------------------------------------------------ #
    #  Filesystem helpers                                                #
    # ------------------------------------------------------------------ #

    def _function_filename(self, name: str) -> str:
        """Return canonical filename for a function (no extensions in name)."""
        safe = name.strip().replace(os.sep, "_")
        return f"{safe}.py"

    def _function_path(self, name: str) -> Optional[Path]:
        if self._functions_dir is None:
            return None
        return self._functions_dir / self._function_filename(name)

    def _write_function_file(self, name: str, source: str) -> Optional[Path]:
        """Atomically write the function source into the functions folder."""
        p = self._function_path(name)
        if p is None:
            return None
        try:
            tmp = p.with_suffix(p.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(source)
            os.replace(tmp, p)
            return p
        except Exception:
            return None

    def _register_function_file(self, name: str, path: Path) -> None:
        """Register function file with FileManager as protected and visible."""
        if self._fm is None:
            return
        display = f"functions/{path.name}"
        try:
            # Idempotent: if already registered under same display, keep it
            if not self._fm.exists(display):
                self._fm.register_existing_file(
                    path,
                    display_name=display,
                    protected=True,
                )
        except Exception:
            # Best-effort registration only
            pass

    def _bootstrap_functions_to_disk(self) -> None:
        """Ensure all existing functions have a file on disk and are registered."""
        if self._functions_dir is None:
            return
        try:
            logs = unify.get_logs(
                context=self._compositional_ctx,
                exclude_fields=list_private_fields(self._compositional_ctx),
            )
            for lg in logs:
                name = lg.entries.get("name")
                impl = lg.entries.get("implementation") or ""
                if not isinstance(name, str) or not impl:
                    continue
                p = self._function_path(name)
                if p is None:
                    continue
                if not p.exists():
                    wrote = self._write_function_file(name, impl)
                    if wrote is not None:
                        self._register_function_file(name, wrote)
                else:
                    # Ensure it's registered as protected even if file already exists
                    self._register_function_file(name, p)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    #  Public API                                                        #
    # ------------------------------------------------------------------ #

    @functools.wraps(BaseFunctionManager.clear, updated=())
    def clear(self) -> None:
        unify.delete_context(self._compositional_ctx)
        unify.delete_context(self._primitives_ctx)
        unify.delete_context(self._venvs_ctx)
        unify.delete_context(self._meta_ctx)

        # Reset any manager-local counters or caches
        try:
            self._next_id = None
            self._primitives_synced = False
            self._custom_venvs_synced = False
            self._custom_functions_synced = False
        except Exception:
            pass

        # Force re-provisioning
        ContextRegistry.refresh(self, "Functions/VirtualEnvs")
        ContextRegistry.refresh(self, "Functions/Compositional")
        ContextRegistry.refresh(self, "Functions/Primitives")
        ContextRegistry.refresh(self, "Functions/Meta")

        # Verify visibility before proceeding
        try:
            import time as _time  # local import to avoid polluting module namespace

            for _ in range(3):
                try:
                    unify.get_fields(context=self._compositional_ctx)
                    break
                except Exception:
                    _time.sleep(0.05)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    #  Primitives sync                                                   #
    # ------------------------------------------------------------------ #

    def _get_stored_primitives_hash(self) -> Optional[str]:
        """Retrieve the primitives hash from the Meta context."""
        try:
            logs = unify.get_logs(
                context=self._meta_ctx,
                filter="meta_id == 1",
                limit=1,
            )
            if logs:
                return logs[0].entries.get("primitives_hash")
        except Exception:
            pass
        return None

    def _store_primitives_hash(self, hash_value: str) -> None:
        """Store the primitives hash in the Meta context."""
        try:
            logs = unify.get_logs(
                context=self._meta_ctx,
                filter="meta_id == 1",
                limit=1,
            )
            if logs:
                unify.update_logs(
                    logs=[logs[0].id],
                    context=self._meta_ctx,
                    entries={"primitives_hash": hash_value},
                    overwrite=True,
                )
            else:
                unity_create_logs(
                    context=self._meta_ctx,
                    entries=[{"meta_id": 1, "primitives_hash": hash_value}],
                    add_to_all_context=self.include_in_multi_assistant_table,
                )
        except Exception as e:
            logger.warning(f"Failed to store primitives hash: {e}")

    def _delete_all_primitives(self) -> None:
        """Delete all rows from the Primitives context."""
        try:
            logs = unify.get_logs(
                context=self._primitives_ctx,
                exclude_fields=list_private_fields(self._primitives_ctx),
            )
            if logs:
                unify.delete_logs(
                    context=self._primitives_ctx,
                    logs=[lg.id for lg in logs],
                )
                logger.debug(f"Deleted {len(logs)} primitive rows")
        except Exception as e:
            logger.warning(f"Failed to delete primitives: {e}")

    def _insert_primitives(self, primitives: Dict[str, Dict[str, Any]]) -> None:
        """Insert primitive rows into the Primitives context with explicit IDs."""
        if not primitives:
            return

        entries = []
        for name, data in primitives.items():
            entry = {
                "name": data["name"],
                "function_id": data[
                    "function_id"
                ],  # Explicit stable ID from collect_primitives()
                "argspec": data["argspec"],
                "docstring": data["docstring"],
                "embedding_text": data["embedding_text"],
                "implementation": None,
                "calls": [],
                "precondition": None,
                "verify": False,
                "is_primitive": True,
                "guidance_ids": [],
                "primitive_class": data.get("primitive_class"),
                "primitive_method": data.get("primitive_method"),
            }
            entries.append(entry)

        try:
            unity_create_logs(
                context=self._primitives_ctx,
                entries=entries,
                batched=True,
                add_to_all_context=self.include_in_multi_assistant_table,
            )
            logger.info(f"Inserted {len(entries)} primitives")
        except Exception as e:
            logger.error(f"Failed to insert primitives: {e}")

    def sync_primitives(self) -> bool:
        """
        Ensure primitives in the database match current Python definitions.

        Uses hash comparison to avoid unnecessary writes. Safe to call
        multiple times; will only perform sync if primitives have changed.

        Returns:
            True if sync was performed, False if already up-to-date.
        """
        if self._primitives_synced:
            return False

        expected = collect_primitives()
        expected_hash = compute_primitives_hash(expected)

        current_hash = self._get_stored_primitives_hash()

        if current_hash == expected_hash:
            logger.debug("Primitives hash matches, skipping sync")
            self._primitives_synced = True
            return False

        logger.info(
            f"Primitives hash mismatch (current={current_hash}, expected={expected_hash}), syncing...",
        )
        self._delete_all_primitives()
        self._insert_primitives(expected)
        self._store_primitives_hash(expected_hash)

        self._primitives_synced = True
        return True

    # ------------------------------------------------------------------ #
    #  Custom Functions Sync                                              #
    # ------------------------------------------------------------------ #

    def _get_stored_custom_functions_hash(self) -> str:
        """Retrieve the stored custom functions hash from the Meta context."""
        try:
            logs = unify.get_logs(
                context=self._meta_ctx,
                filter="meta_id == 1",
                limit=1,
            )
            if logs:
                return logs[0].entries.get("custom_functions_hash", "")
        except Exception as e:
            logger.warning(f"Failed to retrieve custom functions hash: {e}")
        return ""

    def _store_custom_functions_hash(self, hash_value: str) -> None:
        """Store the custom functions hash in the Meta context."""
        try:
            logs = unify.get_logs(
                context=self._meta_ctx,
                filter="meta_id == 1",
                limit=1,
            )
            if logs:
                unify.update_logs(
                    context=self._meta_ctx,
                    logs=[logs[0].id],
                    entries={"custom_functions_hash": hash_value},
                    overwrite=True,
                )
            else:
                # Create the meta row if it doesn't exist
                unity_create_logs(
                    context=self._meta_ctx,
                    entries=[{"meta_id": 1, "custom_functions_hash": hash_value}],
                    add_to_all_context=self.include_in_multi_assistant_table,
                )
        except Exception as e:
            logger.warning(f"Failed to store custom functions hash: {e}")

    def _get_custom_functions_from_db(self) -> Dict[str, Dict[str, Any]]:
        """Get all custom functions from the database (those with custom_hash set)."""
        logs = unify.get_logs(
            context=self._compositional_ctx,
            filter="custom_hash != None",
            exclude_fields=list_private_fields(self._compositional_ctx),
        )
        return {
            lg.entries.get("name"): lg.entries for lg in logs if lg.entries.get("name")
        }

    def _delete_custom_function_by_name(self, name: str) -> bool:
        """Delete a custom function by name."""
        logs = unify.get_logs(
            context=self._compositional_ctx,
            filter=f"name == '{name}' and custom_hash != None",
            limit=1,
        )
        if not logs:
            return False
        unify.delete_logs(
            context=self._compositional_ctx,
            logs=[logs[0].id],
        )
        return True

    def _update_custom_function(
        self,
        function_id: int,
        data: Dict[str, Any],
    ) -> None:
        """Update an existing custom function."""
        log = self._get_log_by_function_id(
            function_id=function_id,
            raise_if_missing=True,
        )
        # Update all fields except function_id (preserve it)
        update_data = {k: v for k, v in data.items() if k != "function_id"}
        unify.update_logs(
            context=self._compositional_ctx,
            logs=[log.id],
            entries=update_data,
            overwrite=True,
        )

    def _insert_custom_function(self, data: Dict[str, Any]) -> int:
        """Insert a new custom function."""
        # Remove function_id if present - let it be auto-assigned
        insert_data = {k: v for k, v in data.items() if k != "function_id"}
        result = unity_create_logs(
            context=self._compositional_ctx,
            entries=[insert_data],
            add_to_all_context=self.include_in_multi_assistant_table,
        )
        # unity_create_logs can return either a dict or a list of Log objects
        if isinstance(result, list) and len(result) > 0:
            log = result[0]
            if hasattr(log, "entries"):
                return log.entries.get("function_id", -1)
        elif isinstance(result, dict):
            log_ids = result.get("log_event_ids", [])
            if log_ids:
                logs = unify.get_logs(
                    context=self._compositional_ctx,
                    filter=f"id == {log_ids[0]}",
                    limit=1,
                )
                if logs and hasattr(logs[0], "entries"):
                    return logs[0].entries.get("function_id")
        return -1

    # ------------------------------------------------------------------ #
    #  Custom Venvs Sync                                                  #
    # ------------------------------------------------------------------ #

    def _get_stored_custom_venvs_hash(self) -> str:
        """Retrieve the stored custom venvs hash from the Meta context."""
        try:
            logs = unify.get_logs(
                context=self._meta_ctx,
                filter="meta_id == 1",
                limit=1,
            )
            if logs:
                return logs[0].entries.get("custom_venvs_hash", "")
        except Exception as e:
            logger.warning(f"Failed to retrieve custom venvs hash: {e}")
        return ""

    def _store_custom_venvs_hash(self, hash_value: str) -> None:
        """Store the custom venvs hash in the Meta context."""
        try:
            logs = unify.get_logs(
                context=self._meta_ctx,
                filter="meta_id == 1",
                limit=1,
            )
            if logs:
                unify.update_logs(
                    context=self._meta_ctx,
                    logs=[logs[0].id],
                    entries={"custom_venvs_hash": hash_value},
                    overwrite=True,
                )
            else:
                unity_create_logs(
                    context=self._meta_ctx,
                    entries=[{"meta_id": 1, "custom_venvs_hash": hash_value}],
                    add_to_all_context=self.include_in_multi_assistant_table,
                )
        except Exception as e:
            logger.warning(f"Failed to store custom venvs hash: {e}")

    def _get_custom_venvs_from_db(self) -> Dict[str, Dict[str, Any]]:
        """Get all custom venvs from the database (those with custom_hash set)."""
        logs = unify.get_logs(
            context=self._venvs_ctx,
            filter="custom_hash != None",
            exclude_fields=list_private_fields(self._venvs_ctx),
        )
        return {
            lg.entries.get("name"): lg.entries for lg in logs if lg.entries.get("name")
        }

    def _delete_custom_venv_by_name(self, name: str) -> bool:
        """Delete a custom venv by name."""
        logs = unify.get_logs(
            context=self._venvs_ctx,
            filter=f"name == '{name}' and custom_hash != None",
            limit=1,
        )
        if not logs:
            return False
        unify.delete_logs(
            context=self._venvs_ctx,
            logs=[logs[0].id],
        )
        return True

    def _update_custom_venv(self, venv_id: int, data: Dict[str, Any]) -> None:
        """Update an existing custom venv."""
        logs = unify.get_logs(
            context=self._venvs_ctx,
            filter=f"venv_id == {venv_id}",
            limit=1,
        )
        if not logs:
            raise ValueError(f"VirtualEnv with ID {venv_id} not found")
        update_data = {k: v for k, v in data.items() if k != "venv_id"}
        unify.update_logs(
            context=self._venvs_ctx,
            logs=[logs[0].id],
            entries=update_data,
            overwrite=True,
        )

    def _insert_custom_venv(self, data: Dict[str, Any]) -> int:
        """Insert a new custom venv."""
        insert_data = {k: v for k, v in data.items() if k != "venv_id"}
        result = unity_create_logs(
            context=self._venvs_ctx,
            entries=[insert_data],
            add_to_all_context=self.include_in_multi_assistant_table,
        )
        # unity_create_logs can return either a dict or a list of Log objects
        if isinstance(result, list) and len(result) > 0:
            log = result[0]
            if hasattr(log, "entries"):
                return log.entries.get("venv_id", -1)
        elif isinstance(result, dict):
            log_ids = result.get("log_event_ids", [])
            if log_ids:
                logs = unify.get_logs(
                    context=self._venvs_ctx,
                    filter=f"id == {log_ids[0]}",
                    limit=1,
                )
                if logs and hasattr(logs[0], "entries"):
                    return logs[0].entries.get("venv_id")
        return -1

    def sync_custom_venvs(self) -> Dict[str, int]:
        """
        Ensure custom venvs in the database match source definitions.

        Scans the custom/venvs/ folder for .toml files and syncs them
        to Functions/VirtualEnvs. Uses hash comparison to minimize writes.

        Behavior:
        - New venvs: inserted with auto-assigned venv_id
        - Changed venvs: updated in place (preserves venv_id)
        - Deleted venvs (in source): deleted from database
        - User-added venvs with same name: overwritten by source version

        Returns:
            Dict mapping venv name to venv_id (for use by sync_custom_functions).
        """
        if self._custom_venvs_synced:
            # Return existing name→id mapping
            db_venvs = self._get_custom_venvs_from_db()
            return {name: v["venv_id"] for name, v in db_venvs.items()}

        source_venvs = collect_custom_venvs()
        expected_hash = compute_custom_venvs_hash()
        current_hash = self._get_stored_custom_venvs_hash()

        # Quick check: if aggregate hash matches, skip detailed sync
        if current_hash == expected_hash:
            logger.debug("Custom venvs hash matches, skipping sync")
            self._custom_venvs_synced = True
            db_venvs = self._get_custom_venvs_from_db()
            return {name: v["venv_id"] for name, v in db_venvs.items()}

        logger.info(
            f"Custom venvs hash mismatch "
            f"(current={current_hash}, expected={expected_hash}), syncing...",
        )

        db_venvs = self._get_custom_venvs_from_db()
        processed_names: Set[str] = set()
        name_to_id: Dict[str, int] = {}

        for name, source_data in source_venvs.items():
            processed_names.add(name)

            if name in db_venvs:
                db_entry = db_venvs[name]
                if db_entry.get("custom_hash") != source_data["custom_hash"]:
                    logger.info(f"Updating custom venv: {name}")
                    self._update_custom_venv(
                        venv_id=db_entry["venv_id"],
                        data=source_data,
                    )
                else:
                    logger.debug(f"Custom venv unchanged: {name}")
                name_to_id[name] = db_entry["venv_id"]
            else:
                # Check for user-added venv with same name
                existing = unify.get_logs(
                    context=self._venvs_ctx,
                    filter=f"name == '{name}'",
                    limit=1,
                )
                if existing:
                    logger.info(f"Overwriting user-added venv with custom: {name}")
                    unify.delete_logs(
                        context=self._venvs_ctx,
                        logs=[existing[0].id],
                    )

                logger.info(f"Inserting custom venv: {name}")
                new_id = self._insert_custom_venv(source_data)
                name_to_id[name] = new_id

        # Delete venvs that are in DB but not in source
        for name in db_venvs:
            if name not in processed_names:
                logger.info(f"Deleting removed custom venv: {name}")
                self._delete_custom_venv_by_name(name)

        self._store_custom_venvs_hash(expected_hash)
        self._custom_venvs_synced = True

        return name_to_id

    def sync_custom_functions(
        self,
        venv_name_to_id: Optional[Dict[str, int]] = None,
    ) -> bool:
        """
        Ensure custom functions in the database match source definitions.

        Scans the custom/functions/ folder for functions decorated with
        @custom_function and syncs them to Functions/Compositional. Uses
        per-function hash comparison to minimize database writes.

        Args:
            venv_name_to_id: Optional mapping from venv name to venv_id.
                             Used to resolve venv_name in decorators.
                             If not provided, venv_name resolution is skipped.

        Behavior:
        - New functions: inserted with auto-assigned function_id
        - Changed functions: updated in place (preserves function_id)
        - Deleted functions (in source): deleted from database
        - User-added functions with same name: overwritten by source version
        - venv_name: resolved to venv_id using venv_name_to_id mapping

        Returns:
            True if sync was performed, False if already up-to-date.
        """
        if self._custom_functions_synced:
            return False

        # Collect source-defined custom functions
        source_functions = collect_custom_functions()
        expected_hash = compute_custom_functions_hash()
        current_hash = self._get_stored_custom_functions_hash()

        # Quick check: if aggregate hash matches, skip detailed sync
        if current_hash == expected_hash:
            logger.debug("Custom functions hash matches, skipping sync")
            self._custom_functions_synced = True
            return False

        logger.info(
            f"Custom functions hash mismatch "
            f"(current={current_hash}, expected={expected_hash}), syncing...",
        )

        venv_name_to_id = venv_name_to_id or {}

        # Get existing custom functions from DB
        db_functions = self._get_custom_functions_from_db()

        # Track what we've processed
        processed_names: Set[str] = set()

        # Sync each source function
        for name, source_data in source_functions.items():
            processed_names.add(name)

            # Resolve venv_name to venv_id
            venv_name = source_data.get("venv_name")
            if venv_name and venv_name in venv_name_to_id:
                source_data["venv_id"] = venv_name_to_id[venv_name]
                logger.debug(
                    f"Resolved venv_name={venv_name} to "
                    f"venv_id={source_data['venv_id']} for {name}",
                )
            # Remove venv_name from source_data (not stored in DB)
            source_data.pop("venv_name", None)

            if name in db_functions:
                db_entry = db_functions[name]
                # Check if hash changed
                if db_entry.get("custom_hash") != source_data["custom_hash"]:
                    logger.info(f"Updating custom function: {name}")
                    self._update_custom_function(
                        function_id=db_entry["function_id"],
                        data=source_data,
                    )
                else:
                    logger.debug(f"Custom function unchanged: {name}")
            else:
                # Check if there's a user-added function with same name
                # (no custom_hash) - if so, we need to delete it first
                existing = unify.get_logs(
                    context=self._compositional_ctx,
                    filter=f"name == '{name}'",
                    limit=1,
                )
                if existing:
                    logger.info(
                        f"Overwriting user-added function with custom: {name}",
                    )
                    unify.delete_logs(
                        context=self._compositional_ctx,
                        logs=[existing[0].id],
                    )

                # Insert new custom function
                logger.info(f"Inserting custom function: {name}")
                self._insert_custom_function(source_data)

        # Delete functions that are in DB but not in source
        for name in db_functions:
            if name not in processed_names:
                logger.info(f"Deleting removed custom function: {name}")
                self._delete_custom_function_by_name(name)

        # Store the new hash
        self._store_custom_functions_hash(expected_hash)

        self._custom_functions_synced = True
        return True

    def sync_custom(self) -> bool:
        """
        Sync all custom venvs and functions from source.

        This is the recommended method for syncing custom definitions.
        It ensures venvs are synced first (so venv_name can be resolved),
        then syncs functions.

        Returns:
            True if any sync was performed, False if everything up-to-date.
        """
        # Sync venvs first to get name→id mapping
        venv_name_to_id = self.sync_custom_venvs()

        # Then sync functions with the mapping
        functions_changed = self.sync_custom_functions(venv_name_to_id)

        # Return True if venvs were newly synced OR functions changed
        # (venv sync always returns a dict, not a bool, so check if hash changed)
        venvs_hash_changed = (
            self._get_stored_custom_venvs_hash() != compute_custom_venvs_hash()
            if not self._custom_venvs_synced
            else False
        )

        return venvs_hash_changed or functions_changed

    def list_primitives(self) -> Dict[str, Dict[str, Any]]:
        """
        Return a mapping of primitive name to primitive metadata.

        Returns primitives from the Primitives context. Call sync_primitives()
        first to ensure the database is up-to-date.

        Returns:
            Dict mapping primitive name to metadata dict (includes function_id).
        """
        entries: Dict[str, Dict[str, Any]] = {}
        try:
            logs = unify.get_logs(
                context=self._primitives_ctx,
                exclude_fields=list_private_fields(self._primitives_ctx),
            )
            for log in logs:
                data = {
                    "function_id": log.entries.get("function_id"),
                    "name": log.entries["name"],
                    "argspec": log.entries.get("argspec", ""),
                    "docstring": log.entries.get("docstring", ""),
                    "is_primitive": True,
                    "primitive_class": log.entries.get("primitive_class"),
                    "primitive_method": log.entries.get("primitive_method"),
                }
                entries[log.entries["name"]] = data
        except Exception as e:
            logger.warning(f"Failed to list primitives: {e}")
        return entries

    # 1. Add / register ------------------------------------------------- #

    @functools.wraps(BaseFunctionManager.add_functions, updated=())
    def add_functions(
        self,
        *,
        implementations: Union[str, List[str]],
        preconditions: Optional[Dict[str, Dict]] = None,
        verify: Optional[Dict[str, bool]] = None,
        overwrite: bool = False,
    ) -> Dict[str, str]:
        """
        Add or update functions in batch.

        Args:
            implementations: Function source code (single string or list of strings).
            preconditions: Optional preconditions for functions.
            verify: Optional verification settings (name -> bool).
            overwrite: If True, update existing functions; if False, skip duplicates.

        Returns:
            Dictionary mapping function names to status ("added", "updated", "skipped", or "error").
        """

        if preconditions is None:
            preconditions = {}
        if verify is None:
            verify = {}
        if isinstance(implementations, str):
            implementations = [implementations]

        parsed: List[Tuple[str, ast.Module, ast.FunctionDef, str]] = []
        parse_errors: Dict[str, str] = {}
        temp_names: Set[str] = set()

        # Parse all implementations
        for i, source in enumerate(implementations):
            try:
                # _parse_implementation validates basic structure (one func at col 0)
                name, tree, node, src = self._parse_implementation(source)
                parsed.append((name, tree, node, src))
                temp_names.add(name)
            except ValueError as e:
                # Associate error with name or index
                potential_name = f"implementation_{i+1}"
                try:
                    name_in_error = ast.parse(source).body[0].name
                except:
                    name_in_error = None
                key = name_in_error or potential_name
                parse_errors[key] = f"error: {e}"

        results: Dict[str, str] = parse_errors

        # Get existing functions for duplicate detection and dependency checking
        try:
            existing_functions = self.list_functions()
            existing_names = set(existing_functions.keys())
            all_known_function_names = existing_names.union(temp_names)
        except Exception as e:
            logger.warning(
                f"Failed to list existing functions for dependency check: {e}",
            )
            existing_functions = {}
            existing_names = set()
            all_known_function_names = temp_names

        # Check for duplicates and separate into new vs. existing functions
        duplicates_to_skip: Set[str] = set()
        existing_to_update: Set[str] = set()

        for name in temp_names:
            if name in existing_names:
                if overwrite:
                    # Mark for in-place update
                    existing_to_update.add(name)
                else:
                    # Skip this function - already exists
                    duplicates_to_skip.add(name)
                    results[name] = "skipped: already exists"

        # Validate dependencies and prepare entries for batch operations
        entries_to_create: List[Dict[str, Any]] = []
        entries_to_update: List[Dict[str, Any]] = []
        log_ids_to_update: List[int] = []
        log_id_to_name: Dict[int, str] = {}
        functions_to_write: List[Tuple[str, str]] = []

        for name, tree, node, source in parsed:
            if name in duplicates_to_skip:
                continue

            try:
                dependencies = self._collect_verified_dependencies(
                    node,
                    all_known_function_names,
                )
                dependencies_list = sorted(list(dependencies))

                all_calls = self._collect_function_calls(node)
                self._validate_function_calls(name, all_calls, temp_names)
                namespace = create_base_globals()
                exec(source, namespace)
                fn_obj = namespace[name]
                signature = str(inspect.signature(fn_obj))
                docstring = inspect.getdoc(fn_obj) or ""
                embedding_text = f"Function Name: {name}\nSignature: {signature}\nDocstring: {docstring}"
                precondition = preconditions.get(name)
                should_verify = verify.get(name, True)

                entry_data = {
                    "argspec": signature,
                    "docstring": docstring,
                    "implementation": source,
                    "calls": dependencies_list,
                    "embedding_text": embedding_text,
                    "precondition": precondition,
                    "verify": should_verify,
                }

                if name in existing_to_update:
                    # Update existing function
                    log_id = self._get_log_by_function_id(
                        function_id=existing_functions[name]["function_id"],
                        raise_if_missing=True,
                    ).id
                    log_ids_to_update.append(log_id)
                    log_id_to_name[log_id] = name
                    entries_to_update.append(entry_data)
                    results[name] = "updated"
                else:
                    # Create new function
                    entry_data["name"] = name
                    entry_data["guidance_ids"] = []
                    entries_to_create.append(entry_data)
                    results[name] = "added"

                functions_to_write.append((name, source))
            except ValueError as e:
                results[name] = f"error: {e}"
            except Exception as e:
                results[name] = f"error: Unexpected error - {e}"
                logger.error(
                    f"Unexpected error processing function {name}: {e}",
                    exc_info=True,
                )

        # Batch create new functions
        if entries_to_create:
            try:
                unity_create_logs(
                    context=self._compositional_ctx,
                    entries=entries_to_create,
                    batched=True,
                    add_to_all_context=self.include_in_multi_assistant_table,
                )
            except Exception as e:
                logger.error(
                    f"Failed to batch create function logs: {e}",
                    exc_info=True,
                )
                for entry in entries_to_create:
                    name = entry["name"]
                    if results.get(name) == "added":
                        results[name] = f"error: Failed to create log - {e}"
                        functions_to_write = [
                            (n, s) for n, s in functions_to_write if n != name
                        ]

        # Batch update existing functions
        if log_ids_to_update and entries_to_update:
            try:
                unify.update_logs(
                    logs=log_ids_to_update,
                    context=self._compositional_ctx,
                    entries=entries_to_update,
                    overwrite=True,
                )
            except Exception as e:
                logger.error(
                    f"Failed to batch update function logs: {e}",
                    exc_info=True,
                )
                for log_id in log_ids_to_update:
                    name = log_id_to_name.get(log_id)
                    if name and results.get(name) == "updated":
                        results[name] = f"error: Failed to update log - {e}"
                        functions_to_write = [
                            (n, s) for n, s in functions_to_write if n != name
                        ]

        # Write function files to disk
        for name, source in functions_to_write:
            p = self._write_function_file(name, source)
            if p is not None:
                self._register_function_file(name, p)

        return results

    # ------------------------------------------------------------------ #
    #  Callable return + dependency injection                             #
    # ------------------------------------------------------------------ #

    def _get_function_data_by_name(self, *, name: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single compositional function record by name.

        Returns the full stored record (as a dict) or ``None`` if not found.
        """
        # Normalize to the Unify filter grammar (and avoid quote-escaping issues).
        try:
            normalized = normalize_filter_expr(f"name == {json.dumps(name)}")
        except Exception:
            normalized = f"name == {json.dumps(name)}"

        last_exc: Exception | None = None
        import time as _time

        # The backend can return 404 for missing contexts in fresh projects/tests.
        for delay in (0.0, 0.05, 0.15):
            if delay:
                _time.sleep(delay)
            try:
                logs = unify.get_logs(
                    context=self._compositional_ctx,
                    filter=normalized,
                    limit=1,
                    exclude_fields=list_private_fields(self._compositional_ctx),
                )
                if logs:
                    return logs[0].entries
                return None
            except _UnifyRequestError as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status == 404:
                    last_exc = e
                    continue
                raise
            except Exception as e:
                last_exc = e
                break

        # Treat missing context as empty library.
        if isinstance(last_exc, _UnifyRequestError):
            status = getattr(getattr(last_exc, "response", None), "status_code", None)
            if status == 404:
                return None
        if last_exc is not None:
            raise last_exc
        return None

        """Convert function records into callables and inject them into ``namespace``."""
        callables: List[Callable[..., Any]] = []
        visited: Set[str] = set()

        for func_data in func_rows:
            name = func_data.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError("Function record missing valid 'name'")

            # Skip primitives (no stored implementation; names often contain dots).
            if func_data.get("is_primitive") is True:
                continue

            # Reuse already-injected callables where possible.
            if name in visited and name in namespace and callable(namespace.get(name)):
                callables.append(namespace[name])
                continue

            visited.add(name)  # Prevent cycles from re-injecting the root function.
            self._inject_dependencies(func_data, namespace=namespace, visited=visited)

            # Create + inject the root callable.
            if func_data.get("venv_id") is not None:
                fn = self._create_venv_callable(func_data, namespace=namespace)
                namespace[name] = fn
            else:
                fn = self._create_in_process_callable(func_data, namespace=namespace)
                namespace[name] = fn

            callables.append(fn)

        return callables

    # 2. Listing -------------------------------------------------------- #

    @functools.wraps(BaseFunctionManager.list_functions, updated=())
    def list_functions(
        self,
        *,
        include_implementations: bool = False,
    ) -> Dict[str, Dict[str, Any]]:

        entries: Dict[str, Dict[str, Any]] = {}
        for log in unify.get_logs(
            context=self._compositional_ctx,
            exclude_fields=list_private_fields(self._compositional_ctx),
        ):
            data = {
                "function_id": log.entries["function_id"],
                "argspec": log.entries["argspec"],
                "docstring": log.entries["docstring"],
                "guidance_ids": log.entries.get("guidance_ids", []),
                "verify": log.entries.get("verify", True),
                "venv_id": log.entries.get("venv_id"),
            }
            if include_implementations:
                data["implementation"] = log.entries["implementation"]
            entries[log.entries["name"]] = data
        return entries

    @functools.wraps(BaseFunctionManager.get_precondition, updated=())
    def get_precondition(self, *, function_name: str) -> Optional[Dict[str, Any]]:
        logs = unify.get_logs(
            context=self._compositional_ctx,
            filter=f"name == '{function_name}'",
            limit=1,
            exclude_fields=list_private_fields(self._compositional_ctx),
        )
        if not logs:
            return None

        return logs[0].entries.get("precondition")

    # 3. Deletion ------------------------------------------------------- #

    @functools.wraps(BaseFunctionManager.delete_function, updated=())
    def delete_function(
        self,
        *,
        function_id: Union[int, List[int]],
        delete_dependents: bool = True,
    ) -> Dict[str, str]:
        """
        Delete one or more functions and optionally their dependents in a single batch operation.

        Args:
            function_id: Function ID (int) or list of function IDs to delete.
            delete_dependents: If True, also delete all functions that depend on target(s).

        Returns:
            Dictionary mapping function names to "deleted" or "already_deleted".
        """
        # Normalize to list
        function_ids = [function_id] if isinstance(function_id, int) else function_id

        if not function_ids:
            return {}

        # Handle single function optimization
        if len(function_ids) == 1:
            log = self._get_log_by_function_id(
                function_id=function_ids[0],
                raise_if_missing=False,
            )
            if log is None:
                return {f"function_{function_ids[0]}": "already_deleted"}

            target_name = log.entries["name"]
            ids_to_delete = {function_ids[0]}
            log_ids_to_delete = [log.id]
            results = {target_name: "deleted"}
        else:
            # Multiple functions - build from all logs
            all_logs = unify.get_logs(
                context=self._compositional_ctx,
                exclude_fields=list_private_fields(self._compositional_ctx),
            )

            id_to_log = {lg.entries["function_id"]: lg for lg in all_logs}
            id_to_name = {
                lg.entries["function_id"]: lg.entries["name"] for lg in all_logs
            }

            ids_to_delete = set(function_ids)
            target_names = {
                id_to_name[fid] for fid in function_ids if fid in id_to_name
            }

            if not target_names:
                return {}

            log_ids_to_delete = [
                id_to_log[fid].id for fid in function_ids if fid in id_to_log
            ]
            results = {
                id_to_name[fid]: "deleted" for fid in function_ids if fid in id_to_name
            }

            function_calls = {
                lg.entries["function_id"]: set(lg.entries.get("calls", []))
                for lg in all_logs
            }

        if delete_dependents:
            # Get all logs if not already loaded
            if len(function_ids) == 1:
                all_logs = unify.get_logs(
                    context=self._compositional_ctx,
                    exclude_fields=list_private_fields(self._compositional_ctx),
                )
                id_to_log = {lg.entries["function_id"]: lg for lg in all_logs}
                id_to_name = {
                    lg.entries["function_id"]: lg.entries["name"] for lg in all_logs
                }
                function_calls = {
                    lg.entries["function_id"]: set(lg.entries.get("calls", []))
                    for lg in all_logs
                }
                target_names = {target_name}

            # BFS to find all transitive dependents
            to_process = set(target_names)
            processed = set()

            while to_process:
                current_name = to_process.pop()
                if current_name in processed:
                    continue
                processed.add(current_name)

                for fid, calls in function_calls.items():
                    if current_name in calls and fid not in ids_to_delete:
                        ids_to_delete.add(fid)
                        if fid in id_to_log:
                            log_ids_to_delete.append(id_to_log[fid].id)
                            dep_name = id_to_name[fid]
                            results[dep_name] = "deleted"
                            to_process.add(dep_name)

        # Batch delete all functions
        if log_ids_to_delete:
            unify.delete_logs(
                context=self._compositional_ctx,
                logs=log_ids_to_delete,
            )

        return results

    # 4. Search --------------------------------------------------------- #

    @functools.wraps(BaseFunctionManager.search_functions, updated=())
    def search_functions(
        self,
        *,
        filter: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:

        normalized = normalize_filter_expr(filter)
        # The underlying Unify backend returns 404 when a context hasn't been created yet.
        # In tests and fresh projects, contexts are created lazily, so we retry briefly and
        # then treat missing context as "no functions" rather than crashing the Actor.
        import time as _time

        last_exc: Exception | None = None
        for delay in (0.0, 0.05, 0.15):
            if delay:
                _time.sleep(delay)
            try:
                logs = unify.get_logs(
                    context=self._compositional_ctx,
                    filter=normalized,
                    offset=offset,
                    limit=limit,
                    exclude_fields=list_private_fields(self._compositional_ctx),
                )
                return [lg.entries for lg in logs]
            except _UnifyRequestError as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status == 404:
                    last_exc = e
                    continue
                raise
            except Exception as e:
                last_exc = e
                break
        # If we still see 404 after retries, treat as empty library.
        if isinstance(last_exc, _UnifyRequestError):
            status = getattr(getattr(last_exc, "response", None), "status_code", None)
            if status == 404:
                return []
        if last_exc is not None:
            raise last_exc
        return []

    # ------------------------------------------------------------------ #
    #  Accessors and disk → context sync                                 #
    # ------------------------------------------------------------------ #

    def get_function_file_path(self, name: str) -> Optional[str]:
        p = self._function_path(name)
        return str(p) if p is not None else None

    def list_function_files(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        try:
            logs = unify.get_logs(
                context=self._compositional_ctx,
                exclude_fields=list_private_fields(self._compositional_ctx),
            )
            for lg in logs:
                nm = lg.entries.get("name")
                if not isinstance(nm, str):
                    continue
                p = self._function_path(nm)
                if p is not None:
                    out[nm] = str(p)
        except Exception:
            pass
        return out

    def sync_from_disk(self, *, prefer_file_when_newer: bool = True) -> List[str]:
        """
        Reconcile function files under functions/ with the context rows.

        Policy: if the on-disk file differs from the stored implementation, update
        the context to the file contents. Returns the list of function names updated.
        """
        updated: List[str] = []
        if self._functions_dir is None:
            return updated
        try:
            # Build a map of name→(log_id, impl)
            rows = unify.get_logs(
                context=self._compositional_ctx,
                exclude_fields=list_private_fields(self._compositional_ctx),
            )
            name_to_log: Dict[str, Tuple[int, str]] = {}
            for lg in rows:
                nm = lg.entries.get("name")
                if isinstance(nm, str):
                    name_to_log[nm] = (lg.id, lg.entries.get("implementation") or "")

            for name, (log_id, stored_impl) in name_to_log.items():
                p = self._function_path(name)
                if p is None or not p.exists():
                    continue
                try:
                    file_text = p.read_text(encoding="utf-8")
                except Exception:
                    continue
                if file_text.strip() == (stored_impl or "").strip():
                    # Ensure it's registered as protected
                    self._register_function_file(name, p)
                    continue

                # Parse and validate file to rebuild signature/docstring/calls
                try:
                    nm2, tree, node, _src = self._parse_implementation(file_text)
                    if nm2 != name:
                        # Skip mismatched names; keep 1:1 name↔file mapping
                        continue
                    namespace = create_base_globals()
                    exec(file_text, namespace)
                    fn_obj = namespace[name]
                    signature = str(inspect.signature(fn_obj))
                    docstring = inspect.getdoc(fn_obj) or ""
                    calls = list(self._collect_function_calls(node))
                    embedding_text = f"Function Name: {name}\nSignature: {signature}\nDocstring: {docstring}"
                    # Update unify row
                    unify.update_logs(
                        logs=[log_id],
                        context=self._compositional_ctx,
                        entries={
                            "argspec": signature,
                            "docstring": docstring,
                            "implementation": file_text,
                            "calls": calls,
                            "embedding_text": embedding_text,
                        },
                        overwrite=True,
                    )
                    # Ensure it's registered as protected
                    self._register_function_file(name, p)
                    updated.append(name)
                except Exception:
                    continue
        except Exception:
            return updated
        return updated

    # 5. Semantic Search ------------------------------------------------ #
    @functools.wraps(BaseFunctionManager.search_functions_by_similarity, updated=())
    def search_functions_by_similarity(
        self,
        *,
        query: str,
        n: int = 5,
        include_primitives: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Search for functions by semantic similarity to a natural-language query.

        Args:
            query: Natural-language text describing the desired function(s).
            n: Number of similar results to return.
            include_primitives: If True (default), sync and include primitives
                in the search results alongside user-defined functions.

        Returns:
            Up to n results ordered by similarity, including both user functions
            and primitives (if include_primitives=True).
        """
        allowed_fields = list(Function.model_fields.keys())

        # Search user-defined functions in the Compositional context
        compositional_rows = table_search_top_k(
            context=self._compositional_ctx,
            references={"embedding_text": query},
            k=n,
            allowed_fields=allowed_fields,
            unique_id_field="function_id",
        )

        if not include_primitives:
            return compositional_rows

        # Sync and search primitives
        self.sync_primitives()

        primitive_rows = table_search_top_k(
            context=self._primitives_ctx,
            references={"embedding_text": query},
            k=n,
            allowed_fields=allowed_fields,
            unique_id_field="function_id",
        )

        # Merge and sort by the private score column (lower distance = better match)
        all_rows = compositional_rows + primitive_rows
        for row in all_rows:
            for key in row.keys():
                if key.startswith("_"):
                    all_rows.sort(key=lambda r, k=key: r.get(k, float("inf")))
                    return all_rows[:n]

        return all_rows[:n]

    # ------------------------------------------------------------------ #
    #  Inverse linkage: Functions → Guidance                              #
    # ------------------------------------------------------------------ #

    def _guidance_context(self) -> str:
        ctxs = unify.get_active_context()
        read_ctx = ctxs.get("read")
        return f"{read_ctx}/Guidance" if read_ctx else "Guidance"

    def _get_guidance_ids_for_function(self, *, function_id: int) -> List[int]:
        # Prefer reading from the function row if present
        try:
            log = self._get_log_by_function_id(function_id=function_id)
            gids = log.entries.get("guidance_ids") or []
            if isinstance(gids, list) and gids:
                return [int(g) for g in gids]
        except Exception:
            pass

        # Fallback: scan Guidance rows that reference this function via function_ids
        gctx = self._guidance_context()
        try:
            rows = unify.get_logs(
                context=gctx,
                filter=f"{int(function_id)} in function_ids",
                exclude_fields=list_private_fields(gctx),
            )
            return [
                int(r.entries.get("guidance_id"))
                for r in rows
                if r.entries.get("guidance_id") is not None
            ]
        except Exception:
            return []

    def _get_guidance_for_function(
        self,
        *,
        function_id: int,
        include_images: bool = True,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return guidance records linked to the function.

        Each dict includes: guidance_id, title, content, images (optional).
        """
        gids = self._get_guidance_ids_for_function(function_id=function_id)
        if not gids:
            return []
        if limit is not None:
            try:
                limit = int(limit)
            except Exception:
                limit = None
            if isinstance(limit, int) and limit >= 0:
                gids = gids[:limit]
        cond = " or ".join(f"guidance_id == {int(g)}" for g in gids)
        gctx = self._guidance_context()
        rows = unify.get_logs(
            context=gctx,
            filter=cond or "False",
            exclude_fields=list_private_fields(gctx),
        )
        out: List[Dict[str, Any]] = []
        for lg in rows:
            ent = lg.entries
            rec: Dict[str, Any] = {
                "guidance_id": ent.get("guidance_id"),
                "title": ent.get("title"),
                "content": ent.get("content"),
            }
            if include_images:
                rec["images"] = ent.get("images") or []
            out.append(rec)
        return out

    def _get_image_handles_for_function_guidance(
        self,
        *,
        function_id: int,
        limit: Optional[int] = None,
    ) -> List[ImageHandle]:
        """Return ImageHandle objects for images referenced by guidance linked to the function."""
        guids = self._get_guidance_for_function(
            function_id=function_id,
            include_images=True,
        )
        image_ids: List[int] = []
        for g in guids:
            imgs = g.get("images") or []
            # Support either raw list (ImageRefs) or a dict with root
            if isinstance(imgs, dict) and "root" in imgs:
                imgs = imgs.get("root") or []
            if not isinstance(imgs, list):
                continue
            for ref in imgs:
                try:
                    if isinstance(ref, dict):
                        # AnnotatedImageRef shape: {"raw_image_ref": {"image_id": X}, "annotation": ...}
                        if "raw_image_ref" in ref and isinstance(
                            ref["raw_image_ref"],
                            dict,
                        ):
                            iid = int(ref["raw_image_ref"].get("image_id"))
                            image_ids.append(iid)
                        elif "image_id" in ref:
                            image_ids.append(int(ref.get("image_id")))
                    else:
                        # If objects leaked through, try attribute access
                        iid = getattr(
                            getattr(ref, "raw_image_ref", ref),
                            "image_id",
                            None,
                        )
                        if iid is not None:
                            image_ids.append(int(iid))
                except Exception:
                    continue
        # Preserve order while de-duplicating
        image_ids = list(dict.fromkeys(image_ids))
        if limit is not None:
            try:
                limit = int(limit)
            except Exception:
                limit = None
            if isinstance(limit, int) and limit >= 0:
                image_ids = image_ids[:limit]

        im = ManagerRegistry.get_image_manager()
        return im.get_images(image_ids)

    def _attach_guidance_images_for_function_to_context(
        self,
        *,
        function_id: int,
        limit: Optional[int] = 3,
    ) -> Dict[str, Any]:
        """Attach images referenced by related guidance into the loop context.

        Returns a dict with keys:
            attached_count: int
            images: list of { meta: {...}, image: <base64> }
        """
        handles = self._get_image_handles_for_function_guidance(
            function_id=function_id,
            limit=limit,
        )
        images: List[Dict[str, Any]] = []
        for h in handles:
            try:
                raw_bytes = h.raw()
            except Exception:
                continue
            import base64

            b64 = base64.b64encode(raw_bytes).decode("utf-8")
            images.append(
                {
                    "meta": {
                        "image_id": int(h.image_id),
                        "caption": h.caption,
                        "timestamp": getattr(h.timestamp, "isoformat", lambda: "")(),
                    },
                    "image": b64,
                },
            )
        return {"attached_count": len(images), "images": images}

    # ------------------------------------------------------------------ #
    #  Virtual Environment Management                                    #
    # ------------------------------------------------------------------ #

    def add_venv(self, *, venv: str) -> int:
        """
        Add a new virtual environment configuration.

        Args:
            venv: The pyproject.toml content as a string.

        Returns:
            The auto-assigned venv_id.
        """
        result = unity_create_logs(
            context=self._venvs_ctx,
            entries=[{"venv": venv}],
            add_to_all_context=self.include_in_multi_assistant_table,
        )
        # unity_create_logs can return either a dict or a list of Log objects
        if isinstance(result, list) and len(result) > 0:
            # List of Log objects - can extract venv_id directly from entries
            log = result[0]
            if hasattr(log, "entries"):
                venv_id = log.entries.get("venv_id")
                if venv_id is not None:
                    return venv_id
        elif isinstance(result, dict):
            log_ids = result.get("log_event_ids", [])
            if log_ids:
                logs = unify.get_logs(
                    context=self._venvs_ctx,
                    filter=f"id == {log_ids[0]}",
                    limit=1,
                )
                if logs and hasattr(logs[0], "entries"):
                    venv_id = logs[0].entries.get("venv_id")
                    if venv_id is not None:
                        return venv_id
        raise RuntimeError("Failed to retrieve venv_id after creation")

    def get_venv(self, *, venv_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a virtual environment by its ID.

        Args:
            venv_id: The unique identifier of the virtual environment.

        Returns:
            Dict with venv_id and venv content, or None if not found.
        """
        logs = unify.get_logs(
            context=self._venvs_ctx,
            filter=f"venv_id == {venv_id}",
            limit=1,
            exclude_fields=list_private_fields(self._venvs_ctx),
        )
        if not logs:
            return None
        return logs[0].entries

    def list_venvs(self) -> List[Dict[str, Any]]:
        """
        List all virtual environments.

        Returns:
            List of dicts, each with venv_id and venv content.
        """
        logs = unify.get_logs(
            context=self._venvs_ctx,
            exclude_fields=list_private_fields(self._venvs_ctx),
        )
        return [lg.entries for lg in logs]

    def delete_venv(self, *, venv_id: int) -> bool:
        """
        Delete a virtual environment by its ID.

        Functions referencing this venv will have their venv_id set to None
        (falling back to the default environment) via the foreign key cascade.

        Args:
            venv_id: The unique identifier of the virtual environment.

        Returns:
            True if deleted, False if not found.
        """
        logs = unify.get_logs(
            context=self._venvs_ctx,
            filter=f"venv_id == {venv_id}",
            limit=1,
        )
        if not logs:
            return False
        unify.delete_logs(
            context=self._venvs_ctx,
            logs=[logs[0].id],
        )
        return True

    def update_venv(self, *, venv_id: int, venv: str) -> bool:
        """
        Update the content of an existing virtual environment.

        Args:
            venv_id: The unique identifier of the virtual environment.
            venv: The new pyproject.toml content.

        Returns:
            True if updated, False if not found.
        """
        logs = unify.get_logs(
            context=self._venvs_ctx,
            filter=f"venv_id == {venv_id}",
            limit=1,
        )
        if not logs:
            return False
        unify.update_logs(
            context=self._venvs_ctx,
            logs=[logs[0].id],
            entries={"venv": venv},
            overwrite=True,
        )
        return True

    def set_function_venv(
        self,
        *,
        function_id: int,
        venv_id: Optional[int],
    ) -> bool:
        """
        Set the virtual environment for a function.

        Args:
            function_id: The function to update.
            venv_id: The venv_id to associate, or None for default environment.

        Returns:
            True if updated, False if function not found.
        """
        log = self._get_log_by_function_id(
            function_id=function_id,
            raise_if_missing=False,
        )
        if log is None:
            return False
        unify.update_logs(
            context=self._compositional_ctx,
            logs=[log.id],
            entries={"venv_id": venv_id},
            overwrite=True,
        )
        return True

    def get_function_venv(self, *, function_id: int) -> Optional[Dict[str, Any]]:
        """
        Get the virtual environment associated with a function.

        Args:
            function_id: The function to query.

        Returns:
            The venv dict if the function has one, None if using default,
            or raises ValueError if function not found.
        """
        log = self._get_log_by_function_id(
            function_id=function_id,
            raise_if_missing=True,
        )
        venv_id = log.entries.get("venv_id")
        if venv_id is None:
            return None
        return self.get_venv(venv_id=venv_id)

    # ------------------------------------------------------------------ #
    #  Virtual Environment Execution Support                             #
    # ------------------------------------------------------------------ #

    def _get_venv_base_dir(self) -> Path:
        """Get the base directory for all custom venvs.

        The path includes the Unify context name to ensure isolation between
        different assistants/users and during parallel test runs.
        """
        # Get current context for isolation
        ctx = unify.get_active_context()
        ctx_name = ctx.get("read") or ctx.get("write") or "default"
        # Sanitize context name for filesystem use
        safe_ctx = ctx_name.replace("/", "_").replace("\\", "_")
        return Path.home() / ".unity" / "venvs" / safe_ctx

    def _get_venv_dir(self, venv_id: int) -> Path:
        """Get the directory for a specific venv."""
        return self._get_venv_base_dir() / str(venv_id)

    def _get_venv_python(self, venv_id: int) -> Path:
        """Get the path to the Python interpreter for a venv."""
        return self._get_venv_dir(venv_id) / ".venv" / "bin" / "python"

    def _get_venv_runner_path(self, venv_id: int) -> Path:
        """Get the path to the runner script for a venv."""
        return self._get_venv_dir(venv_id) / "venv_runner.py"

    def _get_runner_script_content(self) -> str:
        """Get the content of the standalone runner script."""
        runner_path = Path(__file__).parent / "venv_runner.py"
        return runner_path.read_text()

    def is_venv_ready(self, *, venv_id: int) -> bool:
        """
        Check if a virtual environment is ready for execution.

        Args:
            venv_id: The venv to check.

        Returns:
            True if the venv exists and is synced, False otherwise.
        """
        venv_data = self.get_venv(venv_id=venv_id)
        if venv_data is None:
            return False

        venv_dir = self._get_venv_dir(venv_id)
        pyproject_path = venv_dir / "pyproject.toml"
        python_path = self._get_venv_python(venv_id)
        runner_path = self._get_venv_runner_path(venv_id)

        # Check if all required files exist
        if not pyproject_path.exists() or not python_path.exists():
            return False

        # Check if pyproject.toml content matches (normalize line endings)
        stored_content = venv_data["venv"].strip()
        disk_content = pyproject_path.read_text().strip()
        if disk_content != stored_content:
            return False

        # Check if runner script exists
        if not runner_path.exists():
            return False

        return True

    async def prepare_venv(self, *, venv_id: int) -> Path:
        """
        Ensure a virtual environment is created and synced.

        This method is idempotent - if the venv already exists and is up-to-date,
        it returns immediately. Otherwise, it creates/updates the venv.

        Args:
            venv_id: The venv to prepare.

        Returns:
            Path to the Python interpreter in the venv.

        Raises:
            ValueError: If the venv_id does not exist.
            RuntimeError: If venv creation fails.
        """
        venv_data = self.get_venv(venv_id=venv_id)
        if venv_data is None:
            raise ValueError(f"VirtualEnv with ID {venv_id} not found")

        venv_content = venv_data["venv"]
        venv_dir = self._get_venv_dir(venv_id)
        pyproject_path = venv_dir / "pyproject.toml"
        python_path = self._get_venv_python(venv_id)
        runner_path = self._get_venv_runner_path(venv_id)

        # Check if already ready
        needs_sync = False
        if pyproject_path.exists():
            if pyproject_path.read_text().strip() != venv_content.strip():
                needs_sync = True
                logger.info(f"Venv {venv_id}: pyproject.toml changed, re-syncing")
        else:
            needs_sync = True
            logger.info(f"Venv {venv_id}: creating new venv")

        if needs_sync or not python_path.exists():
            # Create directory and write pyproject.toml
            venv_dir.mkdir(parents=True, exist_ok=True)
            pyproject_path.write_text(venv_content)

            # Run uv sync
            logger.info(f"Venv {venv_id}: running 'uv sync'...")
            import asyncio

            process = await asyncio.create_subprocess_exec(
                "uv",
                "sync",
                cwd=str(venv_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else stdout.decode()
                raise RuntimeError(
                    f"Failed to sync venv {venv_id}: {error_msg}",
                )

            logger.info(f"Venv {venv_id}: sync complete")

        # Ensure runner script is present and up-to-date
        runner_content = self._get_runner_script_content()
        if not runner_path.exists() or runner_path.read_text() != runner_content:
            runner_path.write_text(runner_content)
            logger.info(f"Venv {venv_id}: runner script installed")

        return python_path

    async def _handle_rpc_call(
        self,
        path: str,
        kwargs: Dict[str, Any],
        primitives: Optional[Any] = None,
        computer_primitives: Optional[Any] = None,
    ) -> Any:
        """
        Handle an RPC call from a subprocess.

        Args:
            path: The RPC path (e.g., "contacts.ask", "computer.click")
            kwargs: The keyword arguments for the call
            primitives: The Primitives instance for state manager access
            computer_primitives: The ComputerPrimitives instance

        Returns:
            The result of the RPC call
        """
        parts = path.split(".", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid RPC path: {path}")

        manager_name, method_name = parts

        # Handle computer primitives
        if manager_name == "computer":
            if computer_primitives is None:
                raise RuntimeError("computer_primitives not available")
            method = getattr(computer_primitives, method_name, None)
            if method is None:
                raise AttributeError(
                    f"computer_primitives has no method '{method_name}'",
                )
            # ComputerPrimitives methods are sync, but we run in async context
            if asyncio.iscoroutinefunction(method):
                return await method(**kwargs)
            return method(**kwargs)

        # Handle state manager primitives
        if primitives is None:
            raise RuntimeError("primitives not available")

        manager = getattr(primitives, manager_name, None)
        if manager is None:
            raise AttributeError(f"primitives has no manager '{manager_name}'")

        method = getattr(manager, method_name, None)
        if method is None:
            raise AttributeError(
                f"primitives.{manager_name} has no method '{method_name}'",
            )

        if asyncio.iscoroutinefunction(method):
            return await method(**kwargs)
        return method(**kwargs)

    async def execute_in_venv(
        self,
        *,
        venv_id: int,
        implementation: str,
        call_kwargs: Optional[Dict[str, Any]] = None,
        is_async: bool = True,
        primitives: Optional[Any] = None,
        computer_primitives: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Execute a function implementation in a custom virtual environment.

        This method:
        1. Ensures the venv is prepared (lazy creation on first use)
        2. Spawns a subprocess with the venv's Python interpreter
        3. Handles bidirectional RPC for primitives and computer_primitives
        4. Returns the result from the subprocess

        Args:
            venv_id: The virtual environment to use.
            implementation: The function source code.
            call_kwargs: Keyword arguments to pass to the function.
            is_async: Whether the function is async (default True).
            primitives: The Primitives instance for RPC access to state managers.
            computer_primitives: The ComputerPrimitives instance for RPC access.

        Returns:
            Dict with keys: result, error, stdout, stderr

        Raises:
            ValueError: If venv_id does not exist.
            RuntimeError: If execution fails.
        """
        call_kwargs = call_kwargs or {}

        # Ensure venv is ready
        python_path = await self.prepare_venv(venv_id=venv_id)
        runner_path = self._get_venv_runner_path(venv_id)

        # Prepare initial execution request
        execute_msg = (
            json.dumps(
                {
                    "type": "execute",
                    "implementation": implementation,
                    "call_kwargs": call_kwargs,
                    "is_async": is_async,
                },
            )
            + "\n"
        )

        # Execute in subprocess with bidirectional communication
        # Use start_new_session=True to create a new process group, allowing
        # us to kill all child processes (including multiprocessing workers)
        # with a single os.killpg() call.
        # Note: start_new_session is not supported on Windows
        use_process_group = sys.platform != "win32"
        process = await asyncio.create_subprocess_exec(
            str(python_path),
            str(runner_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=use_process_group,
        )

        # Send initial execution request
        process.stdin.write(execute_msg.encode())
        await process.stdin.drain()

        # Handle bidirectional communication
        stderr_output = []

        async def read_stderr():
            """Read stderr in background."""
            while True:
                line = await process.stderr.readline()
                if not line:
                    break
                stderr_output.append(line.decode())

        stderr_task = asyncio.create_task(read_stderr())

        try:
            while True:
                # Read next message from subprocess
                line = await process.stdout.readline()
                if not line:
                    # Process ended without sending complete message
                    await stderr_task
                    return {
                        "result": None,
                        "error": "Subprocess ended unexpectedly",
                        "stdout": "",
                        "stderr": "".join(stderr_output),
                    }

                try:
                    msg = json.loads(line.decode().strip())
                except json.JSONDecodeError as e:
                    continue  # Skip malformed lines

                msg_type = msg.get("type")

                if msg_type == "rpc_call":
                    # Handle RPC call from subprocess
                    request_id = msg.get("id")
                    path = msg.get("path", "")
                    rpc_kwargs = msg.get("kwargs", {})

                    try:
                        result = await self._handle_rpc_call(
                            path=path,
                            kwargs=rpc_kwargs,
                            primitives=primitives,
                            computer_primitives=computer_primitives,
                        )
                        response = (
                            json.dumps(
                                {
                                    "type": "rpc_result",
                                    "id": request_id,
                                    "result": self._make_json_serializable(result),
                                },
                            )
                            + "\n"
                        )
                    except Exception as e:
                        response = (
                            json.dumps(
                                {
                                    "type": "rpc_error",
                                    "id": request_id,
                                    "error": str(e),
                                },
                            )
                            + "\n"
                        )

                    process.stdin.write(response.encode())
                    await process.stdin.drain()

                elif msg_type == "complete":
                    # Subprocess finished
                    await stderr_task
                    return {
                        "result": msg.get("result"),
                        "error": msg.get("error"),
                        "stdout": msg.get("stdout", ""),
                        "stderr": msg.get("stderr", "") + "".join(stderr_output),
                    }

        except asyncio.CancelledError:
            # Task was cancelled (e.g., Actor.stop() was called)
            # Re-raise after cleanup in finally block
            raise
        except Exception as e:
            return {
                "result": None,
                "error": f"RPC error: {e}",
                "stdout": "",
                "stderr": "".join(stderr_output),
            }
        finally:
            # Cancel stderr reader task
            stderr_task.cancel()
            try:
                await stderr_task
            except asyncio.CancelledError:
                pass

            # Ensure process and all its children are terminated
            if process.returncode is None:
                await self._terminate_process_group(process, use_process_group)

    async def _terminate_process_group(
        self,
        process: asyncio.subprocess.Process,
        use_process_group: bool,
    ) -> None:
        """
        Terminate a subprocess and all its children (process group).

        Sends SIGTERM first for graceful shutdown, then SIGKILL if the process
        doesn't terminate within the timeout.

        Args:
            process: The subprocess to terminate.
            use_process_group: Whether the process was started with start_new_session=True.
        """
        try:
            if use_process_group and process.pid is not None:
                # Kill the entire process group (subprocess + all its children)
                try:
                    pgid = os.getpgid(process.pid)
                    # Send SIGTERM for graceful shutdown
                    os.killpg(pgid, signal.SIGTERM)
                except (ProcessLookupError, OSError):
                    # Process already dead or no permission
                    pass
            else:
                # Fall back to terminating just the main process
                process.terminate()

            # Wait for process to terminate
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                # Process didn't terminate gracefully, force kill
                if use_process_group and process.pid is not None:
                    try:
                        pgid = os.getpgid(process.pid)
                        os.killpg(pgid, signal.SIGKILL)
                    except (ProcessLookupError, OSError):
                        pass
                else:
                    process.kill()
                # Wait for kill to complete
                try:
                    await asyncio.wait_for(process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    pass
        except Exception:
            # Best effort cleanup - don't let cleanup errors propagate
            pass

    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert an object to a JSON-serializable form."""
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        if isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        if isinstance(obj, dict):
            return {str(k): self._make_json_serializable(v) for k, v in obj.items()}
        # Handle Pydantic models
        try:
            from pydantic import BaseModel

            if isinstance(obj, BaseModel):
                return self._make_json_serializable(obj.model_dump())
        except ImportError:
            pass
        # For other types, convert to string representation
        return str(obj)
