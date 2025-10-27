import ast
import inspect
import functools
import os
import logging
from pathlib import Path
from typing import Dict, List, Set, Union, Tuple, Any, Optional
import unify
from ..common.embed_utils import list_private_fields
from ..common.search_utils import table_search_top_k
from ..common.sandbox_utils import create_sandbox_globals
from .types.function import Function
from .base import BaseFunctionManager
from ..common.model_to_fields import model_to_fields
from ..common.context_store import TableStore
from ..file_manager.file_manager import FileManager
from ..image_manager.image_manager import ImageManager, ImageHandle
from ..common.filter_utils import normalize_filter_expr


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
        # (We assume obj like action_provider is globally available)

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

class FunctionManager(BaseFunctionManager):
    """
    Keeps a catalogue of user-supplied Python functions that can reference
    one another.  Each function is stored in the `unify` backend so that it
    can be listed, searched and cleanly deleted (optionally cascading to
    dependants).
    """

    # ------------------------------------------------------------------ #
    #  Construction                                                      #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        *,
        daemon: bool = True,
        traced: bool = False,
        file_manager: Optional[FileManager] = None,
    ) -> None:
        # No thread behavior; keep parameter for backward compatibility
        self._daemon = daemon
        # ToDo: expose tools to LLM once needed
        self._tools: Dict[str, callable] = {}

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
        self._ctx = f"{read_ctx}/Functions" if read_ctx else "Functions"

        # Ensure functions context and fields exist deterministically
        self._provision_storage()
        # Add tracing
        if traced:
            self = unify.traced(self)

        # ------------------------------------------------------------------ #
        #  File system mirroring (functions folder under FileManager tmp)     #
        # ------------------------------------------------------------------ #
        try:
            # Resolve a FileManager instance (DI preferred)
            self._fm: Optional[FileManager] = (
                file_manager if file_manager is not None else FileManager()
            )
        except Exception:
            self._fm = None

        self._functions_dir: Optional[Path] = None
        if self._fm is not None:
            try:
                # Create <tmp>/functions
                tmp_dir = getattr(self._fm, "_tmp_dir", None)
                if isinstance(tmp_dir, Path):
                    functions_dir = tmp_dir / "functions"
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
        - Any method calls on objects (e.g., action_provider.*, call_handle.*, call.*)
        - User-defined functions (tracked as dependencies)

        Disallows:
        - Dangerous built-in functions (eval, exec, etc.)
        """
        dangerous = self._dangerous_builtins

        for called in calls:
            # Allow all method calls (anything with a dot)
            # This includes action_provider.*, call_handle.*, obj.method(), etc.
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

    def _provision_storage(self) -> None:
        """Ensure Functions context and schema exist deterministically."""
        self._store = TableStore(
            self._ctx,
            unique_keys={"function_id": "int"},
            auto_counting={"function_id": None},
            description="List of functions, with all function details stored.",
            fields=model_to_fields(Function),
        )
        self._store.ensure_context()

    def _get_log_by_function_id(self, *, function_id: int) -> unify.Log:
        logs = unify.get_logs(
            context=self._ctx,
            filter=f"function_id == {function_id}",
            exclude_fields=list_private_fields(self._ctx),
        )
        assert len(logs) == 1, f"No function with id {function_id!r} exists."
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
                context=self._ctx,
                exclude_fields=list_private_fields(self._ctx),
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
        try:
            unify.delete_context(self._ctx)
        except Exception:
            pass

        # Reset any manager-local counters or caches
        try:
            self._next_id = None
        except Exception:
            pass

        # Force re-provisioning by clearing TableStore ensure memo for this context
        try:
            from ..common.context_store import TableStore as _TS  # local import

            try:
                _TS._ENSURED.discard((unify.active_project(), self._ctx))
            except Exception:
                pass
        except Exception:
            pass

        # Recreate schema
        self._provision_storage()

        # Verify visibility before proceeding
        try:
            import time as _time  # local import to avoid polluting module namespace

            for _ in range(3):
                try:
                    unify.get_fields(context=self._ctx)
                    break
                except Exception:
                    _time.sleep(0.05)
        except Exception:
            pass

    # 1. Add / register ------------------------------------------------- #

    @functools.wraps(BaseFunctionManager.add_functions, updated=())
    def add_functions(
        self,
        *,
        implementations: Union[str, List[str]],
        preconditions: Optional[Dict[str, Dict]] = None,
        overwrite: bool = False,
    ) -> Dict[str, str]:
        """
        Add or update functions in batch.
        
        Args:
            implementations: Function source code (single string or list of strings).
            preconditions: Optional preconditions for functions.
            overwrite: If True, update existing functions; if False, skip duplicates.
            
        Returns:
            Dictionary mapping function names to status ("added", "updated", "skipped", or "error").
        """

        if preconditions is None:
            preconditions = {}
        if isinstance(implementations, str):
            implementations = [implementations]

        parsed: List[Tuple[str, ast.Module, ast.FunctionDef, str]] = []
        parse_errors: Dict[str, str] = {}
        temp_names: Set[str] = set()  # Track names parsed successfully in this batch

        # First pass: Parse all implementations
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
                dependencies = self._collect_verified_dependencies(node, all_known_function_names)
                dependencies_list = sorted(list(dependencies))

                all_calls = self._collect_function_calls(node)
                self._validate_function_calls(name, all_calls, temp_names)
                namespace = create_sandbox_globals()
                exec(source, namespace)
                fn_obj = namespace[name]
                signature = str(inspect.signature(fn_obj))
                docstring = inspect.getdoc(fn_obj) or ""
                embedding_text = f"Function Name: {name}\nSignature: {signature}\nDocstring: {docstring}"
                precondition = preconditions.get(name)

                entry_data = {
                    "argspec": signature,
                    "docstring": docstring,
                    "implementation": source,
                    "calls": dependencies_list,
                    "embedding_text": embedding_text,
                    "precondition": precondition,
                }
                
                if name in existing_to_update:
                    # Update existing function
                    log_id = self._get_log_by_function_id(
                        function_id=existing_functions[name]["function_id"],
                        raise_if_missing=True
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
                unify.create_logs(
                    context=self._ctx,
                    entries=entries_to_create,
                    batched=True,
                )
            except Exception as e:
                logger.error(f"Failed to batch create function logs: {e}", exc_info=True)
                for entry in entries_to_create:
                    name = entry["name"]
                    if results.get(name) == "added":
                        results[name] = f"error: Failed to create log - {e}"
                        functions_to_write = [(n, s) for n, s in functions_to_write if n != name]
        
        # Batch update existing functions
        if log_ids_to_update and entries_to_update:
            try:
                unify.update_logs(
                    logs=log_ids_to_update,
                    context=self._ctx,
                    entries=entries_to_update,
                    overwrite=True,
                )
            except Exception as e:
                logger.error(f"Failed to batch update function logs: {e}", exc_info=True)
                for log_id in log_ids_to_update:
                    name = log_id_to_name.get(log_id)
                    if name and results.get(name) == "updated":
                        results[name] = f"error: Failed to update log - {e}"
                        functions_to_write = [(n, s) for n, s in functions_to_write if n != name]

        # Write function files to disk
        for name, source in functions_to_write:
            p = self._write_function_file(name, source)
            if p is not None:
                self._register_function_file(name, p)

        return results

    # 2. Listing -------------------------------------------------------- #

    @functools.wraps(BaseFunctionManager.list_functions, updated=())
    def list_functions(
        self,
        *,
        include_implementations: bool = False,
    ) -> Dict[str, Dict[str, Any]]:

        entries: Dict[str, Dict[str, Any]] = {}
        for log in unify.get_logs(
            context=self._ctx,
            exclude_fields=list_private_fields(self._ctx),
        ):
            data = {
                "function_id": log.entries["function_id"],
                "argspec": log.entries["argspec"],
                "docstring": log.entries["docstring"],
                "guidance_ids": log.entries.get("guidance_ids", []),
            }
            if include_implementations:
                data["implementation"] = log.entries["implementation"]
            entries[log.entries["name"]] = data
        return entries

    @functools.wraps(BaseFunctionManager.get_precondition, updated=())
    def get_precondition(self, *, function_name: str) -> Optional[Dict[str, Any]]:
        logs = unify.get_logs(
            context=self._ctx,
            filter=f"name == '{function_name}'",
            limit=1,
            exclude_fields=list_private_fields(self._ctx),
        )
        if not logs:
            return None

        return logs[0].entries.get("precondition")

    # 3. Deletion ------------------------------------------------------- #

    @functools.wraps(BaseFunctionManager.delete_function, updated=())
    def delete_function(
        self,
        *,
        function_id: int,
        delete_dependents: bool = True,
    ) -> Dict[str, str]:

        log = self._get_log_by_function_id(function_id=function_id)
        target_name = log.entries["name"]

        # Identify dependants (direct callers)
        if delete_dependents:
            dependants = unify.get_logs(
                context=self._ctx,
                filter=f"'{target_name}' in calls",
            )
            for dep in dependants:
                if dep.entries["function_id"] == function_id:
                    continue  # skip the target itself
                self.delete_function(
                    function_id=dep.entries["function_id"],
                    delete_dependents=True,
                )

        unify.delete_logs(
            context=self._ctx,
            logs=log.id,
        )
        return {target_name: "deleted"}

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
        logs = unify.get_logs(
            context=self._ctx,
            filter=normalized,
            offset=offset,
            limit=limit,
            exclude_fields=list_private_fields(self._ctx),
        )
        return [lg.entries for lg in logs]

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
                context=self._ctx,
                exclude_fields=list_private_fields(self._ctx),
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
                context=self._ctx,
                exclude_fields=list_private_fields(self._ctx),
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
                    namespace = create_sandbox_globals()
                    exec(file_text, namespace)
                    fn_obj = namespace[name]
                    signature = str(inspect.signature(fn_obj))
                    docstring = inspect.getdoc(fn_obj) or ""
                    calls = list(self._collect_function_calls(node))
                    embedding_text = f"Function Name: {name}\nSignature: {signature}\nDocstring: {docstring}"
                    # Update unify row
                    unify.update_logs(
                        logs=[log_id],
                        context=self._ctx,
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
    ) -> List[Dict[str, Any]]:
        allowed_fields = list(Function.model_fields.keys())
        rows = table_search_top_k(
            context=self._ctx,
            references={"embedding_text": query},
            k=n,
            allowed_fields=allowed_fields,
            unique_id_field="function_id",
        )
        return rows

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

        im = ImageManager()
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
