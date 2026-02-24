#!/usr/bin/env python3
"""Copy context data between assistants across users.

Reads logs from the source assistant and writes them into the matching
contexts of every target assistant.  Existing entries are preserved — only
new entries are appended (deduplication is based on non-auto-counting unique
keys, or full data comparison when no natural keys exist).

When --source-assistant / --target-assistants are omitted the script assumes
each user owns exactly one assistant.  If that assumption is wrong, the error
message lists every discovered assistant ID so you can re-run with the
explicit flags.

Usage examples
--------------
# Users with one assistant each — IDs are inferred:
python scripts/copy_contexts.py \\
    --source yasser@unify.ai \\
    --target dan@unify.ai \\
    --contexts Secrets Guidance

# Explicit assistant IDs (required when a user has multiple assistants):
python scripts/copy_contexts.py \\
    --source yasser@unify.ai \\
    --target dan@unify.ai \\
    --source-assistant abc123 \\
    --target-assistants def456 ghi789 \\
    --contexts Secrets Guidance "Functions/Compositional"
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from dotenv import load_dotenv

load_dotenv(override=False)

BASE_URL = os.environ["ORCHESTRA_URL"]
ADMIN_KEY = os.environ["ORCHESTRA_ADMIN_KEY"]
PROJECT = "Assistants"


def _admin_headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {ADMIN_KEY}"}


def _user_headers(api_key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def lookup_user(identifier: str) -> Dict[str, Any]:
    if "@" in identifier:
        r = requests.get(
            f"{BASE_URL}/admin/user/by-email",
            params={"email": identifier},
            headers=_admin_headers(),
        )
    else:
        r = requests.get(
            f"{BASE_URL}/admin/user/by-user-id",
            params={"user_id": identifier},
            headers=_admin_headers(),
        )
    r.raise_for_status()
    return r.json()


def list_contexts(api_key: str) -> List[Dict[str, Any]]:
    r = requests.get(
        f"{BASE_URL}/project/{PROJECT}/contexts",
        headers=_user_headers(api_key),
    )
    r.raise_for_status()
    return r.json()


def _extract_assistant_prefixes(
    contexts: List[Dict[str, Any]],
) -> List[Tuple[str, str]]:
    """Return deduplicated (prefix, assistant_id) pairs.

    Context names follow ``{user_prefix}/{assistant_id}/{suffix}``.
    We ignore aggregation contexts (``All/...``) and the bare
    ``{user_prefix}/{assistant_id}`` root context.
    """
    seen: Set[str] = set()
    results: List[Tuple[str, str]] = []
    for ctx in contexts:
        parts = ctx["name"].split("/")
        if len(parts) < 3:
            continue
        if parts[0] == "All" or (len(parts) >= 2 and parts[1] == "All"):
            continue
        prefix = f"{parts[0]}/{parts[1]}"
        if prefix not in seen:
            seen.add(prefix)
            results.append((prefix, parts[1]))
    return results


def get_logs(
    api_key: str,
    context: str,
) -> List[Dict[str, Any]]:
    """Fetch all logs from a context, paginating in batches of 1000."""
    all_logs: List[Dict[str, Any]] = []
    offset = 0
    batch = 1000
    while True:
        r = requests.get(
            f"{BASE_URL}/logs",
            params={
                "project_name": PROJECT,
                "context": context,
                "limit": batch,
                "offset": offset,
            },
            headers=_user_headers(api_key),
        )
        if r.status_code == 404:
            return []
        r.raise_for_status()
        data = r.json()
        logs = data.get("logs") or []
        all_logs.extend(logs)
        if len(logs) < batch:
            break
        offset += batch
    return all_logs


def post_logs(
    api_key: str,
    context: str,
    entries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    r = requests.post(
        f"{BASE_URL}/logs",
        json={
            "project_name": PROJECT,
            "context": context,
            "entries": entries,
        },
        headers=_user_headers(api_key),
    )
    r.raise_for_status()
    return r.json()


def _auto_counting_fields(ctx_meta: Dict[str, Any]) -> Set[str]:
    ac = ctx_meta.get("auto_counting") or {}
    return set(ac.keys())


def _natural_unique_keys(ctx_meta: Dict[str, Any]) -> List[str]:
    """Unique keys that are NOT auto-counting (i.e. meaningful natural keys)."""
    uks = set(ctx_meta.get("unique_keys") or [])
    ac = _auto_counting_fields(ctx_meta)
    return sorted(uks - ac)


def _entry_signature(
    entry: Dict[str, Any],
    natural_keys: Optional[List[str]],
    auto_fields: Set[str],
) -> str:
    """Produce a hashable signature for deduplication.

    If natural keys exist, use only those.  Otherwise fall back to the full
    entry data with auto-counting fields stripped.
    """
    if natural_keys:
        vals = tuple(entry.get(k) for k in natural_keys)
        return repr(vals)
    stripped = {k: v for k, v in entry.items() if k not in auto_fields}
    return repr(sorted(stripped.items()))


def _strip_auto_fields(
    entry: Dict[str, Any],
    auto_fields: Set[str],
) -> Dict[str, Any]:
    return {k: v for k, v in entry.items() if k not in auto_fields}


def _find_context_meta(
    contexts: List[Dict[str, Any]],
    full_name: str,
) -> Optional[Dict[str, Any]]:
    for ctx in contexts:
        if ctx["name"] == full_name:
            return ctx
    return None


def copy_context(
    source_api_key: str,
    target_api_key: str,
    source_full_ctx: str,
    target_full_ctx: str,
    source_contexts: List[Dict[str, Any]],
    target_contexts: List[Dict[str, Any]],
    dry_run: bool = False,
) -> int:
    """Copy new entries from source context to target context.

    Returns the number of entries written.
    """
    source_meta = _find_context_meta(source_contexts, source_full_ctx)
    target_meta = _find_context_meta(target_contexts, target_full_ctx)

    source_logs = get_logs(source_api_key, source_full_ctx)
    if not source_logs:
        return 0

    auto_fields = _auto_counting_fields(source_meta) if source_meta else set()
    natural_keys = _natural_unique_keys(source_meta) if source_meta else []

    existing_sigs: Set[str] = set()
    if target_meta:
        target_logs = get_logs(target_api_key, target_full_ctx)
        for log in target_logs:
            entries = log.get("entries") or {}
            sig = _entry_signature(entries, natural_keys or None, auto_fields)
            existing_sigs.add(sig)

    new_entries: List[Dict[str, Any]] = []
    for log in source_logs:
        entries = log.get("entries") or {}
        sig = _entry_signature(entries, natural_keys or None, auto_fields)
        if sig in existing_sigs:
            continue
        cleaned = _strip_auto_fields(entries, auto_fields)
        new_entries.append(cleaned)

    if not new_entries:
        return 0

    if dry_run:
        return len(new_entries)

    post_logs(target_api_key, target_full_ctx, new_entries)
    return len(new_entries)


def _resolve_single_assistant(
    prefixes: List[Tuple[str, str]],
    explicit_id: Optional[str],
    *,
    label: str,
    flag: str,
) -> Tuple[str, str]:
    """Pick exactly one (prefix, assistant_id) for the source side.

    If *explicit_id* is given, select the matching entry.  Otherwise require
    that exactly one assistant exists.  On ambiguity, print all discovered IDs
    and the flag the caller should use, then exit.
    """
    if explicit_id is not None:
        for prefix, aid in prefixes:
            if aid == explicit_id:
                return prefix, aid
        ids = [p[1] for p in prefixes]
        print(
            f"ERROR: {label} assistant '{explicit_id}' not found.\n"
            f"  Discovered {label} assistant IDs: {ids}\n"
            f"  Pass one of them via {flag}.",
            file=sys.stderr,
        )
        sys.exit(1)

    if len(prefixes) == 1:
        return prefixes[0]

    ids = [p[1] for p in prefixes]
    print(
        f"ERROR: {label} user has {len(prefixes)} assistants — cannot "
        f"auto-select.\n"
        f"  Discovered {label} assistant IDs: {ids}\n"
        f"  Re-run with {flag} <ID> to specify which one to use.",
        file=sys.stderr,
    )
    sys.exit(1)


def _resolve_target_assistants(
    prefixes: List[Tuple[str, str]],
    explicit_ids: Optional[List[str]],
) -> List[Tuple[str, str]]:
    """Pick one or more (prefix, assistant_id) pairs for the target side.

    If *explicit_ids* is given, select the matching entries (preserving
    order).  Otherwise require exactly one assistant.
    """
    if explicit_ids is not None:
        by_id = {aid: (prefix, aid) for prefix, aid in prefixes}
        selected: List[Tuple[str, str]] = []
        missing: List[str] = []
        for eid in explicit_ids:
            if eid in by_id:
                selected.append(by_id[eid])
            else:
                missing.append(eid)
        if missing:
            ids = [p[1] for p in prefixes]
            print(
                f"ERROR: target assistant(s) not found: {missing}\n"
                f"  Discovered target assistant IDs: {ids}\n"
                f"  Pass valid IDs via --target-assistants.",
                file=sys.stderr,
            )
            sys.exit(1)
        return selected

    if len(prefixes) == 1:
        return prefixes

    ids = [p[1] for p in prefixes]
    print(
        f"ERROR: target user has {len(prefixes)} assistants — cannot "
        f"auto-select.\n"
        f"  Discovered target assistant IDs: {ids}\n"
        f"  Re-run with --target-assistants <ID> [<ID> ...] to specify "
        f"which one(s) to copy into.",
        file=sys.stderr,
    )
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Copy context data between users' assistants.",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Source user (email or user ID).",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Target user (email or user ID).",
    )
    parser.add_argument(
        "--source-assistant",
        default=None,
        help=(
            "Assistant ID for the source user.  "
            "Omit to auto-detect (requires exactly one assistant)."
        ),
    )
    parser.add_argument(
        "--target-assistants",
        nargs="+",
        default=None,
        help=(
            "Assistant ID(s) for the target user.  "
            "Omit to auto-detect (requires exactly one assistant)."
        ),
    )
    parser.add_argument(
        "--contexts",
        nargs="+",
        required=True,
        help='Context suffixes to copy (e.g. Secrets Guidance "Functions/Compositional").',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without writing.",
    )
    args = parser.parse_args()

    # Resolve users
    print(f"Looking up source user: {args.source}")
    source_user = lookup_user(args.source)
    print(
        f"  → {source_user['name']} {source_user.get('last_name', '')} "
        f"(id={source_user['id']})",
    )

    print(f"Looking up target user: {args.target}")
    target_user = lookup_user(args.target)
    print(
        f"  → {target_user['name']} {target_user.get('last_name', '')} "
        f"(id={target_user['id']})",
    )

    source_api_key = source_user["api_key"]
    target_api_key = target_user["api_key"]

    # Discover assistants via context names
    print("\nDiscovering source assistants...")
    source_contexts = list_contexts(source_api_key)
    source_prefixes = _extract_assistant_prefixes(source_contexts)
    if not source_prefixes:
        print("ERROR: No assistants found for source user.", file=sys.stderr)
        sys.exit(1)

    source_prefix, source_aid = _resolve_single_assistant(
        source_prefixes,
        args.source_assistant,
        label="source",
        flag="--source-assistant",
    )
    print(f"  → assistant prefix: {source_prefix}")

    print("Discovering target assistants...")
    target_contexts = list_contexts(target_api_key)
    target_prefixes = _extract_assistant_prefixes(target_contexts)
    if not target_prefixes:
        print("ERROR: No assistants found for target user.", file=sys.stderr)
        sys.exit(1)

    target_selected = _resolve_target_assistants(
        target_prefixes,
        args.target_assistants,
    )
    for prefix, aid in target_selected:
        print(f"  → assistant prefix: {prefix}")

    # Copy each requested context suffix into every target assistant
    total_written = 0
    for suffix in args.contexts:
        source_full = f"{source_prefix}/{suffix}"
        print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Context: {suffix}")
        print(f"  source: {source_full}")

        for target_prefix, target_aid in target_selected:
            target_full = f"{target_prefix}/{suffix}"
            print(f"  target: {target_full} ... ", end="", flush=True)

            try:
                n = copy_context(
                    source_api_key=source_api_key,
                    target_api_key=target_api_key,
                    source_full_ctx=source_full,
                    target_full_ctx=target_full,
                    source_contexts=source_contexts,
                    target_contexts=target_contexts,
                    dry_run=args.dry_run,
                )
                if n > 0:
                    verb = "would write" if args.dry_run else "wrote"
                    print(f"{verb} {n} entries")
                else:
                    print("nothing new to copy")
                total_written += n
            except requests.HTTPError as e:
                print(f"FAILED ({e.response.status_code}: {e.response.text})")
            except Exception as e:
                print(f"FAILED ({e})")

    action = "Would write" if args.dry_run else "Wrote"
    print(f"\nDone. {action} {total_written} total entries.")


if __name__ == "__main__":
    main()
