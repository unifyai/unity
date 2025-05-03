"""
DOM diffing and hashing utilities for the planner verifier.

This module provides utilities for comparing DOM structures and generating
stable hashes of DOM subtrees for efficient comparison.
"""

import hashlib
import json
import difflib
from typing import Dict, Any, List, Optional


def _hash_dom(node: Optional[Dict[str, Any]]) -> str:
    """
    Recursively walk a DOM-like dict and generate a stable hash.

    This function extracts only semantic keys from the DOM structure
    (tag, text, href, alt, aria, role) and generates a stable SHA1 hash
    that can be used for quick comparison of DOM structures.

    Args:
        node: A dictionary representing a DOM node or subtree

    Returns:
        A SHA1 hash string representing the semantic content of the DOM
    """
    if not isinstance(node, dict):
        return hashlib.sha1(str(node).encode()).hexdigest()

    # Extract only semantic keys for hashing
    semantic_keys = ["tag", "text", "href", "alt", "aria", "role"]
    semantic_content = {}

    # Process regular attributes
    for key in semantic_keys:
        if key in node:
            semantic_content[key] = node[key]

    # Process aria attributes (which might be nested under 'attributes')
    if "attributes" in node and isinstance(node["attributes"], dict):
        aria_attrs = {
            k: v
            for k, v in node["attributes"].items()
            if k.startswith("aria-") or k == "role"
        }
        if aria_attrs:
            semantic_content["aria_attributes"] = aria_attrs

    # Process children recursively
    if "children" in node and isinstance(node["children"], list):
        semantic_content["children"] = [_hash_dom(child) for child in node["children"]]

    # Convert to a stable JSON string and hash
    json_str = json.dumps(semantic_content, sort_keys=True)
    return hashlib.sha1(json_str.encode()).hexdigest()


def dom_diff_summary(before: Dict[str, Any], after: Dict[str, Any]) -> str:
    """
    Generate a unified diff summary between two DOM structures.

    This function pretty-prints the two DOM dictionaries and returns a
    unified diff string that highlights the differences between them.

    Args:
        before: The DOM structure before a change
        after: The DOM structure after a change

    Returns:
        A string containing a unified diff of the two DOM structures
    """
    # Convert DOM structures to pretty-printed JSON
    before_str = json.dumps(before, sort_keys=True, indent=2).splitlines()
    after_str = json.dumps(after, sort_keys=True, indent=2).splitlines()

    # Generate unified diff
    diff = difflib.unified_diff(
        before_str, after_str, fromfile="before", tofile="after", lineterm=""
    )

    # Join diff lines into a single string
    return "\n".join(diff)
