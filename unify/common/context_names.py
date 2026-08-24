"""Whether a context name is one the backend will accept.

The backend enforces its own rule and reports a violation by naming the rule,
never the value: *"Invalid context name. Names can only contain alphanumeric
characters, underscores, dashes, and forward slashes. Consecutive slashes are
not allowed."* That is a perfectly good message to receive **synchronously**,
and close to useless to receive from a worker pod four retries into a dispatch,
where the caller cannot see which of fifteen destinations was at fault.

So the same rule is stated here, and checked before anything is published. A
malformed destination then fails at the call that named it, with the value and
the reason, instead of becoming a poison message that a fleet retries
indefinitely.

This deliberately duplicates a backend rule. The alternative -- discovering the
rule from a rejection -- costs a dispatch, a deploy cycle to add logging, and a
worker loop in between.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple

# Alphanumerics, underscore, dash, forward slash. No spaces, no dots.
_ALLOWED = re.compile(r"^[A-Za-z0-9_/-]+$")
_ILLEGAL_RUN = re.compile(r"[^A-Za-z0-9_/-]+")


class InvalidContextName(ValueError):
    """A context name the backend would refuse, refused earlier."""


def why_invalid(name: str) -> Optional[str]:
    """Return why *name* would be rejected, or ``None`` if it is acceptable.

    The message names the value and the specific rule, because "invalid" alone
    sends the reader back to the documentation to work out which of four
    constraints they broke.
    """
    if not isinstance(name, str) or not name.strip():
        return "a context name cannot be empty"
    if name != name.strip():
        return f"{name!r} has leading or trailing whitespace"
    if "//" in name:
        # Almost always an empty path segment from a join whose middle term
        # resolved to "" -- a table label that came back blank, for instance.
        return (
            f"{name!r} contains consecutive slashes, which means an empty path "
            "segment: some part of the path resolved to nothing"
        )
    if name.startswith("/") or name.endswith("/"):
        return f"{name!r} starts or ends with a slash, leaving an empty segment"
    if not _ALLOWED.match(name):
        bad = sorted({m for m in _ILLEGAL_RUN.findall(name)})
        shown = ", ".join(repr(b) for b in bad[:4])
        return (
            f"{name!r} contains characters a context name cannot hold ({shown}). "
            "Only letters, digits, '_', '-' and '/' are allowed -- spaces and "
            "dots are common offenders when a path is built from a file or "
            "folder name"
        )
    return None


def is_valid(name: str) -> bool:
    return why_invalid(name) is None


def sanitise_segment(segment: str) -> str:
    """Turn one arbitrary path segment into an acceptable one.

    For deriving a destination from something outside our control -- a folder
    called ``MH data extract 11th May``, a sheet name with punctuation. Runs of
    disallowed characters collapse to a single underscore rather than
    disappearing, so two distinct names cannot silently become one.

    Returns ``""`` for a segment with nothing usable in it, which the caller
    must treat as a missing name rather than substituting a default: a path
    joined through an empty segment is what produces consecutive slashes.
    """
    cleaned = _ILLEGAL_RUN.sub("_", (segment or "").replace("/", "_")).strip("_-")
    return cleaned


def join_context(*segments: str) -> str:
    """Join segments into a context path, refusing an empty one.

    An empty segment is the specific defect behind "consecutive slashes", and
    dropping it silently would move a table to a different place than the caller
    named. Raising says which position was empty.
    """
    parts: List[str] = []
    for index, segment in enumerate(segments):
        text = (segment or "").strip("/")
        if not text:
            raise InvalidContextName(
                f"segment {index} of the context path is empty, which would "
                "produce consecutive slashes; the value it was built from "
                "resolved to nothing",
            )
        parts.append(text)
    joined = "/".join(parts)
    reason = why_invalid(joined)
    if reason is not None:
        raise InvalidContextName(reason)
    return joined


def check_all(names: Iterable[str]) -> List[Tuple[str, str]]:
    """Return ``(name, reason)`` for every unacceptable name.

    Reports all of them rather than the first: a caller about to create fifteen
    destinations wants one answer listing what to fix, not fifteen round trips
    discovering them one at a time.
    """
    problems: List[Tuple[str, str]] = []
    for name in names:
        reason = why_invalid(name)
        if reason is not None:
            problems.append((name, reason))
    return problems


def assert_all_valid(names: Iterable[str], *, what: str = "destination") -> None:
    """Raise once, naming every unacceptable name.

    Raised before a dispatch is published, so the caller learns about a bad
    destination from the call that named it rather than from a worker log it
    cannot read.
    """
    problems = check_all(names)
    if not problems:
        return
    lines = "\n".join(f"  - {reason}" for _name, reason in problems)
    raise InvalidContextName(
        f"{len(problems)} {what}(s) would be rejected by the backend:\n{lines}",
    )
