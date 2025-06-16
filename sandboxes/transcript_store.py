"""
transcript_store.py
~~~~~~~~~~~~~~~~~~~
Tiny helper that remembers every custom-scenario transcript you create in the
TaskScheduler sandbox and lets you recall – or name – them later on.

•  All data lives in  “~/.task_scheduler_transcripts.json”.
•  Every transcript is pushed on to an append-only *history* list.
•  Named entries are just a dict name → transcript (no fragile indices).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

__all__ = ["TranscriptStore"]

_STORE_PATH = Path.home() / ".task_scheduler_transcripts.json"
_DEFAULT = {"history": [], "named": {}}


class TranscriptStore:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _STORE_PATH
        try:
            self._data = json.loads(self._path.read_text())
            if not isinstance(self._data, dict):
                raise ValueError
        except Exception:
            self._data = _DEFAULT.copy()  # corrupt / missing file → start fresh

    # ------------------------------------------------------------------ #
    #  Public API                                                        #
    # ------------------------------------------------------------------ #

    def add_to_history(self, transcript: str) -> None:
        """Append *transcript* to history and persist."""
        self._data["history"].append(transcript)
        self._save()

    def get(self, key: Union[str, int]) -> str:
        """
        • *negative int*  → pick from history  (-1 last, -2 2nd-last, …)
        • *str*           → look up by name
        """
        if isinstance(key, int):
            if key >= 0:
                raise ValueError("Positive indices are not supported – use -1, -2 …")
            try:
                return self._data["history"][key]
            except IndexError:
                raise KeyError(f"No transcript at history index {key}") from None

        # string name
        try:
            return self._data["named"][key]
        except KeyError:
            raise KeyError(f"No transcript saved as “{key}”") from None

    def save_named(self, name: str, transcript: str) -> None:
        """Bind *name* to *transcript* (overwrites existing)."""
        self._data["named"][name] = transcript
        self._save()

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                  #
    # ------------------------------------------------------------------ #

    def _save(self) -> None:
        self._path.write_text(json.dumps(self._data, indent=2))
