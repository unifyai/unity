from __future__ import annotations

"""
Optional spaCy helpers for robust sentence splitting.

This module is a small, self-contained enhancement layer:
- If spaCy is available, we register a light custom component that merges
  enumeration-only sentence fragments (e.g., "1.", "(a)", "IV.") with the
  following sentence.
- If spaCy is not available, callers should fall back to regex splitting.
"""

from typing import Any

try:
    import spacy  # type: ignore
    from spacy.language import Language  # type: ignore

    SPACY_AVAILABLE = True
except Exception:  # pragma: no cover
    SPACY_AVAILABLE = False
    spacy = None  # type: ignore
    Language = None  # type: ignore


def ensure_sentence_fixes(nlp: Any) -> Any:
    """
    Ensure the enumeration sentence-boundary fix is registered on the pipeline.

    This is safe to call repeatedly; it is a no-op if spaCy is unavailable or the
    component already exists.
    """
    if not SPACY_AVAILABLE:
        return nlp
    try:
        import re as _re

        _ENUM_REGEXES = [
            _re.compile(r"^\\s*[-–—•*]?\\s*\\(?\\d+(?:\\.\\d+)*\\)?[.)]?\\s*$"),
            _re.compile(r"^\\s*[-–—•*]?\\s*\\(?[A-Z]\\)?[.)]?\\s*$"),
            _re.compile(
                r"^\\s*[-–—•*]?\\s*\\(?[IVXLCDM]+\\)?[.)]?\\s*$",
                _re.IGNORECASE,
            ),
        ]

        @Language.component("sent_fix_enums")  # type: ignore[misc]
        def sent_fix_enums(doc):  # type: ignore[no-untyped-def]
            if not len(doc):
                return doc
            starts = [i for i, tok in enumerate(doc) if tok.is_sent_start]
            if not starts:
                return doc
            prev = starts[0]
            for idx in starts[1:]:
                span_text = doc[prev:idx].text.strip()
                if any(rx.match(span_text) for rx in _ENUM_REGEXES):
                    try:
                        doc[idx].is_sent_start = False
                    except Exception:
                        pass
                else:
                    prev = idx
            return doc

        names = [p[0] for p in getattr(nlp, "pipeline", [])]
        if "sent_fix_enums" not in names:
            try:
                nlp.add_pipe("sent_fix_enums", last=True)
            except Exception:
                # Best-effort: never break sentence splitting just for this tweak.
                pass
    except Exception:
        return nlp
    return nlp
