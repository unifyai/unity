# Optional spaCy (lightweight) for robust sentence splitting
try:
    import spacy  # type: ignore
    from spacy.language import Language  # type: ignore

    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False
    spacy = None  # type: ignore
    Language = None  # type: ignore

# Register a lightweight spaCy component to merge enumeration prefixes with following sentences
if SPACY_AVAILABLE:
    import re as _re  # local alias

    _ENUM_REGEXES = [
        _re.compile(r"^\s*[-–—•*]?\s*\(?\d+(?:\.\d+)*\)?[.)]?\s*$"),
        _re.compile(r"^\s*[-–—•*]?\s*\(?[A-Z]\)?[.)]?\s*$"),
        _re.compile(r"^\s*[-–—•*]?\s*\(?[IVXLCDM]+\)?[.)]?\s*$", _re.IGNORECASE),
    ]

    @Language.component("sent_fix_enums")
    def sent_fix_enums(doc):
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
