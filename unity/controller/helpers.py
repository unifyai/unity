import re
import unicodedata


def _slug(text: str) -> str:
    # ➊ Normalise to NFKD, then drop any non‑ASCII code‑points
    ascii_only = (
        unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    )

    # ➋ Collapse every run of “not [A‑Za‑z0‑9_]” into one _
    slug = re.sub(r"\W+", "_", ascii_only, flags=re.ASCII)

    # ➌ Remove leading / trailing _ and force lowercase
    return slug.strip("_").lower()


def _pascal(text: str) -> str:
    """Turn 'my_slug_text' → 'MySlugText' (PascalCase)"""
    return "".join(word.capitalize() for word in text.split("_"))
