from __future__ import annotations

import re
import unicodedata
from pythainlp.util import normalize as pythainlp_normalize

INVISIBLE_CHARS = [
    "\u00ad",  # SOFT HYPHEN
    "\u034f",  # COMBINING GRAPHEME JOINER
    "\u061c",  # ARABIC LETTER MARK
    "\u115f", "\u1160",  # HANGUL FILLERS
    "\u17b4", "\u17b5",  # KHMER VOWEL INHERENT
    "\u180e",  # MONGOLIAN VOWEL SEPARATOR
    "\u200b", "\u200c", "\u200d",  # ZWSP, ZWNJ, ZWJ
    "\u200e", "\u200f",  # LRM, RLM
    "\u2028", "\u2029",  # LINE/PARAGRAPH SEPARATOR
    "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",  # Bidi overrides
    "\u2060",  # WORD JOINER
    "\u2066", "\u2067", "\u2068", "\u2069",  # Bidi isolates
    "\ufeff",  # ZERO WIDTH NO-BREAK SPACE / BOM
]
INVISIBLE_PATTERN = re.compile("|".join(re.escape(c) for c in INVISIBLE_CHARS))
# Any run of horizontal *or* vertical whitespace collapses to a single space. The
# vertical part (newlines, CR, form/vertical feed) matters: by the time this runs the
# text has already been split into pieces and each line break's pause was captured as a
# paragraph seam, so a newline still sitting inside a piece is stray formatting. Left in,
# it reaches VoxCPM2 mid-utterance and the continuation-mode engine drops everything
# after it -- the whole tail of a multi-line chunk goes unspoken.
MULTI_SPACE_PATTERN = re.compile(r"[ \t\r\n\f\v\u00a0\u3000]+")
REPEATED_CHARS_PATTERN = re.compile(r"([^\u0e30-\u0e4e\s])\1{2,}")


def normalize_thai_text(text: str) -> str:
    """
    Encoding-only Thai text hygiene:
    - Unicode NFC normalization
    - Stripping zero-width, BOM, bidi, and invisible control codes
    - Normalizing whitespace & collapsing spaces (newlines included -> single space)
    - Collapsing 3+ repeated characters to 2
    - PyThaiNLP vowel/tone-mark order hygiene
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = INVISIBLE_PATTERN.sub("", text)
    text = MULTI_SPACE_PATTERN.sub(" ", text)
    text = REPEATED_CHARS_PATTERN.sub(r"\1\1", text)
    text = pythainlp_normalize(text)
    return text.strip()
