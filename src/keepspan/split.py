"""A fixed train / validation / test split by document hash.

The split depends only on the document's normalised text, so identical documents always land in the same split,
whatever source they came from, and the split never depends on file order or on a random seed.
"""

from __future__ import annotations

import hashlib
import unicodedata

TRAIN, VALIDATION, TEST = "train", "validation", "test"


def doc_hash(text: str) -> str:
    norm = " ".join(unicodedata.normalize("NFC", text).lower().split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def split_of(h: str) -> str:
    bucket = int(h[:8], 16) % 100
    if bucket < 80:
        return TRAIN
    if bucket < 90:
        return VALIDATION
    return TEST
