"""Span metrics for redaction.

Definitions (character offsets; gold and predicted spans are first trimmed of edge whitespace and punctuation):

- strict: a predicted span is correct when a gold redact span has the same type and the same boundaries.
- lenient: a predicted span is correct when it overlaps a gold redact span by at least one alphanumeric character,
  whatever the types; a gold span is found when a predicted span overlaps it.
- removed: every alphanumeric character of a gold span lies inside some predicted span, so nothing of it is left
  readable. Reported for the core types (NAME, LOC, CONTACT, ID) and for all types.
- keep-span violation: a keep span with at least one alphanumeric character inside a predicted span.
- character precision: the share of predicted alphanumeric characters that lie inside gold redact spans.

Counts are summed over documents (micro); `summarize` also gives per-source results and their macro mean.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .labels import CORE
from .sources import trim

COUNT_KEYS = (
    "docs", "pred", "gold", "gold_core", "strict_tp", "len_pred_tp", "len_gold_tp", "removed", "removed_core",
    "keep", "keep_violated", "pred_chars", "pred_chars_in_gold", "words",
)


def _alnum_mask(text: str) -> list[bool]:
    return [c.isalnum() for c in text]


def doc_counts(text: str, gold: list, keep: list, pred: list) -> dict:
    """Counts for one document. `gold`, `keep`, `pred` are lists of [start, end, type]."""
    alnum = _alnum_mask(text)
    n = len(text)

    def norm(spans):
        out = []
        for s, e, t in spans:
            s, e = trim(text, int(s), int(e))
            if e > s:
                out.append((s, e, t))
        return out

    gold, keep, pred = norm(gold), norm(keep), norm(pred)

    covered = bytearray(n)
    for s, e, _ in pred:
        covered[s:e] = b"\x01" * (e - s)
    in_gold = bytearray(n)
    for s, e, _ in gold:
        in_gold[s:e] = b"\x01" * (e - s)

    def overlaps(a, spans_mask):
        s, e = a[0], a[1]
        return any(spans_mask[i] and alnum[i] for i in range(s, e))

    def fully_removed(a):
        s, e = a[0], a[1]
        has = False
        for i in range(s, e):
            if alnum[i]:
                has = True
                if not covered[i]:
                    return False
        return has

    gold_set = {(s, e, t) for s, e, t in gold}
    c = dict.fromkeys(COUNT_KEYS, 0)
    c["docs"] = 1
    c["pred"] = len(pred)
    c["gold"] = len(gold)
    c["strict_tp"] = sum(1 for p in pred if p in gold_set)
    c["len_pred_tp"] = sum(1 for p in pred if overlaps(p, in_gold))
    c["len_gold_tp"] = sum(1 for g in gold if overlaps(g, covered))
    for g in gold:
        r = fully_removed(g)
        c["removed"] += r
        if g[2] in CORE:
            c["gold_core"] += 1
            c["removed_core"] += r
    c["keep"] = len(keep)
    c["keep_violated"] = sum(1 for k in keep if overlaps(k, covered))
    c["pred_chars"] = sum(1 for i in range(n) if covered[i] and alnum[i])
    c["pred_chars_in_gold"] = sum(1 for i in range(n) if covered[i] and alnum[i] and in_gold[i])
    c["words"] = len(text.split())
    return c


def _div(a: float, b: float) -> float | None:
    return a / b if b else None


def _f1(p: float | None, r: float | None) -> float | None:
    if r == 0:
        return 0.0  # nothing found (including when nothing was predicted)
    if p is None or r is None or p + r == 0:
        return None if p is None or r is None else 0.0
    return 2 * p * r / (p + r)


def rates(c: dict) -> dict:
    p_s, r_s = _div(c["strict_tp"], c["pred"]), _div(c["strict_tp"], c["gold"])
    p_l, r_l = _div(c["len_pred_tp"], c["pred"]), _div(c["len_gold_tp"], c["gold"])
    return {
        "precision_strict": p_s, "recall_strict": r_s, "f1_strict": _f1(p_s, r_s),
        "precision_lenient": p_l, "recall_lenient": r_l, "f1_lenient": _f1(p_l, r_l),
        "removal_core": _div(c["removed_core"], c["gold_core"]),
        "removal_all": _div(c["removed"], c["gold"]),
        "keep_violation": _div(c["keep_violated"], c["keep"]),
        "char_precision": _div(c["pred_chars_in_gold"], c["pred_chars"]),
        "pred_per_1k_words": _div(1000 * c["pred"], c["words"]),
        "docs_with_pred": None,
    }


def add(a: dict, b: dict) -> dict:
    return {k: a.get(k, 0) + b.get(k, 0) for k in COUNT_KEYS}


REDACT_SOURCES = ("openpii", "faker", "wnut17", "tab")


def summarize(docs: Iterable[dict], preds: dict[str, list]) -> dict:
    """docs: records (id, source, text, spans, keep); preds: id -> predicted spans."""
    per_source: dict[str, dict] = defaultdict(lambda: dict.fromkeys(COUNT_KEYS, 0))
    docs_with_pred: dict[str, int] = defaultdict(int)
    for d in docs:
        p = preds.get(d["id"], [])
        c = doc_counts(d["text"], d["spans"], d["keep"], p)
        per_source[d["source"]] = add(per_source[d["source"]], c)
        docs_with_pred[d["source"]] += bool(c["pred"])
    out: dict = {"by_source": {}, "counts": {}}
    for src, c in sorted(per_source.items()):
        r = rates(c)
        r["docs_with_pred"] = docs_with_pred[src] / c["docs"]
        out["by_source"][src] = r
        out["counts"][src] = c
    pooled = dict.fromkeys(COUNT_KEYS, 0)
    for src in REDACT_SOURCES:
        if src in per_source:
            pooled = add(pooled, per_source[src])
    out["pooled"] = rates(pooled)
    out["counts"]["pooled"] = pooled
    macro = {}
    for k in ("precision_strict", "recall_strict", "f1_strict", "precision_lenient", "recall_lenient",
              "f1_lenient", "removal_core", "removal_all", "char_precision"):
        vals = [out["by_source"][s][k] for s in REDACT_SOURCES if s in out["by_source"]
                and out["by_source"][s][k] is not None]
        macro[k] = sum(vals) / len(vals) if vals else None
    out["macro"] = macro
    # The two over-redaction measures.
    out["headline"] = {
        "f1_strict_macro": macro["f1_strict"],
        "f1_lenient_macro": macro["f1_lenient"],
        "removal_core_pooled": out["pooled"]["removal_core"],
        "removal_all_pooled": out["pooled"]["removal_all"],
        "tab_keep_violation": out["by_source"].get("tab", {}).get("keep_violation"),
        "wikineural_keep_violation": out["by_source"].get("wikineural", {}).get("keep_violation"),
        "wikipedia_pred_per_1k_words": out["by_source"].get("wikipedia", {}).get("pred_per_1k_words"),
        "tab_removal_all": out["by_source"].get("tab", {}).get("removal_all"),
    }
    return out
