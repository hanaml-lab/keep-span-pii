"""Read every source into one record shape.

A record is a dict:
    {"id": str, "source": str, "text": str,
     "spans": [[start, end, type], ...],     # character offsets of spans to redact
     "keep":  [[start, end, kind], ...]}     # character offsets of spans that must stay readable

Offsets are Python string indices into `text`, end-exclusive.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator

from . import labels as L

# --------------------------------------------------------------------------------------------------------------
# Detokenisation for the token-level corpora (WNUT-17, WikiNEuRal), with offsets.

_NO_SPACE_BEFORE = {",", ".", ";", ":", "!", "?", ")", "]", "}", "%", "'s", "'S", "n't", "'re", "'ve", "'m", "'ll",
                    "'d", "''", "...", "…"}
_NO_SPACE_AFTER = {"(", "[", "{", "``", "$", "#"}


def detokenize(tokens: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Join tokens into readable text; return the text and each token's (start, end) offsets."""
    out: list[str] = []
    offsets: list[tuple[int, int]] = []
    pos = 0
    prev = None
    quote_open = False
    for tok in tokens:
        space = prev is not None
        if tok in _NO_SPACE_BEFORE or (prev in _NO_SPACE_AFTER):
            space = False
        if tok == '"':
            if quote_open:
                space = False
            quote_open = not quote_open
        elif prev == '"' and quote_open:
            space = False
        if space:
            out.append(" ")
            pos += 1
        offsets.append((pos, pos + len(tok)))
        out.append(tok)
        pos += len(tok)
        prev = tok
    return "".join(out), offsets


def bio_to_spans(tags: list[str]) -> list[tuple[int, int, str]]:
    """Token-index spans (start, end exclusive, type) from BIO tags; an I- after O or another type starts a span."""
    spans: list[tuple[int, int, str]] = []
    cur: list | None = None
    for i, tag in enumerate(tags):
        if tag == "O":
            if cur:
                spans.append(tuple(cur))
                cur = None
            continue
        prefix, typ = tag.split("-", 1)
        if prefix == "B" or cur is None or cur[2] != typ:
            if cur:
                spans.append(tuple(cur))
            cur = [i, i + 1, typ]
        else:
            cur[1] = i + 1
    if cur:
        spans.append(tuple(cur))
    return spans


# --------------------------------------------------------------------------------------------------------------
# Span helpers.

_WS_PUNCT = " \t\n\r.,;:!?\"'()[]{}"


def trim(text: str, start: int, end: int) -> tuple[int, int]:
    """Strip whitespace and edge punctuation from a span; returns (start, end), possibly empty."""
    while start < end and text[start] in _WS_PUNCT:
        start += 1
    while end > start and text[end - 1] in _WS_PUNCT:
        end -= 1
    return start, end


def merge_adjacent(text: str, spans: list[list], types: tuple[str, ...] = ("NAME", "LOC")) -> list[list]:
    """Merge same-type spans separated only by spaces (given name + surname, number + street).

    A TITLE span followed by a NAME is merged into it; a TITLE with no name after it is dropped.
    """
    spans = sorted(spans, key=lambda s: (s[0], s[1]))
    out: list[list] = []
    for s in spans:
        if out:
            p = out[-1]
            gap = text[p[1]:s[0]]
            if gap.strip(" ") == "" and "\n" not in gap:
                if p[2] == s[2] and s[2] in types:
                    p[1] = max(p[1], s[1])
                    continue
                if p[2] == "TITLE" and s[2] == "NAME":
                    p[1], p[2] = s[1], "NAME"
                    continue
        out.append(list(s))
    return [s for s in out if s[2] != "TITLE"]


# --------------------------------------------------------------------------------------------------------------
# Loaders. Each yields records in the shape above.


def load_openpii(path: Path) -> Iterator[dict]:
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            if d.get("language") != "en":
                continue
            text = d["source_text"]
            spans = []
            for m in d["privacy_mask"]:
                typ = L.OPENPII.get(m["label"])
                if typ is None:
                    continue
                spans.append([m["start"], m["end"], typ])
            spans = merge_adjacent(text, spans)
            yield {"id": f"openpii:{d['uid']}", "source": "openpii", "text": text, "spans": spans, "keep": [],
                   "meta": {"region": d.get("region")}}


def load_tab(paths: list[Path]) -> Iterator[dict]:
    """TAB: one record per document, annotated by the first annotator in sorted order."""
    by_doc: dict[str, list[dict]] = {}
    for p in paths:
        for r in json.loads(p.read_text()):
            by_doc.setdefault(r["doc_id"], []).append(r)
    for doc_id in sorted(by_doc):
        rows = sorted(by_doc[doc_id], key=lambda r: r["annotator_id"])
        r = rows[0]
        text = r["text"]
        spans, keep = [], []
        for m in r["entity_mentions"]:
            s, e = m["start_offset"], m["end_offset"]
            typ = L.TAB[m["entity_type"]]
            if m["identifier_type"] in L.TAB_REDACT:
                spans.append([s, e, typ])
            elif m["identifier_type"] in L.TAB_KEEP:
                keep.append([s, e, m["entity_type"]])
        yield {"id": f"tab:{doc_id}", "source": "tab", "text": text, "spans": spans, "keep": keep,
               "meta": {"annotator": r["annotator_id"], "tab_split": r["dataset_type"], "n_annotators": len(rows)}}


def _token_corpus(rows: list[dict], tagnames: tuple[str, ...], source: str, id_prefix: str) -> Iterator[dict]:
    for i, row in enumerate(rows):
        tokens = list(row["tokens"])
        tags = [tagnames[t] for t in row["ner_tags"]]
        text, offs = detokenize(tokens)
        spans, keep = [], []
        for ts, te, typ in bio_to_spans(tags):
            s, e = offs[ts][0], offs[te - 1][1]
            if source == "wnut17":
                mapped = L.WNUT[typ]
                if mapped is not None:
                    spans.append([s, e, mapped])
            else:  # wikineural: every entity is a public entity to keep
                keep.append([s, e, typ])
        yield {"id": f"{id_prefix}:{row.get('id', i)}", "source": source, "text": text, "spans": spans,
               "keep": keep, "meta": {}}


def load_wnut(paths: list[Path]) -> Iterator[dict]:
    import pyarrow.parquet as pq

    for p in paths:
        split = p.parent.name
        rows = pq.read_table(p).to_pylist()
        yield from _token_corpus(rows, L.WNUT_TAGS, "wnut17", f"wnut17:{split}")


def load_wikineural(paths: list[Path]) -> Iterator[dict]:
    import pyarrow.parquet as pq

    for p in paths:
        split = p.name.split("_")[0]
        rows = pq.read_table(p).to_pylist()
        yield from _token_corpus(rows, L.WIKINEURAL_TAGS, "wikineural", f"wikineural:{split}")


_PARA_MIN, _PARA_MAX = 200, 1500


def load_wikipedia(path: Path, wikineural_shingles: set[tuple[str, ...]] | None = None,
                   shingle: int = 8) -> Iterator[dict]:
    """Prose paragraphs of English Wikipedia articles, every token outside a span.

    Paragraphs sharing any 8-word shingle with a WikiNEuRal sentence are dropped, so the Wikipedia negatives never
    contain a sentence of the public-entity evaluation set.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=2000, columns=["id", "title", "text"]):
        for art in batch.to_pylist():
            for j, para in enumerate(art["text"].split("\n")):
                para = para.strip()
                if not (_PARA_MIN <= len(para) <= _PARA_MAX):
                    continue
                if para.count(".") < 2 or "|" in para or para.endswith(":"):
                    continue
                if wikineural_shingles is not None:
                    words = normalize_words(para)
                    if any(tuple(words[k:k + shingle]) in wikineural_shingles
                           for k in range(0, max(0, len(words) - shingle + 1))):
                        continue
                yield {"id": f"wikipedia:{art['id']}:{j}", "source": "wikipedia", "text": para, "spans": [],
                       "keep": [], "meta": {"title": art["title"]}}


_WORD = re.compile(r"[a-z0-9]+")


def normalize_words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def shingles(text: str, n: int = 8) -> set[tuple[str, ...]]:
    w = normalize_words(text)
    return {tuple(w[k:k + n]) for k in range(0, max(0, len(w) - n + 1))}
