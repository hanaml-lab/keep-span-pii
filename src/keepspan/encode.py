"""Tokenise documents into windows with BIO labels, and decode token predictions back into character spans.

Every sub-word token of a span carries the span's label (B- on the span's first token, I- after). Tokens that are
only whitespace are ignored by the loss and skipped when decoding. Keep-span tokens are labelled O when keep-span
negatives are on, and excluded from the loss (IGNORE) when they are off; `keep_mask` records which tokens those are,
so one encoding serves both conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .labels import ID2LABEL, IGNORE, LABEL2ID


@dataclass
class DocEncoding:
    doc_id: str
    input_ids: list[int]
    offsets: list[tuple[int, int]]          # stripped character offsets; (s, s) for whitespace-only tokens
    labels: list[int] = field(default_factory=list)
    keep_mask: list[bool] = field(default_factory=list)


def _strip(text: str, s: int, e: int) -> tuple[int, int]:
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def encode_doc(tokenizer, doc: dict, with_labels: bool = True) -> DocEncoding:
    text = doc["text"]
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    ids = enc["input_ids"]
    offs = [_strip(text, s, e) for s, e in enc["offset_mapping"]]
    de = DocEncoding(doc["id"], ids, offs)
    if not with_labels:
        return de
    n = len(text)
    span_of = np.full(n, -1, dtype=np.int32)
    for i, (s, e, _t) in enumerate(doc["spans"]):
        span_of[s:e] = np.where(span_of[s:e] < 0, i, span_of[s:e])
    keep = np.zeros(n, dtype=bool)
    for s, e, _t in doc.get("keep", []):
        keep[s:e] = True
    started: set[int] = set()
    labels, kmask = [], []
    for s, e in offs:
        if e <= s:
            labels.append(IGNORE)
            kmask.append(False)
            continue
        seg = span_of[s:e]
        hit = seg[seg >= 0]
        if hit.size:
            si = int(np.bincount(hit).argmax())
            typ = doc["spans"][si][2]
            prefix = "I" if si in started else "B"
            started.add(si)
            labels.append(LABEL2ID[f"{prefix}-{typ}"])
            kmask.append(False)
        elif keep[s:e].any():
            labels.append(LABEL2ID["O"])
            kmask.append(True)
        else:
            labels.append(LABEL2ID["O"])
            kmask.append(False)
    de.labels, de.keep_mask = labels, kmask
    return de


def window_starts(n: int, width: int, overlap: int) -> list[int]:
    if n <= width:
        return [0]
    stride = width - overlap
    starts = list(range(0, n - width + 1, stride))
    if starts[-1] + width < n:
        starts.append(n - width)
    return starts


def windows(de: DocEncoding, max_len: int, overlap: int, cls_id: int, sep_id: int,
            keep_negatives: bool = True) -> list[dict]:
    """Training/inference windows: [CLS] tokens [SEP], with labels (special tokens ignored)."""
    width = max_len - 2
    out = []
    for st in window_starts(len(de.input_ids), width, overlap):
        ids = [cls_id] + de.input_ids[st:st + width] + [sep_id]
        item = {"doc_id": de.doc_id, "start": st, "input_ids": ids}
        if de.labels:
            lab = list(de.labels[st:st + width])
            if not keep_negatives:
                km = de.keep_mask[st:st + width]
                lab = [IGNORE if k else x for x, k in zip(lab, km)]
            item["labels"] = [IGNORE] + lab + [IGNORE]
        out.append(item)
    return out


def merge_window_logits(n_tokens: int, pieces: list[tuple[int, np.ndarray]]) -> np.ndarray:
    """Average the logits of overlapping windows; pieces are (start token, logits without special tokens)."""
    k = pieces[0][1].shape[-1]
    acc = np.zeros((n_tokens, k), dtype=np.float32)
    cnt = np.zeros((n_tokens, 1), dtype=np.float32)
    for st, lg in pieces:
        acc[st:st + lg.shape[0]] += lg
        cnt[st:st + lg.shape[0]] += 1
    return acc / np.maximum(cnt, 1)


def decode(text: str, offsets: list[tuple[int, int]], label_ids, id2label: dict | None = None) -> list[list]:
    """Character spans [start, end, type] from per-token label ids; whitespace-only tokens are skipped."""
    id2label = id2label or ID2LABEL
    spans: list[list] = []
    cur: list | None = None
    for (s, e), li in zip(offsets, label_ids):
        if e <= s:
            continue
        lab = id2label[int(li)]
        if lab == "O":
            if cur:
                spans.append(cur)
                cur = None
            continue
        prefix, typ = lab.split("-", 1)
        if prefix == "B" or cur is None or cur[2] != typ:
            if cur:
                spans.append(cur)
            cur = [s, e, typ]
        else:
            cur[1] = e
    if cur:
        spans.append(cur)
    return spans
