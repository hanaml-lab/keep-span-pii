"""The three span models, at the revisions this study used, and windowed inference with PyTorch."""

from __future__ import annotations

import numpy as np
import torch

from .encode import decode, encode_doc, merge_window_logits, windows
from .labels import ID2LABEL, LABEL2ID, LABELS

MODELS: dict[str, dict] = {
    "small": {
        "repo": "gravitee-io/bert-small-pii-detection",
        "revision": "f8c27a85c51c0168f07b9dcf00265bf0a4097939",
        "licence": "Apache-2.0",
        "lr": 1e-4,
        "micro_tokens": 16384,
        "eval_tokens": 32768,
    },
    "deberta": {
        "repo": "microsoft/deberta-v3-base",
        "revision": "8ccc9b6f36199bec6961081d44eb72fb3f7353f3",
        "licence": "MIT",
        "lr": 3e-5,
        "micro_tokens": 4096,
        "eval_tokens": 4096,
        "grad_ckpt": True,
    },
    "modernbert": {
        "repo": "answerdotai/ModernBERT-base",
        "revision": "8949b909ec900327062f0ebf497f51aef5e6f0c8",
        "licence": "Apache-2.0",
        "lr": 5e-5,
        "micro_tokens": 8192,
        "eval_tokens": 8192,
        "grad_ckpt": True,
    },
}

MAX_LEN = 512
OVERLAP = 128


def load_tokenizer(name: str):
    from transformers import AutoTokenizer

    m = MODELS[name]
    return AutoTokenizer.from_pretrained(m["repo"], revision=m["revision"], use_fast=True)


def load_model(name: str, path: str | None = None):
    """A fresh token-classification head over the pretrained encoder (path=None), or a fine-tuned checkpoint."""
    from transformers import AutoModelForTokenClassification

    m = MODELS[name]
    kw = {}
    if name == "modernbert":
        kw["reference_compile"] = False
    if path is not None:
        return AutoModelForTokenClassification.from_pretrained(path, **kw)
    return AutoModelForTokenClassification.from_pretrained(
        m["repo"], revision=m["revision"], num_labels=len(LABELS), id2label=dict(ID2LABEL),
        label2id=dict(LABEL2ID), ignore_mismatched_sizes=True, **kw)


def collate(items: list[dict], pad_id: int, device=None) -> dict:
    n = max(len(x["input_ids"]) for x in items)
    ids = torch.full((len(items), n), pad_id, dtype=torch.long)
    att = torch.zeros((len(items), n), dtype=torch.long)
    lab = None
    if "labels" in items[0]:
        lab = torch.full((len(items), n), -100, dtype=torch.long)
    for i, x in enumerate(items):
        k = len(x["input_ids"])
        ids[i, :k] = torch.tensor(x["input_ids"])
        att[i, :k] = 1
        if lab is not None:
            lab[i, :k] = torch.tensor(x["labels"])
    out = {"input_ids": ids, "attention_mask": att}
    if lab is not None:
        out["labels"] = lab
    if device is not None:
        out = {k: v.to(device) for k, v in out.items()}
    return out


def predict_with(forward, tokenizer, docs: list[dict], batch_tokens: int = 16384, id2label: dict | None = None,
                 max_len: int = MAX_LEN, overlap: int = OVERLAP, return_logits: bool = False):
    """Windowed prediction over whole documents with any forward function (PyTorch or onnxruntime).

    forward(input_ids, attention_mask) takes int64 numpy arrays (batch, length) and returns float logits
    (batch, length, labels). Returns doc id -> character spans [start, end, type]; with return_logits, also
    doc id -> (merged token logits, token offsets).
    """
    encs = {d["id"]: encode_doc(tokenizer, d, with_labels=False) for d in docs}
    items = []
    for de in encs.values():
        items += windows(de, max_len, overlap, tokenizer.cls_token_id, tokenizer.sep_token_id)
    order = sorted(range(len(items)), key=lambda i: len(items[i]["input_ids"]))
    pieces: dict[str, list] = {d: [] for d in encs}

    def flush(batch):
        n = max(len(items[i]["input_ids"]) for i in batch)
        ids = np.full((len(batch), n), tokenizer.pad_token_id, dtype=np.int64)
        att = np.zeros((len(batch), n), dtype=np.int64)
        for j, i in enumerate(batch):
            k = len(items[i]["input_ids"])
            ids[j, :k] = items[i]["input_ids"]
            att[j, :k] = 1
        logits = forward(ids, att)
        for j, i in enumerate(batch):
            k = len(items[i]["input_ids"])
            pieces[items[i]["doc_id"]].append((items[i]["start"], logits[j, 1:k - 1]))

    batch: list[int] = []
    for i in order:
        n = len(items[i]["input_ids"])
        if batch and (len(batch) + 1) * n > batch_tokens:
            flush(batch)
            batch = []
        batch.append(i)
    if batch:
        flush(batch)
    out, logits = {}, {}
    text_of = {d["id"]: d["text"] for d in docs}
    for did, de in encs.items():
        if not de.input_ids:
            out[did] = []
            continue
        lg = merge_window_logits(len(de.input_ids), pieces[did])
        out[did] = decode(text_of[did], de.offsets, lg.argmax(-1), id2label)
        if return_logits:
            logits[did] = (lg, de.offsets)
    return (out, logits) if return_logits else out


@torch.no_grad()
def predict_docs(model, tokenizer, docs: list[dict], device, batch_tokens: int = 16384, autocast: bool = False,
                 id2label: dict | None = None, max_len: int = MAX_LEN, overlap: int = OVERLAP) -> dict[str, list]:
    """predict_with for a PyTorch model."""
    model.eval()

    def forward(ids, att):
        ids_t = torch.from_numpy(ids).to(device)
        att_t = torch.from_numpy(att).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast):
            logits = model(input_ids=ids_t, attention_mask=att_t).logits
        return logits.float().cpu().numpy()

    return predict_with(forward, tokenizer, docs, batch_tokens, id2label, max_len, overlap)


def count_params(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    emb = model.get_input_embeddings().weight.numel()
    return {"total": total, "input_embeddings": emb}


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
