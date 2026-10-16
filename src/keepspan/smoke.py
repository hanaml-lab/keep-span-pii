"""A CPU smoke run of the whole pipeline, with no downloads: CI runs it.

It builds a tiny word-level tokenizer and a tiny random BERT, generates Faker documents and a legal-style document
with keep spans, trains a few hundred steps with and without keep-span negatives, and reports the same metrics as
the study. The numbers say only that every stage runs; the study's results come from scripts/sweep.py.

    python -m keepspan.smoke
"""

from __future__ import annotations

import json
import random
import time

import torch

from . import metrics as M
from .encode import encode_doc, windows
from .labels import ID2LABEL, LABEL2ID, LABELS
from .models import collate, predict_with
from .synth import generate
from .train import make_steps

COURTS = ("the Supreme Court", "the Court of Appeal", "the Government", "the Ministry of Justice", "the Commission")
COUNTRIES = ("Denmark", "Poland", "Turkey", "Italy", "Croatia")


def legal_doc(i: int, rng: random.Random, names: list[str]) -> dict:
    """A short ECHR-style paragraph: the applicant and dates are redacted, institutions and countries are kept."""
    parts, spans, keep = [], [], []

    def add(text, kind=None, typ=None):
        start = sum(len(p) for p in parts)
        parts.append(text)
        if kind == "R":
            spans.append([start, start + len(text), typ])
        elif kind == "K":
            keep.append([start, start + len(text), typ])

    add("The applicant, ")
    add(rng.choice(names), "R", "NAME")
    add(", lodged an application against ")
    add(rng.choice(COUNTRIES), "K", "LOC")
    add(" on ")
    add(f"{rng.randint(1, 28)} March {rng.randint(1990, 2010)}", "R", "DATE")
    add(". ")
    add(rng.choice(COURTS), "K", "ORG")
    add(" dismissed the appeal, and ")
    add(rng.choice(COURTS), "K", "ORG")
    add(" upheld it.")
    return {"id": f"legal:{i}", "source": "tab", "text": "".join(parts), "spans": spans, "keep": keep}


def tiny_tokenizer(docs: list[dict]):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = sorted({w for d in docs for w in pre_tokenizers.BertPreTokenizer().pre_tokenize_str(d["text"])
                    for w in [w[0]]})
    vocab = {t: i for i, t in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]"] + words)}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    return PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]",
                                   sep_token="[SEP]")


def run(keep_negatives: bool, epochs: int = 8, seed: int = 0) -> dict:
    from transformers import BertConfig, BertForTokenClassification

    torch.manual_seed(seed)
    rng = random.Random(seed)
    faker_docs = generate(240, seed=7)
    names = [s_text for d in faker_docs for s_text in [d["text"][s:e] for s, e, t in d["spans"] if t == "NAME"]]
    docs = faker_docs + [legal_doc(i, rng, names) for i in range(240)]
    rng.shuffle(docs)
    train, test = docs[:400], docs[400:]
    tok = tiny_tokenizer(docs)
    cfg = BertConfig(vocab_size=tok.vocab_size, hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=128, max_position_embeddings=128, num_labels=len(LABELS),
                     id2label=dict(ID2LABEL), label2id=dict(LABEL2ID))
    model = BertForTokenClassification(cfg)
    items = []
    for d in train:
        for w in windows(encode_doc(tok, d), 128, 16, tok.cls_token_id, tok.sep_token_id, keep_negatives):
            w["n_labelled"] = sum(1 for x in w["labels"] if x != -100)
            items.append(w)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    model.train()
    for _ in range(epochs):
        for st in make_steps(items, 2048, rng, step_tokens=1024):
            n_tok = sum(items[i]["n_labelled"] for mb in st for i in mb)
            for mb in st:
                b = collate([items[i] for i in mb], tok.pad_token_id)
                logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
                (loss_fn(logits.view(-1, logits.shape[-1]), b["labels"].view(-1)) / max(1, n_tok)).backward()
            opt.step()
            opt.zero_grad()
    model.eval()

    @torch.no_grad()
    def forward(ids, att):
        return model(input_ids=torch.from_numpy(ids), attention_mask=torch.from_numpy(att)).logits.numpy()

    preds = predict_with(forward, tok, test, batch_tokens=4096, max_len=128, overlap=16)
    return M.summarize(test, preds)


def main() -> int:
    t0 = time.time()
    out = {}
    for k in (False, True):
        s = run(k)
        out["with keep-span negatives" if k else "without"] = {
            "f1_lenient_macro": round(s["macro"]["f1_lenient"], 3),
            "removal_all_pooled": round(s["pooled"]["removal_all"], 3),
            "keep_violation": round(s["by_source"]["tab"]["keep_violation"], 3),
        }
    print(json.dumps(out, indent=2))
    print(f"smoke run finished in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
