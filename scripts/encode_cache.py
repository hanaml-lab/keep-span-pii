"""Tokenise the training split once per model (data/cache/<model>-train.pkl) and print its size."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from keepspan.models import MAX_LEN, OVERLAP, load_tokenizer  # noqa: E402
from keepspan.encode import windows  # noqa: E402
from keepspan.train import encoded_train, load_split  # noqa: E402

for name in sys.argv[1:] or ["small", "deberta", "modernbert"]:
    tok = load_tokenizer(name)
    docs = load_split("train")
    encs = encoded_train(name, tok, docs)
    by = Counter()
    win = Counter()
    for d in docs:
        ws = windows(encs[d["id"]], MAX_LEN, OVERLAP, tok.cls_token_id, tok.sep_token_id)
        win[d["source"]] += len(ws)
        by[d["source"]] += sum(len(w["input_ids"]) for w in ws)
    print(name, "tokens by source", dict(by), "windows", dict(win), "total", sum(by.values()), flush=True)
