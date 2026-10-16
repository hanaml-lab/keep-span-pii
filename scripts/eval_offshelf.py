"""Evaluate the off-the-shelf small model, with its own head and labels, on the test split (no fine-tuning).

Its Presidio-style labels are mapped onto this study's seven types (keepspan.labels.OFFSHELF); an honorific next to
a name is merged into it. Writes results/runs/offshelf-small/{metrics_test.json, pred_test.jsonl.gz, config.json}.

    python scripts/eval_offshelf.py
"""

from __future__ import annotations

import gzip
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from keepspan import labels as L  # noqa: E402
from keepspan import metrics as M  # noqa: E402
from keepspan.models import MODELS, predict_with  # noqa: E402
from keepspan.sources import merge_adjacent  # noqa: E402
from keepspan.train import load_split  # noqa: E402


def main() -> int:
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    m = MODELS["small"]
    tok = AutoTokenizer.from_pretrained(m["repo"], revision=m["revision"])
    model = AutoModelForTokenClassification.from_pretrained(m["repo"], revision=m["revision"]).eval()
    torch.set_num_threads(8)
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    @torch.no_grad()
    def forward(ids, att):
        return model(input_ids=torch.from_numpy(ids), attention_mask=torch.from_numpy(att)).logits.numpy()

    docs = load_split("test")
    text = {d["id"]: d["text"] for d in docs}
    t0 = time.time()
    raw = predict_with(forward, tok, docs, batch_tokens=16384, id2label=id2label)
    secs = time.time() - t0
    preds = {}
    for did, spans in raw.items():
        mapped = [[s, e, L.OFFSHELF[t]] for s, e, t in spans if L.OFFSHELF.get(t)]
        preds[did] = merge_adjacent(text[did], mapped)
    out = ROOT / "results" / "runs" / "offshelf-small"
    out.mkdir(parents=True, exist_ok=True)
    met = M.summarize(docs, preds)
    (out / "metrics_test.json").write_text(json.dumps(met, indent=2) + "\n")
    with gzip.open(out / "pred_test.jsonl.gz", "wt") as f:
        for did in sorted(preds):
            f.write(json.dumps({"id": did, "spans": preds[did]}) + "\n")
    (out / "config.json").write_text(json.dumps({
        "run": "offshelf-small", "model": "small", "base": m["repo"], "base_revision": m["revision"],
        "fine_tuned": False, "label_map": L.OFFSHELF, "device": "cpu", "threads": 8, "eval_seconds": round(secs, 1),
    }, indent=2) + "\n")
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in met["headline"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
