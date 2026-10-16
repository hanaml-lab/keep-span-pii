"""Operating curves: shift the O logit by a bias b and decode again, for each run, on the test split.

A positive b makes the model redact less; b = 0 is the model's own argmax decision (and reproduces
metrics_test.json). The curve traces removal against keep-span violation, so a condition that only moves the
operating point lies on the other condition's curve, and one that teaches the distinction lies above it.
Writes results/runs/<run>/curve_test.json.

    python scripts/curves.py small-base-s0 small-K-s0 ...      # or --all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from keepspan import metrics as M  # noqa: E402
from keepspan.encode import decode  # noqa: E402
from keepspan.models import MODELS, load_model, predict_with  # noqa: E402
from keepspan.train import load_split  # noqa: E402

BIASES = (-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
KEYS = ("f1_strict_macro", "f1_lenient_macro", "removal_core_pooled", "removal_all_pooled", "tab_keep_violation",
        "wikineural_keep_violation", "wikipedia_pred_per_1k_words", "tab_removal_all")


def curve(run: str, docs, device, threads: int) -> dict:
    from transformers import AutoTokenizer

    model_name = run.split("-")[0]
    ckpt = ROOT / "checkpoints" / run
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = load_model(model_name, str(ckpt)).to(device)
    torch.set_num_threads(threads)
    t0 = time.time()
    _, logits = predict_docs_logits(model, tok, docs, device, MODELS[model_name].get("eval_tokens", 8192))
    t_inf = time.time() - t0
    text = {d["id"]: d["text"] for d in docs}
    points = []
    for b in BIASES:
        preds = {}
        for did in text:
            if did not in logits:
                preds[did] = []
                continue
            lg, offs = logits[did]
            lg = lg.copy()
            lg[:, 0] += b
            preds[did] = decode(text[did], offs, lg.argmax(-1))
        s = M.summarize(docs, preds)
        pt = {"bias": b, **{k: s["headline"][k] for k in KEYS}}
        tab = s["by_source"].get("tab", {})
        pt["tab_removal_core"] = tab.get("removal_core")
        pt["tab_precision_lenient"] = tab.get("precision_lenient")
        points.append(pt)
    return {"run": run, "device": str(device), "inference_s": round(t_inf, 1), "points": points}


def predict_docs_logits(model, tok, docs, device, batch_tokens):


    model.eval()

    @torch.no_grad()
    def forward(ids, att):
        out = model(input_ids=torch.from_numpy(ids).to(device), attention_mask=torch.from_numpy(att).to(device))
        return out.logits.float().cpu().numpy()

    return predict_with(forward, tok, docs, batch_tokens, return_logits=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--mem-gb", type=float, default=3.0)
    args = ap.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.mem_gb * 1e9 / total))
    runs = args.runs
    if args.all:
        runs = sorted(p.name for p in (ROOT / "checkpoints").iterdir() if (p / "model.safetensors").exists())
    docs = load_split("test")
    for run in runs:
        out = ROOT / "results" / "runs" / run / "curve_test.json"
        if out.exists():
            continue
        c = curve(run, docs, device, args.threads)
        out.write_text(json.dumps(c, indent=2) + "\n")
        p0 = next(p for p in c["points"] if p["bias"] == 0.0)
        ref = json.loads((ROOT / "results" / "runs" / run / "metrics_test.json").read_text())["headline"]
        drift = max(abs((p0[k] or 0) - (ref[k] or 0)) for k in KEYS)
        print(json.dumps({"run": run, "inference_s": c["inference_s"], "max_drift_vs_train_eval": drift}),
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())


