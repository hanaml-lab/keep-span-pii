"""Measure the inputs of each kind of weight product, to explain where per-tensor int8 activations lose precision.

Dynamic int8 quantisation gives each activation tensor one scale, set by its largest absolute value. If a few
values are far larger than the rest, most values fall into a handful of the 256 levels. For each role of weight
product (attention input and output projections, MLP input and output projections) this records, per call, the
ratio of the largest absolute value to the 99th percentile of absolute values, and the share of the 127 positive
int8 levels that the 99th percentile reaches. Run on a few test documents of each source.

    python scripts/activation_ranges.py modernbert-base-s0 small-base-s0
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from keepspan.encode import encode_doc, windows  # noqa: E402
from keepspan.models import load_model  # noqa: E402
from keepspan.train import load_split  # noqa: E402

ROLES = {
    "attention input (Wqkv / query, key, value)": ("attn.Wqkv", "attention.self.query", "attention.self.query_proj"),
    "attention output (Wo / attention.output.dense)": ("attn.Wo", "attention.output.dense"),
    "MLP input (Wi / intermediate.dense)": ("mlp.Wi", "intermediate.dense"),
    "MLP output (Wo / output.dense)": ("mlp.Wo", "layer.{}.output.dense"),
}


def role_of(name: str) -> str | None:
    for role, pats in ROLES.items():
        for p in pats:
            if "{}" in p:
                if any(name.endswith(p.format(i)) for i in range(48)):
                    return role
            elif name.endswith(p):
                return role
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--per-source", type=int, default=5)
    args = ap.parse_args()
    from transformers import AutoTokenizer

    torch.set_num_threads(4)
    docs, seen = [], {}
    for d in load_split("test"):
        if d["source"] in ("openpii", "tab", "wnut17", "wikipedia") and seen.get(d["source"], 0) < args.per_source:
            seen[d["source"]] = seen.get(d["source"], 0) + 1
            docs.append(d)
    out = {}
    for run in args.runs:
        model_name = run.split("-")[0]
        ckpt = ROOT / "checkpoints" / run
        tok = AutoTokenizer.from_pretrained(ckpt)
        model = load_model(model_name, str(ckpt)).eval()
        stats = defaultdict(list)

        layer_worst = defaultdict(float)

        def hook(name):
            role = role_of(name)

            def f(_mod, inp, _out):
                x = inp[0].detach().float().abs().flatten()
                p99 = torch.quantile(x[:: max(1, x.numel() // 200000)], 0.99).item()
                mx = x.max().item()
                ratio = mx / max(p99, 1e-12)
                stats[role].append({"max_over_p99": ratio, "p99_levels": 127 * p99 / max(mx, 1e-12)})
                layer_worst[name] = max(layer_worst[name], ratio)
            return f

        handles = [m.register_forward_hook(hook(n)) for n, m in model.named_modules()
                   if isinstance(m, torch.nn.Linear) and role_of(n)]
        with torch.no_grad():
            for d in docs:
                for w in windows(encode_doc(tok, d, with_labels=False), 512, 128, tok.cls_token_id, tok.sep_token_id):
                    ids = torch.tensor([w["input_ids"]])
                    model(input_ids=ids, attention_mask=torch.ones_like(ids))
        for h in handles:
            h.remove()
        out[run] = {role: {"calls": len(v),
                           "median_max_over_p99": statistics.median(x["max_over_p99"] for x in v),
                           "median_p99_levels": statistics.median(x["p99_levels"] for x in v),
                           "worst_max_over_p99": max(x["max_over_p99"] for x in v),
                           "worst_p99_levels": min(x["p99_levels"] for x in v),
                           "calls_over_100x": sum(x["max_over_p99"] > 100 for x in v) / len(v),
                           "layers_over_100x": sorted(n for n, r in layer_worst.items()
                                                      if role_of(n) == role and r > 100)}
                    for role, v in stats.items()}
        print(run, json.dumps({r: {k: (round(x, 3) if isinstance(x, float) else x) for k, x in s.items()}
                               for r, s in out[run].items()}), flush=True)
    path = ROOT / "results" / "activation_ranges.json"
    prev = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps(prev | out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
