"""Evaluate an int8 recipe that keeps some weight products in fp32, on the full test split.

The default dynamic recipe (scripts/onnx_queue.py) quantises every weight product. A recipe here names the products
left in fp32, chosen from the diagnosis of scripts/int8_variants.py. Writes, per run,
results/runs/<run>/metrics_test_int8_<recipe>.json and onnx_<recipe>.json (sizes and times); the int8 file goes to
checkpoints/<run>/onnx/model_int8_<recipe>.onnx.

    python scripts/eval_int8_recipe.py --recipe fp32-mlp-out modernbert-base-s0 modernbert-K-s0
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from keepspan import metrics as M  # noqa: E402
from keepspan.export import forward_fn, session  # noqa: E402
from keepspan.models import predict_with  # noqa: E402
from keepspan.train import load_split  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from int8_variants import ATTN_OUT, MLP_IN, MLP_OUT, quantize, weight_matmuls  # noqa: E402

# Each recipe: the node-name patterns of the weight products left in fp32.
RECIPES = {
    "fp32-mlp-out": MLP_OUT,
    "fp32-mlp": MLP_OUT + MLP_IN,
    "fp32-attn-out": ATTN_OUT,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--recipe", choices=sorted(RECIPES), required=True)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    from transformers import AutoTokenizer

    docs = load_split("test")
    for run in args.runs:
        out = ROOT / "results" / "runs" / run / f"metrics_test_int8_{args.recipe}.json"
        if out.exists():
            continue
        ckpt = ROOT / "checkpoints" / run
        fp32 = ckpt / "onnx" / "model.onnx"
        q = ckpt / "onnx" / f"model_int8_{args.recipe}.onnx"
        names = weight_matmuls(fp32)
        exclude = [n for n in names if any(re.search(p, n) for p in RECIPES[args.recipe])]
        t0 = time.time()
        quantize(fp32, q, exclude, per_channel=False)
        tok = AutoTokenizer.from_pretrained(ckpt)
        t1 = time.time()
        preds = predict_with(forward_fn(session(q, args.threads)), tok, docs, 8192)
        m = M.summarize(docs, preds)
        out.write_text(json.dumps(m, indent=2) + "\n")
        with gzip.open(out.with_name(f"pred_test_int8_{args.recipe}.jsonl.gz"), "wt") as f:
            for did in sorted(preds):
                f.write(json.dumps({"id": did, "spans": preds[did]}) + "\n")
        info = {"run": run, "recipe": args.recipe, "fp32_products": len(exclude), "weight_products": len(names),
                "int8_onnx_bytes": q.stat().st_size, "quantize_s": round(t1 - t0, 1),
                "eval_s": round(time.time() - t1, 1), "threads": args.threads}
        out.with_name(f"onnx_{args.recipe}.json").write_text(json.dumps(info, indent=2) + "\n")
        print(json.dumps(info), {k: round(v, 4) for k, v in m["headline"].items() if v is not None}, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
