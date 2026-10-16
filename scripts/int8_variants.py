"""Which part of a model does dynamic int8 quantisation break? Compare quantisation recipes on a fixed subset.

For one run, quantise its fp32 ONNX model several ways and evaluate each on a fixed, source-stratified subset of the
test split (the first --per-source documents of each source in hash order), against the fp32 model on the same
subset. Writes results/runs/<run>/int8_variants.json. The study's int8 numbers come from scripts/onnx_queue.py on the
full test split; this script only diagnoses.

    python scripts/int8_variants.py modernbert-base-s0 --per-source 60
"""

from __future__ import annotations

import argparse
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

# Weight products by role, as node-name patterns for ModernBERT (layers.N/mlp/Wo) and BERT or DeBERTa
# (layer.N/output/dense is the MLP output; layer.N/attention/output/dense the attention output).
MLP_OUT = (r"/mlp/Wo/", r"/layer\.\d+/output/dense/")
MLP_IN = (r"/mlp/Wi/", r"/intermediate/dense/")
ATTN_OUT = (r"/attn/Wo/", r"/attention/output/dense/")

KEYS = ("f1_strict_macro", "removal_core_pooled", "tab_removal_all", "tab_keep_violation",
        "wikineural_keep_violation")


def weight_matmuls(path: Path) -> list[str]:
    import onnx

    m = onnx.load(str(path), load_external_data=False)
    init = {i.name for i in m.graph.initializer}
    return [n.name for n in m.graph.node if n.op_type == "MatMul" and n.input[1] in init]


def quantize(fp32: Path, out: Path, exclude: list[str], per_channel: bool, matmul_only: bool = False) -> None:
    """onnxruntime's dynamic int8, leaving `exclude` in fp32; matmul_only leaves embedding gathers in fp32 too."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    kw = {"op_types_to_quantize": ["MatMul"]} if matmul_only else {}
    quantize_dynamic(str(fp32), str(out), weight_type=QuantType.QInt8, per_channel=per_channel,
                     nodes_to_exclude=exclude, **kw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--per-source", type=int, default=60)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--only", nargs="*", default=None, help="names of the variants to run")
    args = ap.parse_args()
    from transformers import AutoTokenizer

    ckpt = ROOT / "checkpoints" / args.run
    fp32 = ckpt / "onnx" / "model.onnx"
    tok = AutoTokenizer.from_pretrained(ckpt)
    docs, seen = [], {}
    for d in load_split("test"):
        if seen.get(d["source"], 0) < args.per_source:
            seen[d["source"]] = seen.get(d["source"], 0) + 1
            docs.append(d)
    names = weight_matmuls(fp32)

    def pick(*parts):
        return [n for n in names if any(re.search(p, n) for p in parts)]

    variants = {
        "default": ([], False, False),
        "per-channel weights": ([], True, False),
        "fp32 MLP output projections": (pick(*MLP_OUT), False, False),
        "fp32 attention output projections": (pick(*ATTN_OUT), False, False),
        "fp32 MLP input projections": (pick(*MLP_IN), False, False),
        "fp32 head and classifier": (pick("/head/", "/classifier/"), False, False),
        "fp32 embeddings": ([], False, True),
        "fp32 embeddings and MLP output projections": (pick(*MLP_OUT), False, True),
    }
    if args.only:
        variants = {k: v for k, v in variants.items() if k in args.only}
    out = {"run": args.run, "docs": len(docs), "per_source": args.per_source, "weight_matmuls": len(names),
           "variants": {}}
    t0 = time.time()
    ref = M.summarize(docs, predict_with(forward_fn(session(fp32, args.threads)), tok, docs, 8192))["headline"]
    out["fp32"] = {k: ref[k] for k in KEYS}
    print("fp32", {k: round(100 * v, 1) for k, v in out["fp32"].items()}, f"{time.time() - t0:.0f}s", flush=True)
    for name, (exclude, per_channel, matmul_only) in variants.items():
        if name != "default" and not exclude and not per_channel and not matmul_only:
            continue
        q = ROOT / "scratch" / f"{args.run}-variant.onnx"
        q.parent.mkdir(parents=True, exist_ok=True)
        quantize(fp32, q, exclude, per_channel, matmul_only)
        h = M.summarize(docs, predict_with(forward_fn(session(q, args.threads)), tok, docs, 8192))["headline"]
        out["variants"][name] = {"excluded_matmuls": len(exclude), "per_channel": per_channel,
                                 "fp32_embeddings": matmul_only,
                                 "bytes": q.stat().st_size, **{k: h[k] for k in KEYS}}
        print(name, len(exclude), {k: round(100 * (h[k] - ref[k]), 1) for k in KEYS}, flush=True)
        q.unlink()
    path = ROOT / "results" / "runs" / args.run / "int8_variants.json"
    if path.exists():  # keep variants scored earlier on the same subset
        prev = json.loads(path.read_text())
        if prev.get("per_source") == out["per_source"]:
            out["variants"] = prev["variants"] | out["variants"]
    path.write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
