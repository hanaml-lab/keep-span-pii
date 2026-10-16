"""Export a fine-tuned checkpoint to ONNX (fp32), quantise it to int8, and run it with onnxruntime.

    python -m keepspan.export checkpoints/<run>            # writes checkpoints/<run>/onnx/{model,model_int8}.onnx

The int8 model uses onnxruntime's dynamic quantisation: weights of MatMul (and Gather) nodes stored as signed 8-bit
integers, activations quantised on the fly. It is the format ONNX Runtime Web runs in a browser on WebAssembly.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def export_fp32(ckpt: Path, out: Path, opset: int = 17) -> Path:
    from transformers import AutoModelForTokenClassification

    cfg = json.loads((ckpt / "config.json").read_text())
    kw = {}
    if cfg.get("model_type") == "modernbert":
        kw = {"reference_compile": False, "attn_implementation": "eager"}
    model = AutoModelForTokenClassification.from_pretrained(ckpt, **kw).eval()
    ids = torch.tensor([[cfg.get("bos_token_id") or 101] + [1000 + i for i in range(14)] + [102]], dtype=torch.long)
    att = torch.ones_like(ids)
    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model, (ids, att), str(out), input_names=["input_ids", "attention_mask"], output_names=["logits"],
            dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}, "attention_mask": {0: "batch", 1: "sequence"},
                          "logits": {0: "batch", 1: "sequence"}},
            opset_version=opset, dynamo=False, do_constant_folding=True)
    return out


def quantize_int8(fp32: Path, out: Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(fp32), str(out), weight_type=QuantType.QInt8)
    return out


def session(path: Path, threads: int = 8):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def forward_fn(sess):
    names = {i.name for i in sess.get_inputs()}

    def forward(ids: np.ndarray, att: np.ndarray) -> np.ndarray:
        feed = {"input_ids": ids, "attention_mask": att}
        return sess.run(["logits"], {k: v for k, v in feed.items() if k in names})[0]

    return forward


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    args = ap.parse_args(argv)
    ckpt = Path(args.ckpt)
    onnx_dir = ckpt / "onnx"
    t0 = time.time()
    fp32 = export_fp32(ckpt, onnx_dir / "model.onnx")
    t1 = time.time()
    q = quantize_int8(fp32, onnx_dir / "model_int8.onnx")
    t2 = time.time()
    info = {"fp32_bytes": fp32.stat().st_size, "int8_bytes": q.stat().st_size, "export_s": round(t1 - t0, 1),
            "quantize_s": round(t2 - t1, 1)}
    (onnx_dir / "export.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
