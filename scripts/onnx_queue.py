"""Export each finished run to ONNX, quantise it to int8, and evaluate it with onnxruntime on CPU.

For every run with weights in checkpoints/<run>/ and no results/runs/<run>/metrics_test_int8.json yet:
  - export fp32 ONNX and the dynamic int8 model (keepspan.export);
  - evaluate the int8 model on the test split -> metrics_test_int8.json and pred_test_int8.jsonl.gz;
  - for <model>-K-s0, also evaluate the fp32 ONNX model -> metrics_test_onnx_fp32.json (a check that export is
    lossless);
  - record sizes and wall times -> onnx.json.
Order: every small-model run (the browser tier), then seed 0 of each condition for the other models, then the
remaining seeds. Polls until --runs runs are done, so it can run beside the training sweep.

    python scripts/onnx_queue.py --threads 16
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from keepspan import metrics as M  # noqa: E402
from keepspan.export import export_fp32, forward_fn, quantize_int8, session  # noqa: E402
from keepspan.models import predict_with  # noqa: E402
from keepspan.train import load_split  # noqa: E402

EVAL_TOKENS = 8192


def evaluate(path: Path, tokenizer, docs, threads: int) -> tuple[dict, dict, float]:
    sess = session(path, threads)
    t0 = time.time()
    preds = predict_with(forward_fn(sess), tokenizer, docs, batch_tokens=EVAL_TOKENS)
    return M.summarize(docs, preds), preds, time.time() - t0


def process(run: str, docs, threads: int) -> dict:
    from transformers import AutoTokenizer

    ckpt = ROOT / "checkpoints" / run
    out = ROOT / "results" / "runs" / run
    onnx_dir = ckpt / "onnx"
    t0 = time.time()
    fp32 = onnx_dir / "model.onnx"
    q = onnx_dir / "model_int8.onnx"
    if not q.exists():
        export_fp32(ckpt, fp32)
        quantize_int8(fp32, q)
    t_export = time.time() - t0
    tok = AutoTokenizer.from_pretrained(ckpt)
    info = {"run": run, "fp32_onnx_bytes": fp32.stat().st_size, "int8_onnx_bytes": q.stat().st_size,
            "export_quantize_s": round(t_export, 1), "threads": threads}
    m8, p8, s8 = evaluate(q, tok, docs, threads)
    (out / "metrics_test_int8.json").write_text(json.dumps(m8, indent=2) + "\n")
    with gzip.open(out / "pred_test_int8.jsonl.gz", "wt") as f:
        for did in sorted(p8):
            f.write(json.dumps({"id": did, "spans": p8[did]}) + "\n")
    info["int8_eval_s"] = round(s8, 1)
    if run.endswith("-K-s0"):
        m32, _, s32 = evaluate(fp32, tok, docs, threads)
        (out / "metrics_test_onnx_fp32.json").write_text(json.dumps(m32, indent=2) + "\n")
        info["fp32_eval_s"] = round(s32, 1)
    (out / "onnx.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--runs", type=int, default=36)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    docs = load_split("test")
    log = (ROOT / "logs" / "onnx_queue.jsonl").open("a")
    while True:
        done, todo = 0, []
        for d in sorted((ROOT / "results" / "runs").iterdir()):
            if args.only and d.name not in args.only:
                continue
            if (d / "metrics_test_int8.json").exists():
                done += 1
            elif (d / "metrics_test.json").exists() and (ROOT / "checkpoints" / d.name / "model.safetensors").exists():
                todo.append(d.name)
        if not todo:
            if done >= (len(args.only) if args.only else args.runs):
                return 0
            time.sleep(60)
            continue
        def priority(name):
            return (0 if name.startswith("small-") else 1 if name.endswith("-s0") else 2, name)

        run = min(todo, key=priority)
        try:
            info = process(run, docs, args.threads)
            info["ok"] = True
        except Exception as e:  # noqa: BLE001
            info = {"run": run, "ok": False, "error": repr(e)[:500]}
        info["end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        log.write(json.dumps(info) + "\n")
        log.flush()
        print(json.dumps(info), flush=True)
        if not info["ok"]:
            return 1


if __name__ == "__main__":
    sys.exit(main())
