"""Time one forward pass of each exported model with onnxruntime on CPU (batch 1).

    taskset -c 0-7 python scripts/latency_cpu.py --runs small-K-s0 deberta-K-s0 modernbert-K-s0 --threads 1 4 8

For each run, precision (fp32, int8), thread count and input length: 10 warm-up passes, then --reps timed passes;
the median and 90th percentile are recorded, with the load average before and after, since the machine is shared.
Writes results/latency_cpu.json.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from keepspan.export import session  # noqa: E402


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 4, 8])
    ap.add_argument("--lengths", nargs="+", type=int, default=[32, 128, 512])
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--out", default=str(ROOT / "results" / "latency_cpu.json"))
    args = ap.parse_args()
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    rows = []
    for run in args.runs:
        for prec, fname in (("fp32", "model.onnx"), ("int8", "model_int8.onnx")):
            path = ROOT / "checkpoints" / run / "onnx" / fname
            for th in args.threads:
                sess = session(path, th)
                names = {i.name for i in sess.get_inputs()}
                for n in args.lengths:
                    ids = rng.integers(1000, 20000, size=(1, n), dtype=np.int64)
                    feed = {"input_ids": ids, "attention_mask": np.ones((1, n), dtype=np.int64)}
                    feed = {k: v for k, v in feed.items() if k in names}
                    for _ in range(10):
                        sess.run(None, feed)
                    load0 = os.getloadavg()[0]
                    ts = []
                    for _ in range(args.reps):
                        t = time.perf_counter()
                        sess.run(None, feed)
                        ts.append((time.perf_counter() - t) * 1000)
                    row = {"run": run, "model": run.split("-")[0], "precision": prec, "threads": th, "length": n,
                           "median_ms": float(np.median(ts)), "p90_ms": float(np.percentile(ts, 90)),
                           "min_ms": float(np.min(ts)), "load_before": load0, "load_after": os.getloadavg()[0],
                           "bytes": path.stat().st_size}
                    rows.append(row)
                    print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}),
                          flush=True)
    meta = {"cpu": cpu_name(), "logical_cpus": os.cpu_count(), "affinity": sorted(os.sched_getaffinity(0)),
            "onnxruntime": ort.__version__, "python": platform.python_version(), "reps": args.reps, "warmup": 10,
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "rows": rows}
    Path(args.out).write_text(json.dumps(meta, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
