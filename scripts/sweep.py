"""Run the study's training sweep, one job at a time, on a shared GPU.

Order: seed 0 of all twelve configurations first, then seeds 1 and 2 of the keep-span ablation (base, K), then
seeds 1 and 2 of the Wikipedia conditions (W, KW). A job whose results/runs/<run>/metrics_test.json exists is
skipped, so the sweep can be stopped and resumed. Before each job the GPU must have --need-gb free; otherwise the
sweep waits. Each job's output goes to logs/runs/<run>.log and one line per job to logs/sweep.jsonl.

    python scripts/sweep.py                 # the whole sweep
    python scripts/sweep.py --models small  # one model
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from keepspan.train import run_name  # noqa: E402

MODELS = ("small", "modernbert", "deberta")
CONDS = ((0, 0), (1, 0), (0, 1), (1, 1))  # (keep, wiki)
MICRO_FALLBACK = {"small": 8192, "modernbert": 4096, "deberta": 2048}


def queue(models) -> list[tuple[str, int, int, int]]:
    jobs = []
    for m in models:
        for k, w in CONDS:
            jobs.append((m, k, w, 0))
    for seed in (1, 2):
        for m in models:
            for k, w in ((0, 0), (1, 0)):
                jobs.append((m, k, w, seed))
    for seed in (1, 2):
        for m in models:
            for k, w in ((0, 1), (1, 1)):
                jobs.append((m, k, w, seed))
    return jobs


def free_gb() -> float:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().split(",")
    return (float(out[0]) - float(out[1])) / 1024


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--need-gb", type=float, default=4.0)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    logs = ROOT / "logs" / "runs"
    logs.mkdir(parents=True, exist_ok=True)
    sweep_log = (ROOT / "logs" / "sweep.jsonl").open("a")
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    for m, k, w, s in queue(args.models):
        name = run_name(m, k, w, s)
        if (ROOT / "results" / "runs" / name / "metrics_test.json").exists():
            continue
        if args.dry:
            print(name)
            continue
        for attempt in range(3):
            waited = 0
            while free_gb() < args.need_gb:
                time.sleep(30)
                waited += 30
            t0 = time.time()
            with (logs / f"{name}.log").open("a") as f:
                f.write(f"# start {time.strftime('%Y-%m-%dT%H:%M:%S')} attempt {attempt} free_gb {free_gb():.2f}\n")
                f.flush()
                cmd = [sys.executable, "-m", "keepspan.train", "--model", m, "--keep", str(k), "--wiki", str(w),
                       "--seed", str(s)]
                if attempt > 0:  # after a failure (an out-of-memory error on a busy GPU), halve the micro-batch
                    cmd += ["--micro-tokens", str(MICRO_FALLBACK[m])]
                rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
            rec = {"run": name, "rc": rc, "attempt": attempt, "seconds": round(time.time() - t0, 1),
                   "waited_s": waited, "end": time.strftime("%Y-%m-%dT%H:%M:%S")}
            sweep_log.write(json.dumps(rec) + "\n")
            sweep_log.flush()
            print(json.dumps(rec), flush=True)
            if rc == 0:
                break
            time.sleep(60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
