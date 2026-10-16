#!/bin/sh
# Time every model on CPU (onnxruntime) and in headless Chromium (onnxruntime-web, WebAssembly), pinned to the
# performance cores 0-7 of the i9-14900KF the study used. Needs the ONNX files of the three -K-s0 runs
# (scripts/onnx_queue.py) and `npm ci` in bench/browser.
set -e
cd "$(dirname "$0")/.."
C=checkpoints
taskset -c 0-7 python scripts/latency_cpu.py --runs small-K-s0 modernbert-K-s0 deberta-K-s0 --threads 1 4 8
cd bench/browser
M=""
V=""
for m in small modernbert deberta; do
  case $m in small) v=30522;; modernbert) v=50368;; deberta) v=128100;; esac
  M="$M,$m-int8=../../$C/$m-K-s0/onnx/model_int8.onnx,$m-fp32=../../$C/$m-K-s0/onnx/model.onnx"
  V="$V,$m-int8=$v,$m-fp32=$v"
done
taskset -c 0-7 node bench.mjs --models "${M#,}" --vocab "${V#,}" --threads 1,4 --lengths 32,128,512 --runs 30 \
  --out ../../results/latency_browser.json
