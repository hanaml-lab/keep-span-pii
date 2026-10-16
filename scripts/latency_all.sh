#!/bin/sh
# Time the models on CPU and in a browser, pinned to the performance cores 0-7 of the i9-14900KF the study used.
#   CPU (onnxruntime): fp32, onnxruntime's default int8 and, for the base models, int8 with the MLP output
#   projections kept in fp32; one and eight threads.
#   Browser (onnxruntime-web on WebAssembly, headless Chromium): the small model in int8 and fp32, and the base models
#   in the mixed int8 they would ship with; one and four threads.
# Needs the ONNX files of the three -K-s0 runs (scripts/onnx_queue.py, scripts/eval_int8_recipe.py) and `npm ci` in
# bench/browser. `--browser-only` skips the CPU part.
set -e
cd "$(dirname "$0")/.."
C=../../checkpoints
if [ "$1" != "--browser-only" ]; then
  taskset -c 0-7 python scripts/latency_cpu.py --runs small-K-s0 modernbert-K-s0 deberta-K-s0 --threads 1 8 --reps 30
fi
cd bench/browser
M="small-int8=$C/small-K-s0/onnx/model_int8.onnx,small-fp32=$C/small-K-s0/onnx/model.onnx"
M="$M,modernbert-int8-mix=$C/modernbert-K-s0/onnx/model_int8_fp32-mlp-out.onnx"
M="$M,deberta-int8-mix=$C/deberta-K-s0/onnx/model_int8_fp32-mlp-out.onnx"
V="small-int8=30522,small-fp32=30522,modernbert-int8-mix=50368,deberta-int8-mix=128100"
taskset -c 0-7 node bench.mjs --models "$M" --vocab "$V" --threads 1,4 --lengths 32,128,512 --runs 20 \
  --out ../../results/latency_browser.json
