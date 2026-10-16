"""Fine-tune one span model in one condition with one seed, then evaluate it on the validation and test splits.

    python -m keepspan.train --model deberta --keep 1 --wiki 0 --seed 0

Conditions: --keep 1 labels keep-span tokens O (keep-span negatives); --keep 0 excludes them from the loss.
--wiki 1 adds the Wikipedia paragraphs (every token O); --wiki 0 leaves them out. Everything else is identical.

Writes results/runs/<run>/ (config, training log, metrics, test predictions as offsets) and the weights to
checkpoints/<run>/ (git-ignored).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import platform
import random
import subprocess
import time
from pathlib import Path

import torch

from . import metrics as M
from .encode import encode_doc, windows
from .models import MAX_LEN, MODELS, OVERLAP, collate, count_params, load_model, load_tokenizer, predict_docs, set_seed

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
CACHE = ROOT / "data" / "cache"

EPOCHS = 3
STEP_TOKENS = 16384        # labelled tokens per optimiser step (gradient accumulation over micro-batches)
WARMUP = 0.06
WEIGHT_DECAY = 0.01
CLIP = 1.0


def run_name(model: str, keep: int, wiki: int, seed: int) -> str:
    cond = ("K" if keep else "") + ("W" if wiki else "")
    return f"{model}-{cond or 'base'}-s{seed}"


def load_split(name: str) -> list[dict]:
    with gzip.open(DATA / f"{name}.jsonl.gz", "rt") as f:
        return [json.loads(line) for line in f]


def encoded_train(model: str, tokenizer, docs: list[dict]):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{model}-train.pkl"
    if path.exists():
        with path.open("rb") as f:
            return pickle.load(f)
    encs = {d["id"]: encode_doc(tokenizer, d) for d in docs}
    with path.open("wb") as f:
        pickle.dump(encs, f)
    return encs


def make_steps(items: list[dict], micro_tokens: int, rng: random.Random,
               step_tokens: int = STEP_TOKENS) -> list[list[list[int]]]:
    """Group example indices into micro-batches (padded size <= micro_tokens, similar lengths) and the micro-batches
    into optimiser steps of about step_tokens labelled tokens."""
    idx = list(range(len(items)))
    rng.shuffle(idx)
    chunk = 4096
    micro: list[list[int]] = []
    for c in range(0, len(idx), chunk):
        part = sorted(idx[c:c + chunk], key=lambda i: len(items[i]["input_ids"]))
        cur: list[int] = []
        longest = 0
        for i in part:
            n = len(items[i]["input_ids"])
            if cur and max(longest, n) * (len(cur) + 1) > micro_tokens:
                micro.append(cur)
                cur, longest = [], 0
            cur.append(i)
            longest = max(longest, n)
        if cur:
            micro.append(cur)
    rng.shuffle(micro)
    steps, cur_step, tok = [], [], 0
    for mb in micro:
        cur_step.append(mb)
        tok += sum(items[i]["n_labelled"] for i in mb)
        if tok >= step_tokens:
            steps.append(cur_step)
            cur_step, tok = [], 0
    if cur_step:
        steps.append(cur_step)
    return steps


def gpu_info() -> dict:
    info = {"device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        info["nvidia_smi"] = out
    except Exception:  # noqa: BLE001
        pass
    return info


def evaluate(model, tokenizer, docs, device, batch_tokens: int = 8192) -> tuple[dict, dict]:
    preds = predict_docs(model, tokenizer, docs, device, batch_tokens=batch_tokens, autocast=False)
    return M.summarize(docs, preds), preds


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--keep", type=int, choices=(0, 1), required=True)
    ap.add_argument("--wiki", type=int, choices=(0, 1), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--mem-gb", type=float, default=3.3, help="cap on the PyTorch allocator (GB)")
    ap.add_argument("--max-train-docs", type=int, default=None, help="smoke tests only")
    ap.add_argument("--max-eval-docs", type=int, default=None, help="smoke tests only")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--micro-tokens", type=int, default=None, help="micro-batch size in padded tokens")
    ap.add_argument("--grad-ckpt", type=int, default=None, help="gradient checkpointing (default: per model)")
    args = ap.parse_args(argv)

    name = run_name(args.model, args.keep, args.wiki, args.seed)
    out = Path(args.out) if args.out else ROOT / "results" / "runs" / name
    ckpt = ROOT / "checkpoints" / name
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(MODELS[args.model])
    if args.micro_tokens:
        cfg["micro_tokens"] = args.micro_tokens
    if args.grad_ckpt is None:
        args.grad_ckpt = int(cfg.get("grad_ckpt", False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.mem_gb * 1e9 / total))
        torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(args.seed)
    t_start = time.time()

    tokenizer = load_tokenizer(args.model)
    train_docs = [d for d in load_split("train") if args.wiki or d["source"] != "wikipedia"]
    val_docs = load_split("validation")
    test_docs = load_split("test")
    if args.max_train_docs:
        random.Random(0).shuffle(train_docs)
        train_docs = train_docs[:args.max_train_docs]
    if args.max_eval_docs:
        val_docs = val_docs[::max(1, len(val_docs) // args.max_eval_docs)]
        test_docs = test_docs[::max(1, len(test_docs) // args.max_eval_docs)]

    if args.max_train_docs:
        encs = {d["id"]: encode_doc(tokenizer, d) for d in train_docs}
    else:
        all_encs = encoded_train(args.model, tokenizer, load_split("train"))
        encs = {d["id"]: all_encs[d["id"]] for d in train_docs}
    items = []
    for d in train_docs:
        for w in windows(encs[d["id"]], MAX_LEN, OVERLAP, tokenizer.cls_token_id, tokenizer.sep_token_id,
                         keep_negatives=bool(args.keep)):
            w["n_labelled"] = sum(1 for x in w["labels"] if x != -100)
            w["source"] = d["source"]
            items.append(w)
    n_keep_tokens = sum(sum(encs[d["id"]].keep_mask) for d in train_docs)

    model = load_model(args.model).to(device)
    # The input word-embedding matrix is frozen in every model: with it trainable, DeBERTa-v3-base's weights,
    # gradients and Adam states alone exceed the study's 4 GB GPU budget. Everything else trains.
    model.get_input_embeddings().weight.requires_grad_(False)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    params = count_params(model)
    params["trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or "norm" in n.lower() or "bias" in n else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": WEIGHT_DECAY},
                             {"params": no_decay, "weight_decay": 0.0}], lr=cfg["lr"], betas=(0.9, 0.999),
                            eps=1e-6, fused=device.type == "cuda")
    rng = random.Random(args.seed)
    epoch_steps = [make_steps(items, cfg["micro_tokens"], rng) for _ in range(args.epochs)]
    total_steps = sum(len(s) for s in epoch_steps)
    warm = max(1, int(WARMUP * total_steps))

    def lr_at(step):
        if step < warm:
            return (step + 1) / warm
        return max(0.0, (total_steps - step) / max(1, total_steps - warm))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    config = {
        "run": name, "model": args.model, "base": cfg["repo"], "base_revision": cfg["revision"],
        "keep_negatives": bool(args.keep), "wikipedia_negatives": bool(args.wiki), "seed": args.seed,
        "epochs": args.epochs, "lr": cfg["lr"], "schedule": f"linear, warmup {WARMUP:.0%}", "optimizer": "AdamW",
        "betas": [0.9, 0.999], "eps": 1e-6, "weight_decay": WEIGHT_DECAY, "grad_clip": CLIP,
        "step_tokens": STEP_TOKENS, "micro_tokens": cfg["micro_tokens"], "max_len": MAX_LEN, "overlap": OVERLAP,
        "precision": "bf16 autocast, fp32 master weights", "frozen": "input word embeddings",
        "gradient_checkpointing": bool(args.grad_ckpt), "train_docs": len(train_docs),
        "train_windows": len(items), "train_labelled_tokens": sum(w["n_labelled"] for w in items),
        "keep_tokens": int(n_keep_tokens), "steps": total_steps, "params": params,
        "train_docs_by_source": {s: sum(1 for d in train_docs if d["source"] == s)
                                 for s in sorted({d["source"] for d in train_docs})},
        "smoke": bool(args.max_train_docs),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    log = (out / "train_log.jsonl").open("w")

    def write(rec):
        log.write(json.dumps(rec) + "\n")
        log.flush()

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    step = 0
    t_train = time.time()
    for ep, steps in enumerate(epoch_steps):
        model.train()
        run_loss, run_tok = 0.0, 0
        for st in steps:
            n_tok = sum(items[i]["n_labelled"] for mb in st for i in mb)
            for mb in st:
                b = collate([items[i] for i in mb], tokenizer.pad_token_id, device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
                loss = loss_fn(logits.float().view(-1, logits.shape[-1]), b["labels"].view(-1)) / max(1, n_tok)
                loss.backward()
                run_loss += loss.item() * n_tok
            run_tok += n_tok
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0 or step == total_steps:
                write({"step": step, "epoch": ep, "loss": run_loss / max(1, run_tok), "grad_norm": float(gn),
                       "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t_train, 1),
                       "max_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if device.type == "cuda"
                       else None})
                run_loss, run_tok = 0.0, 0
        vm, _ = evaluate(model, tokenizer, val_docs, device, cfg['eval_tokens'])
        write({"epoch_end": ep, "step": step, "elapsed_s": round(time.time() - t_train, 1),
               "validation": vm["headline"]})
    train_s = time.time() - t_train

    t_eval = time.time()
    vm, _ = evaluate(model, tokenizer, val_docs, device, cfg['eval_tokens'])
    tm, tp = evaluate(model, tokenizer, test_docs, device, cfg['eval_tokens'])
    eval_s = time.time() - t_eval
    (out / "metrics_validation.json").write_text(json.dumps(vm, indent=2) + "\n")
    (out / "metrics_test.json").write_text(json.dumps(tm, indent=2) + "\n")
    with gzip.open(out / "pred_test.jsonl.gz", "wt") as f:
        for did in sorted(tp):
            f.write(json.dumps({"id": did, "spans": tp[did]}) + "\n")
    if not args.no_save:
        ckpt.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt, safe_serialization=True)
        tokenizer.save_pretrained(ckpt)
    import transformers

    env = {
        "train_seconds": round(train_s, 1), "eval_seconds": round(eval_s, 1),
        "wall_seconds": round(time.time() - t_start, 1),
        "max_mem_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if device.type == "cuda" else None,
        "max_mem_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 3) if device.type == "cuda" else None,
        "torch": torch.__version__, "transformers": transformers.__version__, "python": platform.python_version(),
        "host_cpu": platform.processor() or platform.machine(), "gpu": gpu_info(),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "pid": os.getpid(),
    }
    (out / "env.json").write_text(json.dumps(env, indent=2) + "\n")
    log.close()
    h = tm["headline"]
    print(json.dumps({"run": name, "train_s": round(train_s), **{k: (round(v, 4) if isinstance(v, float) else v)
                                                                  for k, v in h.items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

