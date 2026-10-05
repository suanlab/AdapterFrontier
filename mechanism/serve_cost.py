#!/usr/bin/env python3
"""Measured inference cost of an adapter ensemble (2026-10-04 panel, B-M3).

The paper prices ensembles in adapter forward passes per query and says their
wall-clock cost was never measured (L8). This script measures it for the A3
SNLI pools. One fixed batch of validation queries goes through
  - the single adapter, merged into the base weights (no adapter overhead),
  - the single adapter, unmerged,
  - an ensemble of N members, N in {2, 4, 8}: each member's forward pass on
    the same batch, then probability averaging,
on one GPU, in bf16, with batch size 32 and max length 128. It reports the
median and p95 latency per batch, the throughput and the peak GPU memory.

Two ways of running the members are measured: one after another (PEFT
`set_adapter`, which pays a module-walk per switch), and in one forward pass
over the batch replicated N times with PEFT's mixed-adapter batching
(`adapter_names`), the cheaper of the two without a dedicated multi-LoRA
kernel.
The GPUs are shared with unrelated jobs, so absolute times are noisy; the
ratios between rows measured back to back are the result.

Usage: python3 mechanism/serve_cost.py --backbone {bert,q05}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from real_lora_classify import _normalize_text  # noqa: E402

MODELS = {"bert": "bert-base-uncased", "q05": "Qwen/Qwen2.5-0.5B"}
E_SEEDS = tuple(range(201, 209))
S_SEED = 251
BATCH = 32
WARMUP, TIMED = 10, 50


def batches(tok, n_batches):
    from datasets import load_dataset
    ds = load_dataset("snli", split="validation").filter(lambda x: x["label"] >= 0)
    out = []
    for b in range(n_batches):
        rows = ds.select(range(b * BATCH, (b + 1) * BATCH))
        enc = tok([_normalize_text(v) for v in rows["premise"]],
                  [_normalize_text(v) for v in rows["hypothesis"]],
                  truncation=True, max_length=128, padding=True, return_tensors="pt")
        out.append({k: v.cuda() for k, v in enc.items()})
    return out


def time_fn(fn, data):
    for x in data[:WARMUP]:
        fn(x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    lat = []
    for x in data[WARMUP:WARMUP + TIMED]:
        t0 = time.perf_counter(); fn(x); torch.cuda.synchronize()
        lat.append(time.perf_counter() - t0)
    lat = np.array(lat) * 1000
    return {"p50_ms": float(np.median(lat)), "p95_ms": float(np.percentile(lat, 95)),
            "throughput_qps": float(BATCH / np.median(lat) * 1000),
            "peak_mem_mb": float(torch.cuda.max_memory_allocated() / 2**20)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=list(MODELS), required=True)
    args = ap.parse_args()
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    name = MODELS[args.backbone]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    data = batches(tok, WARMUP + TIMED)

    def base():
        m = AutoModelForSequenceClassification.from_pretrained(name, num_labels=3,
                                                               torch_dtype=torch.bfloat16)
        if m.config.pad_token_id is None:
            m.config.pad_token_id = tok.pad_token_id
        return m

    runs = HERE / "runs"
    rows = {}
    with torch.inference_mode():
        s_dir = runs / f"a3_snli_{args.backbone}_S_s{S_SEED}" / "epoch4" / "adapter"
        single = PeftModel.from_pretrained(base(), s_dir).cuda().eval()
        rows["single_unmerged"] = time_fn(lambda x: single(**x).logits, data)
        merged = single.merge_and_unload().eval()
        rows["single_merged"] = time_fn(lambda x: merged(**x).logits, data)
        del single, merged; torch.cuda.empty_cache()

        ens = None
        for i, s in enumerate(E_SEEDS):
            d = runs / f"a3_snli_{args.backbone}_E_s{s}" / "epoch2" / "adapter"
            if ens is None:
                ens = PeftModel.from_pretrained(base(), d, adapter_name=f"m{i}")
            else:
                ens.load_adapter(d, adapter_name=f"m{i}")
        ens = ens.cuda().eval()
        for n in (2, 4, 8):
            names = [f"m{i}" for i in range(n)]

            def ensemble(x, names=names):
                p = 0
                for a in names:
                    ens.set_adapter(a)
                    p = p + torch.softmax(ens(**x).logits.float(), -1)
                return p / len(names)
            rows[f"ensemble_N{n}"] = time_fn(ensemble, data)

            # one forward pass: the batch replicated n times, each copy routed to
            # one member through PEFT's mixed-adapter batching
            def ensemble_batched(x, names=names):
                xx = {k: v.repeat(len(names), 1) for k, v in x.items()}
                an = [a for a in names for _ in range(x["input_ids"].shape[0])]
                p = torch.softmax(ens(**xx, adapter_names=an).logits.float(), -1)
                return p.view(len(names), -1, p.shape[-1]).mean(0)
            rows[f"ensemble_N{n}_batched"] = time_fn(ensemble_batched, data)

    ref = rows["single_merged"]["p50_ms"]
    for k, v in rows.items():
        v["latency_x_vs_merged_single"] = v["p50_ms"] / ref
        print(f"{k:16s} p50 {v['p50_ms']:7.1f} ms  p95 {v['p95_ms']:7.1f} ms  "
              f"{v['throughput_qps']:7.0f} q/s  mem {v['peak_mem_mb']:6.0f} MB  "
              f"x{v['latency_x_vs_merged_single']:.2f}")
    out = HERE / "results" / f"serve_cost_{args.backbone}.json"
    out.write_text(json.dumps({"model": name, "gpu": torch.cuda.get_device_name(0),
                               "batch": BATCH, "timed_batches": TIMED, "rows": rows,
                               "_note": __doc__.split("\n\n")[1].strip()}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
