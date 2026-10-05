#!/usr/bin/env python3
"""Train one SNLI LoRA adapter for H8 (mechanism/PREREG.md §10, §9 deviation 3).

Two recipes per model: the pool recipe (r=8, alpha=16, 2 epochs) and the
n_rank baseline recipe (r=32, alpha=32, 4 epochs). Batch size and learning
rate follow each model's corpus recipe. One seed drives all randomness, as in
the corpus. Logits for the full filtered SNLI validation split are dumped so
the analysis can use the paper's 40% test slice.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from real_lora_classify import _normalize_text  # noqa: E402

TRAIN_ROWS = 20000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--recipe", choices=["pool", "baseline"], required=True)
    ap.add_argument("--bs", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs-override", type=float, default=None, help="smoke tests only")
    args = ap.parse_args()
    r, alpha, epochs = (8, 16, 2.0) if args.recipe == "pool" else (32, 32, 4.0)
    if args.epochs_override is not None:
        epochs = args.epochs_override

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.json").exists():
        print(f"already done: {out}"); return 0

    train = load_dataset("snli", split="train").filter(lambda x: x["label"] >= 0).select(range(TRAIN_ROWS))
    evald = load_dataset("snli", split="validation").filter(lambda x: x["label"] >= 0)
    labels = np.array(evald["label"], dtype=np.int64)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id
    targets = ["query", "value"] if "bert" in args.model.lower() else ["q_proj", "v_proj"]
    model = get_peft_model(model, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0,
                                             target_modules=targets, bias="none", task_type="SEQ_CLS"))

    def tokenize(b):
        x = tok([_normalize_text(v) for v in b["premise"]],
                [_normalize_text(v) for v in b["hypothesis"]], truncation=True, max_length=128)
        x["labels"] = [int(v) for v in b["label"]]
        return x

    train = train.map(tokenize, batched=True, remove_columns=train.column_names)
    evald = evald.map(tokenize, batched=True, remove_columns=evald.column_names)
    targs = TrainingArguments(
        output_dir=str(out / "_trainer"), per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=64, learning_rate=3e-4, num_train_epochs=epochs,
        warmup_ratio=0.06, weight_decay=0.01, logging_steps=50, save_strategy="no",
        eval_strategy="no", report_to=[], bf16=True, remove_unused_columns=False,
        dataloader_pin_memory=False, seed=args.seed, data_seed=args.seed)
    trainer = Trainer(model=model, args=targs, train_dataset=train,
                      data_collator=DataCollatorWithPadding(tokenizer=tok))
    t0 = time.time(); res = trainer.train(); wall = time.time() - t0
    logits = trainer.predict(evald).predictions
    logits = np.asarray(logits[0] if isinstance(logits, tuple) else logits, dtype=np.float32)
    np.save(out / "logits_validation.npy", logits); np.save(out / "labels_validation.npy", labels)
    loss = float(res.metrics.get("train_loss", float("nan")))
    (out / "metrics.json").write_text(json.dumps({
        "model": args.model, "task": "snli", "recipe": args.recipe, "lora_r": r, "lora_alpha": alpha,
        "epochs": epochs, "batch_size": args.bs, "seed": args.seed, "train_rows": TRAIN_ROWS,
        "train_loss": loss, "train_loss_over_ln3": loss / float(np.log(3)),
        "acc_full_validation": float((logits.argmax(-1) == labels).mean()),
        "train_runtime_s": round(wall, 1)}, indent=2))
    for p in sorted((out / "_trainer").rglob("*"), reverse=True):
        p.unlink() if p.is_file() else p.rmdir()
    (out / "_trainer").rmdir()
    return 0


if __name__ == "__main__":
    sys.exit(main())
