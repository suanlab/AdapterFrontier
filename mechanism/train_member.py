#!/usr/bin/env python3
"""Train one Qwen-2.5-0.5B MNLI LoRA member with its random sources separated.

For H3a/H3b of mechanism/PREREG.md. The recipe is the paper's
`sweep_configs/pool_a_mnli_qwen25_05b.json` (40k train rows, r=8, alpha=16,
dropout 0, LoRA on q_proj/v_proj, lr 3e-4, bs 16, seq 128, warmup 0.06,
wd 0.01, bf16). The paper's runs drew all randomness from one `--seed`. Here
three seeds are set independently, in the order the randomness is consumed:

  --head-seed   seeds torch just before `from_pretrained`, which initialises the
                new classification head (`score`) from the global RNG;
  --lora-seed   seeds torch just before `get_peft_model`, which draws lora_A
                (lora_B starts at zero);
  --data-seed   goes to TrainingArguments(seed, data_seed), which fixes the
                sampler's shuffle order. Trainer reseeds the global RNG with it
                at construction, after the model already exists, so it does
                not touch the two initialisations above.

Instead of saving the adapter, the script dumps logits for the full
`validation_matched` split (n = 9815) so every quantity in the study can be
computed offline on the paper's own test split.
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

MODEL = "Qwen/Qwen2.5-0.5B"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-seed", type=int, required=True)
    ap.add_argument("--lora-seed", type=int, required=True)
    ap.add_argument("--data-seed", type=int, required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.json").exists():
        print(f"already done: {out}")
        return 0

    train = load_dataset("nyu-mll/glue", "mnli", split="train").select(range(40000))
    evald = load_dataset("nyu-mll/glue", "mnli", split="validation_matched")
    labels = np.array(evald["label"], dtype=np.int64)

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    torch.manual_seed(args.head_seed)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=3)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id

    torch.manual_seed(args.lora_seed)
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "v_proj"],
        bias="none", task_type="SEQ_CLS"))

    def tokenize(b):
        r = tok([_normalize_text(x) for x in b["premise"]],
                [_normalize_text(x) for x in b["hypothesis"]],
                truncation=True, max_length=128)
        r["labels"] = [int(x) for x in b["label"]]
        return r

    train = train.map(tokenize, batched=True, remove_columns=train.column_names)
    evald = evald.map(tokenize, batched=True, remove_columns=evald.column_names)

    targs = TrainingArguments(
        output_dir=str(out / "_trainer"), per_device_train_batch_size=16,
        per_device_eval_batch_size=64, learning_rate=3e-4,
        num_train_epochs=args.epochs, warmup_ratio=0.06, weight_decay=0.01,
        logging_steps=50, save_strategy="no", eval_strategy="no", report_to=[],
        bf16=True, remove_unused_columns=False, dataloader_pin_memory=False,
        seed=args.data_seed, data_seed=args.data_seed)
    trainer = Trainer(model=model, args=targs, train_dataset=train,
                      data_collator=DataCollatorWithPadding(tokenizer=tok))

    t0 = time.time()
    res = trainer.train()
    wall = time.time() - t0
    logits = trainer.predict(evald).predictions
    logits = np.asarray(logits[0] if isinstance(logits, tuple) else logits, dtype=np.float32)

    np.save(out / "logits_validation_matched.npy", logits)
    np.save(out / "labels_validation_matched.npy", labels)
    metrics = {
        "model": MODEL, "task": "mnli", "epochs": args.epochs,
        "head_seed": args.head_seed, "lora_seed": args.lora_seed,
        "data_seed": args.data_seed,
        "train_loss": float(res.metrics.get("train_loss", float("nan"))),
        "train_loss_over_ln3": float(res.metrics.get("train_loss", float("nan")) / np.log(3)),
        "acc_full_validation": float((logits.argmax(-1) == labels).mean()),
        "train_runtime_s": round(wall, 1),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    # The trainer scratch dir holds nothing we need once logits are saved.
    for p in sorted((out / "_trainer").rglob("*"), reverse=True):
        p.unlink() if p.is_file() else p.rmdir()
    (out / "_trainer").rmdir()
    print(json.dumps(metrics))
    return 0


if __name__ == "__main__":
    sys.exit(main())
