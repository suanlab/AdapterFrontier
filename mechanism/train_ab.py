#!/usr/bin/env python3
"""Train one SNLI LoRA trajectory for experiment A2 (mechanism/PREREG_AB.md §3).

The H8 recipe, with two changes the pre-registration fixes: lora_alpha = 2r at
every rank (H8's baseline used alpha/r = 1), and one 4-epoch schedule whose
epoch-2 state is saved as a *mid-trajectory* checkpoint. At each checkpoint we
dump logits on all of SNLI validation and on the sealed SNLI test, the LoRA
weights, the GPU seconds and train tokens spent so far, and the train loss.

Test logits are written but never read here; mechanism/ab_common.py refuses to
load them before mechanism/TEST_FREEZE exists (PREREG_AB.md §8).
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
YAHOO_VALIDATION_ROWS = 10000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--r", type=int, required=True)
    ap.add_argument("--bs", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-rows", type=int, default=TRAIN_ROWS, help="smoke tests only")
    # A3 (PREREG_AB.md §9): a full 2-epoch schedule, and the second task
    ap.add_argument("--epochs", type=int, choices=[2, 4], default=4)
    ap.add_argument("--task", choices=["snli", "yahoo"], default="snli")
    # A3b (PREREG_AB.md §9.2): the tuned single searches over learning rate
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    alpha = 2 * args.r
    EPOCHS = float(args.epochs)
    CHECKPOINT_EPOCHS = (2, 4) if args.epochs == 4 else (2,)

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainerCallback,
                              TrainingArguments)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.json").exists():
        print(f"already done: {out}"); return 0

    if args.task == "snli":
        def snli(split):
            return load_dataset("snli", split=split).filter(lambda x: x["label"] >= 0)
        train = snli("train").select(range(args.train_rows))
        evals = {"validation": snli("validation"), "test": snli("test")}
        n_labels, fields = 3, ("premise", "hypothesis")
    else:
        # Yahoo has no validation split: hold out the rows after the training
        # rows of a fixed shuffle of train. The official test stays sealed.
        yt = load_dataset("yahoo_answers_topics", split="train").shuffle(seed=0)
        train = yt.select(range(args.train_rows))
        evals = {"validation": yt.select(range(TRAIN_ROWS, TRAIN_ROWS + YAHOO_VALIDATION_ROWS)),
                 "test": load_dataset("yahoo_answers_topics", split="test")}
        def prep(b):
            return {"text_a": [f"{t} {c}".strip() for t, c in zip(b["question_title"], b["question_content"])],
                    "text_b": b["best_answer"], "label": b["topic"]}
        train = train.map(prep, batched=True)
        evals = {k: v.map(prep, batched=True) for k, v in evals.items()}
        n_labels, fields = 10, ("text_a", "text_b")
    labels = {k: np.array(v["label"], dtype=np.int64) for k, v in evals.items()}

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=n_labels)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id
    targets = ["query", "value"] if "bert" in args.model.lower() else ["q_proj", "v_proj"]
    model = get_peft_model(model, LoraConfig(r=args.r, lora_alpha=alpha, lora_dropout=0.0,
                                             target_modules=targets, bias="none", task_type="SEQ_CLS"))

    def tokenize(b):
        x = tok([_normalize_text(v or "") for v in b[fields[0]]],
                [_normalize_text(v or "") for v in b[fields[1]]], truncation=True, max_length=128)
        x["labels"] = [int(v) for v in b["label"]]
        return x

    train = train.map(tokenize, batched=True, remove_columns=train.column_names)
    train_tokens_per_epoch = int(sum(len(x) for x in train["input_ids"]))
    evals = {k: v.map(tokenize, batched=True, remove_columns=v.column_names) for k, v in evals.items()}

    targs = TrainingArguments(
        output_dir=str(out / "_trainer"), per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=64, learning_rate=args.lr, num_train_epochs=EPOCHS,
        warmup_ratio=0.06, weight_decay=0.01, logging_steps=50, save_strategy="no",
        eval_strategy="no", report_to=[], bf16=True, remove_unused_columns=False,
        dataloader_pin_memory=False, seed=args.seed, data_seed=args.seed)

    checkpoints: dict[int, dict] = {}
    t0 = [0.0]

    class Checkpoint(TrainerCallback):
        def on_train_begin(self, targs_, state, control, **kw):
            torch.cuda.synchronize()
            t0[0] = time.time()

        def on_epoch_end(self, targs_, state, control, **kw):
            ep = int(round(state.epoch))
            if ep not in CHECKPOINT_EPOCHS or ep in checkpoints:
                return
            torch.cuda.synchronize()
            train_seconds = time.time() - t0[0] - sum(c["eval_seconds"] for c in checkpoints.values())
            t_eval = time.time()
            ck = out / f"epoch{ep}"; ck.mkdir(exist_ok=True)
            for split, ds in evals.items():
                pred = trainer.predict(ds).predictions
                logits = np.asarray(pred[0] if isinstance(pred, tuple) else pred, dtype=np.float32)
                np.save(ck / f"logits_{split}.npy", logits)
                np.save(ck / f"labels_{split}.npy", labels[split])
            trainer.model.save_pretrained(ck / "adapter")
            losses = [h["loss"] for h in state.log_history if "loss" in h]
            recent = float(np.mean(losses[-4:])) if losses else float("nan")
            checkpoints[ep] = {
                "epoch": ep, "global_step": state.global_step,
                "train_seconds": round(train_seconds, 1),
                "eval_seconds": round(time.time() - t_eval, 1),
                "train_tokens": train_tokens_per_epoch * ep,
                "train_loss_recent": recent,
                # None when no loss was logged yet (tiny smoke runs)
                "failed": None if np.isnan(recent) else bool(recent >= 0.85 * float(np.log(n_labels))),
            }
            # the training loop resumes in train mode on its next step
            trainer.model.train()

    trainer = Trainer(model=model, args=targs, train_dataset=train,
                      data_collator=DataCollatorWithPadding(tokenizer=tok),
                      callbacks=[Checkpoint()])
    res = trainer.train()
    missing = [e for e in CHECKPOINT_EPOCHS if e not in checkpoints]
    if missing:
        raise SystemExit(f"checkpoints {missing} were not written")

    (out / "metrics.json").write_text(json.dumps({
        "model": args.model, "task": args.task, "lora_r": args.r, "lora_alpha": alpha, "learning_rate": args.lr,
        "epochs_scheduled": EPOCHS, "batch_size": args.bs, "seed": args.seed,
        "train_rows": args.train_rows, "train_loss_mean": float(res.metrics.get("train_loss", float("nan"))),
        "gpu": torch.cuda.get_device_name(0),
        "checkpoints": [checkpoints[e] for e in CHECKPOINT_EPOCHS],
        # validation accuracy only: test logits stay sealed (PREREG_AB.md §8)
        "acc_full_validation": {
            str(e): float((np.load(out / f"epoch{e}/logits_validation.npy").argmax(-1)
                           == labels["validation"]).mean()) for e in CHECKPOINT_EPOCHS},
    }, indent=2))
    for p in sorted((out / "_trainer").rglob("*"), reverse=True):
        p.unlink() if p.is_file() else p.rmdir()
    if (out / "_trainer").exists():
        (out / "_trainer").rmdir()
    return 0


if __name__ == "__main__":
    sys.exit(main())
