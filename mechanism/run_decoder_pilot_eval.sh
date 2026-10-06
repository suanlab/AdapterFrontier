#!/usr/bin/env bash
# Exploratory: cache member logits for the six decoder pools with >=6 surviving
# adapters (the decoder counterpart of the six-pool encoder control).
set -u
cd "$(dirname "$0")/.."
for spec in "pool_a_anli_qwen25_05b_local10 anli 1200" "pool_a_anli_qwen25_15b_local10 anli 1200" \
            "pool_a_agnews_qwen25_05b_local7 agnews 7600" "pool_a_agnews_qwen25_15b_local10 agnews 7600" \
            "pool_a_qnli_qwen25_05b_local6 qnli 5463" "pool_a_qnli_qwen25_15b_local10 qnli 5463"; do
  set -- $spec
  out="ensemble_results/pilot_c1/$1.json"
  [ -f "$out" ] && { echo "skip $1"; continue; }
  echo "[$(date '+%F %T')] start $1"
  CUDA_VISIBLE_DEVICES=1 python3 ensemble_eval.py --pool "pools/pilot_c1/$1.json" --task "$2" \
      --max-eval-samples "$3" --cache-dir ensemble_cache --out "$out" > "mechanism/logs/pilot_eval_$1.log" 2>&1 \
    && echo "[$(date '+%F %T')] done  $1" || echo "[$(date '+%F %T')] FAIL  $1"
done
echo "[$(date '+%F %T')] finished"
