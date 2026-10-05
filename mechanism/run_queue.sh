#!/usr/bin/env bash
# Usage: bash mechanism/run_queue.sh <gpu> <queue-file>
# Runs each line "name head lora data epochs" sequentially on one GPU. Skips
# runs that already finished, so the queue can be restarted after a failure.
set -u
GPU="$1"; QUEUE="$2"; cd "$(dirname "$0")/.."
while read -r name h l d e; do
  [ -z "$name" ] && continue
  out="mechanism/runs/$name"
  if [ -f "$out/metrics.json" ]; then echo "skip $name"; continue; fi
  echo "[$(date '+%F %T')] start $name (gpu $GPU)"
  CUDA_VISIBLE_DEVICES="$GPU" python3 mechanism/train_member.py \
      --head-seed "$h" --lora-seed "$l" --data-seed "$d" --epochs "$e" --out "$out" \
      > "mechanism/logs/$name.log" 2>&1 \
    && echo "[$(date '+%F %T')] done  $name" \
    || echo "[$(date '+%F %T')] FAIL  $name (see mechanism/logs/$name.log)"
done < "$QUEUE"
echo "[$(date '+%F %T')] queue $QUEUE finished"
