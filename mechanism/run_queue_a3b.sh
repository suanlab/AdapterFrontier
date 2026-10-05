#!/usr/bin/env bash
# Usage: bash mechanism/run_queue_a3b.sh <gpu> <queue-file>
#   lines: name task model r lr bs seed     (PREREG_AB.md §9.2; all 4-epoch)
set -u
GPU="$1"; QUEUE="$2"; cd "$(dirname "$0")/.."
while read -r name task model r lr bs seed; do
  [ -z "$name" ] && continue
  out="mechanism/runs/$name"
  if [ -f "$out/metrics.json" ]; then echo "skip $name"; continue; fi
  echo "[$(date '+%F %T')] start $name (gpu $GPU)"
  CUDA_VISIBLE_DEVICES="$GPU" python3 mechanism/train_ab.py --task "$task" --model "$model" \
      --r "$r" --lr "$lr" --epochs 4 --bs "$bs" --seed "$seed" --out "$out" > "mechanism/logs/$name.log" 2>&1 \
    && echo "[$(date '+%F %T')] done  $name" \
    || echo "[$(date '+%F %T')] FAIL  $name (see mechanism/logs/$name.log)"
done < "$QUEUE"
echo "[$(date '+%F %T')] queue $QUEUE finished"
