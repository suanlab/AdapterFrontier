#!/usr/bin/env bash
# Usage: bash mechanism/run_queue_h8.sh <gpu> <queue-file>   (lines: name model recipe bs seed)
set -u
GPU="$1"; QUEUE="$2"; cd "$(dirname "$0")/.."
while read -r name model recipe bs seed; do
  [ -z "$name" ] && continue
  out="mechanism/runs/$name"
  if [ -f "$out/metrics.json" ]; then echo "skip $name"; continue; fi
  echo "[$(date '+%F %T')] start $name (gpu $GPU)"
  CUDA_VISIBLE_DEVICES="$GPU" python3 mechanism/train_h8.py --model "$model" --recipe "$recipe" \
      --bs "$bs" --seed "$seed" --out "$out" > "mechanism/logs/$name.log" 2>&1 \
    && echo "[$(date '+%F %T')] done  $name" \
    || echo "[$(date '+%F %T')] FAIL  $name (see mechanism/logs/$name.log)"
done < "$QUEUE"
echo "[$(date '+%F %T')] queue $QUEUE finished"
