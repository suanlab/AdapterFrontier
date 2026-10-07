#!/usr/bin/env bash
# Start the A3e queues only after both GPUs are free of other users' jobs
# (used memory below 6 GB on each). Checks every 5 minutes.
set -u
cd "$(dirname "$0")/.."
echo "[$(date '+%F %T')] waiting for free GPUs"
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')
  set -- $used
  if [ "${1:-99999}" -lt 6000 ] && [ "${2:-99999}" -lt 6000 ]; then break; fi
  sleep 300
done
echo "[$(date '+%F %T')] GPUs free (used: $used MiB); starting A3e"
nohup setsid bash mechanism/run_queue_a3d.sh 0 mechanism/queue_a3e_gpu0.txt > mechanism/logs/queue_a3e_gpu0.out 2>&1 < /dev/null &
nohup setsid bash mechanism/run_queue_a3d.sh 1 mechanism/queue_a3e_gpu1.txt > mechanism/logs/queue_a3e_gpu1.out 2>&1 < /dev/null &
echo "[$(date '+%F %T')] launched"
