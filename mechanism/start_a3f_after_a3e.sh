#!/usr/bin/env bash
# Start the A3f queues (registration step 9.8) once both A3e queues have finished.
set -u
cd "$(dirname "$0")/.."
echo "[$(date '+%F %T')] waiting for A3e to finish"
until grep -q finished mechanism/logs/queue_a3e_gpu0.out 2>/dev/null && grep -q finished mechanism/logs/queue_a3e_gpu1.out 2>/dev/null; do sleep 300; done
echo "[$(date '+%F %T')] A3e finished; starting A3f"
nohup setsid bash mechanism/run_queue_a3d.sh 0 mechanism/queue_a3f_gpu0.txt > mechanism/logs/queue_a3f_gpu0.out 2>&1 < /dev/null &
nohup setsid bash mechanism/run_queue_a3d.sh 1 mechanism/queue_a3f_gpu1.txt > mechanism/logs/queue_a3f_gpu1.out 2>&1 < /dev/null &
echo "[$(date '+%F %T')] launched"
