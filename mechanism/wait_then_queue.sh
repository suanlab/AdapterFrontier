#!/usr/bin/env bash
# Usage: bash mechanism/wait_then_queue.sh <pid-to-wait-for> <gpu> <queue-file> [runner-script]
set -u
while kill -0 "$1" 2>/dev/null; do sleep 20; done
exec bash "$(dirname "$0")/${4:-run_queue_a3.sh}" "$2" "$3"
