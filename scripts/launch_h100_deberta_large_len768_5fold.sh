#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

mkdir -p logs

timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="logs/h100_deberta_large_len768_5fold_${timestamp}.log"
pid_file="logs/h100_deberta_large_len768_5fold_${timestamp}.pid"

nohup bash scripts/run_h100_deberta_large_len768_5fold.sh \
  > "${log_file}" 2>&1 &

pid="$!"
echo "${pid}" > "${pid_file}"

echo "Started H100 DeBERTa large 5-fold training"
echo "PID: ${pid}"
echo "Log: ${log_file}"
echo "PID file: ${pid_file}"
echo
echo "Follow logs:"
echo "  tail -f ${log_file}"
echo
echo "Stop training:"
echo "  kill ${pid}"
