#!/bin/bash
# Runs every line of results/metrics/m4/jobs.txt as a 1-thread process, P in parallel.
# A job is skipped if its log already ends with a result line (resumable).
cd "$(dirname "$0")"
P=${P:-7}
JOBS=${JOBS:-../results/metrics/m4/jobs.txt}
export TH=${TH:-1}
run() {
  args="$1"; name=$(echo "$args" | tr -d '-' | tr ' ' '_')
  log=../results/metrics/m4/logs/$name.log
  grep -q "train_time_s" "$log" 2>/dev/null && exit 0
  QPINO_THREADS=$TH OMP_NUM_THREADS=$TH python3 m4_experiments.py $args > "$log" 2>&1
}
export -f run
grep -v "^\s*$" "$JOBS" | xargs -P "$P" -I{} bash -c 'run "{}"'
echo ALL_DONE
