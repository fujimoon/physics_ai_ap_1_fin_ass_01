#!/usr/bin/env bash
# Gazebo 上で配達ミッションを N 回反復し (作業員の出発タイミングを変える)、結果を集計する
#   bash tools/eval_gazebo.sh [N]
cd "$(dirname "$0")/.."
N=${1:-5}
DELAYS=(3.0 6.5 10.0 14.0 18.5 1.0 8.0 12.0 16.0 21.0)
for i in $(seq 0 $((N-1))); do
  d=${DELAYS[$i]}
  echo "=== trial $i (worker_delay=$d) ==="
  docker run --rm -v "$PWD":/work pai-gz:humble bash /work/tools/run_sim.sh /work/recordings/eval_$i 600 \
      save_images:=false worker_delay:=$d | tail -3
done
.venv/bin/python tools/summarize_runs.py recordings/eval_*
