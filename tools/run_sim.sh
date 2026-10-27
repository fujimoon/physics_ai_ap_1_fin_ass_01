#!/usr/bin/env bash
# コンテナ内で実行: ビルド -> Gazebo + AI ノード起動 -> ミッション完了で終了
#   docker run --rm -v "$PWD":/work pai-gz:humble bash /work/tools/run_sim.sh [out_dir] [timeout_sec]
set -e
OUT=${1:-/work/recordings/run}
TIMEOUT=${2:-900}
shift 2 2>/dev/null || true
EXTRA="$@"   # 例: save_images:=false worker_delay:=7.0
source /opt/ros/humble/setup.bash
Xvfb :99 -screen 0 1280x800x24 >/dev/null 2>&1 &
export DISPLAY=:99
cd /work/ros2_ws
colcon build --symlink-install --packages-select ai_delivery > /tmp/build.log 2>&1 || { cat /tmp/build.log; exit 1; }
source install/setup.bash
rm -rf "$OUT"; mkdir -p "$OUT"
ros2 launch ai_delivery ai_delivery.launch.py out_dir:="$OUT" $EXTRA > "$OUT/launch.log" 2>&1 &
LPID=$!
SECONDS=0
while [ ! -f "$OUT/FINISHED" ] && [ $SECONDS -lt $TIMEOUT ]; do sleep 5; done
[ -f "$OUT/FINISHED" ] && echo "MISSION FINISHED in ${SECONDS}s wall" || echo "TIMEOUT"
kill -INT $LPID; sleep 3; pkill -9 -f "ign gazebo" || true; pkill -9 -f parameter_bridge || true
grep -E "A\* planned|Arrived|Mission|Stuck|Delivered" "$OUT/launch.log" | sed 's/\x1b\[[0-9;]*m//g' | tail -30
