"""学習 (sim2d.NavEnv) と実行 (mission_node) で共有する観測・行動の定義.

ここを共有することで「学習時と Gazebo 上で観測の作り方が1ビットも違わない」ことを保証する。
gymnasium に依存しないので ROS 2 ノードからそのまま import できる。
"""
import math

import numpy as np

from . import world_def as W

DT = 0.1  # 制御周期 [s] (Gazebo 側ノードも 10Hz で動かす)
SUBSTEPS = 2
GOAL_TOL = 0.20
N_DYN = 2      # 方策に渡す動的障害物の数 (近い順)
DYN_RANGE = 3.5
OBS_DIM = W.LIDAR_SECTORS * 2 + 3 + 2 + N_DYN * 5
SCAN_LAG = 3  # 過去スキャンは 0.3 秒前 (10Hz x 3) を使う: 動く障害物の速度を読み取りやすくする


class ScanHistory:
    """現在と SCAN_LAG ステップ前のセクタ距離を返すリングバッファ."""

    def __init__(self, lag=SCAN_LAG):
        self.lag = lag
        self.buf = []

    def push(self, sec):
        self.buf.append(sec)
        if len(self.buf) > self.lag + 1:
            self.buf.pop(0)
        return self.buf[0]


def sector_pool(ranges, n_sectors=W.LIDAR_SECTORS):
    """全周スキャン (index0 = 正面, 反時計回り) を n_sectors 個の min に縮約."""
    r = np.asarray(ranges, dtype=np.float64)
    r = np.where(np.isfinite(r), r, W.LIDAR_MAX)
    r = np.clip(r, W.LIDAR_MIN, W.LIDAR_MAX)
    n = len(r)
    assert n % n_sectors == 0, "LiDAR のレイ数はセクタ数の倍数にすること"
    k = n // n_sectors
    # 正面セクタが正面を中心に来るよう半セクタ分回す
    r = np.roll(r, k // 2)
    return r[: k * n_sectors].reshape(n_sectors, k).min(axis=1)


def dyn_features(tracks, pose):
    """追跡中の動的障害物 [(x, y, vx, vy) world] -> ロボット座標の特徴 (近い順 N_DYN 個).

    各物体: 相対位置 (前方, 左方) / 3.5m, 速度 (前方, 左方) / 0.5m/s, 有効フラグ
    """
    x, y, th = pose
    c, s = math.cos(-th), math.sin(-th)
    feats = []
    for ox, oy, vx, vy in tracks:
        dx, dy = ox - x, oy - y
        d = math.hypot(dx, dy)
        if d > DYN_RANGE:
            continue
        feats.append((d, [(c * dx - s * dy) / DYN_RANGE, (s * dx + c * dy) / DYN_RANGE,
                          (c * vx - s * vy) / 0.5, (s * vx + c * vy) / 0.5, 1.0]))
    feats.sort(key=lambda f: f[0])
    out = [f[1] for f in feats[:N_DYN]]
    while len(out) < N_DYN:
        out.append([0.0, 0.0, 0.0, 0.0, 0.0])
    return np.clip(np.array(out, dtype=np.float32).ravel(), -1.5, 1.5)


def build_obs(sectors, prev_sectors, goal_dist, goal_ang, v, w, dyn=None):
    if dyn is None:
        dyn = np.zeros(N_DYN * 5, np.float32)
    return np.concatenate([
        sectors / W.LIDAR_MAX,
        prev_sectors / W.LIDAR_MAX,
        [min(goal_dist, 4.0) / 4.0, math.cos(goal_ang), math.sin(goal_ang)],
        [v / W.V_MAX, w / W.W_MAX],
        dyn,
    ]).astype(np.float32)


def action_to_cmd(a):
    a = np.clip(a, -1.0, 1.0)
    v = W.V_MIN + (a[0] + 1.0) * 0.5 * (W.V_MAX - W.V_MIN)
    w = a[1] * W.W_MAX
    return float(v), float(w)


def tracker_goal(tracker, x, y, th, goal):
    """A* 経路上のサブゴールをロボット座標の極座標 (距離, 角度) で返す.

    学習 (NavEnv) と Gazebo 上のノード (mission_node) で全く同じ関数を使う。
    最終区間ではサブゴールではなく本当のゴールまでの距離を渡す。
    """
    sub = tracker.subgoal(x, y)
    dx, dy = sub[0] - x, sub[1] - y
    ang = math.atan2(dy, dx) - th
    ang = math.atan2(math.sin(ang), math.cos(ang))
    final = tracker.length - tracker.progress < 0.3 + tracker.lookahead
    dist = math.hypot(goal[0] - x, goal[1] - y) if final else math.hypot(dx, dy)
    return dist, ang, sub


def remaining(tracker, x, y, goal):
    """進捗報酬のポテンシャル: 経路上の残り距離 (終盤は直線距離)."""
    rem = tracker.length - tracker.progress
    return rem if rem > 0.3 else math.hypot(goal[0] - x, goal[1] - y)
