"""倉庫ワールドの「唯一の定義」(Single Source of Truth).

ここで定義した障害物・配達ステーションから
  * Gazebo 用 world.sdf (tools/gen_world.py)
  * A* 用の占有格子地図 (planner.py)
  * 動画用の俯瞰マップ描画 (video/compose_video.py)
をすべて自動生成する。座標系は Gazebo の world 座標 [m]。
"""

# 部屋の内寸 (x: -ROOM_W/2..ROOM_W/2, y: -ROOM_H/2..ROOM_H/2)
ROOM_W = 8.0
ROOM_H = 6.0
WALL_T = 0.10

# 静的障害物: 箱 (cx, cy, sx, sy, 高さ, 色名)
BOXES = [
    (-1.50, 1.05, 0.50, 2.70, 0.6, "shelf"),   # 左の棚
    (1.00, -1.05, 0.50, 2.70, 0.6, "shelf"),   # 中央の棚
    (2.60, 1.40, 1.60, 0.40, 0.6, "shelf"),    # 右上の棚
    (-0.35, -2.10, 0.60, 0.60, 0.5, "crate"),  # 木箱
    (3.05, -0.55, 0.50, 0.50, 0.5, "crate"),   # 木箱
]
# 静的障害物: 円柱 (cx, cy, r, 高さ)
CYLINDERS = [
    (-2.80, -1.10, 0.22, 1.2),  # 柱
    (0.20, 1.55, 0.18, 1.2),    # 柱
]

# 動く障害物 (作業員): 線分 WORKER_P0 <-> WORKER_P1 を往復 (ロボットの経路がレーン中央付近を横切る配置)
# 作業員は地図に載っておらず、ロボットには位置も動きも一切与えない (LiDAR で見つけるしかない)
WORKER_R = 0.20
WORKER_P0 = (-2.20, -1.00)
WORKER_P1 = (0.30, -1.00)
WORKER_SPEED = 0.30

# 配達ステーション (名前, x, y, 色RGB)
HOME = ("HOME", -3.30, -2.30, (0.2, 0.8, 0.2))
STATIONS = [
    ("A", -0.40, 2.30, (0.95, 0.25, 0.25)),
    ("B", 3.25, -2.25, (0.25, 0.45, 0.95)),
    ("C", 3.30, 2.40, (0.95, 0.75, 0.15)),
]
# 配達ミッション: HOME -> A -> B -> C -> HOME
MISSION = ["A", "B", "C", "HOME"]

# ロボット (参考記事の差動二輪ロボットと同じ車輪パラメータ)
WHEEL_RADIUS = 0.033
WHEEL_SEPARATION = 0.19
ROBOT_RADIUS = 0.14          # 衝突判定用の外接円半径
V_MAX = 0.35                 # [m/s]
V_MIN = -0.12                # [m/s] 後退も許可 (全周 LiDAR なので後方も見える)
W_MAX = 1.6                  # [rad/s]

# LiDAR
LIDAR_RAYS = 240             # 全周 1.5deg 刻み (24セクタ x 10本)
LIDAR_MIN = 0.12
LIDAR_MAX = 3.5
LIDAR_SECTORS = 24           # 方策への入力は 15deg ごとの min-pooling


def station(name):
    if name == HOME[0]:
        return HOME
    for s in STATIONS:
        if s[0] == name:
            return s
    raise KeyError(name)


def wall_boxes():
    """外壁を箱として返す (cx, cy, sx, sy)."""
    hw, hh, t = ROOM_W / 2, ROOM_H / 2, WALL_T
    return [
        (0.0, hh + t / 2, ROOM_W + 2 * t, t),
        (0.0, -hh - t / 2, ROOM_W + 2 * t, t),
        (hw + t / 2, 0.0, t, ROOM_H),
        (-hw - t / 2, 0.0, t, ROOM_H),
    ]


def static_obstacles():
    """(boxes[(cx,cy,sx,sy)], circles[(cx,cy,r)]) を返す（壁込み）."""
    boxes = [(b[0], b[1], b[2], b[3]) for b in BOXES] + wall_boxes()
    circles = [(c[0], c[1], c[2]) for c in CYLINDERS]
    return boxes, circles


def worker_lane():
    """(始点, 単位ベクトル, 長さ) を返す."""
    import math
    dx, dy = WORKER_P1[0] - WORKER_P0[0], WORKER_P1[1] - WORKER_P0[1]
    L = math.hypot(dx, dy)
    return WORKER_P0, (dx / L, dy / L), L
