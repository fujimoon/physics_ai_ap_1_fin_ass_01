"""大域経路計画 (A*) と サブゴール生成.

階層型ナビゲーション:
  大域層 = 既知の静的地図上の A* (遠回りや袋小路を避ける)
  局所層 = 強化学習方策 (動く障害物の回避・滑らかな追従)
A* 経路上で「ロボットから LOOKAHEAD [m] 先の点」をサブゴールとして RL 方策に渡す。
"""
import heapq
import math

import numpy as np

from . import world_def as W

RES = 0.05          # 格子解像度 [m]
INFLATE = W.ROBOT_RADIUS + 0.12
LOOKAHEAD = 1.0


class GridMap:
    def __init__(self, res=RES, inflate=INFLATE, boxes=None, circles=None, half_size=None):
        """引数省略時は world_def の倉庫. 学習時はランダム生成した部屋を渡す."""
        if boxes is None:
            boxes, circles = W.static_obstacles()
            half_size = (W.ROOM_W / 2, W.ROOM_H / 2)
        self.res = res
        self.inflate = inflate
        m = 0.3  # 外壁の外側まで格子を広げる (壁の内部も距離場で表現するため)
        self.ox, self.oy = -half_size[0] - m, -half_size[1] - m
        self.nx = int(round((2 * half_size[0] + 2 * m) / res))
        self.ny = int(round((2 * half_size[1] + 2 * m) / res))
        xs = self.ox + (np.arange(self.nx) + 0.5) * res
        ys = self.oy + (np.arange(self.ny) + 0.5) * res
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        sdf = np.full(X.shape, np.inf)
        for cx, cy, sx, sy in boxes:
            dx = np.abs(X - cx) - sx / 2
            dy = np.abs(Y - cy) - sy / 2
            out = np.hypot(np.maximum(dx, 0), np.maximum(dy, 0))
            sdf = np.minimum(sdf, out + np.minimum(np.maximum(dx, dy), 0))
        for cx, cy, r in circles:
            sdf = np.minimum(sdf, np.hypot(X - cx, Y - cy) - r)
        self.sdf = sdf                    # 符号付き距離場 (障害物内部で負): 自己位置推定で使用
        dist = np.maximum(sdf, 0)
        self.dist = dist                  # 最近障害物までの距離場
        self.occ = dist < inflate         # 膨張済み占有

    def to_cell(self, x, y):
        return (min(max(int((x - self.ox) / self.res), 0), self.nx - 1),
                min(max(int((y - self.oy) / self.res), 0), self.ny - 1))

    def to_world(self, i, j):
        return self.ox + (i + 0.5) * self.res, self.oy + (j + 0.5) * self.res

    def dynamic_layer(self, tracks, horizon=4.0, pad=0.12):
        """追跡中の動的障害物の「現在位置〜horizon 秒後の予測位置」に沿ってコストを置く (Nav2 の obstacle layer 相当).

        通行禁止にすると通路を塞いで計画が失敗/振動するため、近いほど高い「ソフトなコスト」にする。
        """
        extra = np.zeros(self.occ.shape)
        xs = self.ox + (np.arange(self.nx) + 0.5) * self.res
        ys = self.oy + (np.arange(self.ny) + 0.5) * self.res
        rad = W.ROBOT_RADIUS + 0.22 + pad
        for tr in tracks:
            p0 = np.array([tr["x"], tr["y"]])
            p1 = p0 + horizon * np.array([tr["vx"], tr["vy"]])
            lo = np.minimum(p0, p1) - rad
            hi = np.maximum(p0, p1) + rad
            i0, j0 = self.to_cell(*lo)
            i1, j1 = self.to_cell(*hi)
            X, Y = np.meshgrid(xs[i0:i1 + 1], ys[j0:j1 + 1], indexing="ij")
            dvec = p1 - p0
            L2 = float(dvec @ dvec) + 1e-9
            t = np.clip(((X - p0[0]) * dvec[0] + (Y - p0[1]) * dvec[1]) / L2, 0, 1)
            dist = np.hypot(X - (p0[0] + t * dvec[0]), Y - (p0[1] + t * dvec[1]))
            # 予測の先ほど不確かなので重みを下げる
            wgt = 40.0 * (1.0 - 0.6 * t)
            cost = np.where(dist < rad, wgt, wgt * np.exp(-(dist - rad) / 0.15))
            extra[i0:i1 + 1, j0:j1 + 1] = np.maximum(extra[i0:i1 + 1, j0:j1 + 1], cost)
        return extra

    def astar(self, start, goal, extra_cost=None):
        s, g = self.to_cell(*start), self.to_cell(*goal)
        occ = self.occ.copy()
        # 始点・終点付近は膨張を解除（ステーションが壁際にあっても計画できるように）
        for c in (s, g):
            occ[max(c[0] - 3, 0):c[0] + 4, max(c[1] - 3, 0):c[1] + 4] = False
        # 障害物から離れるほど低コスト（通路の中央を通る経路になる）
        pen = 1.0 + 2.0 * np.exp(-np.clip(self.dist - self.inflate, 0, None) / 0.25)
        if extra_cost is not None:
            pen = pen + extra_cost
        nbrs = [(1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
                (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))]
        openq = [(0.0, s)]
        gcost = {s: 0.0}
        parent = {}
        while openq:
            _, cur = heapq.heappop(openq)
            if cur == g:
                break
            for di, dj, c in nbrs:
                n = (cur[0] + di, cur[1] + dj)
                if not (0 <= n[0] < self.nx and 0 <= n[1] < self.ny) or occ[n]:
                    continue
                ng = gcost[cur] + c * pen[n]
                if ng < gcost.get(n, np.inf):
                    gcost[n] = ng
                    parent[n] = cur
                    h = math.hypot(n[0] - g[0], n[1] - g[1])
                    heapq.heappush(openq, (ng + h, n))
        if g not in parent and g != s:
            return None
        path = [g]
        while path[-1] != s:
            path.append(parent[path[-1]])
        path.reverse()
        pts = np.array([self.to_world(*c) for c in path])
        pts[0] = start
        pts[-1] = goal
        return smooth(pts)


def smooth(pts, iters=60, a=0.4, b=0.25):
    """端点固定のラプラシアン平滑化で格子のガタつきを除去."""
    p = pts.copy()
    for _ in range(iters):
        p[1:-1] += a * (pts[1:-1] - p[1:-1]) + b * (p[:-2] + p[2:] - 2 * p[1:-1])
    return p


class PathTracker:
    """A* 経路を保持し、ロボット位置から lookahead 先のサブゴールを返す."""

    def __init__(self, path, lookahead=LOOKAHEAD):
        self.path = np.asarray(path)
        self.lookahead = lookahead
        seg = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        self.s = np.concatenate([[0], np.cumsum(seg)])
        self.progress = 0.0

    @property
    def length(self):
        return float(self.s[-1])

    def subgoal(self, x, y):
        # 現在の進捗付近 (後戻りしない) で最も近い経路点を探す
        win = (self.s >= self.progress - 0.3) & (self.s <= self.progress + 2.0)
        idx = np.where(win)[0]
        d = np.hypot(self.path[idx, 0] - x, self.path[idx, 1] - y)
        k = idx[int(np.argmin(d))]
        self.progress = max(self.progress, float(self.s[k]))
        target_s = min(self.progress + self.lookahead, self.s[-1])
        j = int(np.searchsorted(self.s, target_s))
        j = min(j, len(self.path) - 1)
        return self.path[j]


class DynamicReplanner:
    """動的障害物を考慮して一定周期で A* を引き直す.

    近く (NEAR m 以内) に追跡中の物体があれば period 秒ごとに動的レイヤー込みで再計画し、
    物体がいなくなったら静的地図だけの経路に戻す。計画に失敗したら現在の経路を維持する。
    """
    NEAR = 3.5

    def __init__(self, grid, period=1.0):
        self.g = grid
        self.period = period
        self.t_last = -1e9
        self.dynamic = False
        self.count = 0

    def step(self, t, pose, goal, tracks):
        near = [tr for tr in tracks if math.hypot(tr["x"] - pose[0], tr["y"] - pose[1]) < self.NEAR]
        if not near and not self.dynamic:
            return None
        if t - self.t_last < self.period:
            return None
        self.t_last = t
        extra = self.g.dynamic_layer(near) if near else None
        path = self.g.astar(pose[:2], goal, extra_cost=extra)
        if path is None:
            return None
        self.dynamic = bool(near)
        self.count += 1
        return PathTracker(path)
