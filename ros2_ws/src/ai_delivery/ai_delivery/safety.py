"""動的障害物の検出・追跡と、モデル予測型セーフティフィルタ.

学習方策は前方からの対向にはよく対応するが、斜め後方から追い越してくる作業員には弱い
(2D 評価での衝突の大半がこのケース)。そこで方策の外側に「予測型の安全層」を置く:

  1. 地図差分による動的障害物検出:
       LiDAR 点を world に投影し、地図の距離場で 0.15m 以上離れている点 = 地図に無い物体
       → 角度方向に連続する点をクラスタリング → 中心を推定
  2. 追跡: 最近傍対応 + 指数移動平均で速度推定 (等速モデル)
  3. セーフティフィルタ:
       方策の指令と代替指令 (減速・停止・後退・回避旋回 …) をそれぞれ 1.6 秒先まで
       ロールアウトし、予測した障害物位置・静的地図との最小クリアランスを評価。
       方策の指令が安全ならそのまま通し、危険なら「安全な候補のうち方策に最も近いもの」を選ぶ。
"""
import math

import numpy as np

from . import world_def as W
from .localizer import LIDAR_OFFSET

HORIZON = 2.4
DT_SIM = 0.1
MARGIN = 0.07         # 必要な余裕 [m]
DYN_THRESH = 0.15     # 地図から何 m 離れた点を「地図に無い物体」とみなすか
OBS_R = 0.22          # 検出物体の想定半径


class DynamicTracker:
    def __init__(self, grid):
        self.g = grid
        self.tracks = []   # dict(x, y, vx, vy, age, miss)
        self.t_last = None

    def _sdf(self, X, Y):
        g = self.g
        i = np.clip(((X - g.ox) / g.res).astype(int), 0, g.nx - 1)
        j = np.clip(((Y - g.oy) / g.res).astype(int), 0, g.ny - 1)
        return g.sdf[i, j]

    def detect(self, ranges, angle_min, angle_inc, pose):
        r = np.asarray(ranges, dtype=np.float64)
        a = angle_min + np.arange(len(r)) * angle_inc
        x, y, th = pose
        ok = np.isfinite(r) & (r > 0.13) & (r < 3.4)
        r = np.where(ok, r, 0.0)
        px = LIDAR_OFFSET + r * np.cos(a)
        py = r * np.sin(a)
        c, s = math.cos(th), math.sin(th)
        X = x + c * px - s * py
        Y = y + s * px + c * py
        dyn = ok & (self._sdf(X, Y) > DYN_THRESH)
        clusters = []
        n = len(r)
        idx = np.where(dyn)[0]
        if len(idx) == 0:
            return clusters
        # 角度方向に連続する点をまとめる (全周なので端をまたぐ連結も考慮)
        groups, cur = [], [idx[0]]
        for k in idx[1:]:
            if k - cur[-1] <= 2 and math.hypot(X[k] - X[cur[-1]], Y[k] - Y[cur[-1]]) < 0.25:
                cur.append(k)
            else:
                groups.append(cur)
                cur = [k]
        groups.append(cur)
        if len(groups) > 1 and groups[0][0] == 0 and groups[-1][-1] == n - 1:
            groups[0] = groups[-1] + groups[0]
            groups.pop()
        for gidx in groups:
            if len(gidx) < 2:
                continue
            cx, cy = float(np.mean(X[gidx])), float(np.mean(Y[gidx]))
            # 見えているのは表面なので、ロボットから見て奥へ半径分ずらして中心とする
            b = math.atan2(cy - y, cx - x)
            clusters.append((cx + 0.6 * OBS_R * math.cos(b), cy + 0.6 * OBS_R * math.sin(b)))
        return clusters

    def update(self, clusters, t):
        dt = 0.1 if self.t_last is None else max(1e-3, t - self.t_last)
        self.t_last = t
        for tr in self.tracks:  # 予測
            tr["x"] += tr["vx"] * dt
            tr["y"] += tr["vy"] * dt
            tr["miss"] += 1
        used = set()
        for cx, cy in clusters:
            best, bd = None, 0.5
            for k, tr in enumerate(self.tracks):
                d = math.hypot(tr["x"] - cx, tr["y"] - cy)
                if d < bd and k not in used:
                    best, bd = k, d
            if best is None:
                self.tracks.append({"x": cx, "y": cy, "vx": 0.0, "vy": 0.0, "age": 1, "miss": 0})
                used.add(len(self.tracks) - 1)
            else:
                tr = self.tracks[best]
                ex, ey = cx - tr["x"], cy - tr["y"]   # イノベーション
                tr["x"] += 0.6 * ex
                tr["y"] += 0.6 * ey
                tr["vx"] = float(np.clip(tr["vx"] + 0.35 * ex / dt, -0.8, 0.8))
                tr["vy"] = float(np.clip(tr["vy"] + 0.35 * ey / dt, -0.8, 0.8))
                tr["age"] += 1
                tr["miss"] = 0
                used.add(best)
        self.tracks = [tr for tr in self.tracks if tr["miss"] <= 5]
        return [tr for tr in self.tracks if tr["age"] >= 3]


def _candidates(v, w):
    c = [(v, w), (0.6 * v, w), (0.3 * v, w), (0.0, w), (0.0, 0.0), (W.V_MIN, 0.0), (W.V_MIN, w)]
    for vv in (0.0, 0.15, 0.3):
        for ww in (-1.2, -0.6, 0.6, 1.2):
            c.append((vv, ww))
    c += [(W.V_MIN, 0.8), (W.V_MIN, -0.8)]
    return c


class SafetyFilter:
    def __init__(self, grid):
        self.g = grid
        self.last_intervened = False

    def _static_clear(self, X, Y):
        g = self.g
        i = np.clip(((X - g.ox) / g.res).astype(int), 0, g.nx - 1)
        j = np.clip(((Y - g.oy) / g.res).astype(int), 0, g.ny - 1)
        return g.sdf[i, j] - W.ROBOT_RADIUS

    def _rollout(self, pose, cmd, tracks, static_steps=5):
        x, y, th = pose
        v, w = cmd
        steps = int(HORIZON / DT_SIM)
        ks = np.arange(1, steps + 1) * DT_SIM
        if abs(w) < 1e-3:
            X = x + v * ks * math.cos(th)
            Y = y + v * ks * math.sin(th)
        else:
            X = x + v / w * (np.sin(th + w * ks) - math.sin(th))
            Y = y - v / w * (np.cos(th + w * ks) - math.cos(th))
        # 静的障害物: 方策自身の指令はごく近い未来 (0.5 秒) だけ確認 (方策が LiDAR で見ているため)。
        # 代替指令は全ホライズンで確認する (棚へ後退するなどを防ぐ)
        m_static = float(np.min(self._static_clear(X[:static_steps], Y[:static_steps])))
        m_dyn = 9.0
        for tr in tracks:
            ox = tr["x"] + tr["vx"] * ks
            oy = tr["y"] + tr["vy"] * ks
            d = np.hypot(X - ox, Y - oy) - W.ROBOT_RADIUS - OBS_R
            # 遠い未来ほど予測が不確か: 「近い未来を重視」と「不確かさで相手を大きく見る」の折衷
            m_dyn = min(m_dyn, float(np.min(d + 0.03 * ks)))
        return m_static, m_dyn

    def filter(self, pose, cmd, tracks):
        """(安全化した指令, 介入したか, 評価値) を返す."""
        ms, md = self._rollout(pose, cmd, tracks)
        if md >= MARGIN and ms >= 0.0:
            self.last_intervened = False
            return cmd, False, md
        best, best_score = cmd, -1e9
        for c in _candidates(*cmd):
            ms_c, md_c = self._rollout(pose, c, tracks, static_steps=int(HORIZON / DT_SIM))
            margin = min(md_c, ms_c + MARGIN)
            safe = margin >= MARGIN
            dev = abs(c[0] - cmd[0]) / W.V_MAX + 0.3 * abs(c[1] - cmd[1]) / W.W_MAX
            score = (10.0 if safe else 0.0) + (0.0 if safe else 5.0 * min(margin, 0.3)) - dev
            if score > best_score:
                best, best_score = c, score
        self.last_intervened = True
        return best, True, md
