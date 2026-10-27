"""強化学習用の高速 2D シミュレータ (Gymnasium 環境).

Gazebo は1ステップが重く、PPO の学習 (数百万ステップ) には向かない。
そこで「差動二輪ロボット + 2D LiDAR」だけを numpy で再現した軽量環境で学習し、
学習済み方策を Gazebo にそのまま載せる (Sim-to-Sim 転移)。

転移ギャップを埋めるためのドメインランダマイゼーション:
  * 部屋サイズ・障害物の数/形/配置をエピソード毎にランダム生成
  * 動く障害物 (速度・半径ランダム)
  * LiDAR ノイズ・欠損, 速度応答の遅れ(一次遅れ時定数), 速度ゲイン誤差
"""
import math

import gymnasium as gym
import numpy as np

from . import world_def as W
from .nav_core import (DT, GOAL_TOL, OBS_DIM, SUBSTEPS, ScanHistory, action_to_cmd,  # noqa: F401
                       build_obs, dyn_features, remaining, sector_pool, tracker_goal)
from .planner import GridMap, PathTracker

def raycast(px, py, angles, boxes, circles, max_r=W.LIDAR_MAX):
    """boxes: (N,4) [cx,cy,sx,sy] 軸並行, circles: (M,3) [cx,cy,r]."""
    dx = np.cos(angles)
    dy = np.sin(angles)
    best = np.full(angles.shape, max_r)
    if len(boxes):
        b = np.asarray(boxes)
        x0 = (b[:, 0] - b[:, 2] / 2)[:, None]
        x1 = (b[:, 0] + b[:, 2] / 2)[:, None]
        y0 = (b[:, 1] - b[:, 3] / 2)[:, None]
        y1 = (b[:, 1] + b[:, 3] / 2)[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            idx = 1.0 / np.where(np.abs(dx) < 1e-9, 1e-9, dx)
            idy = 1.0 / np.where(np.abs(dy) < 1e-9, 1e-9, dy)
            tx0 = (x0 - px) * idx
            tx1 = (x1 - px) * idx
            ty0 = (y0 - py) * idy
            ty1 = (y1 - py) * idy
        tmin = np.maximum(np.minimum(tx0, tx1), np.minimum(ty0, ty1))
        tmax = np.minimum(np.maximum(tx0, tx1), np.maximum(ty0, ty1))
        hit = (tmax >= np.maximum(tmin, 0.0))
        t = np.where(tmin > 0, tmin, tmax)  # 内側からなら出口
        t = np.where(hit & (t > 0), t, np.inf)
        best = np.minimum(best, t.min(axis=0))
    if len(circles):
        c = np.asarray(circles)
        ox = (px - c[:, 0])[:, None]
        oy = (py - c[:, 1])[:, None]
        bq = ox * dx + oy * dy
        cq = ox ** 2 + oy ** 2 - (c[:, 2] ** 2)[:, None]
        disc = bq ** 2 - cq
        sq = np.sqrt(np.maximum(disc, 0.0))
        t1 = -bq - sq
        t2 = -bq + sq
        t = np.where(t1 > 0, t1, t2)
        t = np.where((disc >= 0) & (t > 0), t, np.inf)
        best = np.minimum(best, t.min(axis=0))
    return best


def lane_bounce(m):
    """レーン端で折り返す (m = [x, y, vx, vy, r, kind])."""
    p0, u, L = W.worker_lane()
    sl = (m[0] - p0[0]) * u[0] + (m[1] - p0[1]) * u[1]
    vs = m[2] * u[0] + m[3] * u[1]
    if (sl < 0 and vs < 0) or (sl > L and vs > 0):
        m[2], m[3] = -m[2], -m[3]


def clearance(px, py, boxes, circles):
    d = np.inf
    if len(boxes):
        b = np.asarray(boxes)
        qx = np.maximum(np.abs(px - b[:, 0]) - b[:, 2] / 2, 0.0)
        qy = np.maximum(np.abs(py - b[:, 1]) - b[:, 3] / 2, 0.0)
        d = min(d, float(np.min(np.hypot(qx, qy))))
    if len(circles):
        c = np.asarray(circles)
        d = min(d, float(np.min(np.hypot(px - c[:, 0], py - c[:, 1]) - c[:, 2])))
    return d


class NavEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed=None, p_warehouse=0.4):
        super().__init__()
        self.observation_space = gym.spaces.Box(-1.5, 1.5, (OBS_DIM,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.rng = np.random.default_rng(seed)
        self.p_warehouse = p_warehouse
        self.angles_rel = np.arange(W.LIDAR_RAYS) * (2 * math.pi / W.LIDAR_RAYS)

    # ------------------------------------------------------------ world gen
    def _random_world(self):
        rng = self.rng
        self.warehouse = rng.random() < self.p_warehouse
        if self.warehouse:
            boxes, circles = W.static_obstacles()
            self.bounds = (W.ROOM_W / 2, W.ROOM_H / 2)
            p0, u, L = W.worker_lane()
            sl = rng.uniform(0, L)
            sp = W.WORKER_SPEED * rng.choice([-1, 1]) * rng.uniform(0.6, 1.3)
            movers = [[p0[0] + u[0] * sl, p0[1] + u[1] * sl, u[0] * sp, u[1] * sp, W.WORKER_R, "lane"]]
        else:
            hw, hh = rng.uniform(2.5, 4.5), rng.uniform(2.0, 3.5)
            self.bounds = (hw, hh)
            t = W.WALL_T
            boxes = [(0, hh + t / 2, 2 * hw + 2 * t, t), (0, -hh - t / 2, 2 * hw + 2 * t, t),
                     (hw + t / 2, 0, t, 2 * hh), (-hw - t / 2, 0, t, 2 * hh)]
            for _ in range(rng.integers(2, 9)):
                if rng.random() < 0.5:  # 細長い棚
                    sx, sy = rng.uniform(0.3, 0.6), rng.uniform(0.8, 3.0)
                    if rng.random() < 0.5:
                        sx, sy = sy, sx
                else:
                    sx, sy = rng.uniform(0.3, 0.9), rng.uniform(0.3, 0.9)
                boxes.append((rng.uniform(-hw, hw), rng.uniform(-hh, hh), sx, sy))
            circles = [(rng.uniform(-hw, hw), rng.uniform(-hh, hh), rng.uniform(0.1, 0.3))
                       for _ in range(rng.integers(0, 5))]
            movers = []
            for _ in range(rng.integers(0, 3)):
                ang = rng.uniform(0, 2 * math.pi)
                sp = rng.uniform(0.0, 0.45)
                movers.append([rng.uniform(-hw, hw), rng.uniform(-hh, hh),
                               sp * math.cos(ang), sp * math.sin(ang),
                               rng.uniform(0.15, 0.3), "free"])
        self.boxes = np.asarray(boxes, dtype=np.float64)
        self.static_circles = np.asarray(circles, dtype=np.float64).reshape(-1, 3)
        self.movers = movers

    def _add_oncoming(self, path, start):
        """経路に絡む動く障害物 (人) を追加: 対向 / 横断 / 同方向 をランダムに."""
        rng = self.rng
        if rng.random() > 0.75 or self.tracker.length < 1.8:
            return
        for _ in range(rng.integers(1, 3)):
            sp = PathTracker(path)
            s_at = rng.uniform(1.2, min(4.5, self.tracker.length))
            j = min(max(int(np.searchsorted(sp.s, s_at)), 1), len(path) - 1)
            px, py = path[j]
            tx, ty = path[j] - path[j - 1]
            n = math.hypot(tx, ty) + 1e-9
            tx, ty = tx / n, ty / n
            r = rng.uniform(0.15, 0.25)
            kind = rng.choice(["oncoming", "crossing", "same"], p=[0.35, 0.5, 0.15])
            speed = rng.uniform(0.1, 0.4)
            if kind == "crossing":
                # 経路の横から、ロボットとほぼ同時に経路に達するように出発
                side = rng.choice([-1, 1])
                t_meet = s_at / 0.3 * rng.uniform(0.6, 1.4)
                off = min(speed * t_meet, 3.0)
                mx, my = px - side * ty * off, py + side * tx * off
                vx, vy = side * ty * speed, -side * tx * speed
            else:
                sign = -1 if kind == "oncoming" else 1
                lat = rng.normal(0.0, 0.1)
                mx, my = px - ty * lat, py + tx * lat
                vx, vy = sign * tx * speed, sign * ty * speed
            hw, hh = self.bounds
            if abs(mx) > hw - r or abs(my) > hh - r:
                continue
            if clearance(mx, my, self.boxes, self.static_circles) < r + 0.02:
                continue
            if math.hypot(mx - start[0], my - start[1]) < 1.0:
                continue
            self.movers.append([mx, my, vx, vy, r, "free"])

    def _dyn_obs(self, raw):
        """トラッカー出力を模擬: 見えている (遮蔽されていない) 動的障害物のみ + 推定ノイズ + 見落とし."""
        tracks = []
        n = len(raw)
        for m in self.movers:
            dx, dy = m[0] - self.x, m[1] - self.y
            d = math.hypot(dx, dy)
            if d > 3.5 or self.rng.random() < 0.08:
                continue
            k = int(round(((math.atan2(dy, dx) - self.th) % (2 * math.pi)) / (2 * math.pi / n))) % n
            if np.isfinite(raw[k]) and raw[k] < d - m[4] - 0.15:
                continue  # 手前の物体に隠れている
            e = self.rng.normal(0, 1, 4) * [0.05, 0.05, 0.06, 0.06]
            tracks.append((m[0] + e[0], m[1] + e[1], m[2] + e[2], m[3] + e[3]))
        return dyn_features(tracks, (self.x, self.y, self.th))

    def _circles(self):
        mv = np.asarray([[m[0], m[1], m[4]] for m in self.movers]).reshape(-1, 3)
        return np.vstack([self.static_circles, mv])

    def _free_point(self, margin):
        hw, hh = self.bounds
        for _ in range(500):
            x = self.rng.uniform(-hw + margin, hw - margin)
            y = self.rng.uniform(-hh + margin, hh - margin)
            if clearance(x, y, self.boxes, self._circles()) > margin:
                return x, y
        return None

    # ------------------------------------------------------------ gym api
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        while True:
            self._random_world()
            s = self._free_point(W.ROBOT_RADIUS + 0.15)
            if s is None:
                continue
            if self.warehouse and self.rng.random() < 0.6:
                # 本番と同じ「ステーション間の配達区間」(始点は周辺にばらつかせる)
                names = [W.HOME] + W.STATIONS
                a, b = self.rng.choice(len(names), 2, replace=False)
                s = (names[a][1] + self.rng.normal(0, 0.1), names[a][2] + self.rng.normal(0, 0.1))
                g = (names[b][1], names[b][2])
                if clearance(*s, self.boxes, self._circles()) < W.ROBOT_RADIUS + 0.05:
                    continue
            else:
                for _ in range(50):
                    g = self._free_point(W.ROBOT_RADIUS + 0.10)
                    if g is not None and 0.8 < math.hypot(g[0] - s[0], g[1] - s[1]) < 8.0:
                        break
                else:
                    continue
            grid = GridMap(res=0.1, boxes=self.boxes, circles=self.static_circles, half_size=self.bounds)
            path = grid.astar(s, g)
            if path is None or len(path) < 2:
                continue
            self.tracker = PathTracker(path)
            self._add_oncoming(path, s)
            break
        self.x, self.y = s
        self.th = self.rng.uniform(-math.pi, math.pi)
        self.goal = np.array(g)
        self.v = 0.0
        self.w = 0.0
        self.tau = self.rng.uniform(0.05, 0.25)
        self.gain = self.rng.uniform(0.85, 1.1)
        self.noise = self.rng.uniform(0.0, 0.02)
        self.steps = 0
        self.max_steps = int(150 + self.tracker.length / 0.12 / DT * 0.4)
        self.prev_dist = remaining(self.tracker, self.x, self.y, self.goal)
        sec = self._scan()
        self.hist = ScanHistory()
        self.prev_sec = self.hist.push(sec)
        return self._obs(sec), {}

    def _goal_polar(self):
        dx, dy = self.goal[0] - self.x, self.goal[1] - self.y
        ang = math.atan2(dy, dx) - self.th
        return math.hypot(dx, dy), math.atan2(math.sin(ang), math.cos(ang))

    def _scan(self):
        self.raw = self._scan_raw()
        return sector_pool(self.raw)

    def _scan_raw(self):
        r = raycast(self.x, self.y, self.angles_rel + self.th, self.boxes, self._circles())
        r = r + self.rng.normal(0.0, self.noise, r.shape) * r
        drop = self.rng.random(r.shape) < 0.01
        r = np.where(drop, np.inf, r)
        return np.where(r >= W.LIDAR_MAX, np.inf, r)

    def _obs(self, sec):
        d, a, _ = tracker_goal(self.tracker, self.x, self.y, self.th, self.goal)
        return build_obs(sec, self.prev_sec, d, a, self.v, self.w, self._dyn_obs(self.raw))

    def step(self, action):
        v_cmd, w_cmd = action_to_cmd(action)
        w_prev = self.w
        h = DT / SUBSTEPS
        for _ in range(SUBSTEPS):
            k = min(1.0, h / self.tau)
            self.v += (v_cmd * self.gain - self.v) * k
            self.w += (w_cmd * self.gain - self.w) * k
            self.th += self.w * h
            self.x += self.v * math.cos(self.th) * h
            self.y += self.v * math.sin(self.th) * h
            for m in self.movers:
                m[0] += m[2] * h
                m[1] += m[3] * h
                if m[5] == "lane":
                    lane_bounce(m)
                else:
                    hw, hh = self.bounds
                    if clearance(m[0], m[1], self.boxes, self.static_circles) < m[4]:
                        m[0] -= m[2] * h  # 壁・棚に当たったら引き返す
                        m[1] -= m[3] * h
                        m[2], m[3] = -m[2], -m[3]
        self.th = math.atan2(math.sin(self.th), math.cos(self.th))
        self.steps += 1

        sec = self._scan()
        self.prev_sec = self.hist.push(sec)
        obs = self._obs(sec)  # tracker の進捗もここで更新される
        dist = math.hypot(self.goal[0] - self.x, self.goal[1] - self.y)
        rem = remaining(self.tracker, self.x, self.y, self.goal)
        clr = clearance(self.x, self.y, self.boxes, self._circles()) - W.ROBOT_RADIUS

        reward = 25.0 * (self.prev_dist - rem) - 0.02
        reward -= 0.03 * abs(self.w - w_prev)
        if self.v < 0:
            reward -= 0.02  # 後退は必要なときだけ
        if clr < 0.30:
            reward -= 1.0 * (0.30 - clr) / 0.30
        # 動く障害物 (人) には静的物体より大きな距離を取るよう促す
        for m in self.movers:
            dm = math.hypot(m[0] - self.x, m[1] - self.y) - m[4] - W.ROBOT_RADIUS
            if dm < 0.45:
                reward -= 1.2 * (0.45 - dm) / 0.45
        self.prev_dist = rem

        terminated = False
        info = {"is_success": False, "collision": False}
        if clr <= 0.0:
            reward -= 30.0
            terminated = True
            info["collision"] = True
        elif dist < GOAL_TOL:
            reward += 15.0
            terminated = True
            info["is_success"] = True
        truncated = self.steps >= self.max_steps
        return obs, float(reward), terminated, truncated, info
