"""LiDAR スキャンと既知地図の照合による自己位置推定 (尤度場スキャンマッチング).

Gazebo の DiffDrive オドメトリは旋回時の車輪の滑りで徐々にずれる
(実測: 45 秒の走行で約 0.5 m)。地図上で A* 計画しているので、このずれは致命的。

  予測: オドメトリの増分 (odom 座標系での相対移動) を推定姿勢に適用
  補正: スキャン点を world に投影し、地図の符号付き距離場 (SDF) が
        0 になるよう Gauss-Newton で (x, y, θ) を最適化
        → 動く作業員など地図に無い点は距離が大きいので Huber 重みで自動的に無視
"""
import math

import numpy as np

from .planner import GridMap

LIDAR_OFFSET = 0.04  # base_link から LiDAR までの前方オフセット [m]


class LikelihoodFieldLocalizer:
    def __init__(self, init_pose, grid=None, huber=0.06, max_corr=0.30):
        g = grid or GridMap()
        self.g = g
        self.df = np.clip(g.sdf, -0.5, 1.0).astype(np.float64)   # (nx, ny) 符号付き
        self.gx, self.gy = np.gradient(self.df, g.res)
        self.pose = np.array(init_pose, dtype=np.float64)
        self.last_odom = None
        self.huber = huber
        self.max_corr = max_corr
        self.last_inliers = 0

    # ---------------------------------------------------------- prediction
    def predict(self, odom_pose):
        o = np.asarray(odom_pose, dtype=np.float64)
        if self.last_odom is not None:
            dx, dy = o[0] - self.last_odom[0], o[1] - self.last_odom[1]
            c, s = math.cos(-self.last_odom[2]), math.sin(-self.last_odom[2])
            lx, ly = c * dx - s * dy, s * dx + c * dy          # ロボット座標での移動量
            dth = math.atan2(math.sin(o[2] - self.last_odom[2]), math.cos(o[2] - self.last_odom[2]))
            c2, s2 = math.cos(self.pose[2]), math.sin(self.pose[2])
            self.pose += [c2 * lx - s2 * ly, s2 * lx + c2 * ly, dth]
            self.pose[2] = math.atan2(math.sin(self.pose[2]), math.cos(self.pose[2]))
        self.last_odom = o

    # ---------------------------------------------------------- correction
    def _sample(self, X, Y):
        g = self.g
        fx = (X - g.ox) / g.res - 0.5
        fy = (Y - g.oy) / g.res - 0.5
        i0 = np.clip(np.floor(fx).astype(int), 0, g.nx - 2)
        j0 = np.clip(np.floor(fy).astype(int), 0, g.ny - 2)
        ax = np.clip(fx - i0, 0, 1)
        ay = np.clip(fy - j0, 0, 1)

        def bil(F):
            return ((1 - ax) * (1 - ay) * F[i0, j0] + ax * (1 - ay) * F[i0 + 1, j0]
                    + (1 - ax) * ay * F[i0, j0 + 1] + ax * ay * F[i0 + 1, j0 + 1])
        return bil(self.df), bil(self.gx), bil(self.gy)

    def correct(self, ranges, angle_min, angle_inc, range_max, iters=6):
        r = np.asarray(ranges, dtype=np.float64)
        a = angle_min + np.arange(len(r)) * angle_inc
        ok = np.isfinite(r) & (r < range_max - 0.05) & (r > 0.15)
        if ok.sum() < 30:
            return False
        r, a = r[ok], a[ok]
        px = LIDAR_OFFSET + r * np.cos(a)    # ロボット座標でのスキャン点
        py = r * np.sin(a)
        x, y, th = self.pose
        for _ in range(iters):
            c, s = math.cos(th), math.sin(th)
            X = x + c * px - s * py
            Y = y + s * px + c * py
            d, gx, gy = self._sample(X, Y)
            inl = np.abs(d) < self.max_corr
            if inl.sum() < 25:
                return False
            w = np.where(np.abs(d) < self.huber, 1.0, self.huber / np.maximum(np.abs(d), 1e-9)) * inl
            dXdth = -s * px - c * py
            dYdth = c * px - s * py
            J = np.stack([gx, gy, gx * dXdth + gy * dYdth], axis=1)
            H = (J * w[:, None]).T @ J + np.diag([1e-3, 1e-3, 1e-3])
            b = (J * w[:, None]).T @ d
            delta = -np.linalg.solve(H, b)
            delta = np.clip(delta, [-0.1, -0.1, -0.08], [0.1, 0.1, 0.08])
            x, y, th = x + delta[0], y + delta[1], th + delta[2]
            if np.abs(delta).max() < 1e-4:
                break
        self.last_inliers = int(inl.sum())
        jump = math.hypot(x - self.pose[0], y - self.pose[1])
        if jump > 0.4:  # 誤収束の疑い -> 採用しない
            return False
        self.pose = np.array([x, y, math.atan2(math.sin(th), math.cos(th))])
        return True
