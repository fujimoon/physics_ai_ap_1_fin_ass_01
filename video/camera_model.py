"""Gazebo 俯瞰カメラ (overview_cam) のピンホールモデル: world 座標 -> 画素."""
import math

import numpy as np

CAM_POS = np.array([0.0, -4.8, 8.0])
CAM_RPY = (0.0, 1.08, math.pi / 2)
HFOV = 1.15
IMG_W, IMG_H = 1024, 640


def _rot(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


R = _rot(*CAM_RPY)
F = (IMG_W / 2) / math.tan(HFOV / 2)


def project(pts, z=0.0, scale=1.0):
    """pts: (N,2) or (N,3) world -> (N,2) pixel (scale: 出力画像の拡大率)."""
    p = np.asarray(pts, dtype=np.float64)
    if p.ndim == 1:
        p = p[None]
    if p.shape[1] == 2:
        p = np.hstack([p, np.full((len(p), 1), z)])
    d = (p - CAM_POS) @ R  # = R^T (p - C)
    u = IMG_W / 2 - F * d[:, 1] / d[:, 0]
    v = IMG_H / 2 - F * d[:, 2] / d[:, 0]
    return np.stack([u, v], axis=1) * scale
