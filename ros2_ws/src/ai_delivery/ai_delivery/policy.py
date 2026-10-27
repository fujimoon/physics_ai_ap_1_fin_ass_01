"""学習済み PPO 方策を numpy だけで推論する軽量ランタイム.

ロボット側 (ROS 2 ノード) に PyTorch を入れずに済むよう、
学習後に重みを .npz へ書き出し、ここで MLP の順伝播を再実装している。
"""
import numpy as np


class NumpyPolicy:
    def __init__(self, path):
        d = np.load(path)
        self.layers = []
        i = 0
        while f"W{i}" in d:
            self.layers.append((d[f"W{i}"], d[f"b{i}"]))
            i += 1
        self.Wout, self.bout = d["Wout"], d["bout"]
        self.act = str(d["activation"]) if "activation" in d else "tanh"

    def __call__(self, obs):
        h = np.asarray(obs, dtype=np.float32)
        h = h[: self.layers[0][0].shape[1]]  # 旧版 (LiDAR のみ 53 次元) の方策も評価できるように
        for Wm, b in self.layers:
            h = h @ Wm.T + b
            h = np.tanh(h) if self.act == "tanh" else np.maximum(h, 0)
        return np.clip(h @ self.Wout.T + self.bout, -1.0, 1.0)
