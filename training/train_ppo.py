"""PPO で LiDAR ベースの局所ナビゲーション方策を学習し、numpy 形式で書き出す.

usage:
    python training/train_ppo.py --steps 3000000
出力:
    training/out/ppo_nav.zip            (SB3 チェックポイント)
    training/out/progress.csv           (学習曲線)
    ros2_ws/src/ai_delivery/models/policy.npz  (ROS 2 ノードが読む重み)
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "ai_delivery"))
from ai_delivery.sim2d import NavEnv  # noqa: E402

OUT = os.path.join(ROOT, "training", "out")
POLICY_NPZ = os.path.join(ROOT, "ros2_ws", "src", "ai_delivery", "models", "policy.npz")


def make_env(i):
    def _f():
        return NavEnv(seed=1000 + i)
    return _f


class StatsCallback(BaseCallback):
    """エピソード毎の成功率・衝突率を集計して CSV へ."""

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.buf = []
        self.rows = []
        self.t0 = time.time()

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done:
                self.buf.append((info.get("is_success", False), info.get("collision", False),
                                 info.get("episode", {}).get("r", 0.0)))
        return True

    def _on_rollout_end(self):
        if len(self.buf) >= 50:
            b = np.array(self.buf[-300:], dtype=np.float64)
            row = dict(steps=self.num_timesteps, success=b[:, 0].mean(), collision=b[:, 1].mean(),
                       ep_return=b[:, 2].mean(), wall=time.time() - self.t0)
            self.rows.append(row)
            print(f"[{row['steps']:>8}] success={row['success']:.3f} collision={row['collision']:.3f} "
                  f"R={row['ep_return']:.2f} ({row['wall']:.0f}s)", flush=True)
            with open(self.path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                w.writeheader()
                w.writerows(self.rows)
            self.buf = self.buf[-300:]


def export_numpy(model, path):
    pol = model.policy
    layers = [m for m in pol.mlp_extractor.policy_net if isinstance(m, torch.nn.Linear)]
    d = {}
    for i, l in enumerate(layers):
        d[f"W{i}"] = l.weight.detach().cpu().numpy().astype(np.float32)
        d[f"b{i}"] = l.bias.detach().cpu().numpy().astype(np.float32)
    d["Wout"] = pol.action_net.weight.detach().cpu().numpy().astype(np.float32)
    d["bout"] = pol.action_net.bias.detach().cpu().numpy().astype(np.float32)
    d["activation"] = np.array("tanh")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **d)
    print("exported", os.path.relpath(path, ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--init", default="", help="既存チェックポイント (.zip) から追加学習")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(2)

    env = VecMonitor(SubprocVecEnv([make_env(i) for i in range(args.envs)]))
    if args.init:
        model = PPO.load(args.init, env=env, device="cpu", learning_rate=args.lr)
    else:
        model = PPO(
            "MlpPolicy", env, n_steps=1024, batch_size=1024, n_epochs=10, gamma=0.99,
            gae_lambda=0.95, learning_rate=args.lr, ent_coef=0.003, clip_range=0.2,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), log_std_init=-0.5),
            verbose=0, seed=0, device="cpu",
        )
    cb = StatsCallback(os.path.join(OUT, f"progress{args.tag}.csv"))
    model.learn(total_timesteps=args.steps, callback=cb, reset_num_timesteps=not args.init)
    model.save(os.path.join(OUT, f"ppo_nav{args.tag}"))
    export_numpy(model, POLICY_NPZ)


if __name__ == "__main__":
    main()
