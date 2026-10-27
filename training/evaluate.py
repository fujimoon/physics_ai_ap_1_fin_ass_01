"""学習済み方策 (numpy 版) を 2D シミュレータで評価する.

usage: python training/evaluate.py [--episodes 500] [--policy path]
"""
import argparse
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "ai_delivery"))
from ai_delivery.policy import NumpyPolicy  # noqa: E402
from ai_delivery.sim2d import NavEnv, clearance  # noqa: E402
from ai_delivery import world_def as W  # noqa: E402


def run(policy, p_wh, n, seed):
    env = NavEnv(seed=seed, p_warehouse=p_wh)
    res = {"success": 0, "collision": 0, "timeout": 0, "col_mover": 0, "steps": []}
    for _ in range(n):
        o, _ = env.reset()
        while True:
            o, r, te, tr, info = env.step(policy(o))
            if te or tr:
                break
        if info["is_success"]:
            res["success"] += 1
            res["steps"].append(env.steps)
        elif info["collision"]:
            res["collision"] += 1
            st = clearance(env.x, env.y, env.boxes, env.static_circles) - W.ROBOT_RADIUS
            if st > 0:
                res["col_mover"] += 1
        else:
            res["timeout"] += 1
    res["mean_steps"] = float(np.mean(res["steps"])) if res["steps"] else None
    del res["steps"]
    for k in ("success", "collision", "timeout"):
        res[k + "_rate"] = res[k] / n
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--policy", default=os.path.join(ROOT, "ros2_ws/src/ai_delivery/models/policy.npz"))
    ap.add_argument("--out", default=os.path.join(ROOT, "training/out/eval.json"))
    a = ap.parse_args()
    pol = NumpyPolicy(a.policy)
    out = {"warehouse": run(pol, 1.0, a.episodes, 7), "random": run(pol, 0.0, a.episodes, 8)}
    print(json.dumps(out, indent=1))
    json.dump(out, open(a.out, "w"), indent=1)
