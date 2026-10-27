"""配達ミッション全体 (HOME→A→B→C→HOME) を 2D シミュレータで多数回実行し、手法を比較する.

比較する手法:
  1. A* + 純追従 (Pure Pursuit)   : 従来型。LiDAR を使わず経路を追うだけ
  2. PPO のみ (A* なし)             : 目的地を直接ゴールとして渡す
  3. A* + PPO                      : 本作品の階層型
  4. A* + PPO + 前方シールド        : 前方至近で前進を止める単純な安全層
  5. 動的再計画 A* + PPO             : LiDAR で追跡した作業員を A* のコストマップへ
  6. 本作品 (Gazebo で使用した構成)   : 5 + 予測セーフティフィルタ + 前方シールド

usage: python training/ablation.py [--trials 100]
"""
import argparse
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "ai_delivery"))
from ai_delivery import world_def as W  # noqa: E402
from ai_delivery.nav_core import ScanHistory, action_to_cmd, build_obs, dyn_features, tracker_goal  # noqa: E402
from ai_delivery.planner import DynamicReplanner, GridMap, PathTracker  # noqa: E402
from ai_delivery.nav_core import sector_pool  # noqa: E402
from ai_delivery.safety import DynamicTracker, SafetyFilter  # noqa: E402
from ai_delivery.policy import NumpyPolicy  # noqa: E402
from ai_delivery.sim2d import DT, SUBSTEPS, NavEnv, clearance, lane_bounce  # noqa: E402

POLICY = os.path.join(ROOT, "ros2_ws", "src", "ai_delivery", "models", "policy.npz")


def make_warehouse_env(rng):
    env = NavEnv(seed=int(rng.integers(1 << 30)), p_warehouse=1.0)
    env._random_world()
    # 作業員の初期位置・向き・速度を試行ごとにランダム化
    p0, u, L = W.worker_lane()
    sl = rng.uniform(0, L)
    sp = W.WORKER_SPEED * rng.choice([-1, 1])
    env.movers = [[p0[0] + u[0] * sl, p0[1] + u[1] * sl, u[0] * sp, u[1] * sp, W.WORKER_R, "lane"]]
    env.x, env.y, env.th = W.HOME[1], W.HOME[2], math.pi / 2
    env.v = env.w = 0.0
    env.tau, env.gain, env.noise = 0.12, 1.0, 0.01
    return env


def physics_step(env, v_cmd, w_cmd):
    h = DT / SUBSTEPS
    for _ in range(SUBSTEPS):
        k = min(1.0, h / env.tau)
        env.v += (v_cmd * env.gain - env.v) * k
        env.w += (w_cmd * env.gain - env.w) * k
        env.th += env.w * h
        env.x += env.v * math.cos(env.th) * h
        env.y += env.v * math.sin(env.th) * h
        for m in env.movers:
            m[0] += m[2] * h
            m[1] += m[3] * h
            lane_bounce(m)
    env.th = math.atan2(math.sin(env.th), math.cos(env.th))


def run_mission(method, policy, grid, rng, time_limit=60.0):
    env = make_warehouse_env(rng)
    hist = ScanHistory()
    raw = env._scan_raw()
    sec = sector_pool(raw)
    prev = hist.push(sec)
    dyn = DynamicTracker(grid)
    filt = SafetyFilter(grid)
    t_abs = 0.0
    n_int = 0
    t_total = 0.0
    shield_n = 0
    for name in W.MISSION:
        st = W.station(name)
        goal = np.array([st[1], st[2]])
        if method == "ppo_only":
            path = np.array([[env.x, env.y], goal])
        else:
            path = grid.astar((env.x, env.y), goal)
        tracker = PathTracker(path)
        replan = DynamicReplanner(grid)
        t = 0.0
        while math.hypot(goal[0] - env.x, goal[1] - env.y) > 0.22:
            pose = (env.x, env.y, env.th)
            if method != "pure_pursuit":
                # Gazebo と同じく LiDAR から検出・追跡 (真値は使わない)
                tracks = dyn.update(dyn.detect(raw, 0.0, 2 * math.pi / len(raw), pose), t_abs)
                if method in ("astar_ppo_dyn", "full"):
                    nt = replan.step(t_abs, pose, goal, tracks)
                    if nt is not None:
                        tracker = nt
            gd, ga, sub = tracker_goal(tracker, env.x, env.y, env.th, goal)
            if method == "pure_pursuit":
                v = W.V_MAX * max(0.0, math.cos(ga)) ** 2
                w = float(np.clip(2.5 * ga, -W.W_MAX, W.W_MAX))
            else:
                feats = dyn_features([(q["x"], q["y"], q["vx"], q["vy"]) for q in tracks], pose)
                v, w = action_to_cmd(policy(build_obs(sec, prev, gd, ga, env.v, env.w, feats)))
                if method in ("astar_ppo_filter", "full"):
                    (v, w), inter, _ = filt.filter(pose, (v, w), tracks)
                    n_int += inter
                if method in ("astar_ppo_shield", "astar_ppo_filter", "full"):
                    if min(sec[0], sec[1], sec[-1]) < W.ROBOT_RADIUS + 0.06 and v > 0.02:
                        v = 0.0
                        shield_n += 1
            physics_step(env, v, w)
            raw = env._scan_raw()
            sec = sector_pool(raw)
            prev = hist.push(sec)
            t += DT
            t_abs += DT
            if clearance(env.x, env.y, env.boxes, env._circles()) - W.ROBOT_RADIUS <= 0:
                static = clearance(env.x, env.y, env.boxes, env.static_circles) - W.ROBOT_RADIUS > 0
                return {"result": "collision_worker" if static else "collision_static", "time": t_total + t,
                        "leg": name}
            if t > time_limit:
                return {"result": "timeout", "time": t_total + t, "leg": name}
        t_total += t
    return {"result": "success", "time": t_total, "shield": shield_n, "interventions": n_int}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--only", default="")
    ap.add_argument("--policy", default=POLICY)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--suffix", default="", help="結果のキーに付ける接尾辞 (別の方策で評価した場合)")
    a = ap.parse_args()
    policy = NumpyPolicy(a.policy)
    grid = GridMap()
    out = {}
    methods = ["pure_pursuit", "ppo_only", "astar_ppo", "astar_ppo_shield", "astar_ppo_filter", "astar_ppo_dyn",
               "full"]
    if a.only:
        methods = a.only.split(",")
    for method in methods:
        rng = np.random.default_rng(123)  # 全手法で同じ作業員パターン
        rs = [run_mission(method, policy, grid, rng) for _ in range(a.trials)]
        succ = [r for r in rs if r["result"] == "success"]
        out[method + a.suffix] = {
            "trials": a.trials,
            "success_rate": len(succ) / a.trials,
            "collision_worker": sum(r["result"] == "collision_worker" for r in rs) / a.trials,
            "collision_static": sum(r["result"] == "collision_static" for r in rs) / a.trials,
            "timeout": sum(r["result"] == "timeout" for r in rs) / a.trials,
            "mean_time_success": float(np.mean([r["time"] for r in succ])) if succ else None,
            "mean_interventions": float(np.mean([r.get("interventions", 0) for r in succ])) if succ else None,
        }
        print(method + a.suffix, json.dumps(out[method + a.suffix]), flush=True)
    os.makedirs(os.path.join(ROOT, "training", "out"), exist_ok=True)
    if a.no_save:
        return
    p = os.path.join(ROOT, "training", "out", "ablation.json")
    old = json.load(open(p)) if (a.only and os.path.exists(p)) else {}
    old.update(out)
    json.dump(old, open(p, "w"), indent=1)


if __name__ == "__main__":
    main()
