"""Gazebo 上の反復試行ログ (recordings/eval_*/status.jsonl) を集計する.

衝突判定は真値姿勢 (/ground_truth/odom) と world_def の障害物形状から計算する。
usage: python tools/summarize_runs.py recordings/eval_* [--out recordings/eval_summary.json]
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
from ai_delivery.sim2d import clearance  # noqa: E402


def analyse(d):
    st = [json.loads(l) for l in open(os.path.join(d, "status.jsonl")) if l.strip()]
    boxes, circles = W.static_obstacles()
    min_static, min_worker = 9.0, 9.0
    for s in st:
        if not s.get("gt"):
            continue
        x, y = s["gt"][:2]
        min_static = min(min_static, clearance(x, y, np.array(boxes), np.array(circles)) - W.ROBOT_RADIUS)
        if s.get("worker"):
            min_worker = min(min_worker, math.hypot(s["worker"][0] - x, s["worker"][1] - y)
                             - W.ROBOT_RADIUS - W.WORKER_R)
    last = st[-1]
    legs = last.get("legs", [])
    t_mission = sum(l["time"] for l in legs)
    o = last.get("odom") or last["pose"]
    err = math.hypot(last["gt"][0] - o[0], last["gt"][1] - o[1]) if last.get("gt") else None
    le = [math.hypot(s["gt"][0] - s["pose"][0], s["gt"][1] - s["pose"][1]) for s in st if s.get("gt")]
    loc = float(np.mean(le)) * 100 if le else None
    loc_max = float(np.max(le)) * 100 if le else None
    return {"run": os.path.basename(d), "done": last["state"] == "DONE", "legs": len(legs),
            "time": t_mission, "dist": last["dist"], "min_clear_static_m": round(min_static, 3),
            "min_clear_worker_m": round(min_worker, 3), "collision": min_static <= 0 or min_worker <= 0,
            "odom_err_cm": round(err * 100, 2) if err is not None else None,
            "loc_err_mean_cm": round(loc, 2) if loc is not None else None,
            "loc_err_max_cm": round(loc_max, 2) if loc_max is not None else None,
            "shield": last.get("shield_count", 0), "filter": last.get("filter_count", 0),
            "steps": len(st)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=os.path.join(ROOT, "recordings", "eval_summary.json"))
    a = ap.parse_args()
    rows = [analyse(d) for d in a.runs if os.path.exists(os.path.join(d, "status.jsonl"))]
    ok = [r for r in rows if r["done"] and not r["collision"]]
    summary = {"n": len(rows), "success": len(ok), "collisions": sum(r["collision"] for r in rows),
               "mean_time": float(np.mean([r["time"] for r in ok])) if ok else 0.0,
               "mean_odom_err_cm": float(np.mean([r["odom_err_cm"] for r in rows if r["odom_err_cm"] is not None])),
               "mean_shield": float(np.mean([r["shield"] for r in rows])),
               "mean_filter": float(np.mean([r["filter"] for r in rows])),
               "mean_filter_ratio": float(np.mean([r["filter"] / max(r["steps"], 1) for r in rows])),
               "mean_loc_err_cm": float(np.mean([r["loc_err_mean_cm"] for r in rows])),
               "min_clear_worker_m": float(min(r["min_clear_worker_m"] for r in rows)),
               "min_clear_static_m": float(min(r["min_clear_static_m"] for r in rows)),
               "runs": rows}
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    json.dump(summary, open(a.out, "w"), indent=1, ensure_ascii=False)
