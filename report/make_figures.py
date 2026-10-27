"""レポート・動画用の図を生成する (学習曲線, アブレーション, 走行軌跡)."""
import csv
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "ai_delivery"))
from ai_delivery import world_def as W  # noqa: E402
from ai_delivery.planner import GridMap  # noqa: E402

FIG = os.path.join(ROOT, "report", "fig")
for f in ["/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"]:
    if os.path.exists(f):
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
plt.rcParams["axes.unicode_minus"] = False


def learning_curve():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "training", "out", "progress.csv"))))
    s = np.array([float(r["steps"]) for r in rows]) / 1e6
    def sm(k, n=15):
        v = np.array([float(r[k]) for r in rows])
        return np.convolve(np.pad(v, (n // 2, n - 1 - n // 2), mode="edge"), np.ones(n) / n, mode="valid")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), dpi=130)
    ax[0].plot(s, sm("success") * 100, color="#2b8cbe", lw=2, label="成功率")
    ax[0].plot(s, sm("collision") * 100, color="#e34a33", lw=2, label="衝突率")
    ax[0].set_xlabel("学習ステップ数 [百万]")
    ax[0].set_ylabel("[%] (直近300エピソード)")
    ax[0].set_ylim(0, 100)
    ax[0].grid(alpha=0.3)
    ax[0].legend()
    ax[0].set_title("PPO 学習曲線 (ランダム環境込み)")
    ax[1].plot(s, sm("ep_return"), color="#31a354", lw=2)
    ax[1].set_xlabel("学習ステップ数 [百万]")
    ax[1].set_ylabel("平均エピソード報酬")
    ax[1].grid(alpha=0.3)
    ax[1].set_title("エピソード報酬")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "learning_curve.png"))
    plt.close(fig)


def ablation():
    p = os.path.join(ROOT, "training", "out", "ablation.json")
    if not os.path.exists(p):
        return
    d = json.load(open(p))
    names = {"pure_pursuit": "A*+純追従\n(従来型)", "ppo_only": "PPOのみ\n(A*なし)",
             "astar_ppo_lidaronly": "A*+PPO\n(LiDARのみ)", "astar_ppo_filter_lidaronly": "A*+PPO\n(LiDARのみ)\n+予測安全層",
             "astar_ppo": "A*+PPO\n(+人の追跡)", "astar_ppo_filter": "A*+PPO\n(+人の追跡)\n+予測安全層\n=本作品"}
    keys = [k for k in names if k in d]
    succ = [d[k]["success_rate"] * 100 for k in keys]
    cw = [d[k]["collision_worker"] * 100 for k in keys]
    cs = [d[k]["collision_static"] * 100 for k in keys]
    to = [d[k]["timeout"] * 100 for k in keys]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=130)
    ax.bar(x, succ, color="#2b8cbe", label="ミッション成功")
    ax.bar(x, cw, bottom=succ, color="#fdae61", label="作業員と衝突")
    ax.bar(x, cs, bottom=np.add(succ, cw), color="#d7191c", label="静的障害物と衝突")
    ax.bar(x, to, bottom=np.add(np.add(succ, cw), cs), color="#bababa", label="時間切れ(立ち往生)")
    for i, v in enumerate(succ):
        ax.text(i, v / 2, f"{v:.0f}%", ha="center", va="center", color="white", fontsize=13, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([names[k] for k in keys])
    ax.set_ylabel("割合 [%]")
    ax.set_ylim(0, 100)
    ax.set_title(f"配達ミッション全体 (HOME→A→B→C→HOME) の比較  各{d[keys[0]]['trials']}試行 / 2Dシミュレータ")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "ablation.png"))
    plt.close(fig)


def trajectories():
    runs = sorted(glob.glob(os.path.join(ROOT, "recordings", "eval_*"))) + [os.path.join(ROOT, "recordings", "final")]
    g = GridMap()
    fig, ax = plt.subplots(figsize=(9, 6.6), dpi=130)
    ax.imshow(g.occ.T, origin="lower", extent=[g.ox, g.ox + g.nx * g.res, g.oy, g.oy + g.ny * g.res], cmap="Greys", alpha=0.18)
    for cx, cy, sx, sy in W.static_obstacles()[0]:
        ax.add_patch(plt.Rectangle((cx - sx / 2, cy - sy / 2), sx, sy, color="#5b6b8c"))
    for cx, cy, r in W.static_obstacles()[1]:
        ax.add_patch(plt.Circle((cx, cy), r, color="#8a8a8a"))
    ax.plot([W.WORKER_P0[0], W.WORKER_P1[0]], [W.WORKER_P0[1], W.WORKER_P1[1]], color="orange",
            lw=2 * W.WORKER_R * 72 / 1.0 * 0.55, alpha=0.25, solid_capstyle="round", label="作業員の往復範囲")
    cmap = plt.get_cmap("tab10")
    for i, d in enumerate(runs):
        f = os.path.join(d, "status.jsonl")
        if not os.path.exists(f):
            continue
        st = [json.loads(l) for l in open(f) if l.strip()]
        xy = np.array([s["gt"][:2] for s in st if s.get("gt")])
        if len(xy):
            ax.plot(xy[:, 0], xy[:, 1], lw=1.6, color=cmap(i), alpha=0.85, label=os.path.basename(d))
    for name, x, y, rgb in [W.HOME] + W.STATIONS:
        ax.add_patch(plt.Rectangle((x - 0.3, y - 0.3), 0.6, 0.6, color=rgb, alpha=0.6))
        ax.text(x, y, name, ha="center", va="center", weight="bold", color="white")
    ax.set_xlim(-4.1, 4.1)
    ax.set_ylim(-3.1, 3.1)
    ax.set_aspect("equal")
    ax.set_title("Gazebo 上の走行軌跡 (真値) ─ 全試行の重ね描き")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "trajectories.png"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    learning_curve()
    ablation()
    trajectories()
    print("figures written to", FIG)
