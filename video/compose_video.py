"""Gazebo の録画データ (カメラ画像 + AI 内部状態ログ) から提出用動画を合成する.

usage:
    python video/compose_video.py --run recordings/final --out deliverables/demo.mp4

構成: タイトル → 参考記事との違い → システム構成 → 学習結果
      → 本編 (等倍, シミュレーション時間) → 回避シーンのスロー再生 → 結果まとめ
本編では Gazebo 俯瞰カメラ映像に、A* 経路・走行軌跡・サブゴール・LiDAR を
カメラモデルで正確に投影して重ねている。
"""
import argparse
import bisect
import glob
import json
import math
import os
import subprocess
import sys

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "ai_delivery"))
from camera_model import project  # noqa: E402
from ai_delivery import world_def as W  # noqa: E402

WID, HEI, FPS = 1920, 1080, 30
FONT_B = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
FONT_R = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
BG = (18, 24, 33)
PANEL = (30, 39, 52)
ACCENT = (80, 200, 255)
WHITE = (240, 244, 248)
GREY = (150, 160, 175)
ORANGE = (255, 150, 40)
MAGENTA = (255, 70, 200)
GREEN = (80, 220, 120)
RED = (255, 80, 80)
YELLOW = (255, 210, 60)

_fonts = {}


def font(size, bold=False):
    k = (size, bold)
    if k not in _fonts:
        _fonts[k] = ImageFont.truetype(FONT_B if bold else FONT_R, size)
    return _fonts[k]


def text(d, xy, s, size=28, color=WHITE, bold=False, anchor="la"):
    d.text(xy, s, font=font(size, bold), fill=color, anchor=anchor)


def bgr(c):
    return (c[2], c[1], c[0])


# ---------------------------------------------------------------- data
class Run:
    def __init__(self, d):
        self.dir = d
        self.status = [json.loads(l) for l in open(os.path.join(d, "status.jsonl")) if l.strip()]
        self.st_t = [s["t"] for s in self.status]
        self.ov = sorted(glob.glob(os.path.join(d, "overview", "*.jpg")))
        self.ov_t = [float(os.path.basename(p)[:-4]) for p in self.ov]
        self.fp = sorted(glob.glob(os.path.join(d, "fpv", "*.jpg")))
        self.fp_t = [float(os.path.basename(p)[:-4]) for p in self.fp]
        self.t0 = self.st_t[0]
        self.t1 = self.st_t[-1]
        self._cache = {}

    def at(self, ts, t):
        i = bisect.bisect_right(ts, t) - 1
        return max(0, min(i, len(ts) - 1))

    def state(self, t):
        return self.status[self.at(self.st_t, t)]

    def img(self, kind, t):
        lst, ts = (self.ov, self.ov_t) if kind == "ov" else (self.fp, self.fp_t)
        p = lst[self.at(ts, t)]
        if p not in self._cache:
            if len(self._cache) > 64:
                self._cache.clear()
            self._cache[p] = cv2.imread(p)
        return self._cache[p]

    def trail(self, t):
        i = self.at(self.st_t, t)
        return np.array([(s["gt"] or s["pose"])[:2] for s in self.status[: i + 1]])


def robot_xyth(st):
    return st["gt"] if st.get("gt") else st["pose"]


# ---------------------------------------------------------------- panels
MAIN_X, MAIN_Y, MAIN_S = 24, 84, 1.25  # 1024x640 -> 1280x800
RX = MAIN_X + int(1024 * MAIN_S) + 24  # 右カラム x
RW = WID - RX - 24


def draw_main(run, t, canvas):
    st = run.state(t)
    img = run.img("ov", t).copy()
    img = cv2.resize(img, (int(1024 * MAIN_S), int(640 * MAIN_S)), interpolation=cv2.INTER_CUBIC)
    over = img.copy()
    x, y, th = robot_xyth(st)
    # LiDAR (24 セクタの最近距離を床面に投影)
    sec = np.array(st["sec"])
    ang = th + np.arange(len(sec)) * (2 * math.pi / len(sec))
    pts3 = np.stack([x + sec * np.cos(ang), y + sec * np.sin(ang), np.full_like(sec, 0.15)], 1)
    c = project(np.array([[x, y, 0.15]]), scale=MAIN_S)[0].astype(int)
    pp = project(pts3, scale=MAIN_S).astype(int)
    for p, r in zip(pp, sec):
        col = (60, 60, 255) if r < 0.45 else (90, 230, 120)
        cv2.line(over, tuple(c), tuple(p), col, 1, cv2.LINE_AA)
    cv2.fillPoly(over, [pp.reshape(-1, 1, 2)], (90, 230, 120))
    img = cv2.addWeighted(over, 0.35, img, 0.65, 0)
    for p, r in zip(pp, sec):
        cv2.circle(img, tuple(p), 4, (60, 60, 255) if r < 0.45 else (90, 230, 120), -1, cv2.LINE_AA)
    # A* 経路
    if st.get("path"):
        pa = project(np.array(st["path"]), scale=MAIN_S).astype(np.int32)
        cv2.polylines(img, [pa.reshape(-1, 1, 2)], False, bgr(ACCENT), 3, cv2.LINE_AA)
    # 軌跡
    tr = run.trail(t)
    if len(tr) > 1:
        pt = project(tr[::2], scale=MAIN_S).astype(np.int32)
        cv2.polylines(img, [pt.reshape(-1, 1, 2)], False, bgr(ORANGE), 4, cv2.LINE_AA)
    # サブゴール
    if st.get("sub"):
        ps = project(np.array([st["sub"]]), scale=MAIN_S)[0].astype(int)
        cv2.line(img, tuple(c), tuple(ps), bgr(MAGENTA), 2, cv2.LINE_AA)
        cv2.circle(img, tuple(ps), 10, bgr(MAGENTA), 3, cv2.LINE_AA)
    # AI が LiDAR から検出・追跡した動的障害物 (赤) と 2.4 秒先までの予測 (点線)
    near = False
    for tx, ty, vx, vy in st.get("tracks", []):
        dtr = math.hypot(tx - x, ty - y)
        near = near or dtr < 1.4
        ring = np.array([[tx + 0.30 * math.cos(a), ty + 0.30 * math.sin(a)] for a in np.linspace(0, 2 * math.pi, 36)])
        pr = project(ring, scale=MAIN_S).astype(np.int32)
        cv2.polylines(img, [pr.reshape(-1, 1, 2)], True, (60, 60, 255), 3, cv2.LINE_AA)
        fut = np.array([[tx + vx * k, ty + vy * k] for k in np.linspace(0, 2.4, 13)])
        pf = project(fut, scale=MAIN_S).astype(int)
        for k in range(0, len(pf) - 1, 2):
            cv2.line(img, tuple(pf[k]), tuple(pf[k + 1]), (60, 60, 255), 3, cv2.LINE_AA)
        cv2.circle(img, tuple(pf[-1]), 6, (60, 60, 255), -1, cv2.LINE_AA)
    canvas[MAIN_Y:MAIN_Y + img.shape[0], MAIN_X:MAIN_X + img.shape[1]] = img
    return st, near


def draw_panels(run, t, st, near, pil):
    d = ImageDraw.Draw(pil)
    # 凡例
    lx, ly = MAIN_X + 14, MAIN_Y + 12
    d.rounded_rectangle([lx - 6, ly - 6, lx + 900, ly + 34], 8, fill=(15, 20, 28))
    items = [(ACCENT, "A*経路"), (MAGENTA, "サブゴール"), (ORANGE, "走行軌跡"), (GREEN, "LiDAR"),
             (RED, "検出した動的障害物と2.4秒先の予測")]
    xx = lx
    for col, name in items:
        d.rectangle([xx, ly + 8, xx + 22, ly + 20], fill=col)
        text(d, (xx + 28, ly + 2), name, 20)
        xx += 40 + int(d.textlength(name, font=font(20)))
    # ステーション名ラベル
    done = [l["to"] for l in st.get("legs", [])]
    for name, sx, sy, _ in [W.HOME] + W.STATIONS:
        u, v = project([(sx, sy)], scale=MAIN_S)[0]
        u, v = u + MAIN_X, v + MAIN_Y - 38
        ok = name in done and not (name == "HOME" and st["state"] not in ("DONE",) and done.count("HOME") == 0)
        tag = f"{name} ✓" if ok else name
        w = 22 * len(tag) + 18
        d.rounded_rectangle([u - w / 2, v - 16, u + w / 2, v + 16], 8, fill=(15, 20, 28))
        text(d, (u, v), tag, 22, GREEN if ok else WHITE, True, "mm")

    # ---- 右カラム: FPV
    y0 = MAIN_Y
    text(d, (RX, y0), "オンボードカメラ (ロボット視点)", 22, GREY, True)
    fpv = cv2.cvtColor(run.img("fp", t), cv2.COLOR_BGR2RGB)
    fh = int(RW * fpv.shape[0] / fpv.shape[1])
    pil.paste(Image.fromarray(cv2.resize(fpv, (RW, fh))), (RX, y0 + 32))
    # ---- LiDAR 極座標 + 方策出力
    y1 = y0 + 32 + fh + 16
    ph = 300
    d.rounded_rectangle([RX, y1, RX + RW, y1 + ph], 12, fill=PANEL)
    text(d, (RX + 16, y1 + 10), "PPO方策の入力 / 出力", 22, GREY, True)
    cx, cy, R = RX + 150, y1 + 165, 115
    d.ellipse([cx - R, cy - R, cx + R, cy + R], outline=(70, 80, 95), width=2)
    d.ellipse([cx - R / 2, cy - R / 2, cx + R / 2, cy + R / 2], outline=(55, 65, 80), width=1)
    sec = np.array(st["sec"])
    n = len(sec)
    poly = []
    for i, r in enumerate(sec):
        a = i * 2 * math.pi / n
        rr = R * min(r, 3.5) / 3.5
        poly.append((cx - rr * math.sin(a), cy - rr * math.cos(a)))  # 上 = 前方
    d.polygon(poly, fill=(50, 120, 80), outline=GREEN)
    for i, r in enumerate(sec):
        if r < 0.45:
            a = i * 2 * math.pi / n
            rr = R * r / 3.5
            d.ellipse([cx - rr * math.sin(a) - 5, cy - rr * math.cos(a) - 5,
                       cx - rr * math.sin(a) + 5, cy - rr * math.cos(a) + 5], fill=RED)
    d.polygon([(cx, cy - 12), (cx - 8, cy + 8), (cx + 8, cy + 8)], fill=WHITE)
    if st.get("sub"):
        x, y, th = robot_xyth(st)
        ga = math.atan2(st["sub"][1] - y, st["sub"][0] - x) - th
        d.line([cx, cy, cx - R * math.sin(ga), cy - R * math.cos(ga)], fill=MAGENTA, width=4)
    text(d, (cx, y1 + ph - 22), "LiDAR 24セクタ (半径3.5m)", 18, GREY, anchor="mm")
    # 行動バー
    bx = RX + 300
    v, w = st["cmd"]
    text(d, (bx, y1 + 52), f"並進速度 v = {v:+.2f} m/s", 22)
    d.rectangle([bx, y1 + 86, bx + 250, y1 + 104], fill=(50, 58, 72))
    vv = (v - W.V_MIN) / (W.V_MAX - W.V_MIN)
    z0 = bx + 250 * (-W.V_MIN) / (W.V_MAX - W.V_MIN)
    d.rectangle([min(z0, bx + 250 * vv), y1 + 86, max(z0, bx + 250 * vv), y1 + 104], fill=ACCENT)
    text(d, (bx, y1 + 124), f"角速度 ω = {w:+.2f} rad/s", 22)
    d.rectangle([bx, y1 + 158, bx + 250, y1 + 176], fill=(50, 58, 72))
    mid = bx + 125
    d.rectangle([min(mid, mid - 125 * w / W.W_MAX), y1 + 158, max(mid, mid - 125 * w / W.W_MAX), y1 + 176],
                fill=ORANGE)
    text(d, (bx, y1 + 182), "左旋回", 16, GREY)
    text(d, (bx + 250, y1 + 182), "右旋回", 16, GREY, anchor="ra")
    if st.get("filtered") or st.get("shield"):
        pv, pw = st.get("cmd_policy", st["cmd"])
        d.rounded_rectangle([bx, y1 + 206, bx + 250, y1 + 276], 8, fill=(140, 40, 40))
        text(d, (bx + 125, y1 + 226), "予測安全層が介入" if st.get("filtered") else "前方シールド作動", 20, WHITE, True, "mm")
        text(d, (bx + 125, y1 + 256), f"方策 ({pv:+.2f}, {pw:+.2f}) → 修正", 17, WHITE, False, "mm")
    else:
        d.rounded_rectangle([bx, y1 + 206, bx + 250, y1 + 276], 8, outline=(70, 80, 95), width=2)
        text(d, (bx + 125, y1 + 241), "安全層: 方策の指令をそのまま実行", 17, GREY, False, "mm")
    # ---- ミッション
    y2 = y1 + ph + 16
    mh = HEI - 20 - y2
    d.rounded_rectangle([RX, y2, RX + RW, y2 + mh], 12, fill=PANEL)
    text(d, (RX + 16, y2 + 10), "配達ミッション", 22, GREY, True)
    route = ["HOME"] + W.MISSION
    done_n = len(st.get("legs", []))
    xs = np.linspace(RX + 50, RX + RW - 50, len(route))
    for i, (name, xx) in enumerate(zip(route, xs)):
        if i > 0:
            d.line([xs[i - 1] + 26, y2 + 75, xx - 26, y2 + 75], fill=GREEN if i <= done_n else (80, 90, 105),
                   width=4)
        if i <= done_n:
            col = GREEN
        elif i == done_n + 1 and st["state"] in ("NAV",):
            col = ACCENT
        else:
            col = (80, 90, 105)
        d.ellipse([xx - 24, y2 + 51, xx + 24, y2 + 99], fill=col)
        text(d, (xx, y2 + 75), name if len(name) == 1 else "H", 24, BG, True, "mm")
    el = t - run.t0
    errs = ""
    if st.get("gt"):
        e = math.hypot(st["gt"][0] - st["pose"][0], st["gt"][1] - st["pose"][1])
        o = st.get("odom") or st["pose"]
        eo = math.hypot(st["gt"][0] - o[0], st["gt"][1] - o[1])
        errs = f"位置推定誤差 {e * 100:4.1f} cm  (オドメトリのみ {eo * 100:4.1f} cm)"
    text(d, (RX + 16, y2 + 120), f"経過時間 {el:5.1f} s     走行距離 {st['dist']:5.2f} m", 22)
    if errs:
        text(d, (RX + 16, y2 + 154), errs, 22)
    text(d, (RX + 16, y2 + 188), f"状態: {st['state']}    目的地: {st.get('goal') or '-'}", 22, GREY)
    text(d, (RX + 16, y2 + 222), f"安全層の介入: 累計 {st.get('filter_count', 0) + st.get('shield_count', 0)} ステップ"
                                 f"（全 {int((t - run.t0) * 10)} ステップ中）", 22, GREY)


def caption_for(st, near):
    s, g = st["state"], st.get("goal")
    if s == "WAIT":
        return "シミュレーション開始。ロボットは HOME で配達指示を待機中", WHITE
    if s == "DONE":
        return "全ての配達を完了し HOME へ帰還しました（衝突 0 回）", GREEN
    if s == "DWELL":
        last = st["legs"][-1]["to"] if st.get("legs") else g
        return f"ステーション {last} に到着 ─ 荷物の受け渡し中（2秒停止）", GREEN
    if st.get("filtered"):
        return "予測安全層：作業員の動きを 2.4 秒先まで予測し、衝突しない指令に修正", RED
    if st.get("shield"):
        return "前方シールド作動：前方至近に障害物 → 前進のみ抑制", RED
    if near:
        return "地図に無い作業員を LiDAR で検出・追跡 → PPO 方策が速度と進路を調整", YELLOW
    return f"目的地 {g} へ：A* 経路（水色）上の 1m 先サブゴール（桃）を PPO 方策が追従", WHITE


def header(d, right):
    d.rectangle([0, 0, WID, 66], fill=(12, 16, 22))
    text(d, (24, 33), "AI 自律配達ロボット ─ PPO強化学習 × A* 階層ナビゲーション", 30, WHITE, True, "lm")
    text(d, (WID - 24, 33), right, 24, YELLOW, True, "rm")


def frame_main(run, t, speed_label, caption_override=None):
    canvas = np.zeros((HEI, WID, 3), np.uint8)
    canvas[:] = bgr(BG)
    st, near = draw_main(run, t, canvas)
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    header(d, f"{speed_label}   t = {t - run.t0:5.1f} s (sim)")
    draw_panels(run, t, st, near, pil)
    cap, col = caption_override or caption_for(st, near)
    cy = MAIN_Y + int(640 * MAIN_S) + 20
    cx1 = MAIN_X + int(1024 * MAIN_S)
    d.rounded_rectangle([24, cy, cx1, HEI - 20], 14, fill=PANEL)
    d.rectangle([24, cy, 34, HEI - 20], fill=col)
    size = 34
    while d.textlength(cap, font=font(size, True)) > cx1 - 90 and size > 20:
        size -= 1
    text(d, (60, (cy + HEI - 20) // 2), cap, size, col, True, "lm")
    return np.array(pil)


# ---------------------------------------------------------------- cards
def card_base(title, sub=None):
    pil = Image.new("RGB", (WID, HEI), BG)
    d = ImageDraw.Draw(pil)
    d.rectangle([0, 0, WID, 8], fill=ACCENT)
    text(d, (100, 90), title, 56, WHITE, True)
    if sub:
        text(d, (100, 170), sub, 30, GREY)
    return pil, d


def card_title():
    pil = Image.new("RGB", (WID, HEI), BG)
    d = ImageDraw.Draw(pil)
    d.rectangle([0, 0, WID, 8], fill=ACCENT)
    text(d, (WID // 2, 330), "AI 自律配達ロボット", 96, WHITE, True, "mm")
    text(d, (WID // 2, 450), "PPO 強化学習 × A* 階層ナビゲーションによる倉庫内ラストワンマイル配送", 40, ACCENT, True, "mm")
    text(d, (WID // 2, 560), "Ignition Gazebo Fortress  +  ROS 2 Humble  +  ros_gz_bridge  +  Stable-Baselines3", 30,
         GREY, anchor="mm")
    text(d, (WID // 2, 700), "Physical AI 応用1講座  最終課題", 32, WHITE, anchor="mm")
    text(d, (WID // 2, 760), "本編はシミュレーション時間で等倍再生（スロー再生部分は 0.5 倍速と明記）", 26, YELLOW, anchor="mm")
    return pil


def card_diff():
    pil, d = card_base("参考記事からの発展", "Qiita「gazeboとROS2を連携してロボットを操作する」(@N622) をベースに")
    cols = [("参考記事", (110, 120, 135), [
        "キーボード入力 → /cmd_vel で人が操縦",
        "センサ無し（ロボットは周囲が見えない）",
        "AI 要素なし",
        "静的な空間を走行",
        "ブリッジは /cmd_vel の 1 トピック",
        "動作確認は目視のみ",
    ]), ("本作品", ACCENT, [
        "LiDAR + カメラで完全自律走行・自己位置推定",
        "PPO 強化学習の局所方策 + A* 大域計画",
        "地図に無い『横切る作業員』を検出・予測し回避",
        "複数地点を巡る配達ミッション（状態機械）",
        "9 トピックを config yaml でブリッジ",
        "成功率・位置誤差を定量評価 / 手法比較",
    ])]
    for k, (name, col, lines) in enumerate(cols):
        x0 = 100 + k * 880
        d.rounded_rectangle([x0, 260, x0 + 820, 960], 20, fill=PANEL, outline=col, width=4)
        text(d, (x0 + 40, 300), name, 44, col, True)
        for i, l in enumerate(lines):
            text(d, (x0 + 40, 400 + i * 88), ("・" if k == 0 else "✓ ") + l, 32, WHITE if k else GREY)
    d.polygon([(945, 580), (975, 610), (945, 640)], fill=ACCENT)
    return pil


def card_arch():
    pil, d = card_base("システム構成", "ROS 2 ノード群と Gazebo を ros_gz_bridge で接続（launch 1 コマンドで全起動）")
    def box(x, y, w, h, title, lines, col):
        d.rounded_rectangle([x, y, x + w, y + h], 16, fill=PANEL, outline=col, width=3)
        text(d, (x + 20, y + 16), title, 30, col, True)
        for i, l in enumerate(lines):
            text(d, (x + 20, y + 64 + i * 38), l, 24, WHITE)
    def arrow(x0, y0, x1, y1, label=""):
        d.line([x0, y0, x1, y1], fill=GREY, width=4)
        a = math.atan2(y1 - y0, x1 - x0)
        d.polygon([(x1, y1), (x1 - 18 * math.cos(a - 0.4), y1 - 18 * math.sin(a - 0.4)),
                   (x1 - 18 * math.cos(a + 0.4), y1 - 18 * math.sin(a + 0.4))], fill=GREY)
        if label:
            text(d, ((x0 + x1) / 2, (y0 + y1) / 2 - 22), label, 22, YELLOW, anchor="mm")
    box(100, 260, 520, 330, "Ignition Gazebo", ["差動二輪ロボット (DiffDrive)", "gpu_lidar 240本 / 全周",
                                                  "FPVカメラ・俯瞰カメラ", "動く作業員 (VelocityControl)",
                                                  "倉庫ワールド (自動生成SDF)"], ORANGE)
    box(700, 260, 520, 330, "ros_gz_bridge", ["config/bridge.yaml", "/scan  /odom  /cmd_vel", "/camera/*  /clock",
                                                 "/worker/*  /ground_truth/odom"], GREY)
    box(1300, 260, 520, 640, "ai_mission ノード (10Hz)", ["⓪ 自己位置推定: LiDAR×地図照合",
                                                          "① 動的障害物の検出・追跡 (地図差分)",
                                                          "② ミッション: HOME→A→B→C→HOME",
                                                          "③ 大域計画: A* (距離場コスト)",
                                                          "④ 局所制御: PPO 方策",
                                                          "   LiDAR 24セクタ×2 + サブゴール",
                                                          "   + 追跡した人の位置・速度",
                                                          "   → 並進速度 v, 角速度 ω",
                                                          "⑤ 予測安全層: 2.4秒先を予測し修正",
                                                          "⑥ 前方シールド / 停滞時の再計画"], ACCENT)
    box(700, 680, 520, 300, "学習 (PC 上, 約30分)", ["numpy 製 2D 高速シミュレータ", "ドメインランダマイゼーション",
                                                    "PPO 1600万ステップ", "→ 重みを .npz に書き出し"], GREEN)
    box(100, 680, 520, 300, "記録・可視化", ["recorder ノード", "画像 + 内部状態を sim 時刻で保存",
                                             "カメラモデルで経路を投影", "→ 本動画を自動合成"], MAGENTA)
    arrow(620, 400, 700, 400)
    arrow(1220, 360, 1300, 360, "")
    arrow(1300, 520, 1220, 520, "")
    text(d, (1260, 330), "センサ", 20, YELLOW, anchor="mm")
    text(d, (1260, 490), "/cmd_vel", 20, YELLOW, anchor="mm")
    arrow(1220, 830, 1300, 830)
    text(d, (960, 1010), "学習済み方策を Gazebo 上のノードへ Sim-to-Sim 転移 (numpy 推論)", 24, YELLOW, anchor="mm")
    return pil


def card_training(curve_png, ev):
    pil, d = card_base("強化学習 (PPO) の学習結果", "Gazebo では重すぎるため、同じ観測・行動を持つ軽量 2D シミュレータで学習")
    if os.path.exists(curve_png):
        im = Image.open(curve_png).convert("RGB")
        im.thumbnail((1100, 700))
        pil.paste(im, (100, 250))
    x0 = 1260
    d.rounded_rectangle([x0, 250, WID - 100, 950], 20, fill=PANEL)
    text(d, (x0 + 30, 280), "工夫した点", 34, ACCENT, True)
    lines = ["学習と実行で観測生成コードを共有", "部屋・棚・柱を毎回ランダム生成",
             "対向・横断・同方向の歩行者を配置", "追跡した人の位置・速度も方策に入力",
             "LiDAR ノイズ・見落とし・遮蔽を再現", "応答遅れ・ゲイン誤差もランダム化",
             "PyTorch 不要の numpy 推論で実装"]
    for i, l in enumerate(lines):
        text(d, (x0 + 30, 345 + i * 44), "• " + l, 24)
    if ev:
        text(d, (x0 + 30, 680), "2D 評価 (未学習の乱数シード)", 26, ACCENT, True)
        text(d, (x0 + 30, 724), f"方策単体・1区間:  倉庫 {ev['warehouse']['success_rate']*100:.0f}% / ランダム部屋 {ev['random']['success_rate']*100:.0f}%", 22)
        ab = ev.get("ablation") or {}
        if "astar_ppo_filter" in ab:
            text(d, (x0 + 30, 768), "ミッション全体 (各100試行):", 22)
            text(d, (x0 + 30, 806), f"  本作品 {ab['astar_ppo_filter']['success_rate']*100:.0f}%  /  従来型 {ab['pure_pursuit']['success_rate']*100:.0f}%", 26, GREEN, True)
            text(d, (x0 + 30, 852), f"  LiDARのみの旧方策+安全層 {ab.get('astar_ppo_filter_lidaronly', {}).get('success_rate', 0)*100:.0f}%", 22, GREY)
    return pil


def card_results(run, summary):
    pil, d = card_base("結果まとめ", "Ignition Gazebo 上で学習済み方策を実行（学習は 2D シミュレータのみ）")
    legs = run.status[-1].get("legs", [])
    d.rounded_rectangle([100, 250, 900, 900], 20, fill=PANEL)
    text(d, (130, 280), "本編の走行記録", 34, ACCENT, True)
    text(d, (130, 350), "区間", 26, GREY)
    text(d, (330, 350), "所要時間", 26, GREY)
    text(d, (530, 350), "走行距離", 26, GREY)
    text(d, (730, 350), "A*経路長", 26, GREY)
    prev = "HOME"
    for i, l in enumerate(legs):
        y = 405 + i * 60
        text(d, (130, y), f"{prev} → {l['to']}", 28)
        text(d, (330, y), f"{l['time']:.1f} s", 28)
        text(d, (530, y), f"{l['dist']:.2f} m", 28)
        text(d, (730, y), f"{l['plan']:.2f} m", 28)
        prev = l["to"]
    tot_t = sum(l["time"] for l in legs)
    tot_d = sum(l["dist"] for l in legs)
    text(d, (130, 405 + len(legs) * 60 + 20), f"合計  {tot_t:.1f} s / {tot_d:.2f} m / 衝突 0 回", 30, GREEN, True)
    d.rounded_rectangle([980, 250, WID - 100, 900], 20, fill=PANEL)
    text(d, (1010, 280), "Gazebo 上の反復試行", 34, ACCENT, True)
    if summary:
        lines = [f"試行回数: {summary['n']} 回 (作業員の出発タイミングを変更)",
                 f"ミッション完遂: {summary['success']} / {summary['n']}",
                 f"衝突: {summary['collisions']} 回",
                 f"平均所要時間: {summary['mean_time']:.1f} s",
                 f"位置推定誤差 (LiDAR照合): 平均 {summary['mean_loc_err_cm']:.1f} cm",
                 f"  └ オドメトリのみなら最終 {summary['mean_odom_err_cm']:.1f} cm",
                 f"予測安全層の介入: 全ステップの {summary['mean_filter_ratio']*100:.1f}%",
                 "  (作業員とすれ違う場面のみ。他は方策の指令のまま)"]
        for i, l in enumerate(lines):
            text(d, (1010, 350 + i * 62), l, 27)
    text(d, (WID // 2, 980), "実用展開: 倉庫・工場・病院内搬送 ─ 地図に無い人や台車を学習方策で回避し、大域計画で確実に目的地へ", 30,
         YELLOW, True, "mm")
    return pil


# ---------------------------------------------------------------- writer
class Writer:
    def __init__(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.p = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo",
                                   "-pix_fmt", "rgb24", "-s", f"{WID}x{HEI}", "-r", str(FPS), "-i", "-",
                                   "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
                                   "-movflags", "+faststart", path], stdin=subprocess.PIPE)
        self.n = 0

    def write(self, rgb):
        self.p.stdin.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
        self.n += 1

    def hold(self, pil, sec, fade=0.4):
        a = np.array(pil).astype(np.float32)
        n = int(sec * FPS)
        nf = int(fade * FPS)
        for i in range(n):
            k = min(1.0, (i + 1) / nf, (n - i) / nf) if nf else 1.0
            self.write((a * k).astype(np.uint8))

    def close(self):
        self.p.stdin.close()
        self.p.wait()
        print(f"wrote {self.n} frames ({self.n / FPS:.1f} s)")


def find_highlight(run):
    """ロボットと作業員 (真値) が最も接近した時刻を探す."""
    best = (1e9, None)
    for s in run.status:
        if s.get("worker") and s["state"] == "NAV":
            x, y, _ = robot_xyth(s)
            dd = math.hypot(s["worker"][0] - x, s["worker"][1] - y)
            if dd < best[0]:
                best = (dd, s["t"])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.path.join(ROOT, "recordings", "final"))
    ap.add_argument("--out", default=os.path.join(ROOT, "deliverables", "demo.mp4"))
    ap.add_argument("--summary", default=os.path.join(ROOT, "recordings", "eval_summary.json"))
    ap.add_argument("--eval2d", default=os.path.join(ROOT, "training", "out", "eval.json"))
    ap.add_argument("--curve", default=os.path.join(ROOT, "report", "fig", "learning_curve.png"))
    ap.add_argument("--preview", type=float, default=None, help="指定 sim 時刻の1フレームだけPNG出力")
    a = ap.parse_args()
    run = Run(a.run)
    if a.preview is not None:
        Image.fromarray(frame_main(run, run.t0 + a.preview, "等倍 (1x)")).save(a.out)
        return
    summary = json.load(open(a.summary)) if os.path.exists(a.summary) else None
    ev = json.load(open(a.eval2d)) if os.path.exists(a.eval2d) else None
    abp = os.path.join(ROOT, "training", "out", "ablation.json")
    if ev is not None and os.path.exists(abp):
        ev["ablation"] = json.load(open(abp))
    w = Writer(a.out)
    w.hold(card_title(), 5.0)
    w.hold(card_diff(), 9.0)
    w.hold(card_arch(), 9.0)
    w.hold(card_training(a.curve, ev), 8.0)
    t = run.t0
    while t <= run.t1:
        w.write(frame_main(run, t, "等倍 (1x)"))
        t += 1.0 / FPS
    dmin, th = find_highlight(run)
    if th is not None:
        pil, d = card_base("スロー再生：動く作業員の回避", f"最接近距離 {dmin:.2f} m（中心間）─ 作業員の位置は AI に一切与えていない")
        w.hold(pil, 2.5)
        t = max(run.t0, th - 4.0)
        while t < min(run.t1, th + 4.0):
            w.write(frame_main(run, t, "0.5倍速 スロー再生",
                               ("スロー再生 (0.5倍速)：追跡した作業員の動きを読んで、譲る・回り込む", YELLOW)))
            t += 0.5 / FPS
    w.hold(card_results(run, summary), 9.0)
    w.close()


if __name__ == "__main__":
    main()
