"""説明資料 (PDF) を生成する. 数値は学習・評価の JSON から自動で差し込む.

usage: python report/build_report.py --name <omnicampusアカウント名>
出力:  deliverables/説明資料_PAI最終課題_<name>.pdf
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "report")
sys.path.insert(0, os.path.join(ROOT, "video"))
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def load(p):
    p = os.path.join(ROOT, p)
    return json.load(open(p)) if os.path.exists(p) else None


def pct(v):
    return f"{v * 100:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="omnicampusアカウント名")
    ap.add_argument("--video", default="demo.mp4")
    a = ap.parse_args()

    ev = load("training/out/eval.json")
    ab = load("training/out/ablation.json")
    gz = load("recordings/eval_summary.json")
    fin = load("recordings/final_summary.json")
    import csv
    rows = list(csv.DictReader(open(os.path.join(ROOT, "training/out/progress.csv"))))
    steps = int(rows[-1]["steps"])
    wall = float(rows[-1]["wall"])

    # システム構成図は動画のカードをそのまま流用
    from compose_video import card_arch, card_diff
    card_arch().save(os.path.join(REP, "fig", "architecture.png"))
    card_diff().save(os.path.join(REP, "fig", "diff.png"))

    ab_rows = ""
    labels = {"pure_pursuit": "A* + 純追従 (従来型, LiDAR 不使用)", "ppo_only": "PPO のみ (A* なし)",
              "astar_ppo_lidaronly": "A* + PPO（旧方策: LiDAR のみ入力）",
              "astar_ppo_filter_lidaronly": "A* + PPO（旧方策）+ 予測安全層",
              "astar_ppo": "A* + PPO（新方策: + 追跡した人の位置・速度）",
              "astar_ppo_filter": "<b>A* + PPO（新方策）+ 予測安全層 = 本作品</b>"}
    if ab:
        for k, lab in labels.items():
            if k not in ab:
                continue
            r = ab[k]
            mt = f"{r['mean_time_success']:.1f} s" if r["mean_time_success"] else "―"
            ab_rows += (f"<tr><td>{lab}</td><td><b>{pct(r['success_rate'])}</b></td><td>{pct(r['collision_worker'])}</td>"
                        f"<td>{pct(r['collision_static'])}</td><td>{pct(r['timeout'])}</td><td>{mt}</td></tr>")
    gz_rows = ""
    if gz:
        for r in gz["runs"]:
            gz_rows += (f"<tr><td>{r['run']}</td><td>{'✓ 完遂' if r['done'] and not r['collision'] else '✗'}</td>"
                        f"<td>{r['time']:.1f} s</td><td>{r['dist']:.1f} m</td><td>{r['min_clear_worker_m']:.2f} m</td>"
                        f"<td>{r['min_clear_static_m']:.2f} m</td><td>{r['loc_err_mean_cm']:.1f} cm</td>"
                        f"<td>{r['odom_err_cm']:.1f} cm</td></tr>")

    html = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<title>説明資料_PAI最終課題</title>
<style>
@page {{ size: A4; margin: 13mm 15mm 13mm 15mm; }}
body {{ font-family: "Hiragino Kaku Gothic ProN", "Hiragino Sans", sans-serif; font-size: 10pt; line-height: 1.62; color: #1d2733; }}
h1 {{ font-size: 21pt; margin: 0 0 4px; color: #0b3d63; }}
h2 {{ font-size: 13.5pt; border-left: 6px solid #1f8ed6; padding-left: 9px; margin: 20px 0 8px; color: #0b3d63; break-after: avoid; }}
h3 {{ font-size: 11pt; margin: 12px 0 4px; color: #134e7a; break-after: avoid; }}
.sub {{ color: #4a5a6a; font-size: 11pt; margin-bottom: 8px; }}
.box {{ background: #f1f6fb; border: 1px solid #cfe0ef; border-radius: 8px; padding: 9px 13px; margin: 8px 0; }}
.key {{ background: #fff7e6; border: 1px solid #f3d9a4; border-radius: 8px; padding: 9px 13px; margin: 8px 0; }}
table {{ border-collapse: collapse; width: 100%; margin: 6px 0 10px; font-size: 9pt; }}
tr {{ break-inside: avoid; }}
th, td {{ border: 1px solid #c9d3dd; padding: 4px 6px; text-align: left; vertical-align: top; }}
th {{ background: #e3edf6; }}
img {{ max-width: 100%; }}
figure {{ margin: 8px 0 12px; text-align: center; break-inside: avoid; }}
figcaption {{ font-size: 8.5pt; color: #5a6875; }}
code {{ font-family: Menlo, monospace; font-size: 8.6pt; background: #f2f4f6; padding: 0 3px; border-radius: 3px; }}
pre {{ font-family: Menlo, monospace; font-size: 8pt; background: #f4f6f8; border: 1px solid #dde3e8; border-radius: 6px; padding: 7px 9px; white-space: pre-wrap; break-inside: avoid; }}
.pb {{ break-before: page; }}
ul {{ margin: 4px 0 4px 1.2em; padding: 0; }} li {{ margin: 1px 0; }}
.cols {{ display: flex; gap: 12px; }} .cols > div {{ flex: 1; }}
.tag {{ display: inline-block; background: #1f8ed6; color: #fff; border-radius: 4px; padding: 0 6px; font-size: 8.5pt; margin-right: 4px; }}
</style></head><body>

<h1>AI 自律配達ロボット</h1>
<div class="sub">PPO 強化学習 × A* 階層ナビゲーション × LiDAR 自己位置推定 ― Ignition Gazebo + ROS 2 による倉庫内ラストワンマイル搬送</div>
<div class="box">
<b>講座:</b> Physical AI 応用1 最終課題<br>
<b>動画:</b> <code>{a.video}</code>（2分38秒, 1920×1080, 30fps）。<b>本編はシミュレーション時間で等倍（1倍速）再生</b>。
途中の回避シーンのみ <b>0.5倍速スロー再生</b>（画面右上に倍速を常時表示）。倍速（2倍速等）の早送りは使用していない。<br>
<b>使用ツール:</b> ROS 2 Humble, Ignition Gazebo Fortress, ros_gz_bridge, Stable-Baselines3 (PPO), PyTorch（学習時のみ）, numpy, Docker
</div>

<h2>1. 課題の概要と目的</h2>
<p>倉庫・工場・病院などの屋内搬送では、<b>「決められた地図の上を確実に目的地まで行く」能力</b>と、
<b>「地図に載っていない人や台車をその場で避ける」能力</b>の両方が必要になる。本課題では、
差動二輪ロボットが倉庫内の 3 つの配達ステーション（A→B→C）を巡回して HOME に戻る<b>配達ミッションを完全自律で実行</b>するシステムを
Gazebo 上に構築した。倉庫内には地図に載っていない<b>通路を横切って往復する作業員</b>がおり、ロボットは LiDAR だけを頼りに作業員を見つけ、動きを予測して「譲る・後ろを回る」判断をする。</p>
<p>AI は「推論のみ」ではなく、<b>強化学習 (PPO) で自ら学習した局所走行方策</b>を用いる。学習は軽量な自作 2D シミュレータで行い、
得られた方策を <b>Gazebo（物理エンジン・GPU LiDAR レンダリング）へ無改造で転移（Sim-to-Sim）</b>させた点が技術的な核である。</p>

<h2>2. 参考記事との違い</h2>
<p>参考記事（Qiita「gazeboとROS2を連携してロボットを操作する」@N622）は、SDF で差動二輪ロボットに駆動部を付け、
<b>キーボード入力を ROS 2 の Twist に変換し ros_ign_bridge 経由で Gazebo のロボットを人が操縦する</b>までを解説している。
本作品はこの「SDF ロボット → DiffDrive → /cmd_vel → ros_gz_bridge (config yaml) → launch 一括起動」という骨格を踏襲し、
車輪半径 0.033 m・車輪間隔 0.19 m も記事と同じ値を用いたうえで、<b>「人間の指」を「AI」に置き換え</b>、以下を独自に追加した。</p>
<table>
<tr><th style="width:17%">観点</th><th style="width:38%">参考記事</th><th>本作品（独自に追加・改良した点）</th></tr>
<tr><td>操縦主体</td><td>キーボード → teleop ノード → /cmd_vel（人が操縦）</td><td><b>AI ノードが LiDAR とオドメトリから /cmd_vel を 10Hz で生成</b>（完全自律）</td></tr>
<tr><td>センサ</td><td>なし（ロボットは周囲が見えない）</td><td>gpu_lidar 240 本/全周、オンボードカメラ、俯瞰カメラを SDF に追加</td></tr>
<tr><td>AI</td><td>なし</td><td><b>PPO 強化学習</b>で局所走行方策を学習（{steps/1e6:.0f}M ステップ）、numpy で推論</td></tr>
<tr><td>計画</td><td>なし</td><td>距離場コスト付き A* + 平滑化、1m 先サブゴール生成、停滞時の自動再計画</td></tr>
<tr><td>自己位置</td><td>なし</td><td>オドメトリ + <b>LiDAR/地図の尤度場スキャンマッチング</b>（車輪滑りのずれを補正）</td></tr>
<tr><td>環境</td><td>地面だけの空間</td><td>棚・柱・木箱のある倉庫（Python から SDF 自動生成）＋<b>通路を横切る作業員</b>（VelocityControl）</td></tr>
<tr><td>タスク</td><td>前後左右に動かす</td><td>HOME→A→B→C→HOME の配達ミッション（状態機械: 待機/走行/荷降ろし/完了）</td></tr>
<tr><td>ブリッジ</td><td>/cmd_vel の 1 トピック</td><td>config yaml で 9 トピック（/scan, /odom, /clock, 画像, 作業員, 真値 など）</td></tr>
<tr><td>人の認識</td><td>―</td><td>地図差分で LiDAR 点から<b>人を検出・追跡し速度を推定</b>、方策の入力に使用</td></tr>
<tr><td>安全</td><td>―</td><td><b>予測安全層</b>（人の 2.4 秒先を予測して指令を修正）＋前方シールド</td></tr>
<tr><td>評価</td><td>目視確認</td><td>2D 評価・<b>手法比較（アブレーション）</b>・Gazebo 反復試行・位置推定誤差を定量化</td></tr>
<tr><td>環境構築</td><td>Ubuntu 実機にインストール</td><td><b>Docker で ROS 2 + Gazebo を再現</b>（Apple Silicon Mac でも 1 コマンドで動作）</td></tr>
</table>
<p style="font-size:9pt;color:#5a6875">※ 記事の ball joint キャスターは跳ねやすい（記事のトラブルシューティングにも記載）ため、低摩擦の球を base_link に固定する方式に変更した。</p>

<h2>3. システム構成</h2>
<figure><img src="fig/architecture.png" style="width:96%"><figcaption>図1 システム構成（動画内のスライドと同一）</figcaption></figure>
<p>Gazebo 側は SDF のプラグイン（DiffDrive, Sensors, VelocityControl, OdometryPublisher）だけで構成し、
ROS 2 側に <code>ai_mission</code>（AI 本体）、<code>worker_mover</code>（作業員の往復）、<code>recorder</code>（記録）の 3 ノードを置いた。
<code>ros2 launch ai_delivery ai_delivery.launch.py</code> の 1 コマンドで Gazebo・ブリッジ・全ノードが起動する（記事の launch 化の発展）。</p>
<h3>ai_mission ノードの処理（10 Hz）</h3>
<ol>
<li><b>自己位置推定:</b> DiffDrive オドメトリの増分で予測し、LiDAR 点群を既知地図の<b>符号付き距離場</b>へ当てはめる Gauss-Newton 最適化で補正（尤度場モデル）。
Huber 重みと対応距離の閾値により、地図に無い作業員の点は自動的に外れ値として無視される。</li>
<li><b>動的障害物の検出・追跡:</b> 推定位置で LiDAR 点を地図に重ね、地図から 0.15m 以上離れた点＝「地図に無い物体」を角度方向にクラスタリング。
最近傍対応と α-β フィルタで位置・速度を推定する。<b>作業員の位置や速度は AI に一切与えておらず、すべて LiDAR から求めている。</b></li>
<li><b>ミッション層:</b> 配達先キューを持つ状態機械。到着判定（0.22m 以内）→ 2 秒停止（荷降ろし）→ 次の目的地。</li>
<li><b>大域計画層:</b> 5cm 格子・膨張半径 0.26m の A*。障害物に近いほどコストが上がる距離場コストで<b>通路中央を通る経路</b>にし、ラプラシアン平滑化。</li>
<li><b>局所制御層 (AI):</b> 経路上 1m 先のサブゴール・LiDAR・追跡した人の相対位置/速度を PPO 方策に入力し、並進・角速度を出力。</li>
<li><b>予測安全層:</b> 方策の指令と 20 種類の代替指令（減速・停止・後退・回避旋回…）をそれぞれ 2.4 秒先までロールアウトし、
人の等速予測位置・地図との最小隙間を評価。方策の指令が安全ならそのまま通し、危険なら「安全な候補のうち方策に最も近いもの」に差し替える（shielded RL）。</li>
<li><b>前方シールド / 停滞検知:</b> 前方 ±22.5° に 0.20m 以内の物体があれば前進成分を 0 に。8 秒進捗が無ければ再計画。</li>
</ol>

<h2>4. AI（強化学習）の設計</h2>
<div class="cols"><div>
<table>
<tr><th colspan="2">観測 (63 次元)</th></tr>
<tr><td>LiDAR</td><td>240 本を 15° ごと 24 セクタに min-pooling（/3.5m 正規化）</td></tr>
<tr><td>LiDAR (過去)</td><td>0.3 秒前の 24 セクタ → 動く障害物の接近を読み取る</td></tr>
<tr><td>サブゴール</td><td>距離, cos(角度), sin(角度)</td></tr>
<tr><td>自己速度</td><td>v, ω（正規化）</td></tr>
<tr><td>人 (2 体)</td><td>追跡した近い 2 体の相対位置・速度（ロボット座標）＋有効フラグ</td></tr>
<tr><th colspan="2">行動 (連続 2 次元)</th></tr>
<tr><td>v</td><td>−0.12〜0.35 m/s（後退も許可）</td></tr>
<tr><td>ω</td><td>−1.6〜1.6 rad/s</td></tr>
</table></div><div>
<table>
<tr><th colspan="2">報酬</th></tr>
<tr><td>進捗</td><td>+25 × (A* 経路上の残り距離の減少量)</td></tr>
<tr><td>到着</td><td>+15（終了）</td></tr>
<tr><td>衝突</td><td>−30（終了）</td></tr>
<tr><td>接近</td><td>隙間 0.3m 未満で最大 −1.0/step、人とは 0.45m 未満で最大 −1.2/step</td></tr>
<tr><td>滑らかさ</td><td>−0.03 × |Δω|, 後退 −0.02, 時間 −0.02</td></tr>
<tr><th colspan="2">PPO 設定</th></tr>
<tr><td>方策</td><td>MLP 256-256 (tanh), 12 並列環境</td></tr>
<tr><td>学習</td><td>{steps/1e6:.1f}M ステップ, 約 {wall/60:.0f} 分 (Mac CPU)</td></tr>
</table></div></div>
<h3>Sim-to-Sim 転移のための工夫</h3>
<ul>
<li><b>学習と実行で観測生成コードを共有</b>（<code>nav_core.py</code>）。セクタ化・正規化・サブゴール計算が学習時と Gazebo 上で 1 ビットも違わない。</li>
<li><b>ドメインランダマイゼーション:</b> 部屋の大きさ、棚・箱・柱の数/形/配置、動く障害物（半径・速度・向き）をエピソードごとに生成。
LiDAR のノイズ（0〜2%）と欠損（1%）、速度応答の一次遅れ（τ=0.05〜0.25s）、速度ゲイン誤差（0.85〜1.1）も毎回変える。</li>
<li><b>学習課題そのものを本番と同じ形に:</b> 各エピソードで A* 経路を計画し「サブゴール追従」を学習。さらに経路に絡む<b>対向・横断・同方向の歩行者</b>を確率 75% で 1〜2 体配置し、譲る・すれ違う・追い越す判断を重点的に練習させた。</li>
<li><b>人の特徴量は「トラッカーの出力を模擬」して与える:</b> 位置 5cm・速度 6cm/s の推定ノイズ、8% の見落とし、手前の物体による遮蔽を再現し、実行時の LiDAR トラッカーとの差を吸収。</li>
<li>推論は重みを <code>.npz</code> に書き出して <b>numpy だけで実行</b>（ロボット側に PyTorch 不要、1 回 0.1ms 未満）。</li>
</ul>
<figure><img src="fig/learning_curve.png" style="width:92%"><figcaption>図2 学習曲線（直近 300 エピソードの移動平均。ランダム生成環境を含むため難しい課題も混在）</figcaption></figure>
{f'''<p>学習後の 2D 評価（学習に使っていない乱数シード, 各 {ev["warehouse"]["success"] + ev["warehouse"]["collision"] + ev["warehouse"]["timeout"]} エピソード）:
倉庫レイアウトで<b>成功率 {pct(ev["warehouse"]["success_rate"])}</b>（衝突 {pct(ev["warehouse"]["collision_rate"])}）、
未知のランダム部屋で<b>成功率 {pct(ev["random"]["success_rate"])}</b>（衝突 {pct(ev["random"]["collision_rate"])}）。
これは安全層なし・方策単体で、高速な歩行者が死角から接近するような厳しいケースも含む値である。実運用ではこれを次節の予測安全層と組み合わせる。</p>''' if ev else ''}

<h2>5. 実験結果</h2>
<h3>5.1 手法比較（アブレーション, 2D シミュレータ, ミッション全体 HOME→A→B→C→HOME）</h3>
<p>同じ作業員の動きパターン（乱数シード固定）で各手法を {ab['pure_pursuit']['trials'] if ab else ''} 回ずつ実行した。</p>
<table><tr><th>手法</th><th>完遂率</th><th>作業員と衝突</th><th>静的障害物と衝突</th><th>立ち往生</th><th>平均所要時間</th></tr>{ab_rows}</table>
<figure><img src="fig/ablation.png" style="width:86%"><figcaption>図3 手法比較</figcaption></figure>
<div class="key"><b>考察:</b>
①従来型の A*+純追従は経路を正確に追うが、人を見ないため横切る作業員とほぼ衝突する。②PPO 単体（A* なし）は棚の裏の目的地へ直進しようとして棚に衝突する。
③A*+PPO で静的障害物の衝突は 0 になるが、<b>LiDAR だけを見る旧方策は横切る人への対応が弱い</b>（{pct(ab['astar_ppo_lidaronly']['success_rate']) if ab and 'astar_ppo_lidaronly' in ab else ''}）。
④<b>追跡した人の位置・速度を方策に入力</b>すると改善し（{pct(ab['astar_ppo']['success_rate']) if ab else ''}）、⑤さらに<b>予測安全層</b>と組み合わせた本作品で <b>{pct(ab['astar_ppo_filter']['success_rate']) if ab else ''}</b> に達した。
同じ安全層でも旧方策では {pct(ab['astar_ppo_filter_lidaronly']['success_rate']) if ab and 'astar_ppo_filter_lidaronly' in ab else ''} に留まり、「学習方策の質」と「予測による安全層」の両方が必要なことが分かる。</div>
<h3>5.2 試したが採用しなかった方法（開発過程）</h3>
<ul>
<li><b>作業員を A* のコストマップへ入れて 1 秒ごとに再計画</b>（Nav2 の obstacle layer 相当）: 経路が作業員の左右で振動し、学習方策がサブゴールの急変に追従できず、むしろ完遂率が下がった（開発途中の方策・作業員配置での比較で、再計画なし 72% → あり 62%）。局所の回避は学習方策＋安全層に任せる方が良いと判断。</li>
<li><b>既存方策の追加学習（ファインチューニング）だけ</b>: 横断する人に対する改善はほぼ無く、観測に人の速度情報を加える設計変更が決め手だった。</li>
<li>作業員の動線を通路と平行にした初期配置は「Uターンした作業員と鉢合わせる」避けようのない状況が多く、現実的な「通路を横切る」動線に変更した。</li>
</ul>

<h3>5.3 Gazebo 上での実行（Sim-to-Sim 転移の検証）</h3>
{f'''<p>作業員の出発タイミングを変えて（＝ロボットとの出会い方を変えて）Gazebo 上でミッション全体を <b>{gz["n"]} 回</b>実行した。
衝突判定は物理エンジンの真値位置と地図形状から算出。<b>完遂 {gz["success"]}/{gz["n"]}、衝突 {gz["collisions"]} 回</b>、
平均所要時間 {gz["mean_time"]:.1f} 秒。</p>''' if gz else ''}
<table><tr><th>試行</th><th>結果</th><th>所要時間</th><th>走行距離</th><th>作業員との最小隙間</th><th>静的障害物との最小隙間</th><th>位置推定誤差(平均)</th><th>オドメトリのみの最終誤差</th></tr>{gz_rows}</table>
<figure><img src="fig/trajectories.png" style="width:80%"><figcaption>図4 Gazebo 上の全試行の走行軌跡（真値）。作業員との出会い方によって経路が変化している</figcaption></figure>
{f'''<div class="box"><b>自己位置推定の効果:</b> DiffDrive の車輪オドメトリだけでは旋回時の滑りにより最終誤差が平均 <b>{gz["mean_odom_err_cm"]:.1f} cm</b> に達するが、
LiDAR スキャンマッチングを併用した推定位置の誤差は走行中平均 <b>{gz["mean_loc_err_cm"]:.1f} cm</b> に抑えられた。地図上の A* 経路を正しく追うために不可欠な要素であった。</div>''' if gz else ''}

<h2>6. 工夫した点（評価観点ごと）</h2>
<p><span class="tag">創造性</span> 記事の「キー入力 → Twist」の部分を「センサ → AI → Twist」に置き換えるという明快な発展軸を取り、
“人の指”を“学習した方策”に差し替えた。地図に無い作業員を入れることで「計画だけでは足りず、学習した反応と予測が要る」ことをタスク自体で示し、アブレーションで裏付けた。</p>
<p><span class="tag">技術力</span> ①PPO による連続値制御方策の学習（人の追跡情報を含む 63 次元観測）、②A*・サブゴール生成との階層統合、③符号付き距離場を使った Gauss-Newton スキャンマッチングの自作、
④地図差分による人の検出・追跡、⑤ロールアウト型の予測安全層（shielded RL）、⑥学習と実行のコード共有による Sim-to-Sim 転移、⑦SDF の自動生成と Docker による再現可能な環境。
開発中に「LiDAR 180 本を 24 分割した際の端数で後方に死角」「OdometryPublisher が world 座標で出力される仕様」「距離場の内部が 0 で点が壁に沈むバイアス」などを発見・修正した。</p>
<p><span class="tag">実用性</span> 構成は実ロボットの Nav2（大域計画 + 局所制御 + AMCL）と同じ階層で、各層を差し替え可能。方策は numpy だけで動くため
Raspberry Pi 等の小型計算機にも載る。予測安全層で学習方策の不確実さを補う設計（shielded RL）は、実機導入時の安全要件を意識したもの。</p>
<p><span class="tag">表現力</span> 動画では Gazebo の俯瞰カメラ映像にカメラモデル（ピンホール）で A* 経路・サブゴール・軌跡・LiDAR を<b>正確に投影</b>し、
右側に方策の入力（LiDAR 極座標図）と出力（v, ω）、ミッション進捗、位置推定誤差を表示。状態に応じた日本語字幕で「今 AI が何をしているか」を説明し、
回避シーンはスロー再生で見せた。録画もすべてシミュレーション時刻で同期している。</p>

<h2>7. 制約と今後の課題</h2>
<ul>
<li>Gazebo はソフトウェアレンダリング（GPU なしの Docker）で実時間の約 0.7 倍で動作。動画はシミュレーション時刻基準で等倍に再構成している。</li>
<li>作業員は単純な往復運動（人の側はロボットを避けない最悪ケース）。今後は Gazebo の actor による歩行者群、人の行動予測の学習、複数ロボットの協調を扱いたい。</li>
<li>2D シミュレータ上の方策単体の成功率はまだ改善の余地がある（観測の時系列化や LSTM 方策、カリキュラム学習など）。</li>
<li>地図は既知を前提。SLAM による地図生成、Nav2 との統合、実機（TurtleBot 等）への Sim-to-Real 転移が次の段階。</li>
</ul>

<h2>8. 実行方法とファイル構成</h2>
<pre>docker build -t pai-gz:humble docker/                     # ROS 2 Humble + Gazebo Fortress 環境
uv venv .venv && uv pip install -r requirements.txt        # 学習・動画合成用 (Mac)
python tools/gen_world.py                                  # world_def.py → SDF ワールド生成
python training/train_ppo.py --steps 16000000              # PPO 学習 → models/policy.npz
python training/evaluate.py ; python training/ablation.py  # 2D 評価・手法比較
docker run --rm -v "$PWD":/work pai-gz:humble bash /work/tools/run_sim.sh /work/recordings/final
bash tools/eval_gazebo.sh 5                                # Gazebo 反復試行
python report/make_figures.py ; python video/compose_video.py</pre>
<pre>ros2_ws/src/ai_delivery/      ROS 2 パッケージ (ament_python)
  ai_delivery/mission_node.py   AI 本体 (自己位置推定・ミッション・A*・PPO・安全シールド)
  ai_delivery/localizer.py      LiDAR 尤度場スキャンマッチング
  ai_delivery/safety.py         人の検出・追跡 / 予測安全層
  ai_delivery/planner.py        A*・平滑化・サブゴール
  ai_delivery/nav_core.py       観測/行動の定義 (学習と実行で共有)
  ai_delivery/sim2d.py          PPO 学習用 2D シミュレータ (Gymnasium)
  ai_delivery/policy.py         numpy 推論ランタイム
  ai_delivery/world_def.py      倉庫の唯一の定義 (SDF・地図・可視化の生成元)
  ai_delivery/worker_node.py, recorder_node.py
  launch/ai_delivery.launch.py  config/bridge.yaml  worlds/warehouse.sdf  models/policy.npz
training/  train_ppo.py evaluate.py ablation.py     tools/  gen_world.py run_sim.sh eval_gazebo.sh
video/  compose_video.py camera_model.py           docker/Dockerfile</pre>

<h2>参考文献</h2>
<ul style="font-size:9pt">
<li>@N622, 「gazeboとROS2を連携してロボットを操作する」, Qiita, 2023. https://qiita.com/N622/items/9d89f77d85d9da0af29e</li>
<li>Gazebo Fortress Documentation（DiffDrive, Sensors, VelocityControl, ros_gz_bridge）https://gazebosim.org/docs/fortress</li>
<li>J. Schulman et al., "Proximal Policy Optimization Algorithms", arXiv:1707.06347, 2017.</li>
<li>A. Raffin et al., "Stable-Baselines3: Reliable Reinforcement Learning Implementations", JMLR, 2021.</li>
<li>S. Thrun, W. Burgard, D. Fox, "Probabilistic Robotics"（尤度場モデル）, MIT Press, 2005.</li>
<li>J. Tobin et al., "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World", IROS, 2017.</li>
</ul>
</body></html>"""
    out_html = os.path.join(REP, "report.html")
    open(out_html, "w").write(html)
    os.makedirs(os.path.join(ROOT, "deliverables"), exist_ok=True)
    pdf = os.path.join(ROOT, "deliverables", f"説明資料_PAI最終課題_{a.name}.pdf")
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", "file://" + out_html], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("wrote", pdf)


if __name__ == "__main__":
    main()
