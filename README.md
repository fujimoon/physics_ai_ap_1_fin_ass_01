# AI 自律配達ロボット ─ PPO強化学習 × A* 階層ナビゲーション

Physical AI 応用1講座 最終課題「AI×ロボティクスによる創造的タスクの実現」。

Qiita「[gazeboとROS2を連携してロボットを操作する](https://qiita.com/N622/items/9d89f77d85d9da0af29e)」(@N622) の構成
(SDF の差動二輪ロボット → DiffDrive → `/cmd_vel` → ros_gz_bridge → launch 一括起動) を土台に、
**キーボード操縦を、強化学習で獲得した自律走行 AI に置き換えた**作品です。

- 倉庫内で HOME → A → B → C → HOME の配達ミッションを完全自律で実行
- 地図に無い「通路を往復する作業員」を LiDAR だけで検知し、すれ違い・追い越し
- 大域計画 = A*、局所制御 = PPO 方策、自己位置推定 = LiDAR/地図の尤度場スキャンマッチング、安全シールド
- 学習は自作 2D シミュレータ、実行は Ignition Gazebo (Sim-to-Sim 転移)

## 提出物

| ファイル | 内容 |
|---|---|
| `deliverables/demo.mp4` | タスク実行動画 (本編は等倍、回避シーンのみ 0.5 倍速と画面に明記) |
| `deliverables/説明資料_PAI最終課題_<アカウント名>.pdf` | 説明資料 (概要・目的・工夫・参考記事との違い・評価結果) |
| このリポジトリ一式 | 使用したコード |

## 動かし方

```bash
# 1. ROS 2 Humble + Ignition Gazebo Fortress 環境 (Docker; Apple Silicon でも動作)
docker build -t pai-gz:humble docker/

# 2. 学習・動画合成用の Python 環境 (ホスト)
uv venv .venv --python 3.12 && uv pip install -r requirements.txt

# 3. ワールド生成 → 学習 → 評価
.venv/bin/python tools/gen_world.py
.venv/bin/python training/train_ppo.py --steps 16000000 --envs 12
.venv/bin/python training/evaluate.py
.venv/bin/python training/ablation.py --trials 100

# 4. Gazebo 上でミッション実行 (録画あり)
docker run --rm -v "$PWD":/work pai-gz:humble bash /work/tools/run_sim.sh /work/recordings/final
bash tools/eval_gazebo.sh 5        # 反復試行 (画像保存なし) → recordings/eval_summary.json

# 5. 図・動画・レポート
.venv/bin/python report/make_figures.py
.venv/bin/python video/compose_video.py --run recordings/final --out deliverables/demo.mp4
.venv/bin/python report/build_report.py --name <omnicampusアカウント名>
```

GUI で見たい場合 (X サーバのある Linux): `ros2 launch ai_delivery ai_delivery.launch.py gui:=true`

## ディレクトリ

```
docker/Dockerfile                 ROS 2 Humble + ros-humble-ros-gz (Fortress)
ros2_ws/src/ai_delivery/          ROS 2 パッケージ (ament_python)
  ai_delivery/mission_node.py       AI 本体: 自己位置推定 / ミッション / A* / PPO / 安全シールド
  ai_delivery/localizer.py          LiDAR 尤度場スキャンマッチング (符号付き距離場 + Gauss-Newton)
  ai_delivery/planner.py            A* (距離場コスト) / 平滑化 / サブゴール
  ai_delivery/nav_core.py           観測・行動の定義 (学習と実行で共有)
  ai_delivery/sim2d.py              PPO 学習用 2D シミュレータ (Gymnasium, ドメインランダマイゼーション)
  ai_delivery/policy.py             numpy 推論ランタイム
  ai_delivery/world_def.py          倉庫の唯一の定義 (SDF・地図・可視化の生成元)
  ai_delivery/worker_node.py        作業員の往復制御
  ai_delivery/recorder_node.py      カメラ画像 + AI 内部状態の記録
  launch/ai_delivery.launch.py      一括起動
  config/bridge.yaml                ros_gz_bridge 設定 (9 トピック)
  worlds/warehouse.sdf              自動生成ワールド
  models/policy.npz                 学習済み方策
training/                         学習・評価・アブレーション
tools/                            ワールド生成・実行・集計スクリプト
video/                            動画合成 (カメラモデル投影)
report/                           説明資料生成 (HTML → PDF)
```
