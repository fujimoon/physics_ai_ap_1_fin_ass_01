"""AI 配達ミッションノード.

  /scan (LaserScan) + /odom (Odometry)
        │
  [自己位置推定]  オドメトリ予測 + LiDAR/地図の尤度場スキャンマッチング
  [ミッション層]  配達先キュー HOME→A→B→C→HOME, 到着判定, 荷降ろし待機
  [大域計画層]    既知地図上の A*  → 経路
  [局所制御層]    PPO 方策 (numpy 推論) ← LiDAR 24セクタ×2フレーム + サブゴール極座標
                  + LiDAR で追跡した動的障害物 (近い 2 体の相対位置・速度)
  [予測安全層]    地図差分で動的障害物を検出・追跡 → 2.4 秒先まで予測して方策の指令を安全化
  [安全シールド]  前方至近距離なら前進を止める (最後の砦)
        │
  /cmd_vel (Twist)  → ros_gz_bridge → Gazebo DiffDrive

参考記事ではキー入力を Twist に変換していた部分を、
「センサ入力から AI が Twist を生成する」部分に置き換えている。
"""
import json
import math
import os

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from . import world_def as W
from .localizer import LikelihoodFieldLocalizer
from .planner import GridMap, PathTracker
from .safety import DynamicTracker, SafetyFilter
from .policy import NumpyPolicy
from .nav_core import ScanHistory, action_to_cmd, build_obs, dyn_features, sector_pool, tracker_goal

ARRIVE_TOL = 0.22
DWELL_SEC = 2.0
STUCK_SEC = 8.0


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class MissionNode(Node):
    def __init__(self):
        super().__init__("ai_mission")
        share = get_package_share_directory("ai_delivery")
        self.declare_parameter("policy_path", os.path.join(share, "models", "policy.npz"))
        self.declare_parameter("start_delay", 3.0)
        self.declare_parameter("use_localizer", True)
        self.use_loc = bool(self.get_parameter("use_localizer").value)
        self.policy = NumpyPolicy(self.get_parameter("policy_path").value)
        self.start_delay = float(self.get_parameter("start_delay").value)
        self.grid = GridMap()

        # odom 座標 -> world 座標 (ロボットのスポーン姿勢)
        self.spawn = (W.HOME[1], W.HOME[2], math.pi / 2)
        self.loc = LikelihoodFieldLocalizer(self.spawn, grid=self.grid)
        self.declare_parameter("use_safety_filter", True)
        self.use_filter = bool(self.get_parameter("use_safety_filter").value)
        self.dyn = DynamicTracker(self.grid)
        self.filt = SafetyFilter(self.grid)
        self.tracks = []
        self.filter_count = 0
        self.odom_pose = None

        self.pub_cmd = self.create_publisher(Twist, "/cmd_vel", 10)
        self.pub_status = self.create_publisher(String, "/ai_nav/status", 10)
        self.create_subscription(LaserScan, "/scan", self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.create_subscription(Odometry, "/ground_truth/odom", self.on_gt, 10)

        self.scan = None
        self.pose = None
        self.gt = None
        self.vel = (0.0, 0.0)
        self.hist = ScanHistory()
        self.queue = list(W.MISSION)
        self.state = "WAIT"
        self.tracker = None
        self.path = None
        self.t_state = None
        self.last_prog = (0.0, None)
        self.events = []
        self.shield_count = 0
        self.dist_travelled = 0.0
        self.last_xy = None
        self.legs = []
        self.timer = self.create_timer(0.1, self.tick)  # 10 Hz (学習時の DT と一致)
        self.get_logger().info("AI mission node ready: " + " -> ".join(["HOME"] + self.queue))

    # ------------------------------------------------------------- callbacks
    def on_scan(self, msg):
        self.scan = np.asarray(msg.ranges, dtype=np.float64)
        # スキャン受信時点で補正 (制御周期で行うより時刻ずれが小さい)
        if self.use_loc and self.odom_pose is not None:
            self.loc.correct(self.scan, msg.angle_min, msg.angle_increment, msg.range_max)
            self.pose = tuple(self.loc.pose)
            # 地図に無い物体 (= 動的障害物) を検出・追跡
            cl = self.dyn.detect(self.scan, msg.angle_min, msg.angle_increment, self.pose)
            self.tracks = self.dyn.update(cl, self.now())

    def to_world(self, p):
        """odom 座標 (スポーン地点が原点) の姿勢を world 座標へ."""
        ox, oy, oth = p.position.x, p.position.y, yaw_of(p.orientation)
        sx, sy, sth = self.spawn
        c, s = math.cos(sth), math.sin(sth)
        return (sx + c * ox - s * oy, sy + s * ox + c * oy,
                math.atan2(math.sin(sth + oth), math.cos(sth + oth)))

    def on_odom(self, msg):
        # 真値 (/ground_truth/odom) は制御には一切使わず、ログ (誤差評価) 専用
        self.odom_pose = self.to_world(msg.pose.pose)   # 車輪オドメトリのみの推定
        self.loc.predict(self.odom_pose)
        self.pose = tuple(self.loc.pose) if self.use_loc else self.odom_pose
        self.vel = (msg.twist.twist.linear.x, msg.twist.twist.angular.z)

    def on_gt(self, msg):
        # OdometryPublisher は world 座標で出力される
        p = msg.pose.pose
        self.gt = (p.position.x, p.position.y, yaw_of(p.orientation))

    # ------------------------------------------------------------- helpers
    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def event(self, text):
        self.get_logger().info(text)
        self.events.append({"t": round(self.now(), 2), "msg": text})

    def plan_to(self, name):
        st = W.station(name)
        path = self.grid.astar(self.pose[:2], (st[1], st[2]))
        if path is None:
            self.event(f"A* failed to {name}")
            return False
        self.path = path
        self.tracker = PathTracker(path)
        self.goal_name = name
        self.goal_xy = np.array([st[1], st[2]])
        self.last_prog = (0.0, self.now())
        self.event(f"A* planned to {name}: {self.tracker.length:.2f} m, {len(path)} pts")
        return True

    def send(self, v, w):
        t = Twist()
        t.linear.x = float(v)
        t.angular.z = float(w)
        self.pub_cmd.publish(t)

    # ------------------------------------------------------------- main loop
    def tick(self):
        if self.scan is None or self.pose is None:
            return
        now = self.now()
        x, y, th = self.pose
        if self.last_xy is not None:
            self.dist_travelled += math.hypot(x - self.last_xy[0], y - self.last_xy[1])
        self.last_xy = (x, y)
        sec = sector_pool(self.scan)
        prev_sec = self.hist.push(sec)  # 0.3 秒前のスキャン
        cmd = (0.0, 0.0)
        action = [0.0, 0.0]
        sub = None
        shield = False
        filtered = False
        cmd_policy = (0.0, 0.0)

        if self.state == "WAIT":
            if self.t_state is None:
                self.t_state = now
            if now - self.t_state > self.start_delay:
                self.event("Mission start: HOME -> " + " -> ".join(self.queue))
                self.leg_t0 = now
                self.leg_d0 = self.dist_travelled
                self.state = "NAV" if self.plan_to(self.queue[0]) else "FAIL"
        elif self.state == "NAV":
            dist_goal = float(np.hypot(*(self.goal_xy - (x, y))))
            if dist_goal < ARRIVE_TOL:
                self.legs.append({"to": self.goal_name, "time": round(now - self.leg_t0, 2),
                                  "dist": round(self.dist_travelled - self.leg_d0, 2),
                                  "plan": round(self.tracker.length, 2)})
                self.event(f"Arrived at {self.goal_name} (err {dist_goal:.2f} m)")
                self.queue.pop(0)
                self.state = "DWELL"
                self.t_state = now
            else:
                gd, ga, sub = tracker_goal(self.tracker, x, y, th, self.goal_xy)
                # 停滞検知 -> 現在地から再計画
                if self.tracker.progress > self.last_prog[0] + 0.05:
                    self.last_prog = (self.tracker.progress, now)
                elif now - self.last_prog[1] > STUCK_SEC:
                    self.event("Stuck detected -> replanning")
                    self.plan_to(self.goal_name)
                dyn = dyn_features([(t["x"], t["y"], t["vx"], t["vy"]) for t in self.tracks], self.pose)
                obs = build_obs(sec, prev_sec, gd, ga, self.vel[0], self.vel[1], dyn)
                action = self.policy(obs)
                cmd = action_to_cmd(action)
                cmd_policy = cmd
                if self.use_filter:
                    cmd, intervened, _ = self.filt.filter(self.pose, cmd, self.tracks)
                    if intervened:
                        filtered = True
                        self.filter_count += 1
                # 安全シールド: 前方 ±22.5deg に 0.20m 以内の障害物があれば前進を止める
                front = min(sec[0], sec[1], sec[-1])
                if front < W.ROBOT_RADIUS + 0.06 and cmd[0] > 0.02:
                    cmd = (0.0, cmd[1])
                    shield = True
                    self.shield_count += 1
        elif self.state == "DWELL":
            if now - self.t_state > DWELL_SEC:
                if self.queue:
                    self.event(f"Delivered. Next: {self.queue[0]}")
                    self.leg_t0 = now
                    self.leg_d0 = self.dist_travelled
                    self.state = "NAV" if self.plan_to(self.queue[0]) else "FAIL"
                else:
                    self.state = "DONE"
                    self.event("Mission complete!")
        self.send(*cmd)

        st = {
            "t": round(now, 3), "state": self.state,
            "goal": getattr(self, "goal_name", None),
            "queue": self.queue,
            "pose": [round(float(v), 4) for v in self.pose],
            "odom": [round(float(v), 4) for v in self.odom_pose],
            "inliers": self.loc.last_inliers,
            "gt": [round(v, 4) for v in self.gt] if self.gt else None,
            "sub": [round(float(v), 3) for v in sub] if sub is not None else None,
            "cmd": [round(cmd[0], 3), round(cmd[1], 3)],
            "act": [round(float(a), 3) for a in action],
            "sec": [round(float(v), 3) for v in sec],
            "shield": shield, "shield_count": self.shield_count,
            "filtered": filtered, "filter_count": self.filter_count,
            "cmd_policy": [round(cmd_policy[0], 3), round(cmd_policy[1], 3)],
            "tracks": [[round(t["x"], 3), round(t["y"], 3), round(t["vx"], 3), round(t["vy"], 3)] for t in self.tracks],
            "dist": round(self.dist_travelled, 3),
            "legs": self.legs,
            "events": self.events[-6:],
        }
        if self.path is not None and self.state in ("NAV", "DWELL"):
            st["path"] = np.round(self.path[::4], 3).tolist()
        self.pub_status.publish(String(data=json.dumps(st)))


def main():
    rclpy.init()
    node = MissionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.send(0.0, 0.0)
    node.destroy_node()
    rclpy.try_shutdown()
