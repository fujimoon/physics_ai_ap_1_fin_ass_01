"""動く障害物 (作業員) を中央通路で往復させるノード.

Gazebo の VelocityControl システムへ /worker/cmd_vel を送り、
/worker/odom (OdometryPublisher: world 座標) を見て端で折り返す。
ロボット側はこの動きを一切知らず、LiDAR だけで回避する。
"""
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node

from . import world_def as W


class WorkerNode(Node):
    def __init__(self):
        super().__init__("worker_mover")
        self.declare_parameter("start_delay", 3.0)
        self.start_delay = float(self.get_parameter("start_delay").value)  # 試行ごとに変えて出会い方を変える
        self.dir = 1.0
        self.p0, self.u, self.L = W.worker_lane()
        self.s = self.L / 2
        self.t0 = None
        self.pub = self.create_publisher(Twist, "/worker/cmd_vel", 10)
        self.create_subscription(Odometry, "/worker/odom", self.on_odom, 10)
        self.create_timer(0.05, self.tick)

    def on_odom(self, msg):
        p = msg.pose.pose.position
        self.s = (p.x - self.p0[0]) * self.u[0] + (p.y - self.p0[1]) * self.u[1]  # レーン上の位置

    def tick(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.t0 is None:
            self.t0 = now
        t = Twist()
        if now - self.t0 > self.start_delay:
            # モデルはレーン方向を向けて置いてあるので、前進 (linear.x) = レーン方向
            if self.s > self.L and self.dir > 0:
                self.dir = -1.0
            elif self.s < 0 and self.dir < 0:
                self.dir = 1.0
            t.linear.x = self.dir * W.WORKER_SPEED
        self.pub.publish(t)


def main():
    rclpy.init()
    n = WorkerNode()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    rclpy.try_shutdown()
