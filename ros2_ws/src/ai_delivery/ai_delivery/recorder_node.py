"""動画用レコーダ: カメラ画像とAIの内部状態をシミュレーション時刻付きで保存.

保存先 (out_dir):
  overview/<stamp>.jpg, fpv/<stamp>.jpg, status.jsonl
動画は video/compose_video.py でこれらを合成して作る (時刻は全て sim time)。
"""
import json
import os

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import String


class Recorder(Node):
    def __init__(self):
        super().__init__("recorder")
        self.declare_parameter("out_dir", "/work/recordings/run")
        self.out = self.get_parameter("out_dir").value
        for d in ("overview", "fpv"):
            os.makedirs(os.path.join(self.out, d), exist_ok=True)
        self.fstatus = open(os.path.join(self.out, "status.jsonl"), "a")
        self.create_subscription(Image, "/camera/overview", lambda m: self.save(m, "overview"),
                                 qos_profile_sensor_data)
        self.create_subscription(Image, "/camera/fpv", lambda m: self.save(m, "fpv"),
                                 qos_profile_sensor_data)
        self.create_subscription(String, "/ai_nav/status", self.on_status, 10)
        # 作業員の位置は可視化専用 (AI ノードには渡していない)
        self.create_subscription(Odometry, "/worker/odom", self.on_worker, 10)
        self.declare_parameter("save_images", True)
        self.save_images = bool(self.get_parameter("save_images").value)
        self.worker = None
        self.done_at = None

    def on_worker(self, msg):
        p = msg.pose.pose.position
        self.worker = [round(p.x, 3), round(p.y, 3)]

    def save(self, msg, kind):
        if not self.save_images:
            return
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        if msg.encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        cv2.imwrite(os.path.join(self.out, kind, f"{stamp:010.3f}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

    def on_status(self, msg):
        st = json.loads(msg.data)
        st["worker"] = self.worker
        self.fstatus.write(json.dumps(st) + "\n")
        self.fstatus.flush()
        if st["state"] == "DONE":
            now = st["t"]
            if self.done_at is None:
                self.done_at = now
                self.get_logger().info("mission done; recording 4 more seconds")
            elif now - self.done_at > 4.0:
                open(os.path.join(self.out, "FINISHED"), "w").write(str(now))


def main():
    rclpy.init()
    n = Recorder()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    rclpy.try_shutdown()
