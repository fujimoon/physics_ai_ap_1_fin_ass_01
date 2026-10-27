"""world_def.py から Gazebo (Ignition Fortress) 用 SDF ワールドを生成する.

usage: python tools/gen_world.py
出力:  ros2_ws/src/ai_delivery/worlds/warehouse.sdf
       ros2_ws/src/ai_delivery/models/textures/*.png  (ステーションの床ラベル)
SDF 内のテクスチャパスは @@PKG@@ と書き、launch 時に実パスへ置換する。
"""
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "ros2_ws", "src", "ai_delivery")
sys.path.insert(0, PKG)
from ai_delivery import world_def as W  # noqa: E402

COLORS = {"shelf": (0.30, 0.38, 0.55), "crate": (0.62, 0.45, 0.25), "wall": (0.85, 0.85, 0.82)}


def mat(rgb, a=1.0):
    r, g, b = rgb
    return (f"<material><ambient>{r} {g} {b} {a}</ambient><diffuse>{r} {g} {b} {a}</diffuse>"
            f"<specular>0.1 0.1 0.1 1</specular></material>")


def box_model(name, cx, cy, sx, sy, h, rgb):
    geo = f"<geometry><box><size>{sx} {sy} {h}</size></box></geometry>"
    return f"""
    <model name="{name}"><static>true</static><pose>{cx} {cy} {h/2} 0 0 0</pose>
      <link name="link">
        <collision name="c">{geo}</collision>
        <visual name="v">{geo}{mat(rgb)}</visual>
      </link></model>"""


def cyl_model(name, cx, cy, r, h, rgb):
    geo = f"<geometry><cylinder><radius>{r}</radius><length>{h}</length></cylinder></geometry>"
    return f"""
    <model name="{name}"><static>true</static><pose>{cx} {cy} {h/2} 0 0 0</pose>
      <link name="link">
        <collision name="c">{geo}</collision>
        <visual name="v">{geo}{mat(rgb)}</visual>
      </link></model>"""


def make_textures():
    from PIL import Image, ImageDraw, ImageFont
    out = os.path.join(PKG, "models", "textures")
    os.makedirs(out, exist_ok=True)
    font = None
    for f in ["/System/Library/Fonts/Supplemental/Arial Bold.ttf",
              "/System/Library/Fonts/Helvetica.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        if os.path.exists(f):
            font = f
            break
    for name, _, _, rgb in [W.HOME] + W.STATIONS:
        img = Image.new("RGB", (512, 512), tuple(int(c * 255) for c in rgb))
        d = ImageDraw.Draw(img)
        d.rectangle([12, 12, 500, 500], outline=(255, 255, 255), width=18)
        size = 300 if len(name) == 1 else 140
        ft = ImageFont.truetype(font, size) if font else ImageFont.load_default()
        bb = d.textbbox((0, 0), name, font=ft)
        d.text(((512 - (bb[2] - bb[0])) / 2 - bb[0], (512 - (bb[3] - bb[1])) / 2 - bb[1]),
               name, fill=(255, 255, 255), font=ft)
        # SDF の平面は +x が画像の右、+y が画像の上になるよう上下反転しない
        img.save(os.path.join(out, f"station_{name}.png"))


def station_pad(name, x, y, rgb):
    tex = f"@@PKG@@/models/textures/station_{name}.png"
    return f"""
    <model name="station_{name}"><static>true</static><pose>{x} {y} 0.002 0 0 0</pose>
      <link name="link">
        <visual name="v">
          <geometry><plane><normal>0 0 1</normal><size>0.6 0.6</size></plane></geometry>
          <material>
            <ambient>1 1 1 1</ambient><diffuse>1 1 1 1</diffuse>
            <pbr><metal><albedo_map>{tex}</albedo_map><roughness>0.8</roughness></metal></pbr>
          </material>
        </visual>
      </link></model>"""


def robot_model():
    r, sep = W.WHEEL_RADIUS, W.WHEEL_SEPARATION
    wheel = lambda side, yy: f"""
      <link name="wheel_{side}_link">
        <pose>0 {yy} {r} -1.5707 0 0</pose>
        <inertial><mass>0.1</mass><inertia><ixx>1.1e-05</ixx><iyy>1.1e-05</iyy><izz>2.1e-05</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
        <collision name="c"><geometry><cylinder><radius>{r}</radius><length>0.018</length></cylinder></geometry>
          <surface><friction><ode><mu>1.2</mu><mu2>1.2</mu2></ode></friction></surface></collision>
        <visual name="v"><geometry><cylinder><radius>{r}</radius><length>0.018</length></cylinder></geometry>
          {mat((0.08, 0.08, 0.08))}</visual>
      </link>
      <joint name="wheel_{side}_joint" type="revolute">
        <parent>base_link</parent><child>wheel_{side}_link</child>
        <axis><xyz>0 0 1</xyz><limit><lower>-1e16</lower><upper>1e16</upper></limit></axis>
      </joint>"""
    return f"""
    <model name="delivery_bot">
      <pose>{W.HOME[1]} {W.HOME[2]} 0.0 0 0 {math.pi/2}</pose>
      <link name="base_link">
        <pose>0 0 0 0 0 0</pose>
        <inertial><pose>-0.03 0 0.07 0 0 0</pose><mass>1.5</mass>
          <inertia><ixx>0.01</ixx><iyy>0.01</iyy><izz>0.015</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
        <collision name="body_c"><pose>-0.03 0 0.075 0 0 0</pose>
          <geometry><box><size>0.22 0.17 0.09</size></box></geometry></collision>
        <visual name="body_v"><pose>-0.03 0 0.075 0 0 0</pose>
          <geometry><box><size>0.22 0.17 0.09</size></box></geometry>{mat((0.95, 0.95, 0.97))}</visual>
        <visual name="stripe_v"><pose>-0.03 0 0.121 0 0 0</pose>
          <geometry><box><size>0.222 0.172 0.004</size></box></geometry>{mat((0.10, 0.55, 0.95))}</visual>
        <visual name="cargo_v"><pose>-0.06 0 0.165 0 0 0</pose>
          <geometry><box><size>0.11 0.11 0.08</size></box></geometry>{mat((0.78, 0.58, 0.32))}</visual>
        <visual name="lidar_v"><pose>0.04 0 0.135 0 0 0</pose>
          <geometry><cylinder><radius>0.035</radius><length>0.03</length></cylinder></geometry>{mat((0.1, 0.1, 0.1))}</visual>
        <!-- キャスター: 参考記事の ball joint は跳ねやすいので、低摩擦の球を固定して代用 -->
        <collision name="caster_c"><pose>-0.12 0 0.015 0 0 0</pose>
          <geometry><sphere><radius>0.015</radius></sphere></geometry>
          <surface><friction><ode><mu>0.0</mu><mu2>0.0</mu2></ode></friction></surface></collision>
        <visual name="caster_v"><pose>-0.12 0 0.015 0 0 0</pose>
          <geometry><sphere><radius>0.015</radius></sphere></geometry>{mat((0.3, 0.3, 0.3))}</visual>
        <sensor name="lidar" type="gpu_lidar">
          <pose>0.04 0 0.15 0 0 0</pose>
          <topic>/scan</topic>
          <update_rate>10</update_rate>
          <lidar>
            <scan><horizontal><samples>{W.LIDAR_RAYS}</samples><resolution>1</resolution>
              <min_angle>0</min_angle><max_angle>{2*math.pi - 2*math.pi/W.LIDAR_RAYS:.6f}</max_angle></horizontal>
              <vertical><samples>1</samples><resolution>1</resolution><min_angle>0</min_angle><max_angle>0</max_angle></vertical></scan>
            <range><min>{W.LIDAR_MIN}</min><max>{W.LIDAR_MAX}</max><resolution>0.01</resolution></range>
            <noise><type>gaussian</type><mean>0</mean><stddev>0.01</stddev></noise>
          </lidar>
          <visualize>true</visualize>
          <always_on>1</always_on>
        </sensor>
        <sensor name="fpv" type="camera">
          <pose>0.085 0 0.13 0 0.12 0</pose>
          <topic>/camera/fpv</topic>
          <update_rate>10</update_rate>
          <camera><horizontal_fov>1.5</horizontal_fov>
            <image><width>480</width><height>300</height></image>
            <clip><near>0.03</near><far>30</far></clip></camera>
          <always_on>1</always_on>
        </sensor>
      </link>
      {wheel("left", sep/2)}
      {wheel("right", -sep/2)}
      <plugin filename="libignition-gazebo-diff-drive-system.so" name="ignition::gazebo::systems::DiffDrive">
        <left_joint>wheel_left_joint</left_joint>
        <right_joint>wheel_right_joint</right_joint>
        <wheel_separation>{sep}</wheel_separation>
        <wheel_radius>{r}</wheel_radius>
        <topic>/cmd_vel</topic>
        <odom_topic>/odom</odom_topic>
        <odom_publish_frequency>20</odom_publish_frequency>
        <frame_id>odom</frame_id>
        <child_frame_id>base_link</child_frame_id>
      </plugin>
      <!-- 真値 (物理エンジン上の実位置): 車輪オドメトリとの誤差評価用 -->
      <plugin filename="libignition-gazebo-odometry-publisher-system.so" name="ignition::gazebo::systems::OdometryPublisher">
        <odom_topic>/ground_truth/odom</odom_topic><odom_publish_frequency>20</odom_publish_frequency>
      </plugin>
    </model>"""


def worker_model():
    p0, u, L = W.worker_lane()
    x0, y0 = p0[0] + u[0] * L / 2, p0[1] + u[1] * L / 2
    return f"""
    <model name="worker">
      <pose>{x0} {y0} 0.0 0 0 {math.atan2(u[1], u[0])}</pose>
      <link name="body">
        <gravity>false</gravity>
        <inertial><pose>0 0 0.85 0 0 0</pose><mass>60</mass>
          <inertia><ixx>10</ixx><iyy>10</iyy><izz>10</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
        <collision name="c"><pose>0 0 0.85 0 0 0</pose>
          <geometry><cylinder><radius>{W.WORKER_R}</radius><length>1.5</length></cylinder></geometry></collision>
        <visual name="torso"><pose>0 0 0.80 0 0 0</pose>
          <geometry><cylinder><radius>{W.WORKER_R}</radius><length>1.4</length></cylinder></geometry>{mat((1.0, 0.45, 0.05))}</visual>
        <visual name="vest"><pose>0 0 1.05 0 0 0</pose>
          <geometry><cylinder><radius>{W.WORKER_R + 0.005}</radius><length>0.08</length></cylinder></geometry>{mat((0.95, 0.95, 0.2))}</visual>
        <visual name="head"><pose>0 0 1.62 0 0 0</pose>
          <geometry><sphere><radius>0.13</radius></sphere></geometry>{mat((0.95, 0.80, 0.65))}</visual>
        <visual name="helmet"><pose>0 0 1.70 0 0 0</pose>
          <geometry><sphere><radius>0.135</radius></sphere></geometry>{mat((1.0, 0.85, 0.0))}</visual>
      </link>
      <plugin filename="libignition-gazebo-velocity-control-system.so" name="ignition::gazebo::systems::VelocityControl">
        <topic>/worker/cmd_vel</topic>
      </plugin>
      <plugin filename="libignition-gazebo-odometry-publisher-system.so" name="ignition::gazebo::systems::OdometryPublisher">
        <odom_topic>/worker/odom</odom_topic><odom_publish_frequency>20</odom_publish_frequency>
      </plugin>
    </model>"""


def overview_camera():
    # 部屋全体を斜め上から撮る固定カメラ（動画の主画面）
    return """
    <model name="overview_cam"><static>true</static>
      <pose>0.0 -4.8 8.0 0 1.08 1.5708</pose>
      <link name="link">
        <sensor name="overview" type="camera">
          <topic>/camera/overview</topic>
          <update_rate>10</update_rate>
          <camera><horizontal_fov>1.15</horizontal_fov>
            <image><width>1024</width><height>640</height></image>
            <clip><near>0.1</near><far>50</far></clip></camera>
          <always_on>1</always_on>
        </sensor>
      </link></model>"""


def world():
    parts = []
    for i, (cx, cy, sx, sy, h, kind) in enumerate(W.BOXES):
        parts.append(box_model(f"{kind}_{i}", cx, cy, sx, sy, h, COLORS[kind]))
    for i, (cx, cy, r, h) in enumerate(W.CYLINDERS):
        parts.append(cyl_model(f"pillar_{i}", cx, cy, r, h, (0.55, 0.55, 0.58)))
    for i, (cx, cy, sx, sy) in enumerate(W.wall_boxes()):
        parts.append(box_model(f"wall_{i}", cx, cy, sx, sy, 0.6, COLORS["wall"]))
    for name, x, y, rgb in [W.HOME] + W.STATIONS:
        parts.append(station_pad(name, x, y, rgb))
    parts += [robot_model(), worker_model(), overview_camera()]
    body = "\n".join(parts)
    return f"""<?xml version="1.0" ?>
<!-- 自動生成: tools/gen_world.py (編集は world_def.py へ) -->
<sdf version="1.8">
  <world name="warehouse">
    <physics name="1ms" type="ignored"><max_step_size>0.002</max_step_size><real_time_factor>1.0</real_time_factor></physics>
    <plugin filename="libignition-gazebo-physics-system.so" name="ignition::gazebo::systems::Physics"/>
    <plugin filename="libignition-gazebo-user-commands-system.so" name="ignition::gazebo::systems::UserCommands"/>
    <plugin filename="libignition-gazebo-scene-broadcaster-system.so" name="ignition::gazebo::systems::SceneBroadcaster"/>
    <plugin filename="libignition-gazebo-sensors-system.so" name="ignition::gazebo::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <scene><ambient>0.55 0.55 0.55 1</ambient><background>0.75 0.82 0.9 1</background><shadows>true</shadows><grid>false</grid></scene>
    <light type="directional" name="sun"><cast_shadows>true</cast_shadows><visualize>false</visualize><pose>0 0 10 0 0 0</pose>
      <diffuse>0.85 0.85 0.85 1</diffuse><specular>0.2 0.2 0.2 1</specular>
      <direction>-0.4 0.3 -0.9</direction></light>
    <light type="point" name="fill"><cast_shadows>false</cast_shadows><visualize>false</visualize><pose>0 0 4 0 0 0</pose>
      <diffuse>0.3 0.3 0.3 1</diffuse><attenuation><range>20</range><constant>0.6</constant><linear>0.02</linear><quadratic>0.0</quadratic></attenuation></light>
    <model name="ground_plane"><static>true</static>
      <link name="link">
        <collision name="c"><geometry><plane><normal>0 0 1</normal></plane></geometry>
          <surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface></collision>
        <visual name="v"><geometry><plane><normal>0 0 1</normal><size>30 30</size></plane></geometry>
          {mat((0.80, 0.80, 0.77))}</visual>
      </link></model>
{body}
  </world>
</sdf>
"""


if __name__ == "__main__":
    make_textures()
    path = os.path.join(PKG, "worlds", "warehouse.sdf")
    with open(path, "w") as f:
        f.write(world())
    print("wrote", path)
