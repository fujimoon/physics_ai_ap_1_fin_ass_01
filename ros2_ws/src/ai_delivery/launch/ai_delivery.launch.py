"""Gazebo + ros_gz_bridge + AI ノード群を一括起動する launch.

ros2 launch ai_delivery ai_delivery.launch.py [gui:=true] [record:=true] [out_dir:=...]
"""
import os
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_gazebo(context):
    share = get_package_share_directory("ai_delivery")
    src = open(os.path.join(share, "worlds", "warehouse.sdf")).read()
    world = os.path.join(tempfile.gettempdir(), "warehouse_resolved.sdf")
    open(world, "w").write(src.replace("@@PKG@@", share))
    gui = LaunchConfiguration("gui").perform(context) == "true"
    cmd = ["ign", "gazebo", "-r", "-v", "2", world] if gui else ["ign", "gazebo", "-s", "-r", "-v", "2", world]
    return [ExecuteProcess(cmd=cmd, output="screen")]


def generate_launch_description():
    share = get_package_share_directory("ai_delivery")
    sim = {"use_sim_time": True}
    return LaunchDescription([
        DeclareLaunchArgument("gui", default_value="false"),
        DeclareLaunchArgument("record", default_value="true"),
        DeclareLaunchArgument("out_dir", default_value="/work/recordings/run"),
        DeclareLaunchArgument("save_images", default_value="true"),
        DeclareLaunchArgument("worker_delay", default_value="3.0"),
        SetEnvironmentVariable("IGN_GAZEBO_RESOURCE_PATH", os.path.join(share, "models")),
        OpaqueFunction(function=launch_gazebo),
        Node(package="ros_gz_bridge", executable="parameter_bridge", output="screen",
             parameters=[{"config_file": os.path.join(share, "config", "bridge.yaml")}, sim]),
        Node(package="ai_delivery", executable="mission_node", output="screen", parameters=[sim]),
        Node(package="ai_delivery", executable="worker_node", output="screen",
             parameters=[sim, {"start_delay": LaunchConfiguration("worker_delay")}]),
        Node(package="ai_delivery", executable="recorder_node", output="screen",
             condition=IfCondition(LaunchConfiguration("record")),
             parameters=[sim, {"out_dir": LaunchConfiguration("out_dir"),
                               "save_images": LaunchConfiguration("save_images")}]),
    ])
