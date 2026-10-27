from glob import glob

from setuptools import setup

package_name = "ai_delivery"
setup(
    name=package_name,
    version="1.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/worlds", glob("worlds/*.sdf")),
        ("share/" + package_name + "/models", glob("models/*.npz")),
        ("share/" + package_name + "/models/textures", glob("models/textures/*.png")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={"console_scripts": [
        "mission_node = ai_delivery.mission_node:main",
        "worker_node = ai_delivery.worker_node:main",
        "recorder_node = ai_delivery.recorder_node:main",
    ]},
)
