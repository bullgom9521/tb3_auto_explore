from setuptools import setup
import os
from glob import glob

package_name = "tb3_auto_explore"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name + "/launch",
            [
                "launch/explore_bringup.launch.py",
                "launch/explore_game2.launch.py",  # 새로 추가
            ],
        ),
        ("share/" + package_name + "/param", glob("param/*.yaml")),
        # [수정] config 폴더의 slam 설정 파일 설치
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Your Name",
    maintainer_email="you@example.com",
    description="Frontier exploration node driving Nav2 to explore unknown space while mapping.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "frontier_explorer = tb3_auto_explore.frontier_explorer:main",
            "explore_game2   = tb3_auto_explore.explore_game2:main",
            "initial_pose_from_file = tb3_auto_explore.initial_pose_from_file:main",  # ★ 추가
        ],
    },
)
