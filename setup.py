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
        # [수정] launch 폴더의 모든 .launch.py 파일 포함
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        # [수정] ★ 중요: param 폴더의 burger.yaml 설치 (이게 있어야 런치 파일이 읽음)
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
        ],
    },
)
