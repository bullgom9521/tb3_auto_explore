from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.conditions import IfCondition  # ✅ 조건부 실행 import 추가
import os

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # === Launch 인자 정의 ===
    use_sim_time = LaunchConfiguration("use_sim_time", default="true")
    slam = LaunchConfiguration("slam", default="true")

    # === 패키지 경로 설정 ===
    tb3_nav2_dir = get_package_share_directory("turtlebot3_navigation2")
    tb3_nav2_launch = os.path.join(tb3_nav2_dir, "launch", "navigation2.launch.py")
    tb3_auto_dir = get_package_share_directory("tb3_auto_explore")

    # === slam loofsafe 파라미터 ===
    slam_config_file = os.path.join(
        tb3_auto_dir, "config", "slam_toolbox_loopsafe.yaml"
    )

    # === Navigation2 실행 ===
    tb3_nav2_launch_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(tb3_nav2_launch),
        launch_arguments=[
            ("use_sim_time", use_sim_time),
            ("slam", slam),
        ],
    )

    # === SLAM Toolbox 실행 (slam=True일 때만) ===
    slam_toolbox_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("slam_toolbox"),
                "launch",
                "online_async_launch.py",
            )
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "slam_params_file": slam_config_file,
        }.items(),
        condition=IfCondition(slam),
    )

    # === Frontier Explorer 노드 ===
    explorer = Node(
        package="tb3_auto_explore",
        executable="frontier_explorer",
        name="frontier_explorer",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    # === Launch Declare arguments ===
    declare_use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation (Gazebo/Isaac) clock if true",
    )

    declare_slam_arg = DeclareLaunchArgument(
        "slam", default_value="true", description="Whether to run SLAM Toolbox"
    )

    # === LaunchDescription 구성 ===
    ld = LaunchDescription(
        [
            declare_use_sim_time_arg,
            declare_slam_arg,
            tb3_nav2_launch_include,
            explorer,
            slam_toolbox_launch,  # ✅ 콤마 포함
        ]
    )

    return ld
