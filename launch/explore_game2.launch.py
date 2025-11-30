from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os
import yaml  # yaml 모듈 추가

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # 1. 파일에서 초기 좌표 읽어오기
    pose_file_path = "/home/ho/tb3_auto_explore/map/start_pose.yaml"

    # 기본값 (파일 없을 경우 0,0,0)
    init_x, init_y, init_yaw = "0.0", "0.0", "0.0"

    if os.path.exists(pose_file_path):
        try:
            with open(pose_file_path, "r") as f:
                data = yaml.safe_load(f)
                init_x = str(data.get("x", 0.0))
                init_y = str(data.get("y", 0.0))
                init_yaw = str(data.get("yaw", 0.0))
            print(f"Loaded initial pose: x={init_x}, y={init_y}, yaw={init_yaw}")
        except Exception as e:
            print(f"Error reading pose file: {e}")
    else:
        print("Start pose file not found. Using default (0,0,0)")

    # ... (기존 Launch Configuration 설정들) ...
    use_sim_time = LaunchConfiguration("use_sim_time")
    slam = LaunchConfiguration("slam")
    map_yaml = LaunchConfiguration("map")
    found_cubes_file = LaunchConfiguration("found_cubes_file")
    start_pose_file = LaunchConfiguration("start_pose_file")

    # ... (DeclareLaunchArgument 부분은 기존과 동일) ...
    # (생략)
    declare_use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time", default_value="true", description="..."
    )
    declare_slam_arg = DeclareLaunchArgument(
        "slam", default_value="false", description="..."
    )
    declare_map_arg = DeclareLaunchArgument(
        "map",
        default_value="/home/ho/tb3_auto_explore/mission_map.yaml",
        description="...",
    )
    declare_found_cubes_arg = DeclareLaunchArgument(
        "found_cubes_file",
        default_value="/home/ho/tb3_auto_explore/found_cubes.txt",
        description="...",
    )
    declare_start_pose_arg = DeclareLaunchArgument(
        "start_pose_file",
        default_value="/home/ho/tb3_auto_explore/map/start_pose.txt",
        description="Path to start_pose.txt file",
    )

    # ---- TurtleBot3 Nav2 bringup 수정 ----
    tb3_nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("turtlebot3_navigation2"),
                "launch",
                "navigation2.launch.py",
            )
        ),
        launch_arguments=[
            ("use_sim_time", use_sim_time),
            ("slam", slam),
            ("map", map_yaml),
            # [핵심] 여기에 읽어온 좌표를 넘겨줍니다!
            ("x_pose", init_x),
            ("y_pose", init_y),
            ("yaw_pose", init_yaw),
        ],
    )

    # ... (나머지 game2_node 및 return 부분 기존과 동일) ...
    game2_node = Node(
        package="tb3_auto_explore",
        executable="explore_game2",
        name="explore_game2",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "global_frame": "map",
                "found_cubes_file": found_cubes_file,
                # 여기 수정 ↓↓↓
                "red_name": "red-cube",
                "green_name": "green-cube",
                "wait_nav2_sec": 15.0,
            }
        ],
    )

    # ... (생략)
    initial_pose_node = Node(
        package="tb3_auto_explore",
        executable="initial_pose_from_file",
        name="initial_pose_from_file",
        output="screen",
        parameters=[
            {
                "start_pose_file": start_pose_file,
                "frame_id": "map",
                "initial_delay": 5.0,
                "publish_count": 30,
                "publish_interval": 1.0,
                "use_sim_time": use_sim_time,  # <--- [중요] 이 줄을 반드시 추가하세요!
            }
        ],
    )
    # ... (생략)

    return LaunchDescription(
        [
            declare_use_sim_time_arg,
            declare_slam_arg,
            declare_map_arg,
            declare_found_cubes_arg,
            declare_start_pose_arg,  # ★ 추가
            tb3_nav2_launch,
            initial_pose_node,  # ★ 추가
            game2_node,
        ]
    )
