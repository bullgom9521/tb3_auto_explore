import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # === 1. Launch 인자 정의 ===
    use_sim_time = LaunchConfiguration("use_sim_time", default="true")
    slam = LaunchConfiguration("slam", default="true")
    # [추가] RViz 실행 여부 (기본값 True)
    use_rviz = LaunchConfiguration("use_rviz", default="true")

    # 내 패키지 경로
    tb3_auto_dir = get_package_share_directory("tb3_auto_explore")

    # 내 패키지의 burger.yaml 사용
    default_param_file = os.path.join(tb3_auto_dir, "param", "burger.yaml")

    params_file = LaunchConfiguration("params_file")

    # [추가] RViz 설정 파일 경로 (Nav2 기본 설정 사용)
    # 만약 내 패키지에 커스텀 rviz 파일이 있다면 그 경로로 바꿔주면 됩니다.
    nav2_bringup_dir = get_package_share_directory("nav2_bringup")
    rviz_config_file = os.path.join(nav2_bringup_dir, "rviz", "nav2_default_view.rviz")

    # === 2. 패키지 경로 및 설정 ===
    nav2_launch_file = os.path.join(nav2_bringup_dir, "launch", "navigation_launch.py")
    slam_config_file = os.path.join(
        tb3_auto_dir, "config", "slam_toolbox_loopsafe.yaml"
    )

    # === 3. Navigation2 실행 ===
    nav2_launch_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(nav2_launch_file),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "params_file": params_file,
            "autostart": "true",
        }.items(),
    )

    # === 4. SLAM Toolbox 실행 ===
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

    # === 5. Frontier Explorer 노드 ===
    explorer = Node(
        package="tb3_auto_explore",
        executable="frontier_explorer",
        name="frontier_explorer",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
            }
        ],
    )

    # === 6. [추가] RViz2 노드 ===
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config_file],  # 설정 파일 로드
        condition=IfCondition(use_rviz),  # use_rviz가 True일 때만 실행
        parameters=[{"use_sim_time": use_sim_time}],
    )

    # === 7. LaunchDescription 구성 ===
    ld = LaunchDescription()

    ld.add_action(
        DeclareLaunchArgument(
            "use_sim_time", default_value="true", description="Use simulation clock"
        )
    )
    ld.add_action(
        DeclareLaunchArgument("slam", default_value="true", description="Run SLAM")
    )
    # [추가] RViz 인자 선언
    ld.add_action(
        DeclareLaunchArgument(
            "use_rviz", default_value="true", description="Start RViz2 automatically"
        )
    )

    # 파라미터 파일 인자
    ld.add_action(
        DeclareLaunchArgument(
            "params_file",
            default_value=default_param_file,
            description="Full path to the ROS2 parameters file to use for all launched nodes",
        )
    )

    ld.add_action(nav2_launch_include)
    ld.add_action(slam_toolbox_launch)
    ld.add_action(rviz_node)  # RViz 추가
    ld.add_action(explorer)

    return ld
