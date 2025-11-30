# tb3_auto_explore/explore_game2.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
from typing import Dict, Tuple, Optional

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped


class ExploreGame2(Node):
    """
    - found_cubes.txt 에서 red / green 좌표를 읽어서
      1) red -> 2) green 순서로 Nav2 goal을 보내는 노드.
    - 좌표는 map 프레임 기준 (x, y) 라고 가정.
    """

    def __init__(self) -> None:
        super().__init__("explore_game2")

        # ===== Parameters =====
        # launch에서 넘길 수 있게 파라미터로 선언
        self.declare_parameter("global_frame", "map")
        self.declare_parameter(
            "found_cubes_file",
            "/home/ho/tb3_auto_explore/found_cubes.txt",  # 필요시 바꿔도 됨
        )
        # 기존 yellow_name -> red_name 으로 변경
        self.declare_parameter("red_name", "red-cube")
        self.declare_parameter("green_name", "green-cube")
        self.declare_parameter("wait_nav2_sec", 15.0)

        self.global_frame: str = (
            self.get_parameter("global_frame").get_parameter_value().string_value
        )
        self.found_cubes_file: str = (
            self.get_parameter("found_cubes_file").get_parameter_value().string_value
        )
        self.red_name: str = (
            self.get_parameter("red_name").get_parameter_value().string_value
        )
        self.green_name: str = (
            self.get_parameter("green_name").get_parameter_value().string_value
        )
        self.wait_nav2_sec: float = (
            self.get_parameter("wait_nav2_sec").get_parameter_value().double_value
        )

        # ===== Nav2 Action Client =====
        self.nav_client: ActionClient = ActionClient(
            self, NavigateToPose, "navigate_to_pose"
        )
        self.current_goal_result_future = None
        self.goal_sequence = []  # [(name, (x,y)), ...]
        self.current_goal_index = 0

        # 서버 기다리기
        self.get_logger().info("Waiting for Nav2 action server (navigate_to_pose)...")
        if not self.nav_client.wait_for_server(timeout_sec=self.wait_nav2_sec):
            self.get_logger().error("Nav2 action server not available. Abort mission.")
            return

        # 타겟 좌표 읽기
        targets = self.load_targets(self.found_cubes_file)
        if targets is None:
            self.get_logger().error("Failed to load targets. Abort mission.")
            return

        if self.red_name not in targets:
            self.get_logger().error(
                f'No "{self.red_name}" entry in found_cubes file. Abort.'
            )
            return
        if self.green_name not in targets:
            self.get_logger().error(
                f'No "{self.green_name}" entry in found_cubes file. Abort.'
            )
            return

        # 순서: red -> green
        self.goal_sequence = [
            (self.red_name, targets[self.red_name]),
            (self.green_name, targets[self.green_name]),
        ]

        self.get_logger().info(
            f"Mission sequence: "
            f"{self.red_name} {targets[self.red_name]} "
            f"-> {self.green_name} {targets[self.green_name]}"
        )

        # 미션 시작 (red로 이동)
        self.send_next_goal()

    # ------------------------------------------------------------------
    # 파일 파싱
    # ------------------------------------------------------------------
    def load_targets(self, path: str) -> Optional[Dict[str, Tuple[float, float]]]:
        """
        found_cubes.txt CSV 파일을 읽어서
        {target_name: (x, y)} 딕셔너리 리턴
        """
        try:
            targets: Dict[str, Tuple[float, float]] = {}
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row.get("Target", "").strip()
                    try:
                        x = float(row.get("X", "nan"))
                        y = float(row.get("Y", "nan"))
                    except ValueError:
                        self.get_logger().warn(f"Invalid coordinate in row: {row}")
                        continue
                    # 같은 색이 여러 번 기록되면 "가장 마지막 것"으로 덮어쓴다.
                    targets[name] = (x, y)

            if not targets:
                self.get_logger().error(f"No valid rows in {path}")
                return None

            self.get_logger().info(f"Loaded targets from {path}: {targets}")
            return targets

        except FileNotFoundError:
            self.get_logger().error(f"found_cubes file not found: {path}")
            return None
        except Exception as e:
            self.get_logger().error(f"Error reading found_cubes file: {e}")
            return None

    # ------------------------------------------------------------------
    # Nav2 goal 전송 관련
    # ------------------------------------------------------------------
    def send_next_goal(self) -> None:
        """
        goal_sequence에서 다음 목표를 Nav2에 전송.
        """
        if self.current_goal_index >= len(self.goal_sequence):
            self.get_logger().info("All cube goals finished. Mission complete ✅")
            return

        name, (x, y) = self.goal_sequence[self.current_goal_index]
        self.get_logger().info(
            f"[{self.current_goal_index+1}/{len(self.goal_sequence)}] "
            f'Going to {name} at (x={x:.3f}, y={y:.3f}) in frame "{self.global_frame}"'
        )

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.header.frame_id = self.global_frame
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        # 방향은 중요하지 않다고 가정 → yaw = 0
        goal_msg.pose.pose.orientation.w = 1.0

        send_future = self.nav_client.send_goal_async(goal_msg)
        send_future.add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, future) -> None:
        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().warn("Goal rejected by Nav2. Try next goal.")
            # 그냥 다음 목표로 넘어간다.
            self.current_goal_index += 1
            self.send_next_goal()
            return

        self.get_logger().info("Goal accepted, waiting for result...")
        self.current_goal_result_future = goal_handle.get_result_async()
        self.current_goal_result_future.add_done_callback(self._goal_result_cb)

    def _goal_result_cb(self, future) -> None:
        try:
            result = future.result()
            status = result.status
        except Exception as e:
            self.get_logger().error(f"Exception while waiting goal result: {e}")
            status = -1

        self.get_logger().info(f"Goal finished with status={status}")
        # 다음 목표로 넘어가기
        self.current_goal_index += 1
        self.send_next_goal()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ExploreGame2()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
