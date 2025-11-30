# tb3_auto_explore/initial_pose_from_file.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import Tuple

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseWithCovarianceStamped


class InitialPoseFromFile(Node):
    """
    start_pose.txt 파일을 읽어서 /initialpose 로 초기 위치를 여러 번 퍼블리시.
    - 파일 형식 예:
        start_x: 0.0
        start_y: 0.0
        start_yaw: 0.0   # rad
    """

    def __init__(self) -> None:
        super().__init__("initial_pose_from_file")

        # 파라미터 선언
        self.declare_parameter(
            "start_pose_file",
            "/home/ho/tb3_auto_explore/map/start_pose.txt",
        )
        self.declare_parameter("frame_id", "map")

        # AMCL 뜰 때까지 기다리는 시간 (초)
        self.declare_parameter("initial_delay", 5.0)

        # 퍼블리시 횟수 / 간격
        self.declare_parameter("publish_count", 30)
        self.declare_parameter("publish_interval", 1.0)

        self.start_pose_file = (
            self.get_parameter("start_pose_file").get_parameter_value().string_value
        )
        self.frame_id = (
            self.get_parameter("frame_id").get_parameter_value().string_value
        )
        self.initial_delay = (
            self.get_parameter("initial_delay").get_parameter_value().double_value
        )
        self.publish_count = (
            self.get_parameter("publish_count").get_parameter_value().integer_value
        )
        self.publish_interval = (
            self.get_parameter("publish_interval").get_parameter_value().double_value
        )

        self.publisher_ = self.create_publisher(
            PoseWithCovarianceStamped,
            "/initialpose",
            10,
        )

        # 파일에서 pose 읽기
        try:
            self.x, self.y, self.yaw = self.load_start_pose(self.start_pose_file)
            self.get_logger().info(
                f"Loaded start pose from {self.start_pose_file}: "
                f"x={self.x:.4f}, y={self.y:.4f}, yaw={self.yaw:.4f} rad"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to load start pose: {e}")
            raise

        self._publish_left = self.publish_count

        # 1단계: AMCL이 뜰 시간을 주기 위한 딜레이 타이머
        self.get_logger().info(
            f"Waiting {self.initial_delay} sec before publishing initial pose..."
        )
        self.delay_timer = self.create_timer(self.initial_delay, self.start_publishing)

    # ---------------- 파일 파싱 ----------------
    def load_start_pose(self, path: str) -> Tuple[float, float, float]:
        x = y = yaw = 0.0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("start_x"):
                    _, val = line.split(":", 1)
                    x = float(val.strip())
                elif line.startswith("start_y"):
                    _, val = line.split(":", 1)
                    y = float(val.strip())
                elif line.startswith("start_yaw"):
                    _, val = line.split(":", 1)
                    yaw = float(val.strip())
        return x, y, yaw

    # -------- 딜레이 후 퍼블리시 시작 --------
    def start_publishing(self) -> None:
        self.delay_timer.cancel()
        self.get_logger().info(
            f"Start publishing /initialpose every {self.publish_interval} sec "
            f"({self.publish_count} times)"
        )
        self.timer = self.create_timer(self.publish_interval, self.timer_cb)

    # ------------- 타이머 콜백 -------------
    def timer_cb(self) -> None:
        if self._publish_left <= 0:
            self.get_logger().info("Done publishing initial pose. Shutting down node.")
            self.timer.cancel()
            rclpy.shutdown()
            return

        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        msg.pose.pose.position.x = self.x
        msg.pose.pose.position.y = self.y
        msg.pose.pose.position.z = 0.0

        # yaw -> quaternion
        qz = math.sin(self.yaw / 2.0)
        qw = math.cos(self.yaw / 2.0)
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        # 간단한 covariance
        cov = [0.0] * 36
        cov[0] = 0.25  # x
        cov[7] = 0.25  # y
        cov[35] = (math.radians(10.0)) ** 2  # yaw
        msg.pose.covariance = cov

        self.publisher_.publish(msg)
        self.get_logger().info(
            f"Published initial pose x={self.x:.3f}, y={self.y:.3f}, yaw={self.yaw:.3f} "
            f"(left={self._publish_left-1})"
        )
        self._publish_left -= 1


def main(args=None) -> None:
    rclpy.init(args=args)
    node = InitialPoseFromFile()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
