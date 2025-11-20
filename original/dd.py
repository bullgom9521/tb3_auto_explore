#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
from typing import List, Tuple, Optional

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose

import tf2_ros
from tf2_ros import Buffer, TransformListener


def idx(x: int, y: int, width: int) -> int:
    return y * width + x


def map_to_world(gx: int, gy: int, info) -> Tuple[float, float]:
    ox = info.origin.position.x
    oy = info.origin.position.y
    res = info.resolution
    mx = ox + (gx + 0.5) * res
    my = oy + (gy + 0.5) * res
    return mx, my


class FrontierExplorer(Node):
    """
    Simple frontier-based exploration:
    - Subscribes to /map (OccupancyGrid)
    - Detects frontier clusters (free cell adjacent to unknown)
    - Chooses nearest cluster centroid with clearance
    - Sends Nav2 NavigateToPose goals repeatedly
    """

    def __init__(self) -> None:
        super().__init__("frontier_explorer")

        # Parameters
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("global_frame", "map")
        self.declare_parameter("robot_base_frame", "base_link")
        self.declare_parameter("min_frontier_size", 3)  # cells
        self.declare_parameter("goal_clearance_cells", 1)
        self.declare_parameter("timeout_sec", 60.0)
        self.declare_parameter("wait_nav2_sec", 15.0)
        self.declare_parameter("retry_frontier", True)

        self.map_topic: str = (
            self.get_parameter("map_topic").get_parameter_value().string_value
        )
        self.global_frame: str = (
            self.get_parameter("global_frame").get_parameter_value().string_value
        )
        self.base_frame: str = (
            self.get_parameter("robot_base_frame").get_parameter_value().string_value
        )
        self.min_frontier: int = (
            self.get_parameter("min_frontier_size")
            .get_parameter_value()
            .integer_value
        )
        self.clearance: int = (
            self.get_parameter("goal_clearance_cells")
            .get_parameter_value()
            .integer_value
        )
        self.timeout_sec: float = (
            self.get_parameter("timeout_sec").get_parameter_value().double_value
        )
        self.wait_nav2_sec: float = (
            self.get_parameter("wait_nav2_sec").get_parameter_value().double_value
        )
        self.retry_frontier: bool = (
            self.get_parameter("retry_frontier").get_parameter_value().bool_value
        )

        # Subscriptions
        qos = QoSProfile(depth=5)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.history = HistoryPolicy.KEEP_LAST
        self.map_sub = self.create_subscription(
            OccupancyGrid, self.map_topic, self.on_map, qos
        )

        # TF
        self.buffer: Buffer = Buffer()
        self.tf_listener = TransformListener(self.buffer, self)

        # Nav2 action
        self.nav_client: ActionClient = ActionClient(
            self, NavigateToPose, "navigate_to_pose"
        )
        self.current_goal = None
        self.result_future = None
        self.sent_goals = set()  # set of (gx, gy) grid cells used

        self._have_map: bool = False
        self._last_map: Optional[OccupancyGrid] = None

        self.get_logger().info("FrontierExplorer started. Waiting for Nav2...")
        self.nav_client.wait_for_server(timeout_sec=self.wait_nav2_sec)
        if not self.nav_client.server_is_ready():
            self.get_logger().warn(
                "Nav2 action server not ready yet; will continue once map arrives."
            )

    # ------------------ Callbacks & helpers ------------------

    def on_map(self, grid: OccupancyGrid) -> None:
        self._have_map = True
        self._last_map = grid
        # If idle, plan a new goal
        if self.current_goal is None:
            self.plan_and_go(grid)

    def get_robot_pose(self) -> Optional[Tuple[float, float, float]]:
        try:
            tf = self.buffer.lookup_transform(
                self.global_frame,
                self.base_frame,
                rclpy.time.Time(),
                Duration(seconds=0.5),
            )
            t = tf.transform.translation
            q = tf.transform.rotation
            # yaw from quaternion
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            return (t.x, t.y, yaw)
        except Exception as e:
            self.get_logger().warn(f"Failed to get robot pose: {e}")
            return None

    # ------------------ Frontier detection ------------------

    def find_frontiers(self, grid: OccupancyGrid) -> List[List[Tuple[int, int]]]:
        info = grid.info
        data = grid.data
        w = info.width
        h = info.height

        def in_bounds(x: int, y: int) -> bool:
            return 0 <= x < w and 0 <= y < h

        def is_free(x: int, y: int) -> bool:
            return in_bounds(x, y) and data[idx(x, y, w)] == 0

        def is_unknown(x: int, y: int) -> bool:
            return in_bounds(x, y) and data[idx(x, y, w)] == -1

        # mark frontier cells: free and adjacent to unknown
        is_frontier = [[False] * w for _ in range(h)]
        nbs8 = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
        for y in range(h):
            for x in range(w):
                if is_free(x, y):
                    for dx, dy in nbs8:
                        if is_unknown(x + dx, y + dy):
                            is_frontier[y][x] = True
                            break

        # cluster frontiers (4-connectivity BFS)
        from collections import deque

        visited = [[False] * w for _ in range(h)]
        frontiers: List[List[Tuple[int, int]]] = []
        nbs4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for y in range(h):
            for x in range(w):
                if is_frontier[y][x] and not visited[y][x]:
                    q = deque([(x, y)])
                    visited[y][x] = True
                    comp: List[Tuple[int, int]] = []
                    while q:
                        cx, cy = q.popleft()
                        comp.append((cx, cy))
                        for dx, dy in nbs4:
                            nx, ny = cx + dx, cy + dy
                            if (
                                0 <= nx < w
                                and 0 <= ny < h
                                and not visited[ny][nx]
                                and is_frontier[ny][nx]
                            ):
                                visited[ny][nx] = True
                                q.append((nx, ny))
                    if len(comp) >= self.min_frontier:
                        frontiers.append(comp)

        return frontiers

    def cell_clear(self, x: int, y: int, grid: OccupancyGrid) -> bool:
        """Check a small square window around (x,y) for obstacles (value > 50)."""
        info = grid.info
        w = info.width
        h = info.height
        data = grid.data

        for dy in range(-self.clearance, self.clearance + 1):
            for dx in range(-self.clearance, self.clearance + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    if data[idx(nx, ny, w)] > 65:
                        return False
        return True

    def choose_goal(
        self, frontiers: List[List[Tuple[int, int]]], grid: OccupancyGrid
    ):
        pose = self.get_robot_pose()
        if pose is None:
            return None
        rx, ry, _ = pose
        info = grid.info
        data = grid.data
        w, h = info.width, info.height

        def free(x, y):  # free=0
            return 0 <= x < w and 0 <= y < h and data[idx(x, y, w)] == 0

        best = None
        best_dist = 1e9

        for comp in frontiers:
            # 1) 우선 센트로이드 시도
            cx = int(round(sum(p[0] for p in comp) / len(comp)))
            cy = int(round(sum(p[1] for p in comp) / len(comp)))

            def try_clear_goal(gx, gy, rmax=8):
                # r=0..rmax로 확대하며 클리어 셀 찾기
                if self.cell_clear(gx, gy, grid):
                    return gx, gy
                for r in range(1, rmax + 1):
                    for dy in range(-r, r + 1):
                        for dx in range(-r, r + 1):
                            x, y = gx + dx, gy + dy
                            if 0 <= x < w and 0 <= y < h and free(x, y) and self.cell_clear(x, y, grid):
                                return x, y
                return None

            pick = try_clear_goal(cx, cy, rmax=8)

            # 2) 센트로이드가 안 되면: 프론티어 셀 중 로봇 쪽으로 한두 칸 물러난 자유셀 선택
            if pick is None:
                fx, fy = min(
                    comp,
                    key=lambda p: (map_to_world(p[0], p[1], info)[0] - rx) ** 2 +
                                  (map_to_world(p[0], p[1], info)[1] - ry) ** 2
                )
                # 로봇 방향 단위벡터(격자 공간) 근사
                fmx, fmy = map_to_world(fx, fy, info)
                vx, vy = rx - fmx, ry - fmy
                L = math.hypot(vx, vy) + 1e-6
                ux, uy = vx / L, vy / L
                # 1~2 셀 뒤쪽 후보
                candidates = [
                    (int(round(fx + ux * k)), int(round(fy + uy * k)))
                    for k in (1, 2)
                ]
                for gx, gy in candidates:
                    if free(gx, gy) and self.cell_clear(gx, gy, grid):
                        pick = (gx, gy)
                        break

            if pick is None:
                continue

            gx, gy = pick
            if (gx, gy) in self.sent_goals:
                continue

            mx, my = map_to_world(gx, gy, info)
            dist = math.hypot(mx - rx, my - ry)
            if dist < best_dist:
                best_dist = dist
                best = (mx, my, gx, gy)

        return best


    # ------------------ Nav2 interaction ------------------

    def send_goal(self, mx: float, my: float, yaw: float = 0.0) -> None:
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.header.frame_id = self.global_frame
        goal.pose.pose.position.x = mx
        goal.pose.pose.position.y = my

        # orientation from yaw
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        goal.pose.pose.orientation.z = qz
        goal.pose.pose.orientation.w = qw

        self.get_logger().info(f"Sending goal to ({mx:.2f}, {my:.2f})")
        self.current_goal = self.nav_client.send_goal_async(goal)
        self.current_goal.add_done_callback(self._goal_response)

    def _goal_response(self, fut) -> None:
        goal_handle = fut.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Goal rejected; will try another frontier.")
            self.current_goal = None
            if self._last_map:
                self.plan_and_go(self._last_map)
            return
        self.get_logger().info("Goal accepted, waiting for result...")
        self.result_future = goal_handle.get_result_async()
        self.result_future.add_done_callback(self._goal_result)

    def _goal_result(self, fut) -> None:
        _ = fut.result().result
        status = fut.result().status
        self.get_logger().info(f"Goal finished with status={status}")
        self.current_goal = None
        if self._last_map:
            self.plan_and_go(self._last_map)

    # ------------------ Planner loop ------------------

    def plan_and_go(self, grid: OccupancyGrid) -> None:
        if not self.nav_client.server_is_ready():
            self.get_logger().warn("Nav2 not ready yet; waiting a bit...")
            time.sleep(2.0)
            return

        frontiers = self.find_frontiers(grid)
        if not frontiers:
            self.get_logger().info("No frontiers found. Exploration likely complete. ✅")
            return

        goal = self.choose_goal(frontiers, grid)
        if goal is None:
            self.get_logger().warn("No valid frontier goal found (blocked or repeated).")
            return

        mx, my, gx, gy = goal
        self.sent_goals.add((gx, gy))
        self.send_goal(mx, my)


def main() -> None:
    rclpy.init()
    node = FrontierExplorer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

