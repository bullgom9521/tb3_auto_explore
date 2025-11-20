#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
from typing import List, Tuple, Optional, Dict, Set
from collections import deque

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import OccupancyGrid, Path
from nav2_msgs.action import NavigateToPose
from rosgraph_msgs.msg import Clock

# 서비스 호출을 위한 임포트
from slam_toolbox.srv import SaveMap, Pause

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
    def __init__(self) -> None:
        super().__init__("frontier_explorer")

        # ===== Parameters =====
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("global_frame", "map")
        self.declare_parameter("robot_base_frame", "base_link")
        self.declare_parameter("min_frontier_size", 5)  # 기본값 7
        self.declare_parameter("goal_clearance_cells", 1)
        self.declare_parameter("goal_timeout_sec", 30.0)
        self.declare_parameter("blacklist_clear_radius", 0.5)
        self.declare_parameter("max_goal_attempts_per_frontier", 2)

        # 맵 저장 이름 파라미터
        self.declare_parameter("map_name", "my_map")

        # Read Params
        self.map_topic = self.get_parameter("map_topic").value
        self.global_frame = self.get_parameter("global_frame").value
        self.base_frame = self.get_parameter("robot_base_frame").value

        # [변수] 파라미터 값을 변수로 저장 (동적 변경을 위해)
        self.min_frontier = self.get_parameter("min_frontier_size").value

        self.clearance = self.get_parameter("goal_clearance_cells").value
        self.goal_timeout_sec = self.get_parameter("goal_timeout_sec").value
        self.blacklist_clear_radius_m = self.get_parameter(
            "blacklist_clear_radius"
        ).value
        self.max_goal_attempts = self.get_parameter(
            "max_goal_attempts_per_frontier"
        ).value
        self.map_name_val = self.get_parameter("map_name").value

        # ===== Subscriptions & Action Client =====
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid, self.map_topic, self.on_map, qos
        )
        self.clock_sub = self.create_subscription(Clock, "/clock", self.on_clock, 10)
        self.plan_sub = self.create_subscription(Path, "/plan", self.on_plan, 10)

        self.buffer = Buffer()
        self.tf_listener = TransformListener(self.buffer, self)
        self.nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")

        # SLAM Toolbox 제어 클라이언트
        self.save_map_client = self.create_client(SaveMap, "/slam_toolbox/save_map")
        self.pause_slam_client = self.create_client(
            Pause, "/slam_toolbox/pause_new_measurements"
        )

        # ===== State =====
        self._have_map = False
        self._last_map: Optional[OccupancyGrid] = None
        self._last_map_time = self.get_clock().now()

        self.current_goal_future = None
        self.result_future = None
        self._goal_handle = None

        self.current_target_grid: Optional[Tuple[int, int]] = None
        self.goal_start_time = None

        self.force_long_range = False
        self.path_update_count = 0
        self.no_valid_goal_count = 0

        # [추가] 'No frontiers found' 연속 횟수 카운트
        self.no_frontier_count = 0

        self.sent_goals_cells: Set[Tuple[int, int]] = set()
        self.goal_attempts: Dict[Tuple[int, int], int] = {}
        self.blacklist_cells: Set[Tuple[int, int]] = set()

        self.start_pose = None
        self.going_home = False
        self.start_time_sec = None
        self.time_limit_sec = 240.0

        self.timer = self.create_timer(1.0, self.watchdog_cb)
        self.get_logger().info(
            f"Explorer Started. [Auto-Save & Return] Limit: {self.time_limit_sec}s"
        )

    def on_map(self, grid: OccupancyGrid) -> None:
        self._have_map = True
        self._last_map = grid
        self._last_map_time = self.get_clock().now()

        if self.start_pose is None:
            pose = self.get_robot_pose()
            if pose:
                self.start_pose = pose
                self.get_logger().info(f"[Home] Start pose saved: {pose}")

        if self.current_goal_future is None and not self.going_home:
            self.plan_and_go(grid)

    def on_plan(self, msg: Path) -> None:
        if self._goal_handle is not None and not self.going_home:
            self.path_update_count += 1
            if self.path_update_count >= 15:
                self.get_logger().warn(
                    f"[Trigger] Path replanned 15 times! Cancelling."
                )
                if self.current_target_grid:
                    self.blacklist_cells.add(self.current_target_grid)
                self.force_long_range = False
                try:
                    self._goal_handle.cancel_goal_async()
                except:
                    pass
                self.path_update_count = 0
                self.current_goal_future = None
                self._goal_handle = None

    def on_clock(self, msg: Clock) -> None:
        now_sec = msg.clock.sec + msg.clock.nanosec * 1e-9
        if self.start_time_sec is None:
            self.start_time_sec = now_sec
            return
        elapsed = now_sec - self.start_time_sec
        if not self.going_home and self.start_pose and elapsed >= self.time_limit_sec:
            self.get_logger().info(
                f"[Timer] Time Limit ({elapsed:.1f}s). RETURNING HOME."
            )
            self.go_home()

    def watchdog_cb(self) -> None:
        now = self.get_clock().now()
        if self._have_map and self.current_goal_future is None and not self.going_home:
            delta = (now - self._last_map_time).nanoseconds * 1e-9
            if delta > 5.0:
                self.plan_and_go(self._last_map, use_relaxation=True)

        if (
            self._goal_handle is not None
            and self.goal_start_time is not None
            and not self.going_home
        ):
            delta = (now - self.goal_start_time).nanoseconds * 1e-9
            if delta > self.goal_timeout_sec:
                self.get_logger().warn(f"[Watchdog] Timeout ({delta:.1f}s).")
                if self.current_target_grid:
                    self.blacklist_cells.add(self.current_target_grid)
                self.force_long_range = False
                try:
                    self._goal_handle.cancel_goal_async()
                except:
                    pass
                self.current_goal_future = None
                self._goal_handle = None

    def get_robot_pose(self):
        try:
            tf = self.buffer.lookup_transform(
                self.global_frame,
                self.base_frame,
                rclpy.time.Time(),
                Duration(seconds=1.0),
            )
            t = tf.transform.translation
            q = tf.transform.rotation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            return (t.x, t.y, yaw)
        except:
            return None

    def save_and_pause_slam(self):
        # 1. 맵 저장
        if self.save_map_client.wait_for_service(timeout_sec=1.0):
            req = SaveMap.Request()
            req.name.data = self.map_name_val
            future = self.save_map_client.call_async(req)
            self.get_logger().info(f"Requesting Map Save: {self.map_name_val}")
        else:
            self.get_logger().warn("SaveMap service not available!")

        # 2. 매핑 일시정지
        if self.pause_slam_client.wait_for_service(timeout_sec=1.0):
            req = Pause.Request()
            self.pause_slam_client.call_async(req)
            self.get_logger().info("Requesting SLAM Pause (Stop Mapping)")
        else:
            self.get_logger().warn("Pause SLAM service not available!")

    def go_home(self):
        if not self.start_pose:
            self.get_logger().error("Cannot go home: Start pose not saved yet.")
            return

        if self.going_home:
            return

        if self._goal_handle:
            try:
                self._goal_handle.cancel_goal_async()
            except:
                pass

        self.going_home = True

        self.get_logger().info(
            ">>> FINISHING EXPLORATION: Saving Map & Pausing SLAM... <<<"
        )
        self.save_and_pause_slam()

        x, y, yaw = self.start_pose
        self.send_goal(x, y, yaw)

    def find_frontiers(
        self, grid: OccupancyGrid, min_size: int
    ) -> List[List[Tuple[int, int]]]:
        info = grid.info
        data = grid.data
        w = info.width
        h = info.height

        def is_frontier_cell(x, y):
            if not (0 <= x < w and 0 <= y < h):
                return False
            if data[idx(x, y, w)] != 0:
                return False
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    if data[idx(nx, ny, w)] == -1:
                        return True
            return False

        visited = set()
        frontiers = []
        for y in range(h):
            for x in range(w):
                if (x, y) not in visited and is_frontier_cell(x, y):
                    q = deque([(x, y)])
                    visited.add((x, y))
                    cluster = []
                    while q:
                        cx, cy = q.popleft()
                        cluster.append((cx, cy))
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx, ny = cx + dx, cy + dy
                            if (0 <= nx < w and 0 <= ny < h) and (
                                (nx, ny) not in visited
                            ):
                                if is_frontier_cell(nx, ny):
                                    visited.add((nx, ny))
                                    q.append((nx, ny))
                    if len(cluster) >= min_size:
                        frontiers.append(cluster)
        return frontiers

    def choose_goal(self, frontiers, grid, relax_level=0):
        pose = self.get_robot_pose()
        if not pose:
            return None
        rx, ry, _ = pose
        info = grid.info
        data = grid.data
        w = info.width
        req_clearance = max(0, self.clearance - relax_level)
        occupancy_threshold = 70 if relax_level < 2 else 95

        def check_clear(cx, cy):
            for dy in range(-req_clearance, req_clearance + 1):
                for dx in range(-req_clearance, req_clearance + 1):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < info.height:
                        if data[idx(nx, ny, w)] > occupancy_threshold:
                            return False
            return True

        best_cand = None
        best_score = 1e9
        for comp in frontiers:
            cx = sum(c[0] for c in comp) // len(comp)
            cy = sum(c[1] for c in comp) // len(comp)
            if not check_clear(cx, cy):
                valid_points = [p for p in comp if check_clear(p[0], p[1])]
                if not valid_points:
                    continue
                cx, cy = min(
                    valid_points, key=lambda p: (p[0] - rx) ** 2 + (p[1] - ry) ** 2
                )
            if self.is_blacklisted(cx, cy, grid):
                continue
            if (cx, cy) in self.sent_goals_cells:
                continue
            mx, my = map_to_world(cx, cy, info)
            dist = math.hypot(mx - rx, my - ry)
            size = len(comp)
            if self.force_long_range:
                score = -dist
            else:
                score = dist - (size * 0.05)
            if self.current_target_grid:
                curr_gx, curr_gy = self.current_target_grid
                dx_g = abs(cx - curr_gx)
                dy_g = abs(cy - curr_gy)
                if dx_g < 20 and dy_g < 20:
                    score -= 2.0
            if score < best_score:
                best_score = score
                best_cand = (mx, my, cx, cy)
        return best_cand

    def is_blacklisted(self, gx, gy, grid):
        if (gx, gy) in self.blacklist_cells:
            return True
        res = grid.info.resolution
        r_cells = int(self.blacklist_clear_radius_m / res)
        for bx, by in self.blacklist_cells:
            if abs(bx - gx) <= r_cells and abs(by - gy) <= r_cells:
                return True
        return False

    def plan_and_go(self, grid, use_relaxation=False, recursion_depth=0):
        if recursion_depth > 5:
            self.get_logger().warn("Max recursion reached.")
            self.no_valid_goal_count += 1
            if self.no_valid_goal_count >= 20:
                self.get_logger().warn("Recursion limit hit 20+. RETURNING HOME.")
                self.go_home()
            return

        if not self.nav_client.server_is_ready() or self.going_home:
            return

        min_f = max(1, self.min_frontier - (1 if use_relaxation else 0))
        frontiers = self.find_frontiers(grid, min_f)

        if not frontiers:
            self.get_logger().info("No frontiers found.")

            # [추가 로직] 연속 5회 'No frontiers' 발생 시 처리
            self.no_frontier_count += 1
            if self.no_frontier_count >= 5:
                if self.min_frontier > 4:
                    self.get_logger().warn(">>> Lowering min_frontier_size to 4! <<<")
                    self.min_frontier = 4
                    self.no_frontier_count = 0  # 카운트 리셋 후 4로 다시 시도
                else:
                    # 이미 5인데도 5번 연속 못 찾았다면
                    self.get_logger().warn(
                        ">>> No frontiers left even with size 4. GOING HOME <<<"
                    )
                    self.go_home()
            return
        else:
            # 프론티어를 찾았다면 카운트 초기화
            self.no_frontier_count = 0

        goal = self.choose_goal(
            frontiers, grid, relax_level=(2 if use_relaxation else 0)
        )
        if not goal:
            if not use_relaxation:
                self.get_logger().info("Retry with relaxation...")
                self.plan_and_go(
                    grid, use_relaxation=True, recursion_depth=recursion_depth + 1
                )
            else:
                if self.blacklist_cells:
                    self.get_logger().warn("Resetting Blacklist & Retrying!")
                    self.blacklist_cells.clear()
                    self.sent_goals_cells.clear()
                    self.plan_and_go(
                        grid, use_relaxation=True, recursion_depth=recursion_depth + 1
                    )
                else:
                    self.get_logger().warn("No valid goal found.")
                    self.no_valid_goal_count += 1
                    if self.no_valid_goal_count >= 20:
                        self.get_logger().warn("Failed 20+. RETURNING HOME.")
                        self.go_home()
                        return
                    if self.force_long_range:
                        self.force_long_range = False
            return
        mx, my, gx, gy = goal
        self.no_valid_goal_count = 0
        if self.force_long_range:
            self.get_logger().info(">>> ESCAPE MODE <<<")
            self.force_long_range = False
        attempts = self.goal_attempts.get((gx, gy), 0)
        if attempts >= self.max_goal_attempts:
            self.blacklist_cells.add((gx, gy))
            self.plan_and_go(
                grid, use_relaxation=True, recursion_depth=recursion_depth + 1
            )
            return
        self.goal_attempts[(gx, gy)] = attempts + 1
        self.sent_goals_cells.add((gx, gy))
        self.current_target_grid = (gx, gy)
        self.send_goal(mx, my)

    def send_goal(self, x, y, yaw=0.0):
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = self.global_frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        goal.pose.pose.orientation.z = qz
        goal.pose.pose.orientation.w = qw
        if self.going_home:
            self.get_logger().info(f"GOING HOME -> ({x:.2f}, {y:.2f})")
        else:
            self.get_logger().info(f"Going to ({x:.2f}, {y:.2f})")
        self.path_update_count = 0
        self.current_goal_future = self.nav_client.send_goal_async(goal)
        self.current_goal_future.add_done_callback(self._goal_response)
        self.goal_start_time = self.get_clock().now()

    def _goal_response(self, fut):
        h = fut.result()
        if not h or not h.accepted:
            self.get_logger().warn("Goal rejected.")
            self.current_goal_future = None
            return
        self._goal_handle = h
        self.result_future = h.get_result_async()
        self.result_future.add_done_callback(self._goal_result)

    def _goal_result(self, fut):
        status = fut.result().status
        self.get_logger().info(f"Goal finished: {status}")
        self.current_goal_future = None
        self._goal_handle = None
        self.current_target_grid = None
        self.path_update_count = 0
        if self.going_home:
            if status == 4:
                self.get_logger().info("Mission Complete (Arrived Home).")
            else:
                self.get_logger().warn(f"Home return finished with status {status}.")
        elif self._last_map:
            self.plan_and_go(self._last_map)


def main():
    rclpy.init()
    node = FrontierExplorer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
