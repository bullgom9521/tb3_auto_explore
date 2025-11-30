#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
import os
import yaml
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from collections import deque

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.action import ActionClient
from std_msgs.msg import Bool

from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import OccupancyGrid, Path
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import ClearEntireCostmap
from rosgraph_msgs.msg import Clock
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

        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("global_frame", "map")
        self.declare_parameter("robot_base_frame", "base_link")
        self.declare_parameter("min_frontier_size", 3)
        self.declare_parameter("goal_clearance_cells", 1)
        self.declare_parameter("goal_timeout_sec", 30.0)
        self.declare_parameter("blacklist_clear_radius", 0.5)
        self.declare_parameter("max_goal_attempts_per_frontier", 2)

        # [설정] 맵 저장 경로
        self.save_dir = "/home/ho/tb3_auto_explore/map"
        self.map_name_base = "mission_map"
        self.full_map_path = os.path.join(self.save_dir, self.map_name_base)

        # 큐브 좌표 파일 경로
        self.cube_log_path = "/home/ho/tb3_auto_explore/found_cubes.txt"

        # [추가] CSV 저장 경로
        self.csv_output_path = os.path.join(self.save_dir, "found_cubes.csv")

        # [NEW] 시작 좌표 저장 파일 경로 추가
        self.start_pose_file_path = os.path.join(self.save_dir, "start_pose.txt")

        self.map_topic = self.get_parameter("map_topic").value
        self.global_frame = self.get_parameter("global_frame").value
        self.base_frame = self.get_parameter("robot_base_frame").value
        self.min_frontier = self.get_parameter("min_frontier_size").value
        self.clearance = self.get_parameter("goal_clearance_cells").value
        self.goal_timeout_sec = self.get_parameter("goal_timeout_sec").value
        self.blacklist_clear_radius_m = self.get_parameter(
            "blacklist_clear_radius"
        ).value
        self.max_goal_attempts = self.get_parameter(
            "max_goal_attempts_per_frontier"
        ).value

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
        self.create_subscription(Bool, "/emergency_stop", self.stop_signal_callback, 10)

        self.is_paused = False

        self.buffer = Buffer()
        self.tf_listener = TransformListener(self.buffer, self)
        self.nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.save_map_client = self.create_client(SaveMap, "/slam_toolbox/save_map")
        self.pause_slam_client = self.create_client(
            Pause, "/slam_toolbox/pause_new_measurements"
        )

        self.clear_global_costmap_client = self.create_client(
            ClearEntireCostmap, "/global_costmap/clear_entirely_global_costmap"
        )
        self.clear_local_costmap_client = self.create_client(
            ClearEntireCostmap, "/local_costmap/clear_entirely_local_costmap"
        )

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
        self.no_frontier_count = 0
        self.sent_goals_cells: Set[Tuple[int, int]] = set()
        self.goal_attempts: Dict[Tuple[int, int], int] = {}
        self.blacklist_cells: Set[Tuple[int, int]] = set()
        self.start_pose = None
        self.going_home = False
        self.start_time_sec = None
        self.time_limit_sec = 240.0

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.timer = self.create_timer(1.0, self.watchdog_cb)
        self.get_logger().info(f"Explorer Started. Output Dir: {self.save_dir}")

    def clear_costmaps(self):
        self.get_logger().warn("!!! CLEARING COSTMAPS TO FIX PATH PLANNING !!!")
        if self.clear_global_costmap_client.service_is_ready():
            req = ClearEntireCostmap.Request()
            self.clear_global_costmap_client.call_async(req)

        if self.clear_local_costmap_client.service_is_ready():
            req = ClearEntireCostmap.Request()
            self.clear_local_costmap_client.call_async(req)

    def stop_signal_callback(self, msg: Bool):
        if self.going_home:
            return

        if msg.data:
            if not self.is_paused:
                self.is_paused = True
                self.get_logger().warn(">> PAUSE SIGNAL RECEIVED. Cancelling Goal. <<")
                if self._goal_handle:
                    try:
                        self._goal_handle.cancel_goal_async()
                    except:
                        pass
                self.current_goal_future = None
                self._goal_handle = None
        else:
            if self.is_paused:
                self.is_paused = False
                self.get_logger().info(">> RESUME SIGNAL RECEIVED. <<")
                if self._last_map:
                    self.plan_and_go(self._last_map)

    def on_map(self, grid: OccupancyGrid) -> None:
        if self.is_paused:
            return
        self._have_map = True
        self._last_map = grid
        self._last_map_time = self.get_clock().now()

        if self.start_pose is None:
            pose = self.get_robot_pose()
            if pose:
                self.start_pose = pose
                self.get_logger().info(f"[Home] Start pose saved in Memory: {pose}")

                # [NEW] 시작 좌표 파일로 저장하는 로직
                try:
                    with open(self.start_pose_file_path, "w") as f:
                        f.write(f"start_x: {pose[0]}\n")
                        f.write(f"start_y: {pose[1]}\n")
                        f.write(f"start_yaw: {pose[2]}\n")
                    self.get_logger().info(
                        f"[Home] Start pose saved to file: {self.start_pose_file_path}"
                    )
                except Exception as e:
                    self.get_logger().error(f"Failed to write start pose to file: {e}")

        if self.current_goal_future is None and not self.going_home:
            self.plan_and_go(grid)

    def on_plan(self, msg: Path) -> None:
        if self.going_home:
            return

        if self._goal_handle is not None:
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
        if self.is_paused:
            return
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
        if self.is_paused:
            return
        if self.going_home:
            return

        now = self.get_clock().now()
        if self._have_map and self.current_goal_future is None:
            delta = (now - self._last_map_time).nanoseconds * 1e-9
            if delta > 5.0:
                self.plan_and_go(self._last_map, use_relaxation=True)
        if self._goal_handle is not None and self.goal_start_time is not None:
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
        if self.pause_slam_client.wait_for_service(timeout_sec=1.0):
            req = Pause.Request()
            self.pause_slam_client.call_async(req)
            self.get_logger().info("Requesting SLAM Pause...")

        if self.save_map_client.wait_for_service(timeout_sec=1.0):
            req = SaveMap.Request()
            req.name.data = self.full_map_path
            future = self.save_map_client.call_async(req)
            self.get_logger().info(f"Requesting Map Save to: {self.full_map_path}")
            future.add_done_callback(self.on_map_save_complete)
        else:
            self.get_logger().warn("SaveMap service not available!")

    def on_map_save_complete(self, future):
        try:
            response = future.result()
            self.get_logger().info("Map saved successfully. Processing image...")
            time.sleep(2.0)
            self.annotate_map_with_locations()
        except Exception as e:
            self.get_logger().error(f"Failed to save map: {str(e)}")

    def annotate_map_with_locations(self):
        self.get_logger().info(">>> STARTING MAP ANNOTATION (Clean Ver) <<<")

        pgm_path = self.full_map_path + ".pgm"
        yaml_path = self.full_map_path + ".yaml"

        if not os.path.exists(pgm_path) or not os.path.exists(yaml_path):
            self.get_logger().error("Map files not found!")
            return

        with open(yaml_path, "r") as f:
            map_info = yaml.safe_load(f)

        resolution = map_info["resolution"]
        origin = map_info["origin"]
        origin_x = origin[0]
        origin_y = origin[1]

        img = cv2.imread(pgm_path)
        if img is None:
            self.get_logger().error("Failed to load PGM image.")
            return

        height, width, _ = img.shape

        def world_to_pixel(wx, wy):
            px = int((wx - origin_x) / resolution)
            py = int(height - 1 - (wy - origin_y) / resolution)
            return px, py

        # Home 표시
        if self.start_pose:
            sx, sy, _ = self.start_pose
            px, py = world_to_pixel(sx, sy)
            pink_color = (203, 192, 255)
            cv2.circle(img, (px, py), 2, pink_color, -1)
            self.get_logger().info(f"Marked HOME dot at ({px}, {py})")

        # 큐브 표시
        if os.path.exists(self.cube_log_path):
            with open(self.cube_log_path, "r") as f:
                lines = f.readlines()

            for line in lines[1:]:
                parts = line.strip().split(",")
                if len(parts) < 4:
                    continue

                target_name = parts[1].strip()
                try:
                    cx = float(parts[2].strip())
                    cy = float(parts[3].strip())
                except ValueError:
                    continue

                px, py = world_to_pixel(cx, cy)

                color = (255, 255, 255)
                if "red" in target_name:
                    color = (0, 0, 255)
                elif "blue" in target_name:
                    color = (255, 0, 0)
                elif "green" in target_name:
                    color = (0, 255, 0)
                elif "yellow" in target_name:
                    color = (0, 255, 255)

                cv2.circle(img, (px, py), 2, color, -1)
                self.get_logger().info(f"Marked dot for {target_name} at ({px}, {py})")

        # 최종 이미지 저장
        output_path = os.path.join(self.save_dir, "final_mission_map.png")
        cv2.imwrite(output_path, img)
        self.get_logger().info(f"SUCCESS: Clean final map saved to {output_path}")

        # [추가] CSV 변환 실행
        self.export_cubes_to_csv()

    # [새로 추가된 함수] TXT -> CSV 변환 (수식 적용)
    def export_cubes_to_csv(self):
        self.get_logger().info(">>> Exporting found_cubes.csv with formula <<<")

        if not os.path.exists(self.cube_log_path):
            self.get_logger().warn("No found_cubes.txt to convert.")
            return

        try:
            with open(self.cube_log_path, "r") as f_in, open(
                self.csv_output_path, "w"
            ) as f_out:
                lines = f_in.readlines()
                # 헤더 작성
                f_out.write("Timestamp,Target,X,Y\n")

                for line in lines[1:]:  # 헤더 건너뛰기
                    parts = line.strip().split(",")
                    if len(parts) < 4:
                        continue

                    timestamp = parts[0].strip()
                    target = parts[1].strip()
                    try:
                        raw_x = float(parts[2].strip())
                        raw_y = float(parts[3].strip())

                        # [요청하신 수식 적용]
                        # |X| * 10 / 4 -> 반올림 -> 정수
                        new_x = int(round(abs(raw_x) * 10 / 4))
                        new_y = int(round(abs(raw_y) * 10 / 4))

                        f_out.write(f"{timestamp},{target},{new_x},{new_y}\n")
                    except ValueError:
                        continue

            self.get_logger().info(
                f"Successfully exported CSV to {self.csv_output_path}"
            )

        except Exception as e:
            self.get_logger().error(f"Failed to export CSV: {e}")

    def go_home(self):
        if not self.start_pose:
            self.get_logger().error("Cannot go home: Start pose not saved yet.")
            return

        self.clear_costmaps()

        if self.going_home:
            pass
        else:
            if self._goal_handle:
                try:
                    self._goal_handle.cancel_goal_async()
                except:
                    pass
            self.going_home = True
            self.get_logger().info(
                ">>> FINISHING EXPLORATION: Going Home (FORCE MODE) <<<"
            )

        x, y, yaw = self.start_pose
        self.send_goal(x, y, yaw)

    def find_frontiers(
        self, grid: OccupancyGrid, min_size: int
    ) -> List[List[Tuple[int, int]]]:
        # [원복 완료] 안전장치(margin) 제거된 기존 코드
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
        if self.is_paused:
            return
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
            self.no_frontier_count += 1
            if self.no_frontier_count >= 5:
                if self.min_frontier > 2:
                    self.get_logger().warn(">>> Lowering min_frontier_size to 2! <<<")
                    self.min_frontier = 2
                    self.no_frontier_count = 0
                else:
                    self.get_logger().warn(">>> No frontiers left. GOING HOME <<<")
                    self.go_home()
            return
        else:
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
            if self.going_home:
                self.get_logger().warn(
                    "Home Goal Rejected immediately! Clearing Map & Retrying..."
                )
                self.clear_costmaps()
                time.sleep(1.0)
                self.go_home()
            return
        self._goal_handle = h
        self.result_future = h.get_result_async()
        self.result_future.add_done_callback(self._goal_result)

    def _goal_result(self, fut):
        if self.is_paused:
            return
        result = fut.result()
        status = result.status
        self.get_logger().info(f"Goal finished: {status}")
        self.current_goal_future = None
        self._goal_handle = None
        self.current_target_grid = None
        self.path_update_count = 0

        if self.going_home:
            if status == 4:
                self.get_logger().info(
                    "Mission Complete (Arrived Home). Saving Map & Pausing SLAM."
                )
                self.save_and_pause_slam()
            else:
                self.get_logger().warn(
                    f"Home return failed (Status: {status}). FORCE RETRYING..."
                )
                self.clear_costmaps()
                time.sleep(1.5)
                if self.start_pose:
                    x, y, yaw = self.start_pose
                    self.send_goal(x, y, yaw)
                else:
                    self.get_logger().error("Start pose lost! Saving anyway.")
                    self.save_and_pause_slam()

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
