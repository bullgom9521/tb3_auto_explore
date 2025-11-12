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

# NEW: parameter service messages
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter as ParamMsg, ParameterValue
from rcl_interfaces.msg import ParameterType as PT


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
    Simple frontier-based exploration with Nav2 param injection:
    - Subscribes to /map (OccupancyGrid)
    - Detects frontier clusters (free cell adjacent to unknown)
    - Chooses nearest cluster centroid with clearance
    - Sends Nav2 NavigateToPose goals repeatedly
    - NEW: on startup, pushes Nav2 tuning params via SetParameters
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

        # NEW: push Nav2 params (same values as your YAML)
        self.push_nav2_params()

        if not self.nav_client.server_is_ready():
            self.get_logger().warn(
                "Nav2 action server not ready yet; will continue once map arrives."
            )

    # ------------------ NEW: Nav2 param injection ------------------

    def _make_param(self, name: str, value) -> ParamMsg:
        v = ParameterValue()
        if isinstance(value, bool):
            v.type = PT.PARAMETER_BOOL
            v.bool_value = value
        elif isinstance(value, int):
            v.type = PT.PARAMETER_INTEGER
            v.integer_value = value
        elif isinstance(value, float):
            v.type = PT.PARAMETER_DOUBLE
            v.double_value = value
        elif isinstance(value, str):
            v.type = PT.PARAMETER_STRING
            v.string_value = value
        elif isinstance(value, (list, tuple)):
            # assume list of strings (for controller_plugins)
            if all(isinstance(x, str) for x in value):
                v.type = PT.PARAMETER_STRING_ARRAY
                v.string_array_value = list(value)
            elif all(isinstance(x, float) for x in value):
                v.type = PT.PARAMETER_DOUBLE_ARRAY
                v.double_array_value = list(value)
            elif all(isinstance(x, int) for x in value):
                v.type = PT.PARAMETER_INTEGER_ARRAY
                v.integer_array_value = list(value)
            else:
                raise ValueError(f"Unsupported array type for param {name}: {value}")
        else:
            raise ValueError(f"Unsupported param type for {name}: {type(value)}")

        p = ParamMsg()
        p.name = name
        p.value = v
        return p

    def _call_set_params(self, srv_name: str, kv: dict) -> None:
        """Call /<node>/set_parameters with a dict of {name: value}."""
        cli = self.create_client(SetParameters, f"{srv_name}/set_parameters")
        if not cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn(f"[Param] Service not available: {srv_name}/set_parameters")
            return
        req = SetParameters.Request()
        req.parameters = [self._make_param(k, v) for k, v in kv.items()]
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        if fut.result() is None:
            self.get_logger().warn(f"[Param] Failed to set on {srv_name} (timeout)")
        else:
            # Check results per-parameter if needed
            self.get_logger().info(f"[Param] Applied {len(req.parameters)} params to {srv_name}")

    def push_nav2_params(self) -> None:
        """
        Push exactly the parameters you used during the hotfix:
        - controller_server FollowPath tuning
        - local/global costmap transform_tolerance/inflation, etc.
        - planner_server frequency
        - bt_navigator progress checker
        """
        # controller_server
        self._call_set_params("/controller_server", {
            "use_sim_time": True,
            "controller_plugins": ["FollowPath"],
            # DWB plugin namespace params: use "FollowPath.<param>"
            "FollowPath.plugin": "dwb_core::DWBLocalPlanner",
            "FollowPath.forward_point_distance": 0.35,
            "FollowPath.prune_distance": 0.5,
            "FollowPath.max_vel_x": 0.22,
            "FollowPath.min_vel_x": 0.05,
            "FollowPath.acc_lim_x": 0.5,
            "FollowPath.acc_lim_theta": 1.0,
            "FollowPath.path_distance_bias": 32.0,
            "FollowPath.goal_distance_bias": 20.0,
            "FollowPath.occdist_scale": 0.05,
        })

        # local_costmap (separate lifecycle node)
        self._call_set_params("/local_costmap", {
            "update_frequency": 10.0,
            "publish_frequency": 10.0,
            "resolution": 0.05,
            "transform_tolerance": 0.25,
            "footprint_padding": 0.02,
            "plugins": ["obstacle_layer", "inflation_layer"],
            "obstacle_layer.plugin": "nav2_costmap_2d::ObstacleLayer",
            "obstacle_layer.observation_sources": "scan",
            "obstacle_layer.scan.topic": "/scan",
            "obstacle_layer.scan.clearing": True,
            "obstacle_layer.scan.marking": True,
            "obstacle_layer.scan.inf_is_valid": True,
            "obstacle_layer.scan.max_obstacle_height": 2.0,
            "inflation_layer.plugin": "nav2_costmap_2d::InflationLayer",
            "inflation_layer.inflation_radius": 0.6,
            "inflation_layer.cost_scaling_factor": 3.0,
        })

        # global_costmap
        self._call_set_params("/global_costmap", {
            "update_frequency": 2.0,
            "publish_frequency": 1.0,
            "resolution": 0.05,
            "transform_tolerance": 0.25,
            "plugins": ["obstacle_layer", "inflation_layer"],
            "obstacle_layer.plugin": "nav2_costmap_2d::ObstacleLayer",
            "obstacle_layer.observation_sources": "scan",
            "obstacle_layer.scan.topic": "/scan",
            "obstacle_layer.scan.clearing": True,
            "obstacle_layer.scan.marking": True,
            "obstacle_layer.scan.inf_is_valid": True,
            "obstacle_layer.scan.max_obstacle_height": 2.0,
            "inflation_layer.plugin": "nav2_costmap_2d::InflationLayer",
            "inflation_layer.inflation_radius": 0.6,
            "inflation_layer.cost_scaling_factor": 3.0,
        })

        # planner_server
        self._call_set_params("/planner_server", {
            "expected_planner_frequency": 1.0,
            "planner_plugins": ["GridBased"],
            "GridBased.plugin": "nav2_navfn_planner/NavfnPlanner",
        })

        # bt_navigator
        self._call_set_params("/bt_navigator", {
            "use_sim_time": True,
            "progress_checker.plugin": "nav2_controller::SimpleProgressChecker",
            "progress_checker.required_movement_speed": 0.05,
            "progress_checker.time_allowance": 5.0,
        })

        # (선택) slam_toolbox도 여기서 조정하고 싶다면 주석 해제
        # self._call_set_params("/slam_toolbox", {
        #     "use_sim_time": True,
        #     "scan_topic": "/scan",
        #     "minimum_laser_range": 0.12,
        #     "max_laser_range": 3.5,
        # })

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
        """Check a small square window around (x,y) for obstacles (value > 65)."""
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
            # 1) centroid
            cx = int(round(sum(p[0] for p in comp) / len(comp)))
            cy = int(round(sum(p[1] for p in comp) / len(comp)))

            def try_clear_goal(gx, gy, rmax=8):
                # r=0..rmax grow window, find nearest clear free cell
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

            if pick is None:
                # 2) pull-back from frontier toward robot (1~2 cells)
                fx, fy = min(
                    comp,
                    key=lambda p: (map_to_world(p[0], p[1], info)[0] - rx) ** 2 +
                                  (map_to_world(p[0], p[1], info)[1] - ry) ** 2
                )
                fmx, fmy = map_to_world(fx, fy, info)
                vx, vy = rx - fmx, ry - fmy
                L = math.hypot(vx, vy) + 1e-6
                ux, uy = vx / L, vy / L
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

