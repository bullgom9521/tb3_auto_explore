#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FrontierExplorer (improved):
- 프런티어 탐사 + Nav2 파라미터 주입은 그대로 유지
- "No valid frontier goal found (blocked or repeated)"로 멈추는 문제 완화:
  1) 목표 타임아웃/취소(watchdog)
  2) 블랙리스트(반경) + 재시도 거리 로직
  3) 점진적 완화 전략(relaxation level: clearance↓, min_frontier_size↓)
  4) 맵 정체 시 강제 재탐색(타이머)
"""

import math
import time
from typing import List, Tuple, Optional, Dict, Set

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


def world_to_map(mx: float, my: float, info) -> Tuple[int, int]:
    ox = info.origin.position.x
    oy = info.origin.position.y
    res = info.resolution
    gx = int((mx - ox) / res)
    gy = int((my - oy) / res)
    return gx, gy


class FrontierExplorer(Node):
    def __init__(self) -> None:
        super().__init__("frontier_explorer")

        # ===== Parameters (declare + get) =====
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("global_frame", "map")
        self.declare_parameter("robot_base_frame", "base_link")

        self.declare_parameter("min_frontier_size", 10)
        self.declare_parameter("goal_clearance_cells", 5)

        # 탐사 멈춤 방지 신규 파라미터
        self.declare_parameter(
            "frontier_search_timeout", 5
        )  # s, 맵 업데이트/프런티어 정체 감지
        self.declare_parameter("goal_timeout_sec", 10.0)  # s, goal 도달 타임아웃
        self.declare_parameter("blacklist_clear_radius", 5)  # m, 블랙리스트 반경(월드)
        self.declare_parameter("retry_distance", 1)  # m, 프런티어에서 뒤로 당길 거리
        self.declare_parameter(
            "max_goal_attempts_per_frontier", 2
        )  # 프런티어 재시도 횟수 제한

        # Nav2 대기
        self.declare_parameter("wait_nav2_sec", 5.0)

        # Read
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
            self.get_parameter("min_frontier_size").get_parameter_value().integer_value
        )
        self.clearance: int = (
            self.get_parameter("goal_clearance_cells")
            .get_parameter_value()
            .integer_value
        )
        self.wait_nav2_sec: float = (
            self.get_parameter("wait_nav2_sec").get_parameter_value().double_value
        )

        self.frontier_search_timeout: float = (
            self.get_parameter("frontier_search_timeout")
            .get_parameter_value()
            .double_value
        )
        self.goal_timeout_sec: float = (
            self.get_parameter("goal_timeout_sec").get_parameter_value().double_value
        )
        self.blacklist_clear_radius_m: float = (
            self.get_parameter("blacklist_clear_radius")
            .get_parameter_value()
            .double_value
        )
        self.retry_distance_m: float = (
            self.get_parameter("retry_distance").get_parameter_value().double_value
        )
        self.max_goal_attempts: int = (
            self.get_parameter("max_goal_attempts_per_frontier")
            .get_parameter_value()
            .integer_value
        )

        # ===== Subscriptions =====
        qos = QoSProfile(depth=5)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.history = HistoryPolicy.KEEP_LAST
        self.map_sub = self.create_subscription(
            OccupancyGrid, self.map_topic, self.on_map, qos
        )

        # ===== TF =====
        self.buffer: Buffer = Buffer()
        self.tf_listener = TransformListener(self.buffer, self)

        # ===== Nav2 action =====
        self.nav_client: ActionClient = ActionClient(
            self, NavigateToPose, "navigate_to_pose"
        )
        self.current_goal_future = None
        self.result_future = None
        self._goal_handle = None

        # ===== State =====
        self._have_map: bool = False
        self._last_map: Optional[OccupancyGrid] = None
        self._last_map_time = self.get_clock().now()

        # 시도/블랙리스트 관리(셀 단위)
        self.sent_goals_cells: Set[Tuple[int, int]] = set()
        self.goal_attempts: Dict[Tuple[int, int], int] = {}
        self.blacklist_cells: Set[Tuple[int, int]] = set()

        # Watchdog timer
        self.timer = self.create_timer(2.0, self.watchdog_cb)

        self.get_logger().info("FrontierExplorer started. Waiting for Nav2...")
        self.nav_client.wait_for_server(timeout_sec=self.wait_nav2_sec)
        self.push_nav2_params()
        if not self.nav_client.server_is_ready():
            self.get_logger().warn(
                "Nav2 action server not ready yet; will continue once map arrives."
            )

    # ===== Nav2 param helpers =====
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
        cli = self.create_client(SetParameters, f"{srv_name}/set_parameters")
        if not cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn(
                f"[Param] Service not available: {srv_name}/set_parameters"
            )
            return
        req = SetParameters.Request()
        req.parameters = [self._make_param(k, v) for k, v in kv.items()]
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        if fut.result() is None:
            self.get_logger().warn(f"[Param] Failed to set on {srv_name} (timeout)")
        else:
            self.get_logger().info(
                f"[Param] Applied {len(req.parameters)} params to {srv_name}"
            )

    def push_nav2_params(self) -> None:
        # controller_server
        self._call_set_params(
            "/controller_server",
            {
                "use_sim_time": True,
                "controller_plugins": ["FollowPath"],
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
            },
        )
        # local_costmap
        self._call_set_params(
            "/local_costmap",
            {
                "update_frequency": 10.0,
                "publish_frequency": 10.0,
                "resolution": 0.05,
                "transform_tolerance": 0.35,
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
                "inflation_layer.inflation_radius": 1.0,
                "inflation_layer.cost_scaling_factor": 2.0,
            },
        )
        # global_costmap
        self._call_set_params(
            "/global_costmap",
            {
                "update_frequency": 2.0,
                "publish_frequency": 1.0,
                "resolution": 0.05,
                "transform_tolerance": 0.35,
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
            },
        )
        # planner_server
        self._call_set_params(
            "/planner_server",
            {
                "expected_planner_frequency": 1.0,
                "planner_plugins": ["GridBased"],
                "GridBased.plugin": "nav2_navfn_planner/NavfnPlanner",
            },
        )
        # bt_navigator
        self._call_set_params(
            "/bt_navigator",
            {
                "use_sim_time": True,
                "progress_checker.plugin": "nav2_controller::SimpleProgressChecker",
                "progress_checker.required_movement_speed": 0.05,
                "progress_checker.time_allowance": 5.0,
            },
        )

    # ===== Callbacks / timers =====
    def on_map(self, grid: OccupancyGrid) -> None:
        self._have_map = True
        self._last_map = grid
        self._last_map_time = self.get_clock().now()
        if self.current_goal_future is None:
            self.plan_and_go(grid)

    def watchdog_cb(self) -> None:
        """탐사 정체/goal 타임아웃 감시"""
        now = self.get_clock().now()
        # 1) 맵 정체: 일정 시간 맵 업데이트 없고 goal도 없는 경우 → 재탐색
        if self._have_map and self.current_goal_future is None:
            if (
                now - self._last_map_time
            ).nanoseconds * 1e-9 > self.frontier_search_timeout:
                self.get_logger().warn(
                    "[Watchdog] Map stale; retry planning with relaxation."
                )
                self.plan_and_go(self._last_map, use_relaxation=True)
        # 2) goal 타임아웃: 진행 중인데 너무 오래 걸리면 취소/블랙리스트 후 재계획
        if self._goal_handle is not None and self.result_future is not None:
            # Goal 시작 시각을 goal handle 생성 시각으로 간주 (ROS2에서 명시 타임스탬프 없음)
            # 간단히 last_map_time 기준으로 근사: 최근 센서 갱신이 없고 시간이 길어지면 취소
            if (now - self._last_map_time).nanoseconds * 1e-9 > self.goal_timeout_sec:
                self.get_logger().warn("[Watchdog] Goal timeout; cancel and replan")
                try:
                    self._goal_handle.cancel_goal_async()
                except Exception:
                    pass
                self.current_goal_future = None
                self.result_future = None
                self._goal_handle = None
                if self._last_map is not None:
                    self.plan_and_go(self._last_map, use_relaxation=True)

    # ===== SLAM/TF helpers =====
    def get_robot_pose(self) -> Optional[Tuple[float, float, float]]:
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
        except Exception as e:
            self.get_logger().warn(f"Failed to get robot pose: {e}")
            return None

    # ===== Frontier detection =====
    def find_frontiers(
        self, grid: OccupancyGrid, min_size: int
    ) -> List[List[Tuple[int, int]]]:
        info = grid.info
        data = grid.data
        w, h = info.width, info.height

        def in_bounds(x: int, y: int) -> bool:
            return 0 <= x < w and 0 <= y < h

        def is_free(x: int, y: int) -> bool:
            return in_bounds(x, y) and data[idx(x, y, w)] == 0

        def is_unknown(x: int, y: int) -> bool:
            return in_bounds(x, y) and data[idx(x, y, w)] == -1

        # --- N=4 이웃만 사용 ---
        nbs4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # 프런티어 마스크 생성
        is_frontier = [[False] * w for _ in range(h)]
        for y in range(h):
            for x in range(w):
                if is_free(x, y):
                    for dx, dy in nbs4:
                        if is_unknown(x + dx, y + dy):
                            is_frontier[y][x] = True
                            break

        # N=4 이웃으로 군집화
        from collections import deque

        visited = [[False] * w for _ in range(h)]
        frontiers: List[List[Tuple[int, int]]] = []

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
                    if len(comp) >= min_size:
                        frontiers.append(comp)

        return frontiers

    def cell_clear(
        self, x: int, y: int, grid: OccupancyGrid, clearance_cells: int
    ) -> bool:
        info = grid.info
        w, h = info.width, info.height
        data = grid.data
        for dy in range(-clearance_cells, clearance_cells + 1):
            for dx in range(-clearance_cells, clearance_cells + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    if data[idx(nx, ny, w)] > 70:
                        return False
        return True

    def is_blacklisted(self, gx: int, gy: int, grid: OccupancyGrid) -> bool:
        if (gx, gy) in self.blacklist_cells:
            return True
        # 반경 기반: blacklist_clear_radius_m 내에 블랙리스트 셀이 있으면 True
        info = grid.info
        res = info.resolution
        r_cells = max(1, int(self.blacklist_clear_radius_m / res))
        for bx, by in self.blacklist_cells:
            if abs(bx - gx) <= r_cells and abs(by - gy) <= r_cells:
                return True
        return False

    def choose_goal(
        self,
        frontiers: List[List[Tuple[int, int]]],
        grid: OccupancyGrid,
        relax_level: int = 0,
    ):
        pose = self.get_robot_pose()
        if pose is None:
            return None
        rx, ry, _ = pose
        info = grid.info
        data = grid.data
        w, h = info.width, info.height

        def free(x, y):
            return 0 <= x < w and 0 <= y < h and data[idx(x, y, w)] == 0

        # Relaxation 전략
        clearance_cells = max(
            0, self.clearance - relax_level
        )  # 레벨마다 1셀씩 느슨하게
        best = None
        best_dist = 1e9

        for comp in frontiers:
            # 중심 셀
            cx = int(round(sum(p[0] for p in comp) / len(comp)))
            cy = int(round(sum(p[1] for p in comp) / len(comp)))

            def try_clear_goal(gx, gy, rmax=8):
                if free(gx, gy) and self.cell_clear(gx, gy, grid, clearance_cells):
                    return gx, gy
                for r in range(1, rmax + 1):
                    for dy in range(-r, r + 1):
                        for dx in range(-r, r + 1):
                            x, y = gx + dx, gy + dy
                            if (
                                0 <= x < w
                                and 0 <= y < h
                                and free(x, y)
                                and self.cell_clear(x, y, grid, clearance_cells)
                            ):
                                return x, y
                return None

            pick = try_clear_goal(cx, cy, rmax=8)

            if pick is None:
                # 로봇 방향으로 retry_distance_m 만큼 당겨서 후보 생성(셀 변환)
                fx, fy = min(
                    comp,
                    key=lambda p: (map_to_world(p[0], p[1], info)[0] - rx) ** 2
                    + (map_to_world(p[0], p[1], info)[1] - ry) ** 2,
                )
                fmx, fmy = map_to_world(fx, fy, info)
                vx, vy = rx - fmx, ry - fmy
                L = math.hypot(vx, vy) + 1e-6
                ux, uy = vx / L, vy / L
                back = self.retry_distance_m
                cand_m = [(fmx + ux * back, fmy + uy * back)]
                if relax_level >= 1:
                    cand_m.append((fmx + ux * (back * 1.5), fmy + uy * (back * 1.5)))
                for cmx, cmy in cand_m:
                    cgx, cgy = world_to_map(cmx, cmy, info)
                    if (
                        0 <= cgx < w
                        and 0 <= cgy < h
                        and free(cgx, cgy)
                        and self.cell_clear(cgx, cgy, grid, clearance_cells)
                    ):
                        pick = (cgx, cgy)
                        break

            if pick is None:
                continue

            gx, gy = pick
            # 블랙리스트/중복 스킵
            if self.is_blacklisted(gx, gy, grid) or (gx, gy) in self.sent_goals_cells:
                continue

            mx, my = map_to_world(gx, gy, info)
            dist = math.hypot(mx - rx, my - ry)
            if dist < best_dist:
                best_dist = dist
                best = (mx, my, gx, gy)

        return best

    # ===== Nav2 interaction =====
    def send_goal(self, mx: float, my: float, yaw: float = 0.0) -> None:
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.header.frame_id = self.global_frame
        goal.pose.pose.position.x = mx
        goal.pose.pose.position.y = my
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        goal.pose.pose.orientation.z = qz
        goal.pose.pose.orientation.w = qw
        self.get_logger().info(f"Sending goal to ({mx:.2f}, {my:.2f})")
        self.current_goal_future = self.nav_client.send_goal_async(goal)
        self.current_goal_future.add_done_callback(self._goal_response)

    def _goal_response(self, fut) -> None:
        goal_handle = fut.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().warn("Goal rejected; will try another frontier.")
            self.current_goal_future = None
            if self._last_map:
                self.plan_and_go(self._last_map, use_relaxation=True)
            return
        self.get_logger().info("Goal accepted, waiting for result...")
        self._goal_handle = goal_handle
        self.result_future = goal_handle.get_result_async()
        self.result_future.add_done_callback(self._goal_result)

    def _goal_result(self, fut) -> None:
        try:
            res = fut.result()
            status = res.status
        except Exception:
            status = -1
        self.get_logger().info(f"Goal finished with status={status}")
        self.current_goal_future = None
        self.result_future = None
        self._goal_handle = None
        if self._last_map:
            self.plan_and_go(self._last_map)

    # ===== Planner loop =====
    def plan_and_go(self, grid: OccupancyGrid, use_relaxation: bool = False) -> None:
        if not self.nav_client.server_is_ready():
            self.get_logger().warn("Nav2 not ready yet; waiting a bit...")
            time.sleep(2.0)
            return

        # Relaxation 단계에 따라 min_frontier 조정
        min_frontier = self.min_frontier
        if use_relaxation:
            min_frontier = max(1, self.min_frontier - 1)

        frontiers = self.find_frontiers(grid, min_frontier)
        if not frontiers:
            self.get_logger().info(
                "No frontiers found. Exploration likely complete. ✅"
            )
            return

        # 1차 시도
        goal = self.choose_goal(frontiers, grid, relax_level=0)
        # 실패 시 완화
        if goal is None and use_relaxation:
            self.get_logger().warn("No valid goal; trying with relaxation level 1.")
            goal = self.choose_goal(frontiers, grid, relax_level=1)
        if goal is None and use_relaxation:
            self.get_logger().warn("No valid goal; trying with relaxation level 2.")
            goal = self.choose_goal(frontiers, grid, relax_level=2)

        if goal is None:
            self.get_logger().warn(
                "No valid frontier goal found (blocked or repeated)."
            )
            return

        mx, my, gx, gy = goal

        # 시도 횟수 체크 → 초과 시 블랙리스트
        attempts = self.goal_attempts.get((gx, gy), 0)
        if attempts >= self.max_goal_attempts:
            self.get_logger().warn(f"Goal ({gx},{gy}) exceeded attempts; blacklisting.")
            self.blacklist_cells.add((gx, gy))
            # 다른 목표로 즉시 재시도
            self.sent_goals_cells.add((gx, gy))
            self.plan_and_go(grid, use_relaxation=True)
            return

        self.goal_attempts[(gx, gy)] = attempts + 1
        self.sent_goals_cells.add((gx, gy))
        self.send_goal(mx, my)


def main() -> None:
    rclpy.init()
    node = FrontierExplorer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
