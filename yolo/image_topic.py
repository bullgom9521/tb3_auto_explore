#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
import cv2
import time
from rclpy.qos import qos_profile_sensor_data
from ultralytics import YOLO

# ==========================================
# [사용자 설정] 속도 파라미터
# ==========================================
SPEED_ROI_1 = 0.03  # 전진 (느림)
SPEED_ROI_3 = 0.12  # 전진 (빠름)
SPEED_ANGULAR = 0.261  # 회전 (약 15도/초)

# [후진 관련 설정]
BACKUP_SPEED = -0.1  # 후진 속도 (m/s)
BACKUP_DISTANCE = 0.2  # 후진 거리 (m)
BACKUP_DURATION = abs(BACKUP_DISTANCE / BACKUP_SPEED)

# ==========================================
# 1. ROI 설정 (순서 중요)
# ==========================================
ROIS = [
    (10, 410, 410, 710),  # ROI 0 (Left) -> 제자리 좌회전
    (470, 410, 810, 710),  # ROI 1 (Bottom Center) -> 전진 (느림) ★ 필수 조건
    (880, 410, 1260, 710),  # ROI 2 (Right) -> 제자리 우회전
    (470, 200, 810, 400),  # ROI 3 (Middle Center) -> 전진 (빠름)
    (10, 200, 410, 400),  # ROI 4 (Middle Left) -> 전진(1) + 좌회전(0)
    (880, 200, 1260, 400),  # ROI 5 (Middle Right) -> 전진(1) + 우회전(2)
]

ROI_COLOR_DEFAULT = (255, 0, 0)
ROI_COLOR_TRIGGERED = (0, 255, 255)


def rect_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class ImageViewer(Node):
    def __init__(self):
        super().__init__("image_viewer")
        self.model = YOLO("/home/ho/tb3_auto_explore/yolo/YOLOCUBE_ClassModified.pt")

        self.sub = self.create_subscription(
            Image, "/rgb", self.callback, qos_profile_sensor_data
        )
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback, 10
        )

        self.stop_pub = self.create_publisher(Bool, "/emergency_stop", 10)
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # ==========================================
        # [상태 변수들]
        # ==========================================
        self.current_pose = None
        self.target_classes = ["red-cube", "blue-cube", "green-cube", "yellow-cube"]
        self.found_locations = {}
        self.locked_target = None

        self.detect_counter = 0
        self.potential_target = None

        # ROI 1 도달 확인용 플래그
        self.has_reached_roi1 = False

        # 후진 상태 관리
        self.is_backing_up = False
        self.backup_start_time = 0.0

        self.get_logger().info(
            "Mission Started: Find (Stable x3) -> Check ROI 1 -> Save -> Backup -> Resume."
        )

    def odom_callback(self, msg: Odometry):
        self.current_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def callback(self, msg: Image):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        results = self.model(img_bgr, verbose=False)
        r = results[0]

        # ----------------------------------------------------
        # [Priority 0] 후진(Backing Up) 상태 처리
        # ----------------------------------------------------
        if self.is_backing_up:
            elapsed_time = time.time() - self.backup_start_time
            if elapsed_time < BACKUP_DURATION:
                twist = Twist()
                twist.linear.x = BACKUP_SPEED
                self.cmd_vel_pub.publish(twist)
                cv2.putText(
                    img_bgr,
                    f"BACKING UP... {elapsed_time:.1f}s",
                    (50, 300),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3,
                )
                cv2.imshow("YOLO Multi-ROI Controller", img_bgr)
                cv2.waitKey(1)
                return
            else:
                self.get_logger().warn(">>> BACKUP COMPLETE. Resuming Exploration. <<<")
                self.cmd_vel_pub.publish(Twist())
                self.stop_pub.publish(Bool(data=False))
                self.is_backing_up = False
                self.locked_target = None
                return

        # ----------------------------------------------------
        # 1. 현재 프레임 타겟 분석
        # ----------------------------------------------------
        roi_triggered = [False] * len(ROIS)
        detected_targets_in_roi = []

        for box in r.boxes:
            cls_id = int(box.cls[0])
            class_name = self.model.names[cls_id]

            if class_name not in self.target_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            det_rect = (x1, y1, x2, y2)

            in_roi = False
            for i, roi in enumerate(ROIS):
                if rect_iou(det_rect, roi) > 0:
                    roi_triggered[i] = True
                    in_roi = True

            if in_roi:
                detected_targets_in_roi.append(class_name)

        # ----------------------------------------------------
        # 2. 상태 머신 (State Machine)
        # ----------------------------------------------------

        # [상태 A] 탐색 모드
        if self.locked_target is None:
            if len(detected_targets_in_roi) > 0:
                target = detected_targets_in_roi[0]
                if target == self.potential_target:
                    self.detect_counter += 1
                else:
                    self.potential_target = target
                    self.detect_counter = 1

                # ★★★ [수정됨] 3회 연속 감지 시 락온 ★★★
                if self.detect_counter >= 3:
                    self.locked_target = target

                    # 락온 시 ROI 1 도달 여부 초기화
                    self.has_reached_roi1 = False

                    self.stop_pub.publish(Bool(data=True))
                    self.get_logger().warn(
                        f">>> FOUND {target} (Stable x3)! Locking on & Pausing Exploration."
                    )
                    self.detect_counter = 0
                    self.potential_target = None
            else:
                self.detect_counter = 0
                self.potential_target = None

        # [상태 B] 제어 모드
        else:
            target = self.locked_target

            if target in detected_targets_in_roi:
                # ROI 1 도달 체크
                if roi_triggered[1]:
                    self.has_reached_roi1 = True

                twist = Twist()

                # ROI 우선순위 동작 로직
                if roi_triggered[4]:  # ROI 4
                    twist.linear.x = SPEED_ROI_1
                    twist.angular.z = SPEED_ANGULAR

                elif roi_triggered[5]:  # ROI 5
                    twist.linear.x = SPEED_ROI_1
                    twist.angular.z = -SPEED_ANGULAR

                elif roi_triggered[0]:  # ROI 0
                    twist.angular.z = SPEED_ANGULAR

                elif roi_triggered[2]:  # ROI 2
                    twist.angular.z = -SPEED_ANGULAR

                elif roi_triggered[1]:  # ROI 1
                    twist.linear.x = SPEED_ROI_1

                elif roi_triggered[3]:  # ROI 3
                    twist.linear.x = SPEED_ROI_3

                else:
                    twist.linear.x = 0.0

                self.cmd_vel_pub.publish(twist)

            else:
                # 타겟 소실 (Lost) -> 조건 검사 (ROI 1 도달했었나?)
                if self.has_reached_roi1:
                    self.get_logger().warn(
                        f"<<< {target} FINISHED (ROI 1 Verified). Saving. >>>"
                    )

                    if self.current_pose:
                        save_key = f"{target}_lo"
                        self.found_locations[save_key] = self.current_pose
                        self.get_logger().error(
                            f"★ SAVED: {save_key} = {self.current_pose}"
                        )

                    if target in self.target_classes:
                        self.target_classes.remove(target)
                        self.get_logger().info(f"Remaining: {self.target_classes}")

                    # 성공 -> 후진
                    self.is_backing_up = True
                    self.backup_start_time = time.time()
                    self.cmd_vel_pub.publish(Twist())

                else:
                    # 실패 -> 무시하고 탐색 재개
                    self.get_logger().error(
                        f"XXX {target} LOST without reaching ROI 1. IGNORED. XXX"
                    )

                    self.stop_pub.publish(Bool(data=False))
                    self.cmd_vel_pub.publish(Twist())
                    self.locked_target = None

        # ----------------------------------------------------
        # 화면 그리기
        # ----------------------------------------------------
        annotated = r.plot()

        if self.locked_target:
            status_text = "VERIFIED" if self.has_reached_roi1 else "APPROACHING..."
            color = (0, 255, 0) if self.has_reached_roi1 else (0, 0, 255)
            cv2.putText(
                annotated,
                f"LOCKED: {self.locked_target} [{status_text}]",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

        elif self.potential_target:
            # 카운터 3으로 표시
            cv2.putText(
                annotated,
                f"Detecting {self.potential_target}: {self.detect_counter}/3",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

        for i, roi in enumerate(ROIS):
            rx1, ry1, rx2, ry2 = roi
            color = ROI_COLOR_TRIGGERED if roi_triggered[i] else ROI_COLOR_DEFAULT
            cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), color, 2)
            label = f"ROI {i}"
            cv2.putText(
                annotated,
                label,
                (rx1, ry1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        cv2.imshow("YOLO Multi-ROI Controller", annotated)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = ImageViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_vel_pub.publish(Twist())
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
