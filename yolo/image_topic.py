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
import os
from rclpy.qos import qos_profile_sensor_data
from ultralytics import YOLO

# ==========================================
# [설정] 속도 파라미터
# ==========================================
SPEED_ROI_1 = 0.03  # 도착/저속
SPEED_ROI_3 = 0.12  # 접근/고속
SPEED_ANGULAR = 0.261  # 회전

# [설정] 후진 파라미터
BACKUP_SPEED = -0.1
BACKUP_DISTANCE = 0.2
BACKUP_DURATION = abs(BACKUP_DISTANCE / BACKUP_SPEED)

# [설정] 좌표 제어(Gap Control) 파라미터
CENTER_MARGIN = 60

# [설정] 저장 파일 경로 & 오프셋 거리
SAVE_FILE_NAME = "/home/ho/tb3_auto_explore/found_cubes.txt"
CUBE_OFFSET_DIST = 0.20  # 로봇 중심에서 큐브까지의 거리 (m)

# ==========================================
# ROI 설정
# ==========================================
shrink = 85
ROIS = [
    (10, 410, 410 - shrink, 710),  # ROI 0 (Left)
    (470, 410, 810, 710),  # ROI 1 (Bottom Center) -> [도착 구역]
    (880 + shrink, 410, 1260, 710),  # ROI 2 (Right)
    (470, 200, 810, 400),  # ROI 3 (Middle Center)
    (10, 200, 410 - shrink, 400),  # ROI 4 (Middle Left)
    (880 + shrink, 200, 1260, 400),  # ROI 5 (Middle Right)
]

ROI_COLOR_DEFAULT = (255, 0, 0)
ROI_COLOR_TRIGGERED = (0, 255, 255)

# Finish Line 설정 (ROI 1의 75% 지점)
ROI1_TOP = ROIS[1][1]  # 410
ROI1_BOTTOM = ROIS[1][3]  # 710
ROI1_HEIGHT = ROI1_BOTTOM - ROI1_TOP
FINISH_LINE_Y = int(ROI1_TOP + (ROI1_HEIGHT * 0.75))


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

        self.current_pose = None
        self.current_yaw = 0.0  # [중요] Yaw 초기화

        self.all_possible_targets = [
            "red-cube",
            "blue-cube",
            "green-cube",
            "yellow-cube",
        ]
        self.target_classes = ["red-cube", "blue-cube", "green-cube", "yellow-cube"]

        self.locked_target = None
        self.detect_counter = 0
        self.potential_target = None
        self.has_reached_roi1 = False
        self.is_backing_up = False
        self.backup_start_time = 0.0

        self.init_save_file()
        self.get_logger().info("Mission Started: Offset Save + Ignore Logic.")

    def init_save_file(self):
        if not os.path.exists(SAVE_FILE_NAME):
            with open(SAVE_FILE_NAME, "w") as f:
                f.write("Timestamp, Target, X, Y\n")

    def odom_callback(self, msg: Odometry):
        # 1. 위치 저장
        self.current_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y)

        # 2. [중요] 쿼터니언 -> Yaw 변환 (이게 있어야 오프셋 계산 가능)
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = np.arctan2(siny_cosp, cosy_cosp)

    def save_cube_location(self, target_name):
        """로봇 위치에서 바라보는 방향으로 20cm 앞 좌표를 저장"""  # 터틀봇 15cm + 큐브 5cm 감안해서..
        if self.current_pose:
            rx, ry = self.current_pose

            # [좌표 보정] 로봇 중심 + (거리 * 방향)
            cube_x = rx + CUBE_OFFSET_DIST * np.cos(self.current_yaw)
            cube_y = ry + CUBE_OFFSET_DIST * np.sin(self.current_yaw)

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            log_line = f"{timestamp}, {target_name}, {cube_x:.4f}, {cube_y:.4f}\n"
            try:
                with open(SAVE_FILE_NAME, "a") as f:
                    f.write(log_line)
                self.get_logger().info(
                    f"★ SAVED (Offset Applied): {target_name} at ({cube_x:.2f}, {cube_y:.2f})"
                )
            except Exception as e:
                self.get_logger().error(f"File Save Error: {e}")

    def callback(self, msg: Image):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        results = self.model(img_bgr, verbose=False)
        r = results[0]

        center_x = msg.width // 2
        center_y = msg.height // 2
        annotated = img_bgr.copy()

        # [Priority 0] 후진 처리
        if self.is_backing_up:
            elapsed = time.time() - self.backup_start_time
            if elapsed < BACKUP_DURATION:
                twist = Twist()
                twist.linear.x = BACKUP_SPEED
                self.cmd_vel_pub.publish(twist)
                cv2.putText(
                    annotated,
                    f"BACKING UP... {elapsed:.1f}s",
                    (50, 300),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3,
                )
                cv2.imshow("YOLO Hybrid Controller", annotated)
                cv2.waitKey(1)
                return
            else:
                self.cmd_vel_pub.publish(Twist())
                self.stop_pub.publish(Bool(data=False))
                self.is_backing_up = False
                self.locked_target = None
                return

        # ----------------------------------------------------
        # 1. 탐색 및 데이터 추출
        # ----------------------------------------------------
        roi_triggered = [False] * len(ROIS)
        detected_targets_in_roi = []
        target_box_data = None

        for box in r.boxes:
            cls_id = int(box.cls[0])
            class_name = self.model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # [처리 완료된 큐브 무시]
            if (
                class_name in self.all_possible_targets
                and class_name not in self.target_classes
            ):
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (128, 128, 128), 2)
                cv2.putText(
                    annotated,
                    f"{class_name} (DONE)",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (128, 128, 128),
                    1,
                )
                continue

            # Lock On 타겟 좌표 저장
            if self.locked_target and class_name == self.locked_target:
                target_box_data = [x1, y1, x2, y2]

            if class_name not in self.target_classes:
                continue

            # 그리기 & ROI 체크
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                class_name,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

            det_rect = (x1, y1, x2, y2)
            in_roi = False
            for i, roi in enumerate(ROIS):
                if rect_iou(det_rect, roi) > 0:
                    roi_triggered[i] = True
                    in_roi = True
            if in_roi:
                detected_targets_in_roi.append(class_name)

        # ----------------------------------------------------
        # 2. State Machine
        # ----------------------------------------------------
        twist = Twist()

        # [상태 A] 탐색 모드
        if self.locked_target is None:
            if len(detected_targets_in_roi) > 0:
                target = detected_targets_in_roi[0]
                if target == self.potential_target:
                    self.detect_counter += 1
                else:
                    self.potential_target = target
                    self.detect_counter = 1

                if self.detect_counter >= 3:
                    self.locked_target = target
                    self.has_reached_roi1 = False
                    self.stop_pub.publish(Bool(data=True))
                    self.detect_counter = 0
                    self.potential_target = None
            else:
                self.detect_counter = 0
                self.potential_target = None

        # [상태 B] 제어 모드
        else:
            target = self.locked_target

            # (1) ROI 제어
            if target in detected_targets_in_roi:
                if roi_triggered[1]:
                    # ROI 1 내부 -> Finish Line 체크
                    cube_bottom_y = 0
                    if target_box_data:
                        cube_bottom_y = target_box_data[3]

                    if cube_bottom_y > FINISH_LINE_Y:
                        # [도착 완료]
                        self.get_logger().info(
                            f">>> {target} FINISHED! Saving & Removing."
                        )
                        twist.linear.x = 0.0
                        twist.angular.z = 0.0
                        self.cmd_vel_pub.publish(twist)

                        self.save_cube_location(target)  # 오프셋 적용된 좌표 저장

                        if target in self.target_classes:
                            self.target_classes.remove(target)

                        self.is_backing_up = True
                        self.backup_start_time = time.time()
                        return
                    else:
                        twist.linear.x = SPEED_ROI_1  # 접근 중

                elif roi_triggered[4]:
                    twist.linear.x = SPEED_ROI_1
                    twist.angular.z = SPEED_ANGULAR
                elif roi_triggered[5]:
                    twist.linear.x = SPEED_ROI_1
                    twist.angular.z = -SPEED_ANGULAR
                elif roi_triggered[0]:
                    twist.angular.z = SPEED_ANGULAR
                elif roi_triggered[2]:
                    twist.angular.z = -SPEED_ANGULAR
                elif roi_triggered[3]:
                    twist.linear.x = SPEED_ROI_3
                else:
                    if not roi_triggered[1]:
                        twist.linear.x = 0.0
                self.cmd_vel_pub.publish(twist)

            # (2) Gap Control
            elif target_box_data is not None:
                x1, y1, x2, y2 = target_box_data
                cx = (x1 + x2) // 2
                cv2.line(annotated, (cx, y1), (cx, y2), (0, 165, 255), 2)

                if cx < center_x - CENTER_MARGIN:
                    twist.angular.z = SPEED_ANGULAR
                elif cx > center_x + CENTER_MARGIN:
                    twist.angular.z = -SPEED_ANGULAR
                else:
                    twist.linear.x = SPEED_ROI_3
                self.cmd_vel_pub.publish(twist)

            # (3) Lost
            else:
                self.stop_pub.publish(Bool(data=False))
                self.cmd_vel_pub.publish(Twist())
                self.locked_target = None

        # ----------------------------------------------------
        # 3. 시각화
        # ----------------------------------------------------
        cv2.line(
            annotated,
            (center_x - 20, center_y),
            (center_x + 20, center_y),
            (0, 255, 0),
            2,
        )
        cv2.line(
            annotated,
            (center_x, center_y - 20),
            (center_x, center_y + 20),
            (0, 255, 0),
            2,
        )

        if self.locked_target:
            cv2.line(
                annotated,
                (center_x - CENTER_MARGIN, 0),
                (center_x - CENTER_MARGIN, msg.height),
                (0, 255, 255),
                1,
            )
            cv2.line(
                annotated,
                (center_x + CENTER_MARGIN, 0),
                (center_x + CENTER_MARGIN, msg.height),
                (0, 255, 255),
                1,
            )

        r1_x1, r1_y1, r1_x2, r1_y2 = ROIS[1]
        step = (r1_y2 - r1_y1) // 4
        for k in range(1, 4):
            line_y = r1_y1 + (step * k)
            if k == 3:
                color = (0, 0, 255)
                cv2.putText(
                    annotated,
                    "FINISH",
                    (r1_x1 + 5, line_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )
                thickness = 2
            else:
                color = (0, 255, 255)
                thickness = 1
            cv2.line(
                annotated, (r1_x1 + 20, line_y), (r1_x2 - 20, line_y), color, thickness
            )

        for i, roi in enumerate(ROIS):
            color = ROI_COLOR_TRIGGERED if roi_triggered[i] else ROI_COLOR_DEFAULT
            cv2.rectangle(annotated, (roi[0], roi[1]), (roi[2], roi[3]), color, 2)
            cv2.putText(
                annotated,
                f"R{i}",
                (roi[0], roi[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        if self.locked_target:
            cv2.putText(
                annotated,
                f"LOCKED: {self.locked_target}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        cv2.imshow("YOLO Hybrid Controller", annotated)
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
