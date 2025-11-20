#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import cv2
from rclpy.qos import qos_profile_sensor_data
from ultralytics import YOLO

# -----------------------------
# 1. ROI 설정 및 IoU 함수
# -----------------------------

# Define rectangular ROIs as (x1, y1, x2, y2)

ROIS = [
    (10, 510, 450, 710),  # ROI 0
    (500, 510, 780, 710),  # ROI 1
    (880, 510, 1270, 710),  # ROI 2
    (10, 300, 450, 500),  # ROI 3
    (500, 300, 780, 500),  # ROI 4
    (880, 300, 1270, 500),  # ROI 5
    (10, 10, 450, 290),  # ROI 6
    (500, 10, 780, 290),  # ROI 7
    (880, 10, 1270, 290),  # ROI 8
]

ROI_COLOR_DEFAULT = (255, 0, 0)  # Blue (BGR)
ROI_COLOR_TRIGGERED = (0, 255, 255)  # Yellow-ish


def rect_iou(a, b):
    # a,b are (x1,y1,x2,y2)
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
    union = area_a + area_b - inter if (area_a + area_b - inter) > 0 else 1e-6
    return inter / union


# -----------------------------
# 2. ROS2 노드 정의
# -----------------------------


class ImageViewer(Node):
    def __init__(self):
        super().__init__("image_viewer")

        self.model = YOLO("/home/ho/tb3_auto_explore/yolo/YOLOCUBE_ClassModified.pt")

        self.sub = self.create_subscription(
            Image, "/rgb", self.callback, qos_profile_sensor_data
        )
        self.get_logger().info("Image viewer started.")

    def callback(self, msg: Image):
        # msg.data → numpy 이미지 변환
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3
        )  # encoding = rgb8 기준

        # OpenCV는 BGR을 쓰므로 변환 (원하면 생략 가능)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # (선택) 카메라 해상도 확인 (ROI 해상도랑 맞는지)
        # self.get_logger().info(f"h={msg.height}, w={msg.width}")

        # results = self.model(img_bgr, verbose=False)
        # annotated = results[0].plot()

        # # cv2.imshow("IsaacSim RGB", img_bgr) # 이건 그냥 rgb 토픽의 image데이터 시각화해서 보여주는거
        # # cv2.waitKey(1)
        # cv2.imshow("YOLO on IsaacSim RGB", annotated) # 이건 yolo가 인식한거 보여주는거..
        # cv2.waitKey(1)

        # 3) YOLO 추론
        results = self.model(img_bgr, verbose=False)
        r = results[0]

        # 4) ROI 트리거 체크 배열
        roi_triggered = [False] * len(ROIS)

        # 5) 각 detection box에 대해 ROI와 IoU 계산
        for box in r.boxes:
            coords_tensor = box.xyxy
            coords_list = coords_tensor.tolist()[0]
            x1, y1, x2, y2 = map(int, coords_list)
            det_rect = (x1, y1, x2, y2)

            for i, roi in enumerate(ROIS):
                if rect_iou(det_rect, roi) > 0:
                    roi_triggered[i] = True

        # 6) YOLO가 그린 기본 bbox 들어간 이미지
        annotated = r.plot()  # 이게 BGR 이미지

        # 7) ROI 박스도 같이 그리기
        for i, roi in enumerate(ROIS):
            x1, y1, x2, y2 = roi
            color = ROI_COLOR_TRIGGERED if roi_triggered[i] else ROI_COLOR_DEFAULT
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"ROI {i} {'TRIGGER' if roi_triggered[i] else ''}"
            cv2.putText(
                annotated,
                label,
                (int(x1), int(y1) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        # (선택) 터미널에 어떤 ROI가 트리거됐는지 출력
        triggered_indices = [
            i for i, triggered in enumerate(roi_triggered) if triggered
        ]
        if triggered_indices:
            self.get_logger().info(f"ROI triggered: {triggered_indices}")

        # 8) 화면 출력
        cv2.imshow("YOLO Multi-ROI on IsaacSim RGB", annotated)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = ImageViewer()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
