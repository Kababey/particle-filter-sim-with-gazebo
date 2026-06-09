#!/usr/bin/env python3
"""Real camera AR-tag detector.

Subscribes to the robot's forward camera, runs an AprilTag (tag36h11) detector
on each frame, and publishes the relative position of every detected tag in the
base_link frame. Because all 8 wall tags are the SAME tag id, the detector
simply reports "a tag is here" several times per frame -- it cannot say which
physical tag each detection belongs to. That ambiguity is handed, unresolved, to
the particle filter's multi-hypothesis sensor model.

Output (identical contract to sim_detector_node):
    /tag_detections : geometry_msgs/PoseArray in the base_link frame.

Pose recovery uses the pinhole intrinsics from /camera/camera_info and the known
physical tag size, so each detection carries a metric (range, bearing).
"""

from __future__ import annotations

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles

from geometry_msgs.msg import PoseArray, Pose
from sensor_msgs.msg import Image, CameraInfo

try:
    import cv2
    from cv_bridge import CvBridge
    from pupil_apriltags import Detector
    _IMPORT_ERR = None
except Exception as exc:  # pragma: no cover - reported at runtime
    _IMPORT_ERR = exc


class TagDetectorNode(Node):
    def __init__(self):
        super().__init__('tag_detector_node')
        if _IMPORT_ERR is not None:
            self.get_logger().error(
                f'Vision deps unavailable ({_IMPORT_ERR}). Install with: '
                'pip install --break-system-packages pupil-apriltags ; '
                'apt install python3-opencv ros-jazzy-cv-bridge. '
                'Falling back is possible with detection_mode:=sim.')
            raise RuntimeError(_IMPORT_ERR)

        p = self.declare_parameter
        self.tag_size = float(p('tag_size', 0.40).value)
        self.camera_x = float(p('camera_x', 0.15).value)
        self.camera_z = float(p('camera_z', 0.20).value)
        self.max_range = float(p('max_range', 6.0).value)
        decimate = float(p('decimate', 1.0).value)
        self.min_margin = float(p('min_decision_margin', 30.0).value)
        self.base_frame = p('base_frame', 'base_link').value
        family = p('tag_family', 'tag36h11').value
        self.show_window = bool(p('show_window', True).value)  # pop a cv2 window
        self.tag_id = int(p('tag_id', 0).value)

        self.bridge = CvBridge()
        self.detector = Detector(families=family, nthreads=4,
                                 quad_decimate=decimate, decode_sharpening=0.25)
        self.K = None  # (fx, fy, cx, cy)
        self._win = 'Robot camera  -  AprilTag detections'
        self._win_ok = self.show_window

        qos = QoSPresetProfiles.SENSOR_DATA.value
        self.create_subscription(CameraInfo, 'camera_info', self.info_cb, qos)
        self.create_subscription(Image, 'image', self.image_cb, qos)
        self.pub = self.create_publisher(PoseArray, 'tag_detections', 10)
        # annotated camera image (also viewable in rqt_image_view / RViz)
        self.pub_img = self.create_publisher(Image, 'camera/detections', 5)
        self.get_logger().info(f'Camera AprilTag detector running (family={family}).')

    def info_cb(self, msg: CameraInfo):
        self.K = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])

    def image_cb(self, msg: Image):
        if self.K is None:
            return
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        # gz publishes rgb8; build a grayscale for detection and a BGR canvas to draw on
        if img.ndim == 3:
            if msg.encoding == 'bgr8':
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                canvas = img.copy()
            else:  # rgb8
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                canvas = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            gray = img
            canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        gray = np.ascontiguousarray(gray, dtype=np.uint8)

        results = self.detector.detect(
            gray, estimate_tag_pose=True,
            camera_params=list(self.K), tag_size=self.tag_size)

        out = PoseArray()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.base_frame

        n_used = 0
        for r in results:
            t = r.pose_t.reshape(3)
            base_x = self.camera_x + float(t[2])
            base_y = -float(t[0])
            rng = math.hypot(base_x, base_y)
            accepted = (r.decision_margin >= self.min_margin
                        and rng <= self.max_range)

            # --- highlight the tag in the camera image ---
            corners = r.corners.astype(np.int32).reshape(-1, 1, 2)
            color = (0, 255, 0) if accepted else (0, 165, 255)  # green / orange (BGR)
            cv2.polylines(canvas, [corners], True, color, 2)
            cx, cy = int(r.center[0]), int(r.center[1])
            cv2.circle(canvas, (cx, cy), 4, color, -1)
            if accepted:
                cv2.putText(canvas, f'id{self.tag_id}  {rng:.1f} m',
                            (int(r.corners[:, 0].min()), int(r.corners[:, 1].min()) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
                pose = Pose()
                pose.position.x = base_x
                pose.position.y = base_y
                pose.position.z = 0.0
                pose.orientation.w = 1.0
                out.poses.append(pose)
                n_used += 1

        self.pub.publish(out)

        # banner + publish/show the annotated image
        cv2.putText(canvas, f'tags seen: {n_used}', (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if n_used else (60, 60, 255), 2, cv2.LINE_AA)
        try:
            self.pub_img.publish(self.bridge.cv2_to_imgmsg(canvas, encoding='bgr8'))
        except Exception:
            pass
        if self._win_ok:
            try:
                cv2.imshow(self._win, canvas)
                cv2.waitKey(1)
            except Exception as exc:  # no display / headless
                self.get_logger().warn(f'camera window disabled: {exc}')
                self._win_ok = False


def main(args=None):
    rclpy.init(args=args)
    node = TagDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
