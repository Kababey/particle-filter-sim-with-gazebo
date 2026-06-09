#!/usr/bin/env python3
"""Simulated AR-tag detector (the reliable fallback / ground-truth oracle).

Instead of running computer vision on the rendered image, this node reads the
robot's ground-truth pose and synthesizes the detections the camera *would*
produce: for every tag that is in range, inside the field of view, and on its
visible front face, it emits a noisy range/bearing measurement -- WITHOUT a tag
id, exactly like the real detector. This makes the filter pipeline fully
deterministic and decoupled from the GPU render path, which is ideal for tuning
the filter and for a guaranteed demo.

Output (identical contract to tag_detector_node):
    /tag_detections : geometry_msgs/PoseArray in the base_link frame, each pose
                      being the position of an anonymous detected tag.
"""

from __future__ import annotations

import math
import os

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles

from geometry_msgs.msg import PoseArray, Pose, PoseStamped

from .tag_map import load_tag_map
from .utils import wrap_to_pi


class SimDetectorNode(Node):
    def __init__(self):
        super().__init__('sim_detector_node')
        p = self.declare_parameter
        self.tag_map_path = p('tag_map_path', '').value
        self.rate_hz = float(p('rate_hz', 10.0).value)
        self.max_range = float(p('max_range', 6.0).value)
        self.half_fov = 0.5 * float(p('hfov_rad', 1.20).value)
        self.cos_view = math.cos(math.radians(
            float(p('max_view_angle_deg', 65.0).value)))
        self.sigma_range = float(p('sigma_range', 0.12).value)
        self.sigma_bearing = float(p('sigma_bearing', 0.05).value)
        self.detection_prob = float(p('detection_prob', 0.95).value)
        self.base_frame = p('base_frame', 'base_link').value

        if not self.tag_map_path or not os.path.exists(self.tag_map_path):
            raise RuntimeError(f'tag_map_path not found: {self.tag_map_path!r}')
        self.tmap = load_tag_map(self.tag_map_path)
        self.rng = np.random.default_rng()
        self.gt = None  # (x, y, theta)

        qos = QoSPresetProfiles.SENSOR_DATA.value
        self.create_subscription(PoseStamped, 'ground_truth', self.gt_cb, qos)
        self.pub = self.create_publisher(PoseArray, 'tag_detections', 10)
        self.create_timer(1.0 / self.rate_hz, self.tick)
        self.get_logger().info('Simulated tag detector running (ground-truth based).')

    def gt_cb(self, msg: PoseStamped):
        q = msg.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.gt = (msg.pose.position.x, msg.pose.position.y,
                   math.atan2(siny, cosy))

    def tick(self):
        if self.gt is None:
            return
        x, y, th = self.gt
        c, s = math.cos(th), math.sin(th)

        out = PoseArray()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.base_frame

        for i in range(self.tmap.num_tags):
            tx, ty = self.tmap.positions[i]
            nx, ny = self.tmap.normals[i]
            dx, dy = tx - x, ty - y
            # express tag in the robot base frame
            rel_x = c * dx + s * dy
            rel_y = -s * dx + c * dy
            rng = math.hypot(rel_x, rel_y)
            bearing = math.atan2(rel_y, rel_x)

            if rng > self.max_range or abs(bearing) > self.half_fov:
                continue
            # robot must be on the tag's front face
            if (-dx * nx - dy * ny) / max(rng, 1e-6) < self.cos_view:
                continue
            if self.rng.random() > self.detection_prob:
                continue

            r_meas = rng + self.rng.normal(0.0, self.sigma_range)
            b_meas = bearing + self.rng.normal(0.0, self.sigma_bearing)
            pose = Pose()
            pose.position.x = r_meas * math.cos(b_meas)
            pose.position.y = r_meas * math.sin(b_meas)
            pose.position.z = 0.0
            pose.orientation.w = 1.0
            out.poses.append(pose)

        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = SimDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
