#!/usr/bin/env python3
"""Odometry corruption node.

In simulation the wheel odometry from the diff-drive plugin is almost perfect,
which would hide the whole point of the filter. This node turns that clean
odometry into realistic dead-reckoning: it adds a small systematic scale/heading
error plus a random walk, accumulated over the path. The result, /odom_noisy:

  * is what the particle filter consumes as its motion input, and
  * drives the "odometry-only trajectory" in the visualization, which visibly
    drifts away from ground truth while the filter stays locked on.

Toggle off (publish odometry unchanged) by setting the noise parameters to 0.
"""

from __future__ import annotations

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles

from nav_msgs.msg import Odometry

from .utils import wrap_to_pi, yaw_from_quat, quat_from_yaw


class OdomNoiseNode(Node):
    def __init__(self):
        super().__init__('odom_noise_node')
        p = self.declare_parameter
        self.trans_scale = float(p('trans_scale', 1.03).value)
        self.rot_scale = float(p('rot_scale', 1.0).value)
        self.rot_per_trans = float(p('rot_per_trans', 0.012).value)
        self.tn_per_m = float(p('trans_noise_per_m', 0.03).value)
        self.rn_per_m = float(p('rot_noise_per_m', 0.015).value)
        self.rn_per_rad = float(p('rot_noise_per_rad', 0.03).value)
        seed = int(p('seed', 0).value)
        self.rng = np.random.default_rng(seed if seed != 0 else None)

        self.last_clean = None         # (x, y, theta)
        self.noisy = None              # (x, y, theta) accumulated

        qos = QoSPresetProfiles.SENSOR_DATA.value
        self.sub = self.create_subscription(Odometry, 'odom_in', self.cb, qos)
        self.pub = self.create_publisher(Odometry, 'odom_out', 10)
        self.get_logger().info('Odometry-noise node running.')

    def cb(self, msg: Odometry):
        q = msg.pose.pose.orientation
        cx, cy = msg.pose.pose.position.x, msg.pose.pose.position.y
        cth = yaw_from_quat(q.x, q.y, q.z, q.w)
        clean = (cx, cy, cth)

        if self.last_clean is None:
            self.last_clean = clean
            self.noisy = clean
            self.publish(msg, clean)
            return

        # clean incremental motion
        dx = cx - self.last_clean[0]
        dy = cy - self.last_clean[1]
        trans = math.hypot(dx, dy)
        rot1 = wrap_to_pi(math.atan2(dy, dx) - self.last_clean[2]) if trans > 1e-4 else 0.0
        rot2 = wrap_to_pi(cth - self.last_clean[2] - rot1)

        # corrupt the increment
        rot1_n = (rot1 * self.rot_scale
                  + self.rng.normal(0.0, self.rn_per_rad * math.sqrt(abs(rot1))
                                    + self.rn_per_m * math.sqrt(trans) + 1e-9))
        trans_n = (trans * self.trans_scale
                   + self.rng.normal(0.0, self.tn_per_m * math.sqrt(trans) + 1e-9))
        rot2_n = (rot2 * self.rot_scale
                  + self.rng.normal(0.0, self.rn_per_rad * math.sqrt(abs(rot2))
                                    + self.rn_per_m * math.sqrt(trans) + 1e-9))

        nx, ny, nth = self.noisy
        nth = wrap_to_pi(nth + rot1_n)
        nx += trans_n * math.cos(nth)
        ny += trans_n * math.sin(nth)
        nth = wrap_to_pi(nth + rot2_n + self.rot_per_trans * trans)
        self.noisy = (nx, ny, nth)
        self.last_clean = clean
        self.publish(msg, self.noisy)

    def publish(self, src: Odometry, pose):
        out = Odometry()
        out.header = src.header
        out.child_frame_id = src.child_frame_id
        out.pose.pose.position.x = pose[0]
        out.pose.pose.position.y = pose[1]
        out.pose.pose.position.z = 0.0
        qx, qy, qz, qw = quat_from_yaw(pose[2])
        out.pose.pose.orientation.x = qx
        out.pose.pose.orientation.y = qy
        out.pose.pose.orientation.z = qz
        out.pose.pose.orientation.w = qw
        out.twist = src.twist  # velocities are not corrupted
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = OdomNoiseNode()
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
