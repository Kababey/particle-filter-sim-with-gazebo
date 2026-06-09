#!/usr/bin/env python3
"""Visualization node: everything the assignment's Task 3 view must contain
(except the particle cloud, which the filter publishes directly since it owns
the weights). In a single RViz view this node renders, in the map frame:

  * the room outline and the 8 AR-tag positions (static markers),
  * the odometry-only trajectory  (/pf/path_odom)      -- drifts,
  * the particle-filter estimate trajectory (/pf/path_estimate) -- stays locked,
  * the ground-truth trajectory   (/pf/path_ground_truth) -- reference,
  * arrow markers for the estimated and true robot poses.

Putting the odometry path and the filter path in the same view is exactly what
makes the benefit of the filter visible.
"""

from __future__ import annotations

import math
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, QoSPresetProfiles

from geometry_msgs.msg import PoseStamped, Point, PoseArray
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from .tag_map import load_tag_map
from .utils import yaw_from_quat, quat_from_yaw, wrap_to_pi


def _stamp_s(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def _append_downsampled(path: Path, x, y, yaw, stamp, frame, trail=0.0,
                        min_d=0.02, min_a=0.035, max_jump=0.0):
    """Append a pose and keep only a rolling time window (a fading trail).

    trail > 0 drops poses older than `trail` seconds, so the trajectory shown in
    RViz is wiped after a while. trail <= 0 keeps the full path (capped at 4000).

    max_jump > 0 wipes the trail when the new point is farther than `max_jump`
    from the last one: the pose estimate can hop between clusters (much faster
    than the robot can move), which would otherwise draw a long, ugly straight
    line across the room. Dropping the old trail keeps each segment local.
    """
    now_t = _stamp_s(stamp)
    if trail and trail > 0.0:
        while path.poses and now_t - _stamp_s(path.poses[0].header.stamp) > trail:
            path.poses.pop(0)
    path.header.stamp = stamp
    path.header.frame_id = frame

    if max_jump > 0.0 and path.poses:
        last = path.poses[-1].pose.position
        if math.hypot(x - last.x, y - last.y) > max_jump:
            path.poses = []   # wipe the "splash" jump; restart the trail here

    if path.poses:
        last = path.poses[-1].pose.position
        lyaw = yaw_from_quat(*(lambda o: (o.x, o.y, o.z, o.w))(
            path.poses[-1].pose.orientation))
        if math.hypot(x - last.x, y - last.y) < min_d and abs(wrap_to_pi(yaw - lyaw)) < min_a:
            return
    ps = PoseStamped()
    ps.header.stamp = stamp
    ps.header.frame_id = frame
    ps.pose.position.x = x
    ps.pose.position.y = y
    qx, qy, qz, qw = quat_from_yaw(yaw)
    ps.pose.orientation.x = qx
    ps.pose.orientation.y = qy
    ps.pose.orientation.z = qz
    ps.pose.orientation.w = qw
    path.poses.append(ps)
    if len(path.poses) > 4000:
        path.poses.pop(0)


class VizNode(Node):
    def __init__(self):
        super().__init__('viz_node')
        p = self.declare_parameter
        self.tag_map_path = p('tag_map_path', '').value
        self.map_frame = p('map_frame', 'map').value
        self.draw_gt = bool(p('draw_ground_truth', True).value)
        # rolling trail length for the trajectory paths [s]; 0 = keep full path
        self.trail_seconds = float(p('trail_seconds', 20.0).value)
        # break the trail when a point jumps farther than this [m] (anti-splash)
        self.max_jump = float(p('trail_max_jump', 0.6).value)
        # camera visibility (must match the detector) -> which tags are "seen"
        self.half_fov = 0.5 * float(p('hfov_rad', 1.50).value)
        self.max_range = float(p('max_range', 8.0).value)
        self.cos_view = math.cos(math.radians(
            float(p('max_view_angle_deg', 75.0).value)))
        self.camera_x = float(p('camera_x', 0.18).value)
        if not self.tag_map_path or not os.path.exists(self.tag_map_path):
            raise RuntimeError(f'tag_map_path not found: {self.tag_map_path!r}')
        self.tmap = load_tag_map(self.tag_map_path)

        self.spawn = None  # (x, y, theta) from first ground-truth message
        self.last_gt = None
        self.det_n = 0          # number of current camera detections
        self.det_t = 0.0        # sim-time of the last detection message
        self.path_odom = Path()
        self.path_est = Path()
        self.path_gt = Path()

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        sensor_qos = QoSPresetProfiles.SENSOR_DATA.value

        self.pub_static = self.create_publisher(MarkerArray, 'viz/static', latched)
        self.pub_robots = self.create_publisher(MarkerArray, 'viz/robots', 10)
        self.pub_visible = self.create_publisher(MarkerArray, 'viz/visible_tags', 10)
        self.pub_path_odom = self.create_publisher(Path, 'pf/path_odom', 10)
        self.pub_path_est = self.create_publisher(Path, 'pf/path_estimate', 10)
        self.pub_path_gt = self.create_publisher(Path, 'pf/path_ground_truth', 10)

        self.create_subscription(Odometry, 'odom_noisy', self.odom_cb, sensor_qos)
        self.create_subscription(PoseStamped, 'pf/estimate', self.est_cb, 10)
        self.create_subscription(PoseStamped, 'ground_truth', self.gt_cb, sensor_qos)
        self.create_subscription(PoseArray, 'tag_detections', self.det_cb, sensor_qos)

        self.create_timer(1.0, self.publish_static)  # latched static map
        self.create_timer(0.1, self.publish_visible_tags)  # which tags are in view
        self.publish_static()
        self.get_logger().info('Visualization node running.')

    # ---------------------------------------------------------------- callbacks
    def det_cb(self, msg: PoseArray):
        self.det_n = len(msg.poses)
        self.det_t = _stamp_s(msg.header.stamp)

    def gt_cb(self, msg: PoseStamped):
        q = msg.pose.orientation
        x, y = msg.pose.position.x, msg.pose.position.y
        th = yaw_from_quat(q.x, q.y, q.z, q.w)
        self.last_gt = (x, y, th)
        if self.spawn is None:
            self.spawn = (x, y, th)
        if self.draw_gt:
            _append_downsampled(self.path_gt, x, y, th, msg.header.stamp,
                                 self.map_frame, self.trail_seconds,
                                 max_jump=self.max_jump)
            self.pub_path_gt.publish(self.path_gt)
        self.publish_robot_marker('truth', x, y, th, ColorRGBA(r=0.1, g=0.8, b=0.1, a=0.9))

    def est_cb(self, msg: PoseStamped):
        q = msg.pose.orientation
        x, y = msg.pose.position.x, msg.pose.position.y
        th = yaw_from_quat(q.x, q.y, q.z, q.w)
        _append_downsampled(self.path_est, x, y, th, msg.header.stamp,
                             self.map_frame, self.trail_seconds,
                             max_jump=self.max_jump)
        self.pub_path_est.publish(self.path_est)
        self.publish_robot_marker('estimate', x, y, th, ColorRGBA(r=1.0, g=0.2, b=0.0, a=0.9))

    def odom_cb(self, msg: Odometry):
        if self.spawn is None:
            return
        # odom frame starts at the spawn pose; compose to express it in map
        q = msg.pose.pose.orientation
        ox, oy = msg.pose.pose.position.x, msg.pose.pose.position.y
        oth = yaw_from_quat(q.x, q.y, q.z, q.w)
        sx, sy, sth = self.spawn
        c, s = math.cos(sth), math.sin(sth)
        wx = sx + ox * c - oy * s
        wy = sy + ox * s + oy * c
        wth = wrap_to_pi(sth + oth)
        _append_downsampled(self.path_odom, wx, wy, wth, msg.header.stamp,
                             self.map_frame, self.trail_seconds,
                             max_jump=self.max_jump)
        self.pub_path_odom.publish(self.path_odom)

    # ----------------------------------------------------- visible-tag highlight
    def visible_tags(self, x, y, th):
        """Indices of tags currently in the camera's view from pose (x, y, th)
        (in range, in FOV, and on the tag's front face). Same gate as the
        detector, evaluated from ground truth -- a visualization aid."""
        c, s = math.cos(th), math.sin(th)
        out = []
        for i in range(self.tmap.num_tags):
            tx, ty = self.tmap.positions[i]
            nx, ny = self.tmap.normals[i]
            dx, dy = tx - x, ty - y
            rel_x = c * dx + s * dy
            rel_y = -s * dx + c * dy
            rng = math.hypot(rel_x, rel_y)
            if rng > self.max_range or abs(math.atan2(rel_y, rel_x)) > self.half_fov:
                continue
            if (-dx * nx - dy * ny) / max(rng, 1e-6) < self.cos_view:
                continue
            out.append(i)
        return out

    def publish_visible_tags(self):
        """Highlight the tags the camera is currently seeing, with a sight line
        from the camera to each. Gated on actually having recent detections, so
        the highlight matches what the camera reports."""
        arr = MarkerArray()
        clr = Marker()
        clr.header.frame_id = self.map_frame
        clr.action = Marker.DELETEALL
        arr.markers.append(clr)

        now = self.get_clock().now()
        fresh = (self.det_n > 0
                 and (_stamp_s(now.to_msg()) - self.det_t) < 0.5)
        if self.last_gt is not None and fresh:
            x, y, th = self.last_gt
            cam_x = x + self.camera_x * math.cos(th)
            cam_y = y + self.camera_x * math.sin(th)
            mid = 0
            for i in self.visible_tags(x, y, th):
                tx, ty = float(self.tmap.positions[i][0]), float(self.tmap.positions[i][1])
                tz = float(self.tmap.mount_height)
                halo = Marker()
                halo.header.frame_id = self.map_frame
                halo.header.stamp = now.to_msg()
                halo.ns = 'visible_tag'
                halo.id = mid
                mid += 1
                halo.type = Marker.SPHERE
                halo.action = Marker.ADD
                halo.pose.position.x = tx
                halo.pose.position.y = ty
                halo.pose.position.z = tz
                halo.pose.orientation.w = 1.0
                halo.scale.x = halo.scale.y = halo.scale.z = 0.55
                halo.color = ColorRGBA(r=1.0, g=0.95, b=0.1, a=0.6)
                arr.markers.append(halo)

                line = Marker()
                line.header.frame_id = self.map_frame
                line.header.stamp = now.to_msg()
                line.ns = 'sight_line'
                line.id = mid
                mid += 1
                line.type = Marker.LINE_LIST
                line.action = Marker.ADD
                line.scale.x = 0.03
                line.pose.orientation.w = 1.0
                line.color = ColorRGBA(r=1.0, g=0.9, b=0.0, a=0.85)
                line.points = [Point(x=cam_x, y=cam_y, z=0.2),
                               Point(x=tx, y=ty, z=tz)]
                arr.markers.append(line)
        self.pub_visible.publish(arr)

    # ------------------------------------------------------------------ markers
    def publish_robot_marker(self, ns, x, y, th, color):
        arr = MarkerArray()
        m = Marker()
        m.header.frame_id = self.map_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns
        m.id = 0
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = 0.1
        qx, qy, qz, qw = quat_from_yaw(th)
        m.pose.orientation.x = qx
        m.pose.orientation.y = qy
        m.pose.orientation.z = qz
        m.pose.orientation.w = qw
        m.scale.x = 0.5
        m.scale.y = 0.08
        m.scale.z = 0.08
        m.color = color
        arr.markers.append(m)
        self.pub_robots.publish(arr)

    def publish_static(self):
        arr = MarkerArray()
        sx, sy = self.tmap.size_x, self.tmap.size_y

        # room outline
        outline = Marker()
        outline.header.frame_id = self.map_frame
        outline.header.stamp = self.get_clock().now().to_msg()
        outline.ns = 'room'
        outline.id = 0
        outline.type = Marker.LINE_STRIP
        outline.action = Marker.ADD
        outline.scale.x = 0.04
        outline.color = ColorRGBA(r=0.5, g=0.5, b=0.55, a=1.0)
        outline.pose.orientation.w = 1.0
        for (px, py) in [(0, 0), (sx, 0), (sx, sy), (0, sy), (0, 0)]:
            outline.points.append(Point(x=float(px), y=float(py), z=0.02))
        arr.markers.append(outline)

        # tag plaques + labels
        for i in range(self.tmap.num_tags):
            tx, ty = self.tmap.positions[i]
            yaw = float(self.tmap.normal_yaw[i])
            plaque = Marker()
            plaque.header.frame_id = self.map_frame
            plaque.header.stamp = outline.header.stamp
            plaque.ns = 'tags'
            plaque.id = i
            plaque.type = Marker.CUBE
            plaque.action = Marker.ADD
            plaque.pose.position.x = float(tx)
            plaque.pose.position.y = float(ty)
            plaque.pose.position.z = float(self.tmap.mount_height)
            qx, qy, qz, qw = quat_from_yaw(yaw)
            plaque.pose.orientation.x = qx
            plaque.pose.orientation.y = qy
            plaque.pose.orientation.z = qz
            plaque.pose.orientation.w = qw
            plaque.scale.x = 0.03
            plaque.scale.y = float(self.tmap.tag_size)
            plaque.scale.z = float(self.tmap.tag_size)
            plaque.color = ColorRGBA(r=0.05, g=0.05, b=0.05, a=1.0)
            arr.markers.append(plaque)

            label = Marker()
            label.header.frame_id = self.map_frame
            label.header.stamp = outline.header.stamp
            label.ns = 'tag_labels'
            label.id = i
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = float(tx)
            label.pose.position.y = float(ty)
            label.pose.position.z = float(self.tmap.mount_height) + 0.35
            label.scale.z = 0.22
            label.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.9)
            label.text = f'tag{self.tmap.tag_id}'
            arr.markers.append(label)

        self.pub_static.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = VizNode()
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
