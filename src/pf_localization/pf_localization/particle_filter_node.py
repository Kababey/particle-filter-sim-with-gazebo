#!/usr/bin/env python3
"""Monte-Carlo localization node (the particle filter itself).

Bayesian recursive filter, one cycle = predict then update:

    prior   bel(x_{t-1})  -- the particle set carried from the last step
    predict bel_bar(x_t) = INT p(x_t | u_t, x_{t-1}) bel(x_{t-1}) dx_{t-1}
                          -- OdometryMotionModel.sample() pushes every particle
                             through the noisy odometry increment u_t
    update  bel(x_t)     ~ p(z_t | x_t) bel_bar(x_t)
                          -- MultiHypothesisSensorModel reweights each particle
                             by the likelihood of the anonymous tag detections
    resample            -- draw the next prior in proportion to the weights when
                             the effective sample size collapses

Inputs : /odom_noisy (nav_msgs/Odometry), /tag_detections (geometry_msgs/PoseArray)
Outputs: /pf/estimate (PoseStamped), /pf/estimate_cov (PoseWithCovarianceStamped),
         /pf/particle_markers (visualization_msgs/Marker, colored by weight)
"""

from __future__ import annotations

import math
import os

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles

from geometry_msgs.msg import (PoseArray, Pose, PoseStamped,
                               PoseWithCovarianceStamped)
from nav_msgs.msg import Odometry
from std_msgs.msg import ColorRGBA, Float32
from visualization_msgs.msg import Marker, MarkerArray

from .tag_map import load_tag_map
from .motion_model import OdometryMotionModel
from .sensor_model import MultiHypothesisSensorModel
from .resampling import (effective_sample_size, systematic_resample,
                         sample_uniform_poses)
from .utils import wrap_to_pi, yaw_from_quat, quat_from_yaw, circular_mean


class ParticleFilterNode(Node):
    def __init__(self):
        super().__init__('particle_filter_node')

        # ---- parameters ----
        p = self.declare_parameter
        self.tag_map_path = p('tag_map_path', '').value
        self.num_particles = int(p('num_particles', 3000).value)
        self.init_mode = p('init_mode', 'global').value
        a1 = p('alpha1', 0.08).value
        a2 = p('alpha2', 0.02).value
        a3 = p('alpha3', 0.06).value
        a4 = p('alpha4', 0.02).value
        min_t = p('min_motion_trans', 0.002).value
        min_r = p('min_motion_rot', 0.002).value
        self.sigma_range = p('sigma_range', 0.20).value
        self.sigma_bearing = p('sigma_bearing', 0.10).value
        self.z_clutter = p('z_clutter', 0.05).value
        self.max_range = p('max_range', 6.0).value
        self.hfov = p('hfov_rad', 1.20).value
        self.max_view_angle = p('max_view_angle_deg', 65.0).value
        self.ess_ratio = p('ess_ratio_threshold', 0.5).value
        # Augmented-MCL adaptive injection: inject random particles only when the
        # measurement likelihood collapses (i.e. the filter is actually lost).
        self.max_inject_frac = p('random_inject_frac', 0.10).value
        self.a_slow = p('w_slow_alpha', 0.01).value
        self.a_fast = p('w_fast_alpha', 0.20).value
        # deadband: only inject once the likelihood has dropped by this fraction
        # (avoids "re-guessing" particles on minor post-convergence fluctuations)
        self.inject_threshold = p('inject_threshold', 0.25).value
        # the robot centre cannot be closer than this to a wall (map constraint);
        # particles that leave the room are resampled back inside
        self.wall_margin = p('wall_margin', 0.15).value
        self.map_frame = p('map_frame', 'map').value

        if not self.tag_map_path or not os.path.exists(self.tag_map_path):
            raise RuntimeError(
                f'tag_map_path not found: {self.tag_map_path!r}. '
                'Pass it via the launch file.')

        self.tmap = load_tag_map(self.tag_map_path)
        m = self.wall_margin
        self.bounds = (m, self.tmap.size_x - m, m, self.tmap.size_y - m)
        self.rng = np.random.default_rng()

        self.motion = OdometryMotionModel(a1, a2, a3, a4, min_t, min_r, self.rng)
        self.sensor = MultiHypothesisSensorModel(
            self.tmap.positions, self.tmap.normals,
            self.sigma_range, self.sigma_bearing,
            self.max_range, self.hfov, self.max_view_angle, self.z_clutter)

        # ---- particle set (the belief) ----
        self.particles = sample_uniform_poses(
            self.num_particles, self.bounds, self.rng)
        self.weights = np.full(self.num_particles, 1.0 / self.num_particles)
        self.last_odom = None      # (x, y, theta) of previous odom message
        self.gt_pose = None        # latest ground truth, for error logging only
        self.w_slow = 0.0          # slow EMA of the mean measurement likelihood
        self.w_fast = 0.0          # fast EMA of the mean measurement likelihood
        self.p_inject = 0.0        # current adaptive injection probability

        # ---- I/O ----
        sensor_qos = QoSPresetProfiles.SENSOR_DATA.value
        self.create_subscription(Odometry, 'odom', self.odom_cb, sensor_qos)
        self.create_subscription(PoseArray, 'tag_detections',
                                 self.detections_cb, sensor_qos)
        self.create_subscription(PoseStamped, 'ground_truth',
                                 self.gt_cb, sensor_qos)

        self.pub_est = self.create_publisher(PoseStamped, 'pf/estimate', 10)
        self.pub_cov = self.create_publisher(
            PoseWithCovarianceStamped, 'pf/estimate_cov', 10)
        self.pub_particles = self.create_publisher(
            MarkerArray, 'pf/particle_markers', 10)
        self.pub_err = self.create_publisher(Float32, 'pf/error', 10)

        # decouple visualization rate from the (bursty) detection rate
        self.create_timer(0.1, self.publish_belief)

        self.get_logger().info(
            f'Particle filter up: {self.num_particles} particles, '
            f'{self.tmap.num_tags} tags, init={self.init_mode}.')

    # ------------------------------------------------------------------ inputs
    def odom_cb(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        th = yaw_from_quat(q.x, q.y, q.z, q.w)
        curr = (x, y, th)

        if self.last_odom is None:
            self.last_odom = curr
            return

        rot1, trans, rot2 = self.motion.decompose(self.last_odom, curr)
        if self.motion.is_significant(rot1, trans, rot2):
            self.particles = self.motion.sample(
                self.particles, self.last_odom, curr)
            self.enforce_bounds()
            self.last_odom = curr

    def enforce_bounds(self):
        """Map constraint: the robot is inside the room, so any particle that
        leaves it is invalid. Resample those out-of-bounds particles from the
        valid (in-room) ones, so the cloud can never bleed through the walls."""
        xmin, xmax, ymin, ymax = self.bounds
        P = self.particles
        inside = ((P[:, 0] >= xmin) & (P[:, 0] <= xmax)
                  & (P[:, 1] >= ymin) & (P[:, 1] <= ymax))
        n_out = int((~inside).sum())
        if n_out == 0:
            return
        in_idx = np.nonzero(inside)[0]
        if in_idx.size == 0:
            # everything left the room (should not happen) -> reinitialize
            self.particles = sample_uniform_poses(
                self.num_particles, self.bounds, self.rng)
            self.weights = np.full(self.num_particles, 1.0 / self.num_particles)
            return
        out_idx = np.nonzero(~inside)[0]
        w_in = self.weights[in_idx]
        s = w_in.sum()
        probs = (w_in / s) if s > 0 else None
        pick = self.rng.choice(in_idx, size=n_out, p=probs)
        self.particles[out_idx] = self.particles[pick]
        self.weights[out_idx] = self.weights[pick]
        total = self.weights.sum()
        if total > 0:
            self.weights /= total

    def detections_cb(self, msg: PoseArray):
        dets = []
        for pose in msg.poses:
            dx = pose.position.x
            dy = pose.position.y
            r = math.hypot(dx, dy)
            phi = math.atan2(dy, dx)
            dets.append((r, phi))
        if not dets:
            return

        # --- UPDATE: reweight by the multi-hypothesis likelihood ---
        meas = self.sensor.update_weights(self.particles, dets)

        # Augmented-MCL: track slow/fast EMAs of the average measurement
        # likelihood; injection probability rises only when the fast average
        # falls below the slow one (the evidence stopped matching the belief).
        w_avg = float(np.mean(meas))
        if self.w_slow <= 0.0:
            self.w_slow = self.w_fast = w_avg
        else:
            self.w_slow += self.a_slow * (w_avg - self.w_slow)
            self.w_fast += self.a_fast * (w_avg - self.w_fast)
        drop = max(0.0, 1.0 - (self.w_fast / self.w_slow
                               if self.w_slow > 0 else 1.0))
        # only inject on a real, sustained collapse; ignore minor dips so a
        # converged cloud stays put instead of re-scattering particles
        self.p_inject = drop if drop > self.inject_threshold else 0.0

        w = self.weights * meas
        total = w.sum()
        if total <= 0 or not np.isfinite(total):
            # total filter failure -> reinitialize globally
            self.get_logger().warn('Weights collapsed; reinitializing globally.')
            self.particles = sample_uniform_poses(
                self.num_particles, self.bounds, self.rng)
            self.weights = np.full(self.num_particles, 1.0 / self.num_particles)
            self.w_slow = self.w_fast = 0.0
            return
        self.weights = w / total

        # --- RESAMPLE when the effective sample size collapses ---
        ess = effective_sample_size(self.weights)
        if ess < self.ess_ratio * self.num_particles:
            self.resample()

    def gt_cb(self, msg: PoseStamped):
        q = msg.pose.orientation
        self.gt_pose = (msg.pose.position.x, msg.pose.position.y,
                        yaw_from_quat(q.x, q.y, q.z, q.w))

    # -------------------------------------------------------------- resampling
    def resample(self):
        n = self.num_particles
        # adaptive number of injected particles (capped so a transient bad frame
        # cannot wipe out a good lock)
        p = min(self.p_inject, self.max_inject_frac)
        n_inject = int(self.rng.binomial(n, p)) if p > 0.0 else 0
        n_keep = n - n_inject

        idx = systematic_resample(self.weights, self.rng)[:n_keep]
        new = self.particles[idx]

        if n_inject > 0:
            injected = sample_uniform_poses(n_inject, self.bounds, self.rng)
            new = np.vstack([new, injected])
            # injecting fresh global particles means the next likelihoods will
            # dip; relax the fast EMA so we do not over-inject on the next frame
            self.w_fast = self.w_slow

        self.particles = new
        self.weights = np.full(n, 1.0 / n)

    # ------------------------------------------------------------- estimate/io
    def estimate(self):
        """Pose estimate = mean of the DOMINANT cluster.

        While the belief is still multi-modal a global weighted mean would sit
        between clusters and be meaningless. We instead locate the densest region
        with a coarse spatial histogram (robust even right after resampling, when
        all weights are equal) and average the particles in a window around it.
        """
        P = self.particles
        w = self.weights
        bs = 0.5
        nx = max(1, int(round(self.tmap.size_x / bs)))
        ny = max(1, int(round(self.tmap.size_y / bs)))
        ix = np.clip((P[:, 0] / bs).astype(int), 0, nx - 1)
        iy = np.clip((P[:, 1] / bs).astype(int), 0, ny - 1)
        counts = np.bincount(ix * ny + iy, minlength=nx * ny)
        pix, piy = divmod(int(np.argmax(counts)), ny)
        cx, cy = (pix + 0.5) * bs, (piy + 0.5) * bs

        mask = (np.abs(P[:, 0] - cx) <= 1.5) & (np.abs(P[:, 1] - cy) <= 1.5)
        if mask.sum() < 0.05 * self.num_particles:
            mask = np.ones(self.num_particles, dtype=bool)
        self._cluster_mask = mask

        ww = w[mask]
        s = ww.sum()
        ww = ww / s if s > 0 else np.full(mask.sum(), 1.0 / mask.sum())
        x = float(np.sum(ww * P[mask, 0]))
        y = float(np.sum(ww * P[mask, 1]))
        th = circular_mean(P[mask, 2], ww)
        # fraction of particles in the dominant cluster -> a convergence indicator
        self.cluster_frac = float(mask.mean())
        return x, y, th

    def covariance_2d(self, mx, my):
        mask = getattr(self, '_cluster_mask', np.ones(self.num_particles, bool))
        P = self.particles[mask]
        w = self.weights[mask]
        s = w.sum()
        w = w / s if s > 0 else np.full(len(w), 1.0 / len(w))
        dx = P[:, 0] - mx
        dy = P[:, 1] - my
        cxx = float(np.sum(w * dx * dx))
        cyy = float(np.sum(w * dy * dy))
        cxy = float(np.sum(w * dx * dy))
        return cxx, cyy, cxy

    def publish_belief(self):
        now = self.get_clock().now().to_msg()
        mx, my, mth = self.estimate()

        est = PoseStamped()
        est.header.stamp = now
        est.header.frame_id = self.map_frame
        est.pose.position.x = mx
        est.pose.position.y = my
        qx, qy, qz, qw = quat_from_yaw(mth)
        est.pose.orientation.x = qx
        est.pose.orientation.y = qy
        est.pose.orientation.z = qz
        est.pose.orientation.w = qw
        self.pub_est.publish(est)

        cxx, cyy, cxy = self.covariance_2d(mx, my)
        cov = PoseWithCovarianceStamped()
        cov.header = est.header
        cov.pose.pose = est.pose
        c = [0.0] * 36
        c[0] = cxx
        c[1] = cxy
        c[6] = cxy
        c[7] = cyy
        c[35] = 0.25  # nominal yaw variance for display
        cov.pose.covariance = c
        self.pub_cov.publish(cov)

        self.pub_particles.publish(self.particle_markers(now))

        if self.gt_pose is not None:
            err = Float32()
            err.data = float(math.hypot(mx - self.gt_pose[0],
                                        my - self.gt_pose[1]))
            self.pub_err.publish(err)

    def particle_markers(self, stamp) -> MarkerArray:
        """Two markers: the cloud as colored POINTS (position + weight), and the
        particle headings as tiny arrows (a LINE_LIST stick per particle pointing
        along theta), so the heading distribution is visible too."""
        from geometry_msgs.msg import Point
        arr = MarkerArray()
        P = self.particles

        # Colour by LOCAL DENSITY, not instantaneous weight. Weights reset to
        # uniform at each resample and peak right after an update, which makes a
        # converged cloud flicker blue; density (how concentrated the particles
        # are) is stable, so a converged cluster stays solid red and a spread
        # cloud stays blue. dense -> red, sparse -> blue.
        bs = 0.5
        nx = max(1, int(round(self.tmap.size_x / bs)))
        ny = max(1, int(round(self.tmap.size_y / bs)))
        ix = np.clip((P[:, 0] / bs).astype(int), 0, nx - 1)
        iy = np.clip((P[:, 1] / bs).astype(int), 0, ny - 1)
        cell = ix * ny + iy
        counts = np.bincount(cell, minlength=nx * ny)
        dens = counts[cell].astype(float)
        denom = max(dens.max(), 0.15 * self.num_particles)
        t = np.sqrt(np.clip(dens / denom, 0.0, 1.0))

        # per-particle colour: YELLOW, brighter where particles are more
        # concentrated (higher posterior probability), dimmer where sparse.
        # (Brightness tracks density, which is stable across resampling, so it
        # does not flicker. The estimate pose/trajectory stay red, set elsewhere.)
        bri = 0.3 + 0.7 * t
        cols = [ColorRGBA(r=float(bri[i]), g=float(bri[i]), b=0.0, a=0.9)
                for i in range(self.num_particles)]

        # (1) positions as points
        pm = Marker()
        pm.header.stamp = stamp
        pm.header.frame_id = self.map_frame
        pm.ns = 'particles'
        pm.id = 0
        pm.type = Marker.POINTS
        pm.action = Marker.ADD
        pm.scale.x = 0.05
        pm.scale.y = 0.05
        pm.pose.orientation.w = 1.0
        pm.points = [Point(x=float(P[i, 0]), y=float(P[i, 1]), z=0.05)
                     for i in range(self.num_particles)]
        pm.colors = cols
        arr.markers.append(pm)

        # (2) headings as tiny arrows (sticks). Down-sample so RViz stays light.
        stride = max(1, self.num_particles // 1500)
        length = 0.18
        cos = np.cos(P[:, 2])
        sin = np.sin(P[:, 2])
        hm = Marker()
        hm.header.stamp = stamp
        hm.header.frame_id = self.map_frame
        hm.ns = 'particle_headings'
        hm.id = 1
        hm.type = Marker.LINE_LIST
        hm.action = Marker.ADD
        hm.scale.x = 0.012  # line width
        hm.pose.orientation.w = 1.0
        hpts = []
        hcols = []
        for i in range(0, self.num_particles, stride):
            x, y = float(P[i, 0]), float(P[i, 1])
            hpts.append(Point(x=x, y=y, z=0.06))
            hpts.append(Point(x=x + length * float(cos[i]),
                              y=y + length * float(sin[i]), z=0.06))
            hcols.append(cols[i])
            hcols.append(cols[i])
        hm.points = hpts
        hm.colors = hcols
        arr.markers.append(hm)
        return arr


def main(args=None):
    rclpy.init(args=args)
    node = ParticleFilterNode()
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
