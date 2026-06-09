"""Multi-hypothesis measurement model (the particle-filter UPDATE step).

This is the graded core of the project. Every AR tag shares the same ID, so an
observation z does NOT say which physical tag was seen. The likelihood of an
observation given a particle pose must therefore marginalize over *which* tag
produced it:

        p(z | x) = sum_{i=1..8} P(tag = i) * p(z | x, tag_i)
                 = (1/8) * sum_{i=1..8} p(z | x, tag_i)          (uniform prior)

We deliberately do NOT collapse this to the single nearest tag -- that shortcut
is forbidden by the assignment and destroys the filter's ability to represent
several competing pose hypotheses while the room still looks ambiguous from the
tags seen so far.

A tag only contributes if it could actually have been seen from the particle's
pose (inside camera range + field of view, and the robot is on the tag's front
face). This visibility gate is exactly the generative assumption behind the
real camera detector, so the model and the data are consistent.

Each detection is a range/bearing pair (r, phi) of an anonymous tag expressed in
the robot base frame. Detections within a frame are treated as conditionally
independent, so their per-particle likelihoods multiply.
"""

from __future__ import annotations

import math

import numpy as np

from .utils import wrap_to_pi


class MultiHypothesisSensorModel:
    def __init__(self, tag_positions, tag_normals,
                 sigma_range, sigma_bearing,
                 max_range, hfov_rad, max_view_angle_deg,
                 z_clutter):
        self.tags = np.asarray(tag_positions, dtype=float)      # (T, 2)
        self.normals = np.asarray(tag_normals, dtype=float)     # (T, 2)
        self.num_tags = self.tags.shape[0]
        self.sr = float(sigma_range)
        self.sb = float(sigma_bearing)
        self.max_range = float(max_range)
        self.half_fov = 0.5 * float(hfov_rad)
        self.cos_view = math.cos(math.radians(max_view_angle_deg))
        self.z_clutter = float(z_clutter)
        # Gaussian normalizer (constant across particles; kept so the clutter
        # term sits at a physically meaningful fraction of the peak likelihood).
        self._norm = 1.0 / (2.0 * math.pi * self.sr * self.sb)

    def _geometry(self, particles):
        """Per (particle, tag) expected range, bearing and visibility mask."""
        px = particles[:, 0][:, None]      # (N, 1)
        py = particles[:, 1][:, None]
        th = particles[:, 2][:, None]

        dx = self.tags[:, 0][None, :] - px   # (N, T)
        dy = self.tags[:, 1][None, :] - py
        rng = np.hypot(dx, dy)               # (N, T) expected range
        bearing = wrap_to_pi(np.arctan2(dy, dx) - th)  # (N, T) expected bearing

        # Visibility: in range, in field of view, and robot on the tag's front.
        in_range = rng <= self.max_range
        in_fov = np.abs(bearing) <= self.half_fov
        # unit vector tag->robot dotted with the tag's outward normal
        inv = 1.0 / np.maximum(rng, 1e-6)
        to_robot_x = -dx * inv
        to_robot_y = -dy * inv
        front = (to_robot_x * self.normals[:, 0][None, :]
                 + to_robot_y * self.normals[:, 1][None, :]) >= self.cos_view

        visible = in_range & in_fov & front
        return rng, bearing, visible

    def update_weights(self, particles, detections):
        """Return the per-particle likelihood multiplier for one camera frame.

        particles  : (N, 3) [x, y, theta]
        detections : (K, 2) [range, bearing] anonymous tag measurements
        returns    : (N,) unnormalized weight multiplier (product over detections)
        """
        n = particles.shape[0]
        if detections is None or len(detections) == 0:
            return np.ones(n)

        rng, bearing, visible = self._geometry(particles)   # (N, T) each
        vis_f = visible.astype(float)

        weights = np.ones(n)
        inv_sr2 = 1.0 / (self.sr * self.sr)
        inv_sb2 = 1.0 / (self.sb * self.sb)

        for (r_obs, phi_obs) in detections:
            dr = r_obs - rng                                 # (N, T)
            dphi = wrap_to_pi(phi_obs - bearing)             # (N, T)
            # p(z | x, tag_i): Gaussian in range & bearing, zero if not visible
            per_tag = self._norm * np.exp(
                -0.5 * (dr * dr * inv_sr2 + dphi * dphi * inv_sb2))
            per_tag *= vis_f
            # marginalize over the 8 tag hypotheses (uniform prior) + clutter
            p_z = per_tag.sum(axis=1) / self.num_tags + self.z_clutter  # (N,)
            weights *= p_z

        return weights
