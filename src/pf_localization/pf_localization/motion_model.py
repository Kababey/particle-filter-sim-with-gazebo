"""Probabilistic odometry motion model (the particle-filter PREDICTION step).

This is the sampling form of the odometry motion model from Thrun, Burgard &
Fox, *Probabilistic Robotics*, Table 5.6 (``sample_motion_model_odometry``),
vectorized so that all N particles are propagated in a single numpy call.

Bayesian role: it draws x_t^[m] ~ p(x_t | u_t, x_{t-1}^[m]), i.e. it turns the
previous belief (the particle set) into the *predicted* belief by pushing every
particle through the noisy control u_t = (odom_{t-1}, odom_t).
"""

from __future__ import annotations

import math

import numpy as np

from .utils import wrap_to_pi


class OdometryMotionModel:
    def __init__(self, alpha1, alpha2, alpha3, alpha4,
                 min_trans=0.0, min_rot=0.0, rng=None):
        self.a1 = float(alpha1)
        self.a2 = float(alpha2)
        self.a3 = float(alpha3)
        self.a4 = float(alpha4)
        self.min_trans = float(min_trans)
        self.min_rot = float(min_rot)
        self.rng = rng if rng is not None else np.random.default_rng()

    def decompose(self, prev_pose, curr_pose):
        """Decompose an odometry step into (rot1, trans, rot2)."""
        dx = curr_pose[0] - prev_pose[0]
        dy = curr_pose[1] - prev_pose[1]
        trans = math.hypot(dx, dy)
        # Guard against spurious rotation when essentially standing still.
        if trans < 1e-4:
            rot1 = 0.0
        else:
            rot1 = wrap_to_pi(math.atan2(dy, dx) - prev_pose[2])
        rot2 = wrap_to_pi(curr_pose[2] - prev_pose[2] - rot1)
        return float(rot1), float(trans), float(rot2)

    def is_significant(self, rot1, trans, rot2) -> bool:
        return (trans >= self.min_trans
                or abs(rot1) >= self.min_rot
                or abs(rot2) >= self.min_rot)

    def sample(self, particles: np.ndarray, prev_pose, curr_pose) -> np.ndarray:
        """Propagate every particle through one noisy odometry increment.

        particles : (N, 3) array of [x, y, theta]
        returns    : (N, 3) propagated particles
        """
        rot1, trans, rot2 = self.decompose(prev_pose, curr_pose)
        n = particles.shape[0]

        # Per-particle noise. Variances follow the alpha parameterization;
        # std = sqrt(variance). np.random.normal takes a std (scale) argument.
        std_rot1 = math.sqrt(self.a1 * rot1 * rot1 + self.a2 * trans * trans)
        std_trans = math.sqrt(self.a3 * trans * trans
                              + self.a4 * (rot1 * rot1 + rot2 * rot2))
        std_rot2 = math.sqrt(self.a1 * rot2 * rot2 + self.a2 * trans * trans)

        rot1_hat = rot1 - self.rng.normal(0.0, std_rot1 + 1e-12, n)
        trans_hat = trans - self.rng.normal(0.0, std_trans + 1e-12, n)
        rot2_hat = rot2 - self.rng.normal(0.0, std_rot2 + 1e-12, n)

        theta = particles[:, 2]
        new = np.empty_like(particles)
        new[:, 0] = particles[:, 0] + trans_hat * np.cos(theta + rot1_hat)
        new[:, 1] = particles[:, 1] + trans_hat * np.sin(theta + rot1_hat)
        new[:, 2] = wrap_to_pi(theta + rot1_hat + rot2_hat)
        return new
