"""Small math helpers shared across nodes (kept dependency-free on purpose)."""

from __future__ import annotations

import math

import numpy as np


def wrap_to_pi(a):
    """Wrap angle(s) to (-pi, pi]. Works on scalars and numpy arrays."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    """Extract yaw (rotation about z) from a quaternion."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def quat_from_yaw(yaw: float):
    """Return (x, y, z, w) for a pure-yaw rotation."""
    return (0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5))


def circular_mean(angles, weights=None) -> float:
    """Weighted circular mean of a set of angles."""
    angles = np.asarray(angles)
    if weights is None:
        s = np.sin(angles).mean()
        c = np.cos(angles).mean()
    else:
        w = np.asarray(weights)
        s = np.sum(w * np.sin(angles))
        c = np.sum(w * np.cos(angles))
    return math.atan2(s, c)
