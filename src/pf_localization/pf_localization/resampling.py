"""Resampling utilities for the particle filter.

Low-variance (a.k.a. systematic) resampling draws N new particles in O(N) with a
single random offset, which keeps far less sampling noise than naive
multinomial resampling (Probabilistic Robotics, Table 4.4).

Resampling is only triggered when the effective sample size drops, which avoids
needlessly throwing away particle diversity on uninformative frames. A small
fraction of globally-random particles is injected on each resample so the filter
can recover if every particle has drifted away from the truth.
"""

from __future__ import annotations

import numpy as np


def effective_sample_size(weights: np.ndarray) -> float:
    """ESS = 1 / sum(w_i^2) for normalized weights; in [1, N]."""
    w = np.asarray(weights, dtype=float)
    s = w.sum()
    if s <= 0:
        return 0.0
    w = w / s
    return 1.0 / np.sum(w * w)


def systematic_resample(weights: np.ndarray, rng) -> np.ndarray:
    """Return resampled indices using a single low-variance comb."""
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0  # guard against floating-point shortfall
    return np.searchsorted(cumulative, positions)


def sample_uniform_poses(n: int, bounds, rng) -> np.ndarray:
    """Draw n global poses uniformly over the room rectangle (x, y, theta)."""
    xmin, xmax, ymin, ymax = bounds
    out = np.empty((n, 3))
    out[:, 0] = rng.uniform(xmin, xmax, n)
    out[:, 1] = rng.uniform(ymin, ymax, n)
    out[:, 2] = rng.uniform(-np.pi, np.pi, n)
    return out
