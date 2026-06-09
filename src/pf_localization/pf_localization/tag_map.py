"""Loader for the tag map / room description (the single source of truth).

The same YAML file (config/tag_map.yaml) is consumed by:
  * generate_world.py   -> to emit the Gazebo world,
  * particle_filter_node -> as the filter's KNOWN tag map,
  * sim_detector_node    -> to synthesize ground-truth detections.

Keeping one file guarantees the simulated world and the filter's map agree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import yaml


@dataclass
class TagMap:
    # Room
    size_x: float
    size_y: float
    wall_height: float
    wall_thickness: float
    # Tag family / size
    family: str
    tag_id: int
    tag_size: float
    mount_height: float
    # Tag geometry (arrays, one row per tag)
    positions: np.ndarray   # (N, 2) world (x, y) of each tag center
    normals: np.ndarray     # (N, 2) unit outward normal (direction tag faces)
    normal_yaw: np.ndarray  # (N,)   yaw of the normal, radians

    @property
    def num_tags(self) -> int:
        return self.positions.shape[0]


def load_tag_map(path: str) -> TagMap:
    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    room = data['room']
    tag = data['tag']
    tags = data['tags']

    pos = np.array([[t['x'], t['y']] for t in tags], dtype=float)
    yaw = np.array([math.radians(t['normal_deg']) for t in tags], dtype=float)
    normals = np.stack([np.cos(yaw), np.sin(yaw)], axis=1)

    return TagMap(
        size_x=float(room['size_x']),
        size_y=float(room['size_y']),
        wall_height=float(room['wall_height']),
        wall_thickness=float(room['wall_thickness']),
        family=str(tag['family']),
        tag_id=int(tag['id']),
        tag_size=float(tag['size']),
        mount_height=float(tag['mount_height']),
        positions=pos,
        normals=normals,
        normal_yaw=yaw,
    )
