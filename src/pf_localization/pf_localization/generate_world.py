#!/usr/bin/env python3
"""Generate the Gazebo world (and the AprilTag texture) from tag_map.yaml.

This is the bit of machinery that guarantees the simulated room and the filter's
known map are the SAME thing: both are derived from config/tag_map.yaml. Run it
once (it is also run automatically by the launch file if the world is missing):

    python3 generate_world.py            # uses package-relative defaults
    python3 generate_world.py --help     # to override paths

It writes:
    worlds/room.sdf                                   -- the world
    models/apriltag_marker/materials/textures/<tag>.png -- the tag texture
    materials/textures/<tag>.png                        -- a copy for reference
"""

from __future__ import annotations

import argparse
import os
import shutil

import numpy as np

try:                                  # when imported as part of the package
    from .tag_map import load_tag_map
except ImportError:                   # when run directly as a script
    from tag_map import load_tag_map


# The generated texture surrounds the black AprilTag with a white quiet zone so
# the detector can find it. QUIET_RATIO = (full texture) / (black tag) edge. The
# plaque box is scaled by this ratio so the BLACK tag on the wall is exactly
# tag.size metres -- which is what the detector is told (tag_size). Keeping this
# in one constant guarantees the rendered tag size and the detector agree.
QUIET_RATIO = 1.2


# --------------------------------------------------------------------- texture
def generate_tag_texture(out_path: str, tag_id: int, family: str,
                         px: int = 800, quiet_bits: int = 1) -> None:
    """Render an AprilTag image with a white quiet zone and save it.

    Tries OpenCV's ArUco AprilTag dictionary first, then the moms-apriltag
    package. Either produces a tag36h11 pattern that pupil-apriltags can decode.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img = None

    # --- attempt 1: OpenCV aruco AprilTag dictionary ---
    try:
        import cv2
        dict_name = {
            'tag36h11': 'DICT_APRILTAG_36h11',
            'tag25h9': 'DICT_APRILTAG_25h9',
            'tag16h5': 'DICT_APRILTAG_16h5',
        }.get(family, 'DICT_APRILTAG_36h11')
        aruco = cv2.aruco
        dic = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
        inner = px
        if hasattr(aruco, 'generateImageMarker'):
            marker = aruco.generateImageMarker(dic, tag_id, inner)
        else:  # OpenCV < 4.7
            marker = aruco.drawMarker(dic, tag_id, inner)
        # ArUco AprilTag markers already include the black border; add a white
        # quiet zone (QUIET_RATIO) so the detector reliably finds the tag.
        pad = int(round(inner * (QUIET_RATIO - 1.0) / 2.0))
        img = np.full((inner + 2 * pad, inner + 2 * pad), 255, dtype=np.uint8)
        img[pad:pad + inner, pad:pad + inner] = marker
        cv2.imwrite(out_path, img)
        return
    except Exception as exc:  # noqa: BLE001
        last_err = exc

    # --- attempt 2: moms_apriltag ---
    try:
        from moms_apriltag import TagGenerator2
        from PIL import Image
        tg = TagGenerator2(family)
        tag = tg.generate(tag_id)  # 2D uint8 array, values {0,255}
        tag = np.kron(tag, np.ones((px // tag.shape[0], px // tag.shape[1]),
                                   dtype=np.uint8))
        pad = int(round(tag.shape[0] * (QUIET_RATIO - 1.0) / 2.0))
        img = np.full((tag.shape[0] + 2 * pad, tag.shape[1] + 2 * pad), 255,
                      dtype=np.uint8)
        img[pad:pad + tag.shape[0], pad:pad + tag.shape[1]] = tag
        Image.fromarray(img).save(out_path)
        return
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            'Could not generate an AprilTag texture. Install OpenCV '
            '(apt install python3-opencv) or moms-apriltag (pip install '
            f'moms-apriltag pillow). OpenCV error was: {last_err}; '
            f'moms error was: {exc}')


# ----------------------------------------------------------------------- world
def _wall(name, x, y, z, sx, sy, sz):
    return f"""
    <model name="{name}">
      <static>true</static>
      <link name="link">
        <pose>{x:.4f} {y:.4f} {z:.4f} 0 0 0</pose>
        <collision name="c"><geometry><box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box></geometry></collision>
        <visual name="v">
          <geometry><box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box></geometry>
          <material>
            <ambient>0.8 0.8 0.82 1</ambient>
            <diffuse>0.8 0.8 0.82 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
      </link>
    </model>"""


def make_marker_model_sdf(tmap, tex_name: str) -> str:
    # plaque is larger than the black tag by the quiet-zone ratio, so the black
    # AprilTag rendered on the wall is exactly tmap.tag_size metres.
    s = tmap.tag_size * QUIET_RATIO
    return f"""<?xml version="1.0" ?>
<!-- Regenerated by generate_world.py so size/texture match config/tag_map.yaml.
     The +X face carries the tag. -->
<sdf version="1.10">
  <model name="apriltag_marker">
    <static>true</static>
    <link name="link">
      <visual name="tag">
        <pose>0 0 0 0 0 0</pose>
        <geometry>
          <box><size>0.02 {s:.3f} {s:.3f}</size></box>
        </geometry>
        <material>
          <ambient>1 1 1 1</ambient>
          <diffuse>1 1 1 1</diffuse>
          <specular>0.05 0.05 0.05 1</specular>
          <pbr>
            <metal>
              <albedo_map>materials/textures/{tex_name}</albedo_map>
              <emissive_map>materials/textures/{tex_name}</emissive_map>
              <metalness>0.0</metalness>
              <roughness>0.95</roughness>
            </metal>
          </pbr>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""


def make_world_sdf(tmap, model_uri='model://apriltag_marker') -> str:
    sx, sy = tmap.size_x, tmap.size_y
    h = tmap.wall_height
    t = tmap.wall_thickness

    walls = ''.join([
        _wall('wall_south', sx / 2, 0.0, h / 2, sx + 2 * t, t, h),
        _wall('wall_north', sx / 2, sy, h / 2, sx + 2 * t, t, h),
        _wall('wall_west', 0.0, sy / 2, h / 2, t, sy, h),
        _wall('wall_east', sx, sy / 2, h / 2, t, sy, h),
    ])

    tags = ''
    for i in range(tmap.num_tags):
        x, y = tmap.positions[i]
        yaw = float(tmap.normal_yaw[i])
        z = tmap.mount_height
        tags += f"""
    <include>
      <uri>{model_uri}</uri>
      <name>ar_tag_{i}</name>
      <pose>{x:.4f} {y:.4f} {z:.4f} 0 0 {yaw:.6f}</pose>
    </include>"""

    return f"""<?xml version="1.0" ?>
<sdf version="1.10">
  <world name="ar_room">
    <physics name="default" type="dart">
      <max_step_size>0.004</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>

    <scene>
      <ambient>0.6 0.6 0.6 1</ambient>
      <background>0.85 0.88 0.92 1</background>
      <grid>false</grid>
    </scene>

    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.4 0.3 -0.9</direction>
    </light>
    <light type="point" name="lamp">
      <pose>{sx/2:.2f} {sy/2:.2f} {h+0.5:.2f} 0 0 0</pose>
      <diffuse>0.6 0.6 0.6 1</diffuse>
      <attenuation><range>30</range><linear>0.05</linear></attenuation>
    </light>

    <model name="ground">
      <static>true</static>
      <link name="link">
        <!-- centered under the room so the floor spans the whole room + margin -->
        <pose>{sx/2:.3f} {sy/2:.3f} 0 0 0 0</pose>
        <collision name="c">
          <geometry><plane><normal>0 0 1</normal><size>{sx+4:.1f} {sy+4:.1f}</size></plane></geometry>
        </collision>
        <visual name="v">
          <geometry><plane><normal>0 0 1</normal><size>{sx+4:.1f} {sy+4:.1f}</size></plane></geometry>
          <material>
            <ambient>0.3 0.3 0.33 1</ambient>
            <diffuse>0.35 0.35 0.38 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
{walls}
{tags}
  </world>
</sdf>
"""


# ------------------------------------------------------------------------- cli
def _default_paths():
    here = os.path.dirname(os.path.abspath(__file__))      # .../pf_localization/pf_localization
    pkg = os.path.dirname(here)                            # .../src/pf_localization
    return {
        'tag_map': os.path.join(pkg, 'config', 'tag_map.yaml'),
        'world': os.path.join(pkg, 'worlds', 'room.sdf'),
        'model_tex': os.path.join(pkg, 'models', 'apriltag_marker',
                                  'materials', 'textures'),
        'pkg_tex': os.path.join(pkg, 'materials', 'textures'),
    }


def run(tag_map_path, world_path, model_tex_dir, pkg_tex_dir):
    tmap = load_tag_map(tag_map_path)
    tex_name = f'{tmap.family}_{tmap.tag_id}.png'
    model_tex = os.path.join(model_tex_dir, tex_name)
    generate_tag_texture(model_tex, tmap.tag_id, tmap.family)
    os.makedirs(pkg_tex_dir, exist_ok=True)
    shutil.copyfile(model_tex, os.path.join(pkg_tex_dir, tex_name))

    # keep the marker model's box size + texture in sync with the map
    # model_tex_dir = .../apriltag_marker/materials/textures -> up two levels
    marker_dir = os.path.dirname(os.path.dirname(model_tex_dir))
    model_sdf = os.path.join(marker_dir, 'model.sdf')
    with open(model_sdf, 'w') as f:
        f.write(make_marker_model_sdf(tmap, tex_name))

    os.makedirs(os.path.dirname(world_path), exist_ok=True)
    with open(world_path, 'w') as f:
        f.write(make_world_sdf(tmap))
    print(f'Wrote world  -> {world_path}')
    print(f'Wrote texture-> {model_tex}')
    print(f'Wrote model  -> {model_sdf}')
    return tex_name


def main(argv=None):
    d = _default_paths()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tag-map', default=d['tag_map'])
    ap.add_argument('--world', default=d['world'])
    ap.add_argument('--model-tex-dir', default=d['model_tex'])
    ap.add_argument('--pkg-tex-dir', default=d['pkg_tex'])
    args = ap.parse_args(argv)
    run(args.tag_map, args.world, args.model_tex_dir, args.pkg_tex_dir)


if __name__ == '__main__':
    main()
