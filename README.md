# Particle Filter Localization with AR Tags (Gazebo, ROS 2)

Monte-Carlo localization of a teleoperated mobile robot in a simulated room. The
room's four walls carry **8 identical AR tags** (tag36h11, *same id*, placed
**asymmetrically**, 2 per wall). The robot's forward camera detects the tags, but
because every tag looks the same, a detection never says *which* tag it is. The
filter therefore uses a **multi-hypothesis sensor model** that explains every
observation as a sum over all 8 candidate tags, and the particle cloud collapses
from a room-filling spread onto the true pose as the asymmetric layout is
gradually revealed by driving.

> Built for Ubuntu 24.04 with **ROS 2 Jazzy** + **Gazebo Harmonic**.

📖 **Full documentation** (architecture, every parameter, how-to recipes, the
algorithm, topics, troubleshooting, extending): see [`DOCUMENTATION.md`](DOCUMENTATION.md).
The project report is in [`report/report.md`](report/report.md).

```
 teleop ──/cmd_vel──▶ Gazebo (room + 8 tags + robot+camera)
                         │            │                 │
                      /odom      /camera/*        /ground_truth
                         │            │                 │
                  odom_noise_node   detector      (viz + sim detector)
                         │       (camera|sim)
                    /odom_noisy        │
                         └────────┐    └──/tag_detections──┐
                                  ▼                        ▼
                          ┌───────────────────────────────────┐
                          │        particle_filter_node        │
                          │  predict (odom) → update (tags) →  │
                          │  resample → estimate + cloud       │
                          └───────────────────────────────────┘
                                  │            │
                            /pf/estimate  /pf/particle_markers
                                  │            │
                                  ▼            ▼   +  viz_node (room, tags, paths)
                                       RViz (single view)
```

## 1. Install (one-time, needs sudo)

```bash
cd pf_ws
sudo bash setup_env.sh
```

This installs ROS 2 Jazzy, Gazebo Harmonic, the bridges, teleop, OpenCV/cv_bridge
and the `pupil-apriltags` detector. It is the only step that needs root.

## 2. Build

```bash
cd pf_ws
source /opt/ros/jazzy/setup.bash
# generate the world + tag texture from the single source of truth (config/tag_map.yaml):
python3 src/pf_localization/pf_localization/generate_world.py
colcon build --symlink-install
source install/setup.bash
```

## 3. Run

```bash
# Reliable, deterministic detections (recommended first run):
ros2 launch pf_localization bringup.launch.py detection_mode:=sim

# Real camera AprilTag detection through the robot's camera:
ros2 launch pf_localization bringup.launch.py detection_mode:=camera
```

A Gazebo window, an RViz window, and a small **teleop** terminal open. Click the
teleop window and drive with the `teleop_twist_keyboard` keys (`i`/`,` forward/back,
`j`/`l` turn, `k` stop). Watch the RViz view:

* **Particle cloud** (points, blue→red by weight) starts spread across the whole
  room, forms a few competing clusters, then collapses onto the true pose.
* **Blue path** = odometry only (drifts away).
* **Red path** = particle-filter estimate (stays locked on).
* **Green path** = ground truth (reference).

## Nodes & topics

| Node | Subscribes | Publishes |
|------|------------|-----------|
| `odom_noise_node` | `/odom` | `/odom_noisy` |
| `tag_detector_node` (camera) | `/camera`, `/camera/camera_info` | `/tag_detections` |
| `sim_detector_node` (sim) | `/ground_truth` | `/tag_detections` |
| `particle_filter_node` | `/odom_noisy`, `/tag_detections`, `/ground_truth` | `/pf/estimate`, `/pf/estimate_cov`, `/pf/particle_markers`, `/pf/error` |
| `viz_node` | `/odom_noisy`, `/pf/estimate`, `/ground_truth` | `/viz/static`, `/viz/robots`, `/pf/path_*` |

## Key parameters (`src/pf_localization/config/filter_params.yaml`)

| Group | Parameter | Meaning |
|-------|-----------|---------|
| particles | `num_particles` | size of the particle set (default 5000) |
| resample | `random_inject_frac` | cap on Augmented-MCL adaptive injection (default 5%) |
| resample | `inject_threshold` | only inject on a severe likelihood collapse — keeps a converged cloud from re-guessing |
| viz | `trail_seconds` | rolling trajectory trail length in RViz (default 20 s; 0 = full path) |
| viz | `trail_max_jump` | wipe the trail if a point jumps > this [m] (anti-"splash" long lines) |
| motion | `alpha1..alpha4` | odometry motion-model noise (Thrun Table 5.6) |
| sensor | `sigma_range`, `sigma_bearing` | measurement noise std |
| sensor | `max_range`, `hfov_rad`, `max_view_angle_deg` | camera visibility (shared with detector) |
| sensor | `z_clutter` | uniform clutter floor per detection |
| resample | `ess_ratio_threshold` | resample when ESS/N drops below |

The room and the 8 tag poses live in `config/tag_map.yaml` — the **single source
of truth** that `generate_world.py` turns into the Gazebo world, and that the
filter loads as its known map.

## Capturing media for the report

```bash
# screen-record the RViz window (initial spread / partial / converged):
# (uses ffmpeg; install with: sudo apt install ffmpeg)
ffmpeg -video_size 1500x900 -framerate 25 -f x11grab -i :1+0,0 report/media/run.mp4
```

Save three stills (initial spread, partial convergence, final) into
`report/media/` and reference them from `report/report.md`.

## Repository layout

```
pf_ws/
├── setup_env.sh                     # one-shot sudo install
├── requirements.txt                 # pip deps (pupil-apriltags, ...)
├── report/report.md                 # the project report
└── src/pf_localization/
    ├── pf_localization/             # the nodes + algorithms
    ├── config/                      # tag_map.yaml, filter_params.yaml, rviz, bridge
    ├── launch/bringup.launch.py
    ├── worlds/ models/ materials/   # Gazebo assets (room, robot, tags, textures)
```

## Troubleshooting

* **No camera detections** — confirm the bridged topic names with `gz topic -l`
  and `ros2 topic list`; the camera image is bridged to `/camera` and info to
  `/camera/camera_info` (adjust `config/ros_gz_bridge.yaml` / the launch remaps
  if your Gazebo scopes them differently). When in doubt, use `detection_mode:=sim`.
* **Gazebo can't find the tag model** — the launch file sets `GZ_SIM_RESOURCE_PATH`
  to the installed `models/` dir; make sure you ran `colcon build` after editing
  models, and re-`source install/setup.bash`.
* **`pupil_apriltags` import error** — re-run the pip line from `setup_env.sh`
  (`pip install --break-system-packages pupil-apriltags`).

## License

MIT.
