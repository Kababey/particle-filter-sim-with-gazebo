# Particle Filter Localization with AR Tags — Full Documentation

Monte-Carlo (particle filter) localization of a teleoperated mobile robot inside a
simulated room whose walls carry **8 identical AR tags**. Because every tag looks
the same, a camera detection says *a tag is there*, not *which* tag — so the filter
uses a **multi-hypothesis** sensor model that explains each observation as a sum
over all 8 candidate tags. Built on **ROS 2 Jazzy + Gazebo Harmonic** (Ubuntu 24.04).

---

## Table of contents

1. [Concept](#1-concept)
2. [System architecture](#2-system-architecture)
3. [Requirements](#3-requirements)
4. [Installation](#4-installation)
5. [Building](#5-building)
6. [Running](#6-running)
7. [Driving the robot](#7-driving-the-robot)
8. [The visualization](#8-the-visualization)
9. [Configuration reference](#9-configuration-reference)
10. [How-to recipes](#10-how-to-recipes)
11. [Algorithm details](#11-algorithm-details)
12. [Topics & messages](#12-topics--messages)
13. [Generating figures / capturing media](#13-generating-figures--capturing-media)
14. [Troubleshooting](#14-troubleshooting)
15. [Repository layout](#15-repository-layout)
16. [Extending the project](#16-extending-the-project)
17. [FAQ](#17-faq)

---

## 1. Concept

The robot starts at a **random, unknown** pose. The particle filter holds a set of
weighted hypotheses (particles) about the pose `(x, y, θ)` and refines them every
cycle:

* **Predict** — push every particle through the noisy odometry motion (the cloud spreads).
* **Update** — reweight particles by how well they explain the camera's anonymous
  tag detections (the cloud sharpens).
* **Resample** — concentrate particles where the posterior mass is.

Because the tags are indistinguishable, early on several pose clusters survive (the
room looks the same from multiple places). As the robot drives and sees more of the
**asymmetric** tag layout, the wrong clusters die and the cloud collapses onto the
true pose.

---

## 2. System architecture

```
 teleop_twist_keyboard ──/cmd_vel──▶ Gazebo Harmonic
                                       │   (room + 8 tags + diff-drive robot + camera)
        ┌───────────────┬─────────────┼──────────────────┬───────────────┐
     /odom          /camera     /camera/camera_info   /ground_truth     /clock
        │            (image)          │                    │               │
   odom_noise_node   │           (bridged)            (bridged)       (bridged)
        │            │                │                    │
   /odom_noisy       └──┬─────────────┘                    │
        │      tag_detector_node (camera)   OR   sim_detector_node (sim)
        │               └────────── /tag_detections ───────┘
        │                                  │
        ▼                                  ▼
   ┌──────────────────────────────────────────────────────────┐
   │                  particle_filter_node                     │
   │  predict(odom) → update(tags, multi-hypothesis) →         │
   │  resample(+Augmented-MCL injection) → estimate(cluster)   │
   └──────────────────────────────────────────────────────────┘
        │                 │                    │
  /pf/estimate     /pf/particle_markers    /pf/error
        │                 │
        └───── viz_node (tags, room, 3 trajectories) ──── RViz (single view)
```

### Nodes

| Node | File | Role |
|------|------|------|
| `particle_filter_node` | `particle_filter_node.py` | The filter: predict, update, resample, estimate. |
| `tag_detector_node` | `tag_detector_node.py` | Real camera AprilTag detector (pupil-apriltags). |
| `sim_detector_node` | `sim_detector_node.py` | Ground-truth-based detector (deterministic fallback). |
| `odom_noise_node` | `odom_noise_node.py` | Adds realistic drift to the perfect sim odometry. |
| `viz_node` | `viz_node.py` | Room + tag markers, robot markers, and the 3 trajectory paths. |
| `generate_world` | `generate_world.py` | Tool: emits `room.sdf` + tag texture from `tag_map.yaml`. |

The filter, detectors, and the visualization are decoupled by **topics**; you can
swap detectors with one launch argument because both publish the same
`/tag_detections` contract.

### Coordinate frames

* `map` — the fixed world frame (RViz fixed frame). All markers/paths are in `map`.
* `odom` — the robot's odometry frame; a static `map→odom` identity TF is published
  so RViz has a frame to anchor on. The filter uses odometry *increments* only, so
  the odom frame offset never matters.
* `base_link` — the robot body; detections are expressed here (range/bearing).

---

## 3. Requirements

* **Ubuntu 24.04** (Noble), x86_64.
* A GPU + working desktop OpenGL (the simulation renders the camera). An Intel iGPU
  is fine; on dual-GPU laptops run with the **GUI** (default), not headless.
* ~5 GB disk for ROS 2 + Gazebo.
* Internet access for the install.

The software stack (installed by `setup_env.sh`):

* **ROS 2 Jazzy Jalisco** (rclpy, RViz2, tf2, teleop).
* **Gazebo Harmonic** via `ros_gz` (sim, bridge, image bridge).
* **OpenCV** (`python3-opencv`, used to generate the tag texture) and **cv_bridge**.
* **pupil-apriltags** (pip, the AprilTag detector).

---

## 4. Installation

One-time, needs `sudo` (this is the only step that does):

```bash
cd pf_ws
sudo bash setup_env.sh
```

What it does, step by step:

1. Locale + prerequisites (`curl`, `gnupg`, `software-properties-common`, …).
2. Adds the official ROS 2 apt repository (`ros2-apt-source` .deb).
3. Installs `ros-jazzy-desktop`, `ros-jazzy-ros-gz`, `ros-jazzy-cv-bridge`,
   `ros-jazzy-teleop-twist-keyboard`, `python3-colcon-common-extensions`,
   `python3-opencv`, `xterm`, …
4. `rosdep init/update`.
5. `pip install --break-system-packages --no-deps pupil-apriltags`.

> **Why `--no-deps`?** Ubuntu 24.04 ships numpy 1.26 (Debian) and ROS/cv_bridge are
> built against it. Letting pip pull `pupil-apriltags`' dependencies would upgrade
> numpy to 2.x and break ROS. `--no-deps` installs only the detector; numpy stays.

The script is idempotent — safe to re-run.

---

## 5. Building

```bash
cd pf_ws
source /opt/ros/jazzy/setup.bash
# generate the world + tag texture from the single source of truth:
python3 src/pf_localization/pf_localization/generate_world.py
colcon build --symlink-install
source install/setup.bash
```

`--symlink-install` symlinks Python files and data, so editing a node or a YAML in
`src/` takes effect on the next run **without rebuilding** (you still rebuild after
adding/renaming files or changing `setup.py`).

`generate_world.py` writes three things from `config/tag_map.yaml`:
`worlds/room.sdf`, the AprilTag PNG texture, and the (size-matched) tag marker model.
Re-run it any time you edit the room or tag layout.

---

## 6. Running

```bash
source /opt/ros/jazzy/setup.bash && source install/setup.bash

# real camera AprilTag detection (default):
ros2 launch pf_localization bringup.launch.py

# deterministic ground-truth detector (recommended for a guaranteed demo):
ros2 launch pf_localization bringup.launch.py detection_mode:=sim
```

This opens **Gazebo**, **RViz**, and a small **teleop** terminal. The robot spawns at
a **random** pose (printed at startup).

### Launch arguments

| Argument | Default | Meaning |
|----------|---------|---------|
| `detection_mode` | `camera` | `camera` (pupil-apriltags on the rendered image) or `sim` (ground-truth detections). |
| `spawn_x` | random | Robot start x [m]. Each launch it is placed close to a random wall, facing a tag-free section of it (so it initially sees nothing and the cloud stays fully spread). Set to pin it. |
| `spawn_y` | random | Robot start y [m]. |
| `spawn_yaw` | random | Robot start heading [rad] (faces the nearby wall). |
| `rviz` | `true` | Open RViz with the project layout. |
| `teleop` | `true` | Open an `xterm` running `teleop_twist_keyboard`. |
| `headless` | `false` | Run Gazebo without its GUI (server only). See Troubleshooting. |

Examples:

```bash
# repeatable demo at a fixed start pose, no teleop window (drive from another shell):
ros2 launch pf_localization bringup.launch.py \
    detection_mode:=sim spawn_x:=2.0 spawn_y:=1.5 spawn_yaw:=0.5 teleop:=false

# camera mode without RViz (e.g. when recording only Gazebo):
ros2 launch pf_localization bringup.launch.py detection_mode:=camera rviz:=false
```

---

## 7. Driving the robot

A `teleop_twist_keyboard` window opens (when `teleop:=true`). **Click it to give it
focus**, then:

```
   u    i    o
   j    k    l        i / , : forward / backward
   m    ,    .        j / l : turn left / right
                      k     : stop
                      q / z : increase / decrease max speeds
```

It publishes `geometry_msgs/Twist` on `/cmd_vel`. To drive without the window (e.g.
scripted), publish to `/cmd_vel` yourself:

```bash
ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.3}, angular: {z: 0.2}}'
```

Drive around so the camera sweeps different walls — that is what disambiguates the
pose. Driving across the room (facing walls) yields more simultaneous tags than
driving parallel to a wall.

---

## 8. The visualization

RViz opens with `config/pf.rviz` (fixed frame `map`), showing everything Task 3
requires in one view:

| Display | Topic | Looks like |
|---------|-------|-----------|
| Tags & Room | `/viz/static` | room outline + 8 black tag plaques with labels |
| Particle Cloud | `/pf/particle_markers` | each particle as a point **and a tiny heading arrow**, colored **yellow** with brightness by local density — **bright = more probable**, dim = sparse (stable, no per-frame flicker). The pose estimate (red trajectory + big red arrow) marks the most-probable pose. |
| Robot Poses | `/viz/robots` | green arrow = ground truth, red arrow = estimate |
| Tags In View | `/viz/visible_tags` | yellow halos on the tags the camera currently sees + gold sight-lines from the robot to them (camera mode) |
| Odometry-only Path | `/pf/path_odom` | **blue** line (drifts) |
| Filter Estimate Path | `/pf/path_estimate` | **red** line (tracks truth) |
| Ground Truth Path | `/pf/path_ground_truth` | **green** line |
| Robot Camera (off by default) | `/camera` | the live camera image |

Enable the camera image inside RViz by ticking **Robot Camera** in the Displays
panel.

**Separate camera window (camera mode).** `tag_detector_node` opens its own window
("Robot camera - AprilTag detections") showing the live camera with every detected
tag outlined in green, labelled `id0` + range, and a "tags seen: N" banner. The
same annotated image is published on `/camera/detections` (view it anywhere with
`ros2 run rqt_image_view rqt_image_view /camera/detections`). Disable the pop-up
window with the `tag_detector_node.show_window: false` parameter.

Live numbers:

```bash
ros2 topic echo /pf/error          # |estimate − ground truth|  in metres
```

---

## 9. Configuration reference

Everything tunable lives in two YAML files (plus launch arguments).

### 9.1 `config/tag_map.yaml` — room + tags (single source of truth)

```yaml
room:
  size_x: 8.0          # interior length along +x  [m]
  size_y: 6.0          # interior length along +y  [m]
  wall_height: 1.5
  wall_thickness: 0.10

tag:
  family: tag36h11     # AprilTag family
  id: 0                # ALL tags share this id (that's the point)
  size: 0.40           # black-square side length [m]; identical for all tags
  mount_height: 0.55   # z of every tag centre [m]

tags:                  # 8 tags, 2 per wall. normal_deg = direction the face points
  - {x: 0.83, y: 0.06, normal_deg:  90}   # south wall, faces +y
  ...
```

This file is consumed by `generate_world.py` (to build the world) **and** by the
filter / sim detector (as the known map), so the simulated world and the filter's
map can never disagree. **After editing it, re-run `generate_world.py` and
`colcon build`.**

> The tag positions were chosen to maximize asymmetry (no symmetric "ghost" pose).
> If you change them, keep them clearly asymmetric — see recipe 10.2.

### 9.2 `config/filter_params.yaml` — runtime parameters

Each top-level key is a node name; ROS hands each node its own section.

**`particle_filter_node`**

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `num_particles` | `5000` | Particle-set size. More = better multi-modal coverage, slower. |
| `init_mode` | `global` | `global` = uniform over the room (the interesting case). |
| `alpha1..alpha4` | `0.08, 0.02, 0.06, 0.02` | Odometry motion-model noise (Thrun Table 5.6). α1 rot-from-rot, α2 rot-from-trans, α3 trans-from-trans, α4 trans-from-rot. |
| `min_motion_trans` / `min_motion_rot` | `0.002` | Skip the predict step below this much motion. |
| `sigma_range` | `0.20` | Range measurement std [m] assumed by the likelihood. |
| `sigma_bearing` | `0.10` | Bearing measurement std [rad]. |
| `z_clutter` | `0.05` | Uniform clutter floor added per detection (robustness to false positives). |
| `max_range` | `8.0` | Tag visible only within this range [m]. **Must match the detector.** |
| `hfov_rad` | `1.50` | Camera horizontal FOV [rad]. **Must match the detector & robot camera.** |
| `max_view_angle_deg` | `75.0` | Tag visible only if viewed within this angle of its normal. **Must match the detector.** |
| `ess_ratio_threshold` | `0.5` | Resample when ESS/N drops below this. |
| `random_inject_frac` | `0.05` | **Cap** on Augmented-MCL global injection (0 disables recovery). |
| `inject_threshold` | `0.2` | Deadband: inject only when the mean likelihood collapses by more than this fraction (a stuck lock / kidnap). Raise it (e.g. 0.55) to make a converged cloud stick harder; lower it for faster recovery. |
| `w_slow_alpha` | `0.01` | Slow EMA rate of the mean measurement likelihood. |
| `w_fast_alpha` | `0.06` | Fast EMA rate; inject only when fast < slow (sustained drop). |
| `wall_margin` | `0.15` | Map constraint [m]: the robot centre must stay this far inside the walls. Particles that leave the room are resampled back inside, so the cloud never bleeds through the walls. |
| `map_frame` | `map` | Frame id for published markers/poses. |

**`sim_detector_node`** (used when `detection_mode:=sim`)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `rate_hz` | `10.0` | Detection publish rate. |
| `max_range`, `hfov_rad`, `max_view_angle_deg` | `8.0, 1.50, 75.0` | Visibility (shared with the filter). |
| `sigma_range` | `0.12` | True range noise injected (≤ the filter's assumed σ). |
| `sigma_bearing` | `0.05` | True bearing noise. |
| `detection_prob` | `0.95` | Probability of detecting a visible tag (dropout). |

**`tag_detector_node`** (used when `detection_mode:=camera`)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `tag_size` | `0.40` | The **black-square** size [m]; must equal `tag.size`. (The plaque is rendered 1.2× larger for the quiet zone, so the black tag is exactly this.) |
| `camera_x` | `0.18` | Camera forward mount offset on `base_link` [m]; matches `model.sdf`. |
| `camera_z` | `0.20` | Camera height on `base_link` [m]. |
| `max_range` | `8.0` | Reject detections farther than this. |
| `decimate` | `1.0` | AprilTag quad decimation (higher = faster, less sensitive). |
| `min_decision_margin` | `30.0` | Reject weak detections below this margin. |

**`odom_noise_node`** (turns perfect sim odometry into realistic drift)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `trans_scale` | `1.03` | Systematic odometer scale error. |
| `rot_per_trans` | `0.012` | Systematic heading drift per metre [rad/m]. |
| `trans_noise_per_m` | `0.03` | Random-walk translation std. |
| `rot_noise_per_m` | `0.015` | Random-walk heading std per metre. |
| `rot_noise_per_rad` | `0.03` | Random-walk heading std per radian rotated. |
| `seed` | `0` | RNG seed (0 = nondeterministic). |

Set the noise terms to 0 (and `trans_scale`/`rot_scale` to 1) to feed the filter
near-perfect odometry.

**`viz_node`**

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `trail_seconds` | `20.0` | Rolling trajectory trail length [s]. The odom/estimate/ground-truth paths show only the last `trail_seconds`; older points are wiped. Set `0` to keep the full path (e.g. for the trajectory-comparison figure). |
| `trail_max_jump` | `0.6` | Anti-"splash" [m]: if a path point jumps farther than this from the previous one (e.g. the estimate hops between clusters), the old trail is wiped instead of drawing a long straight line across the room. |
| `draw_ground_truth` | `true` | Draw the ground-truth path/arrow. |

### 9.3 Launch arguments

See [section 6](#launch-arguments).

### 9.4 `config/ros_gz_bridge.yaml`

Maps Gazebo topics to ROS 2 topics (`/cmd_vel`, `/odom`, `/clock`,
`/model/pf_robot/pose → /ground_truth`, `/camera_info → /camera/camera_info`). The
camera **image** is bridged separately by `ros_gz_image` in the launch file. Edit
this if your Gazebo version scopes the camera-info topic differently (see
Troubleshooting).

---

## 10. How-to recipes

### 10.1 Change the room size
Edit `room.size_x` / `room.size_y` in `tag_map.yaml`, move the tags to stay on the
new walls, then:
```bash
python3 src/pf_localization/pf_localization/generate_world.py && colcon build --symlink-install
```
The floor, walls, filter bounds, and random-spawn range all follow automatically.

### 10.2 Move / add / remove tags
Edit the `tags:` list in `tag_map.yaml` (each entry is `{x, y, normal_deg}`; keep
tags on the interior wall face). To verify the layout has **no ghost pose**, check
that it stays strongly asymmetric — a quick script:
```python
# saves you from a symmetric layout the filter would lock onto wrongly
# (rotate/mirror the tag set; every transform should move at least one tag far away)
```
After editing, re-run `generate_world.py` + `colcon build`. The number of tags is
not fixed at 8 — the sensor model sums over however many you define.

### 10.3 Change the physical tag size
Edit `tag.size` in `tag_map.yaml` **and** `tag_detector_node.tag_size` in
`filter_params.yaml` to the same value, then regenerate + rebuild. The plaque is
auto-scaled by the quiet-zone ratio so the black tag matches.

### 10.4 Change the robot or camera
Edit `models/pf_robot/model.sdf`:
* camera FOV: `<horizontal_fov>` — **also** update `hfov_rad` in *both* the filter
  and `sim_detector` sections of `filter_params.yaml` to match.
* camera resolution: `<image><width>/<height>`.
* wheel size / track: `<wheel_radius>`, `<wheel_separation>` in the DiffDrive plugin
  (and the wheel link geometry).
Then `colcon build`.

### 10.5 Make the start pose deterministic
Pass explicit spawn args: `spawn_x:=2.0 spawn_y:=1.5 spawn_yaw:=0.5`. Omit them for a
fresh random pose every launch.

### 10.6 Switch detection mode
`detection_mode:=sim` (deterministic, no vision) or `detection_mode:=camera` (real
AprilTag detection). Both feed the identical `/tag_detections`.

### 10.7 Feed the filter perfect odometry (no drift)
In `filter_params.yaml → odom_noise_node`, set `trans_scale: 1.0`, `rot_scale: 1.0`,
and the three noise terms to `0.0`. The odometry-only path then matches ground truth.

### 10.8 Make convergence faster / slower or more / less stable
* **Faster, greedier**: lower `sigma_range`/`sigma_bearing` (sharper likelihood).
* **Stop the cloud "re-guessing" after it converges**: raise `inject_threshold`
  (e.g. 0.7) so injection fires only on a severe collapse, or lower
  `random_inject_frac`. Setting `random_inject_frac: 0` disables injection entirely
  (cleanest once locked, but then it cannot recover from a wrong lock).
* **Better global coverage / harder rooms**: raise `num_particles`.
* **Kidnapped-robot recovery**: lower `inject_threshold` and/or raise
  `random_inject_frac` so it re-localizes faster if teleported.

### 10.10 Shorten / lengthen the trajectory trail
`viz_node.trail_seconds` controls how long the odom/estimate/ground-truth paths
persist in RViz (default 20 s rolling tail). Set it to `0` to keep the full path
(useful for the odom-vs-filter trajectory figure). `viz_node.trail_max_jump`
(default 0.6 m) wipes the trail when a point jumps farther than that, so the
estimate hopping between clusters never draws a long straight line.

### 10.9 Run two terminals (manual drive)
```bash
# terminal A
ros2 launch pf_localization bringup.launch.py detection_mode:=sim teleop:=false
# terminal B
source install/setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

---

## 11. Algorithm details

The filter is a sample-based Bayes filter. With `bel(xₜ) = p(xₜ | z₁:ₜ, u₁:ₜ)`:

### Prediction (motion model) — `motion_model.py`
Sampling odometry model (Probabilistic Robotics, Table 5.6). The odometry increment
is decomposed into rotate–translate–rotate `(δrot1, δtrans, δrot2)`; each particle
gets a noisy copy (variances `αᵢ·(…)²`), then is moved forward. Vectorized over all
particles. This realizes `x̄ₜ^[m] ~ p(xₜ | uₜ, xₜ₋₁^[m])`.

### Update (multi-hypothesis sensor model) — `sensor_model.py`
Each detection is an anonymous range/bearing `(r, φ)` in `base_link`. For a particle
and tag *i* the expected `(r̂ᵢ, φ̂ᵢ)` is computed; the per-tag likelihood is a
Gaussian, **gated** by whether tag *i* is actually visible from that particle (in
range, in FOV, on the tag's front face). The observation likelihood marginalizes
over the unknown tag identity:

```
p(z | x) = (1/8) Σ_{i=1..8} 1[tagᵢ visible from x] · N(r; r̂ᵢ, σ_r) · N(φ; φ̂ᵢ, σ_φ) + z_clutter
```

Multiple detections in a frame multiply (assumed independent). The filter **never**
collapses this to the single nearest tag (that would destroy the multi-hypothesis
behaviour). Fully vectorized as an `N×8` broadcast.

### Resampling + recovery — `resampling.py`
Low-variance (systematic) resampling, triggered only when `ESS = 1/Σwᵢ² < 0.5N`.
**Augmented MCL**: track slow/fast EMAs of the mean measurement likelihood and inject
global-uniform particles with probability `max(0, 1 − w_fast/w_slow)` (capped). When
well-localized the EMAs agree → no injection; when the evidence stops matching the
belief → inject and recover.

### Estimate
The reported pose is the **dominant cluster**: the densest cell of a coarse spatial
histogram (robust even right after resampling, when weights are equal), averaged
(circular mean for θ). This is meaningful even while the belief is multi-modal,
unlike a global weighted mean that would sit between clusters.

### Bayesian mapping

| Bayes filter | Here |
|---|---|
| prior `bel(xₜ₋₁)` | the particle set from the last step |
| prediction | `OdometryMotionModel.sample` |
| likelihood `p(zₜ|xₜ)` | `MultiHypothesisSensorModel.update_weights` |
| update | reweight + normalize |
| posterior | the reweighted (and resampled) set |
| resampling | `systematic_resample` |

---

## 12. Topics & messages

| Topic | Type | Direction / Producer |
|-------|------|----------------------|
| `/cmd_vel` | `geometry_msgs/Twist` | teleop → Gazebo |
| `/odom` | `nav_msgs/Odometry` | Gazebo (wheel odom) |
| `/odom_noisy` | `nav_msgs/Odometry` | `odom_noise_node` (filter input) |
| `/ground_truth` | `geometry_msgs/PoseStamped` | Gazebo (eval/viz only) |
| `/camera` | `sensor_msgs/Image` | Gazebo camera |
| `/camera/camera_info` | `sensor_msgs/CameraInfo` | Gazebo camera |
| `/clock` | `rosgraph_msgs/Clock` | Gazebo (sim time) |
| `/tag_detections` | `geometry_msgs/PoseArray` | detector (anonymous tag positions in `base_link`) |
| `/camera/detections` | `sensor_msgs/Image` | `tag_detector_node` (annotated camera, camera mode) |
| `/pf/estimate` | `geometry_msgs/PoseStamped` | filter |
| `/pf/estimate_cov` | `geometry_msgs/PoseWithCovarianceStamped` | filter |
| `/pf/particle_markers` | `visualization_msgs/Marker` | filter (cloud, colored by weight) |
| `/pf/error` | `std_msgs/Float32` | filter (vs ground truth) |
| `/pf/path_odom` `/pf/path_estimate` `/pf/path_ground_truth` | `nav_msgs/Path` | `viz_node` |
| `/viz/static` `/viz/robots` `/viz/visible_tags` | `visualization_msgs/MarkerArray` | `viz_node` |

Inspect anything live:
```bash
ros2 topic list
ros2 topic echo /pf/error
ros2 topic hz /tag_detections
ros2 node list
```

---

## 13. Generating figures / capturing media

The report figures in `report/media/` are rendered from live filter data with
matplotlib (no screen-grabbing). To recreate them, run the stack (sim mode) and a
small capture node that subscribes to `/pf/particle_markers`, `/pf/path_*`,
`/ground_truth` and draws the room/tags from `tag_map.yaml` — see the report for the
three stages (initial spread, partial convergence, converged) and the trajectory plot.

To record a **video** of RViz/Gazebo, use a desktop screen recorder (e.g. the GNOME
screen recorder, `Ctrl+Alt+Shift+R`) or install one:
```bash
sudo apt install ffmpeg     # then:
ffmpeg -video_size 1500x900 -framerate 25 -f x11grab -i :1+0,0 report/media/run.mp4
```

---

## 14. Troubleshooting

**Gazebo dies / no `/clock` when headless.** On dual-GPU laptops the headless
camera-render (EGL) path can fail. **Run with the GUI** (the default, `headless:=false`)
— it uses the working desktop OpenGL. Verify the sim is live with
`ros2 topic echo /clock --once`.

**No camera detections (`detection_mode:=camera`).**
* Confirm the image is flowing: `ros2 topic hz /camera`.
* Confirm topic names: `gz topic -l` and `ros2 topic list`. The image is bridged to
  `/camera`, info to `/camera/camera_info`. If your Gazebo scopes them differently,
  fix `config/ros_gz_bridge.yaml` and the launch `image_bridge` arg, then re-source.
* Quick fallback that always works: `detection_mode:=sim`.

**Filter converges to the wrong place (a "ghost").** The tag layout is too symmetric.
Use a strongly asymmetric layout (recipe 10.2) and keep `random_inject_frac > 0` so a
contradicted hypothesis can be abandoned.

**`pupil_apriltags` import error.** Re-run the pip line:
`pip install --break-system-packages --no-deps pupil-apriltags`.

**numpy/ROS errors after pip installs.** Don't let pip upgrade numpy. Always use
`--no-deps` for `pupil-apriltags`. The system numpy must stay at 1.26 (Debian).

**Gazebo can't find the tag/robot model.** The launch sets `GZ_SIM_RESOURCE_PATH` to
the installed `models/` dir; make sure you ran `colcon build` after editing models and
re-`source install/setup.bash`.

**Teleop keys do nothing.** Click the teleop `xterm` window to focus it. If no window
appears, ensure `xterm` is installed, or run teleop in a second terminal (recipe 10.9).

---

## 15. Repository layout

```
pf_ws/
├── setup_env.sh                     # one-shot sudo install
├── requirements.txt                 # pip deps (pupil-apriltags, --no-deps)
├── README.md                        # quick start
├── DOCUMENTATION.md                 # this file
├── report/report.md                 # the project report
├── report/media/*.png               # result figures
└── src/pf_localization/
    ├── package.xml, setup.py, setup.cfg
    ├── pf_localization/
    │   ├── particle_filter_node.py  # the filter
    │   ├── motion_model.py          # prediction
    │   ├── sensor_model.py          # multi-hypothesis update
    │   ├── resampling.py            # systematic resample + ESS + injection
    │   ├── tag_detector_node.py     # camera AprilTag detector
    │   ├── sim_detector_node.py     # ground-truth detector
    │   ├── odom_noise_node.py       # odometry drift
    │   ├── viz_node.py              # tags/room/trajectories
    │   ├── tag_map.py               # loads tag_map.yaml
    │   ├── utils.py                 # angle/quaternion helpers
    │   └── generate_world.py        # world + texture generator
    ├── config/  tag_map.yaml, filter_params.yaml, pf.rviz, ros_gz_bridge.yaml
    ├── launch/  bringup.launch.py
    ├── worlds/  room.sdf            # generated
    ├── models/  pf_robot/, apriltag_marker/
    └── materials/textures/          # generated tag texture
```

---

## 16. Extending the project

* **More/fewer tags**: just edit `tag_map.yaml`; the sensor model sums over all of them.
* **A different family**: set `tag.family` (and the detector's `tag_family`); the
  texture is generated to match.
* **Different sensor**: anything that publishes `/tag_detections`
  (`geometry_msgs/PoseArray` of tag positions in `base_link`) drops straight in.
* **Real robot (bonus)**: replace the Gazebo source with a real camera + AprilTag
  node and a real odometry source publishing the same topics; the filter is unchanged.
* **Publish `map→odom` TF** (AMCL-style) by extending `particle_filter_node` if you
  want the standard localization TF tree for navigation.

---

## 17. FAQ

**Why do all tags share one id?** That's the assignment's core challenge: a detection
can't be associated to a specific tag, so the filter must consider all of them.

**Why does the cloud sometimes briefly jump after converging?** The multi-hypothesis
filter momentarily re-entertains an alternative when an ambiguous view arrives, then
the asymmetric layout refutes it. Lower `w_fast_alpha`/`random_inject_frac` to damp it.

**Sim vs camera mode — which should I demo?** `sim` is deterministic and rock-solid;
`camera` is the faithful "robot sees with its camera" pipeline. Both use the same
filter. Record `camera` for realism, keep `sim` as the reliable backup.

**Does the filter know where the robot started?** No — it initializes uniformly over
the whole room. The random spawn makes that genuine.

**Where is the one source of truth for the room?** `config/tag_map.yaml`. Change it,
re-run `generate_world.py`, rebuild — world and filter stay in sync.
