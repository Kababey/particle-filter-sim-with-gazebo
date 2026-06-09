#!/usr/bin/env bash
# =============================================================================
# setup_env.sh -- one-shot system setup for the AR-tag particle-filter project.
#
# Installs ROS 2 Jazzy + Gazebo Harmonic + the build/teleop/vision tooling on
# Ubuntu 24.04. This is the ONLY step that needs root. Run it once:
#
#     sudo bash setup_env.sh
#
# After it finishes, everything else (colcon build, ros2 launch) runs without
# sudo. The script is safe to re-run.
# =============================================================================
set -euo pipefail

if [ "$(id -u)" -eq 0 ]; then SUDO=""; else SUDO="sudo"; fi
# the unprivileged user we should run rosdep/pip as (so they land in the right place)
TARGET_USER="${SUDO_USER:-$(id -un)}"

echo "==> [1/7] Checking Ubuntu version"
. /etc/os-release
if [ "${VERSION_CODENAME:-}" != "noble" ]; then
  echo "WARNING: this script targets Ubuntu 24.04 (noble); found '${VERSION_CODENAME:-?}'." >&2
  echo "         ROS 2 Jazzy + Gazebo Harmonic are intended for noble. Continuing anyway." >&2
fi

echo "==> [2/7] Locale + prerequisites"
$SUDO apt-get update
$SUDO apt-get install -y locales curl gnupg lsb-release software-properties-common ca-certificates
$SUDO locale-gen en_US en_US.UTF-8 || true
$SUDO update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 || true
$SUDO add-apt-repository -y universe

echo "==> [3/7] Adding the ROS 2 apt repository (ros2-apt-source)"
ROS_APT_SOURCE_VERSION="$(curl -fsSL https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
  | grep -F '"tag_name"' | awk -F'"' '{print $4}')"
DEB="/tmp/ros2-apt-source.deb"
curl -fsSL -o "$DEB" \
  "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.${VERSION_CODENAME}_all.deb"
$SUDO apt-get install -y "$DEB"

echo "==> [4/7] apt update"
$SUDO apt-get update

echo "==> [5/7] Installing ROS 2 Jazzy, Gazebo Harmonic and tooling"
$SUDO apt-get install -y \
  ros-jazzy-desktop \
  ros-jazzy-ros-gz \
  ros-jazzy-cv-bridge \
  ros-jazzy-image-transport \
  ros-jazzy-teleop-twist-keyboard \
  ros-dev-tools \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-opencv \
  python3-numpy \
  python3-scipy \
  python3-yaml \
  python3-pip \
  python3-pil \
  xterm

echo "==> [6/7] rosdep init/update"
if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
  $SUDO rosdep init || true
fi
# rosdep update must run as a normal user, not root
sudo -u "$TARGET_USER" rosdep update || rosdep update || true

echo "==> [7/7] Python AprilTag detector (pip, system-wide so ROS can import it)"
# Ubuntu 24.04 marks the system Python 'externally managed' (PEP 668). Install the
# one extra dependency with --no-deps so it CANNOT pull numpy 2.x: a numpy upgrade
# would break ROS / cv_bridge, which are built against the Debian numpy 1.26.
# (The tag-image generator uses the apt OpenCV AprilTag dictionary, so we do NOT
# need moms-apriltag / opencv-contrib-python.)
$SUDO python3 -m pip install --break-system-packages --no-deps pupil-apriltags

cat <<'EOF'

============================================================
 Setup complete.

 Next (no sudo needed), from the pf_ws directory:

   source /opt/ros/jazzy/setup.bash
   python3 src/pf_localization/pf_localization/generate_world.py   # build world+texture
   colcon build --symlink-install
   source install/setup.bash
   ros2 launch pf_localization bringup.launch.py detection_mode:=sim

 Use detection_mode:=camera for the real camera AprilTag pipeline.
============================================================
EOF
