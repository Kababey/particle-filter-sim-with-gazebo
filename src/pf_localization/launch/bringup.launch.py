"""Bring up the whole simulation: Gazebo + robot + bridges + detector + filter +
visualization + teleop.

Launch arguments (all optional):
  detection_mode := camera | sim   (default: camera; 'sim' uses the ground-truth
                                     fallback detector and needs no vision deps)
  spawn_x, spawn_y, spawn_yaw      robot's (unknown) start pose
  rviz   := true | false           open RViz with the project layout
  teleop := true | false           open a teleop_twist_keyboard window (xterm)
  gui    := true | false           Gazebo GUI

Example:
  ros2 launch pf_localization bringup.launch.py
  ros2 launch pf_localization bringup.launch.py detection_mode:=sim
"""

import os
import math
import random

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            ExecuteProcess, SetEnvironmentVariable, LogInfo)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('pf_localization')
    models_dir = os.path.join(pkg, 'models')
    world = os.path.join(pkg, 'worlds', 'room.sdf')
    robot_sdf = os.path.join(models_dir, 'pf_robot', 'model.sdf')
    tag_map = os.path.join(pkg, 'config', 'tag_map.yaml')
    params = os.path.join(pkg, 'config', 'filter_params.yaml')
    rviz_cfg = os.path.join(pkg, 'config', 'pf.rviz')
    bridge_cfg = os.path.join(pkg, 'config', 'ros_gz_bridge.yaml')

    # --- random spawn: close to a wall, FACING AN EMPTY (tag-free) section ---
    # Re-rolled every launch. The robot starts near a wall looking at a part of
    # that wall that carries no tag, so it initially detects NOTHING and the
    # particle cloud stays at the full global spread. Driving away / turning then
    # reveals tags and you can watch the cloud converge. Override with spawn_x:=...
    with open(tag_map) as f:
        _m = yaml.safe_load(f)
    rsx, rsy = float(_m['room']['size_x']), float(_m['room']['size_y'])
    _tags = _m['tags']

    def _empty_wall_spawn():
        standoff, edge, clear = 0.7, 1.0, 1.3   # [m] from wall, from corner, from a tag
        walls = ['south', 'north', 'west', 'east']
        random.shuffle(walls)
        for w in walls:
            if w in ('south', 'north'):
                dim, fixed = rsx, (0.06 if w == 'south' else rsy - 0.06)
                tag_along = [t['x'] for t in _tags if abs(t['y'] - fixed) < 0.25]
            else:
                dim, fixed = rsy, (0.06 if w == 'west' else rsx - 0.06)
                tag_along = [t['y'] for t in _tags if abs(t['x'] - fixed) < 0.25]
            steps = [edge + 0.2 * i for i in range(int((dim - 2 * edge) / 0.2) + 1)]
            empty = [c for c in steps if all(abs(c - t) > clear for t in tag_along)]
            if not empty:
                continue
            pos = random.choice(empty)
            if w == 'south':
                return pos, standoff, -math.pi / 2     # facing -y (the south wall)
            if w == 'north':
                return pos, rsy - standoff, math.pi / 2  # facing +y (the north wall)
            if w == 'west':
                return standoff, pos, math.pi            # facing -x (the west wall)
            return rsx - standoff, pos, 0.0              # facing +x (the east wall)
        # fallback: anywhere in the room
        return (random.uniform(edge, rsx - edge),
                random.uniform(edge, rsy - edge),
                random.uniform(-math.pi, math.pi))

    _spx, _spy, _spyaw = _empty_wall_spawn()
    rand_x = f'{_spx:.2f}'
    rand_y = f'{_spy:.2f}'
    rand_yaw = f'{_spyaw:.3f}'

    # --- arguments ---
    args = [
        DeclareLaunchArgument('detection_mode', default_value='camera'),
        DeclareLaunchArgument('spawn_x', default_value=rand_x),
        DeclareLaunchArgument('spawn_y', default_value=rand_y),
        DeclareLaunchArgument('spawn_yaw', default_value=rand_yaw),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('teleop', default_value='true'),
        DeclareLaunchArgument('headless', default_value='false'),
    ]
    detection_mode = LaunchConfiguration('detection_mode')
    headless = LaunchConfiguration('headless')
    gz_flags = PythonExpression(
        ["' -r -s -v 3' if '", headless, "' == 'true' else ' -r -v 3'"])
    is_camera = IfCondition(PythonExpression(["'", detection_mode, "' == 'camera'"]))
    is_sim = IfCondition(PythonExpression(["'", detection_mode, "' == 'sim'"]))

    common = [params, {'tag_map_path': tag_map, 'use_sim_time': True}]

    # --- Gazebo can find the included tag/robot models ---
    set_resource = SetEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        models_dir + os.pathsep + os.environ.get('GZ_SIM_RESOURCE_PATH', ''))

    # --- Gazebo Harmonic ---
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': [world, gz_flags]}.items())

    # --- spawn the robot at the (unknown) start pose ---
    spawn = Node(
        package='ros_gz_sim', executable='create', output='screen',
        arguments=['-name', 'pf_robot', '-file', robot_sdf,
                   '-x', LaunchConfiguration('spawn_x'),
                   '-y', LaunchConfiguration('spawn_y'),
                   '-z', '0.0',
                   '-Y', LaunchConfiguration('spawn_yaw')])

    # --- bridges ---
    bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge', output='screen',
        parameters=[{'config_file': bridge_cfg, 'use_sim_time': True}])
    image_bridge = Node(
        package='ros_gz_image', executable='image_bridge', output='screen',
        arguments=['camera'], parameters=[{'use_sim_time': True}])

    # --- fixed frame so RViz has a 'map' (markers carry absolute coords) ---
    static_tf = Node(
        package='tf2_ros', executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'])

    # --- odometry corruption (drives the filter and the odom-only path) ---
    odom_noise = Node(
        package='pf_localization', executable='odom_noise_node', output='screen',
        parameters=common,
        remappings=[('odom_in', '/odom'), ('odom_out', '/odom_noisy')])

    # --- detectors (one active, chosen by detection_mode) ---
    tag_detector = Node(
        package='pf_localization', executable='tag_detector_node', output='screen',
        condition=is_camera, parameters=common,
        remappings=[('image', '/camera'),
                    ('camera_info', '/camera/camera_info'),
                    ('tag_detections', '/tag_detections')])
    sim_detector = Node(
        package='pf_localization', executable='sim_detector_node', output='screen',
        condition=is_sim, parameters=common,
        remappings=[('ground_truth', '/ground_truth'),
                    ('tag_detections', '/tag_detections')])

    # --- the particle filter ---
    pf = Node(
        package='pf_localization', executable='particle_filter_node', output='screen',
        parameters=common,
        remappings=[('odom', '/odom_noisy'),
                    ('tag_detections', '/tag_detections'),
                    ('ground_truth', '/ground_truth')])

    # --- visualization helper (tags, room, three trajectories) ---
    viz = Node(
        package='pf_localization', executable='viz_node', output='screen',
        parameters=common,
        remappings=[('odom_noisy', '/odom_noisy'),
                    ('pf/estimate', '/pf/estimate'),
                    ('ground_truth', '/ground_truth')])

    rviz = Node(
        package='rviz2', executable='rviz2', output='screen',
        condition=IfCondition(LaunchConfiguration('rviz')),
        arguments=['-d', rviz_cfg], parameters=[{'use_sim_time': True}])

    teleop = ExecuteProcess(
        condition=IfCondition(LaunchConfiguration('teleop')),
        cmd=['xterm', '-title', 'teleop  (drive the robot)', '-geometry', '80x25',
             '-e', 'ros2', 'run', 'teleop_twist_keyboard', 'teleop_twist_keyboard'],
        output='screen')

    return LaunchDescription(args + [
        LogInfo(msg=['Bringing up pf_localization | detection_mode=', detection_mode,
                     ' | spawn x=', LaunchConfiguration('spawn_x'),
                     ' y=', LaunchConfiguration('spawn_y'),
                     ' yaw=', LaunchConfiguration('spawn_yaw')]),
        set_resource, gz_sim, spawn, bridge, image_bridge, static_tf,
        odom_noise, tag_detector, sim_detector, pf, viz, rviz, teleop,
    ])
