"""One autonomous mapping run: sim + SLAM, Nav2 + frontier explorer (no keepout), our nodes, recording.

  ros2 launch semantic_mapper mapping_run.launch.py run:=run5
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction, ExecuteProcess
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, EnvironmentVariable
from launch_ros.actions import Node


def generate_launch_description():
    sim_pkg = get_package_share_directory('mallbot_sim')
    run = LaunchConfiguration('run')
    out = PathJoinSubstitution([EnvironmentVariable('HOME'), 'mallbot_ws', 'runs', run])
    sim_time = {'use_sim_time': True}

    return LaunchDescription([
        DeclareLaunchArgument('run', default_value='run5', description='name of this run (output folder)'),
        DeclareLaunchArgument('record', default_value='true'),

        DeclareLaunchArgument('look_around', default_value='false'),
        DeclareLaunchArgument('depth_cloud', default_value='true',
                              description='false = no /camera/points (costmaps then use the lidar only)'),
        DeclareLaunchArgument('rviz', default_value='true',
                              description='false for measured runs (RViz costs 50-120% CPU); open it separately to check'),
        TimerAction(period=30.0, actions=[
            ExecuteProcess(cmd=['ros2', 'param', 'set', '/frontier_explorer', 'look_around',
                                LaunchConfiguration('look_around')]),
        ]),
        
        ExecuteProcess(cmd=['mkdir', '-p', out]),

        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(sim_pkg, 'launch', 'sim.launch.py')),
                                 launch_arguments={'gz_extra': '-s', 'rviz': LaunchConfiguration('rviz')}.items()),

        TimerAction(period=15.0, actions=[      # let Gazebo and SLAM start first
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(sim_pkg, 'launch', 'nav.launch.py')),
                launch_arguments={'explore': 'true', 'keepout': 'false'}.items()),
        ]),

        TimerAction(period=5.0, actions=[
            # depth image -> /camera/points, 2nd obstacle source of both Nav2 costmaps (P1: café tabletops)
            Node(package='depth_image_proc', executable='point_cloud_xyz_node', name='depth_to_points',
                 condition=IfCondition(LaunchConfiguration('depth_cloud')),
                 output='screen', parameters=[sim_time],
                 remappings=[('image_rect', '/camera/depth_image'), ('camera_info', '/camera/camera_info'),
                             ('points', '/camera/points')]),
            Node(package='semantic_mapper', executable='detection_node', output='screen', parameters=[sim_time]),
            Node(package='semantic_mapper', executable='object_map_node', name='object_map_box', output='screen',
                 parameters=[sim_time, {'save_path': PathJoinSubstitution([out, 'map_box.json'])}]),
            Node(package='semantic_mapper', executable='object_map_node', name='object_map_centroid', output='screen',
                 parameters=[sim_time, {'save_path': PathJoinSubstitution([out, 'map_centroid.json'])}],
                 remappings=[('/semantic_mapper/detections', '/semantic_mapper/detections_centroid'),
                             ('/semantic_mapper/object_map', '/semantic_mapper/object_map_centroid')]),
            ExecuteProcess(condition=IfCondition(LaunchConfiguration('record')),
                           cmd=['ros2', 'bag', 'record', '-o', PathJoinSubstitution([out, 'bag']),
                                '/semantic_mapper/detections', '/semantic_mapper/detections_centroid', '/clock']),
        ]),
    ])
