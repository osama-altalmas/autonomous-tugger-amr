"""V1 static-map planning-only launch. Does not start controller or navigation executor."""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    share = get_package_share_directory('tugger_navigation')
    cfg = os.path.join(share, 'config', 'nav2_v1_static.yaml')
    mapfile = os.path.join(share, 'maps', 'warehouse_map.yaml')
    publish_static_map_to_odom = LaunchConfiguration(
        'publish_static_map_to_odom'
    )
    return LaunchDescription([
        DeclareLaunchArgument(
            'publish_static_map_to_odom',
            default_value='false',
            description=(
                'Publish an identity map -> odom transform for offline planning '
                'only. Leave false when SLAM or simulator ground-truth '
                'localization is running.'
            ),
        ),
        # Simulator localization or SLAM owns map -> odom by default.
        # The identity transform remains available solely for offline analysis.
        Node(package='tf2_ros', executable='static_transform_publisher',
             name='v1_map_to_odom_initial', output='screen',
             condition=IfCondition(publish_static_map_to_odom),
             arguments=['--x', '0', '--y', '0', '--z', '0',
                        '--roll', '0', '--pitch', '0', '--yaw', '0',
                        '--frame-id', 'map', '--child-frame-id', 'odom']),
        Node(package='nav2_map_server', executable='map_server',
             name='map_server', output='screen',
             parameters=[{'use_sim_time': True, 'yaml_filename': mapfile}]),
        Node(package='nav2_planner', executable='planner_server',
             name='planner_server', output='screen', parameters=[cfg]),
        TimerAction(
            period=5.0,
            actions=[
                Node(
                    package='nav2_lifecycle_manager',
                    executable='lifecycle_manager',
                    name='lifecycle_manager_v1_planning',
                    output='screen',
                    parameters=[{
                        'use_sim_time': True,
                        'autostart': True,
                        'node_names': ['map_server', 'planner_server'],
                        'service_timeout': 20.0,
                        'bond_timeout': 10.0,
                    }]
                )
            ]
        ),
    ])
