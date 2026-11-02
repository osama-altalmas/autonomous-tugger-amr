from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():

    config_file = os.path.join(
        get_package_share_directory('tugger_navigation'),
        'config',
        'slam.yaml'
    )

    slam_toolbox = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[config_file]
    )

    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_slam',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'autostart': True,
            'node_names': ['slam_toolbox']
        }]
    )

    return LaunchDescription([
        Node(
            package='tugger_navigation',
            executable='slam_scan_filter.py',
            name='slam_scan_filter',
            output='screen',
            parameters=[{'use_sim_time': True}]
        ),

        slam_toolbox,
        lifecycle_manager
    ])
