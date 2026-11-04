from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from nav2_common.launch import RewrittenYaml

import os


def generate_launch_description():

    package_share = get_package_share_directory('tugger_navigation')

    config_file = os.path.join(
        package_share,
        'config',
        'nav2.yaml'
    )

    bt_file = os.path.join(
        package_share,
        'behavior_trees',
        'navigate_to_pose_tugger.xml'
    )

    # Precision docking has its own controller and costmap parameters.  The
    # normal navigation controller must keep the full warehouse footprint.
    docking_config = RewrittenYaml(
        source_file=config_file,
        root_key='docking',
        param_rewrites={'footprint_padding': '0.03'},
        convert_types=True)

    nodes = [
        Node(
            package='nav2_controller',
            executable='controller_server',
            namespace='docking',
            name='controller_server',
            output='screen',
            remappings=[('cmd_vel', '/tugger/dock_nav_cmd_vel')],
            parameters=[docking_config]
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            namespace='docking',
            name='lifecycle_manager_docking',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'autostart': True,
                'node_names': ['controller_server'],
            }]
        ),
        Node(
            package='nav2_controller',
            executable='controller_server',
            name='controller_server',
            output='screen',
            parameters=[config_file],
            remappings=[('cmd_vel', '/tugger/nav_cmd_vel')]
        ),

        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[config_file]
        ),

        Node(
            package='nav2_smoother',
            executable='smoother_server',
            name='smoother_server',
            output='screen',
            parameters=[config_file]
        ),

        Node(
            package='nav2_behaviors',
            executable='behavior_server',
            name='behavior_server',
            output='screen',
            remappings=[('cmd_vel', '/tugger/nav_cmd_vel')],
            parameters=[config_file]
        ),

        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            name='bt_navigator',
            output='screen',
            parameters=[
                config_file,
                {'default_nav_to_pose_bt_xml': bt_file}
            ]
        ),

        Node(
            package='nav2_waypoint_follower',
            executable='waypoint_follower',
            name='waypoint_follower',
            output='screen',
            parameters=[config_file]
        ),

        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[config_file]
        ),
    ]

    return LaunchDescription(nodes)
