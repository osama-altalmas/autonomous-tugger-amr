from launch import LaunchDescription

from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os
import yaml


def generate_launch_description():

    tugger_control_share = get_package_share_directory(
        'tugger_control'
    )

    ekf_config = os.path.join(
        tugger_control_share,
        'config',
        'ekf.yaml'
    )

    speed_config = os.path.join(
        tugger_control_share,
        'config',
        'sim_speed.yaml'
    )

    four_ws_controller = Node(
        package='tugger_control',
        executable='four_ws_controller',
        name='four_ws_controller',
        output='screen',
        remappings=[('/cmd_vel', '/tugger/guarded_cmd_vel')],
        parameters=[
            yaml.safe_load(
                open(speed_config, encoding='utf-8')
            )['four_ws_controller']['ros__parameters'],
            {'use_sim_time': True}
        ]
    )

    four_ws_odometry = Node(
        package='tugger_control',
        executable='four_ws_odometry',
        name='four_ws_odometry',
        output='screen',
        parameters=[
            {'use_sim_time': True}
        ]
    )

    ekf = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[
            ekf_config,
            {'use_sim_time': True}
        ],
        remappings=[
            ('odometry/filtered', '/odometry/filtered')
        ]
    )

    qr_docking_vision = Node(
        package='tugger_control',
        executable='qr_docking_vision',
        name='qr_docking_vision',
        output='screen',
        parameters=[{'use_sim_time': True, 'qr_grid_size_m': 0.064}],
    )

    sim_motion_gate = Node(
        package='tugger_control', executable='sim_motion_gate',
        name='tugger_sim_motion_gate', output='screen',
        parameters=[{'use_sim_time': True}],
    )

    return LaunchDescription([
        four_ws_controller,
        four_ws_odometry,
        ekf,
        qr_docking_vision,
        sim_motion_gate,
    ])
