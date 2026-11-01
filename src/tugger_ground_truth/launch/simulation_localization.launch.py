"""Simulator-only truth correction for the tugger localization stack.

Gazebo provides the true model pose only in simulation.  The correction node
uses it to publish map -> odom, while the EKF remains the sole publisher of
odom -> base_footprint.  Do not run this launch on the physical robot.
"""

from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # The running Gazebo server in this project is bound to loopback.
        SetEnvironmentVariable(name='GZ_IP', value='127.0.0.1'),
        SetEnvironmentVariable(name='IGN_IP', value='127.0.0.1'),
        Node(
            package='tugger_ground_truth',
            executable='ground_truth_bridge',
            name='tugger_ground_truth_bridge',
            output='screen',
            parameters=[{'use_sim_time': True}],
        ),
        Node(
            package='tugger_ground_truth',
            executable='ground_truth_localization.py',
            name='tugger_ground_truth_localization',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'position_tolerance_m': 0.03,
                'yaw_tolerance_rad': 0.05236,
            }],
        ),
    ])
