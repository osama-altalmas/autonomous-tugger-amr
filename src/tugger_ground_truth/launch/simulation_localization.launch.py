"""Simulator ground truth: the true pose for evaluation, and optionally map->odom.

Gazebo provides the true model pose only in simulation. Since Phase B the
robot localizes with AMCL (tugger_navigation/localization.launch.py); this
launch then only bridges the true pose for evaluation and logging. With
truth_localization:=true the old truth correction publishes map -> odom
instead (do not run it together with AMCL). Never run it on the robot.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('truth_localization', default_value='false'),
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
            condition=IfCondition(LaunchConfiguration('truth_localization')),
            parameters=[{
                'use_sim_time': True,
                'position_tolerance_m': 0.03,
                'yaw_tolerance_rad': 0.05236,
            }],
        ),
    ])
