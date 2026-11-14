"""Real localization: AMCL on the merged safety-scanner scan (Phase B3),
refined by a scan-to-map match on both raw scans (refined_localization,
which publishes map -> odom in AMCL's place).

AMCL matches scans against its own localization map (accurate walls, only
what the scanners see; tools/warehouse_validation/export_localization_map.py),
served here on /localization_map. The conservative navigation map (/map) is
started separately for the planners.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('tugger_navigation')
    config = os.path.join(share, 'config', 'amcl.yaml')
    localization_map = os.path.join(share, 'maps', 'localization_map.yaml')
    # Same image, origin half a cell on, for AMCL only: nav2_amcl rounds when
    # it looks cells up, and would otherwise see every wall 1 cm south-west
    # (tools/warehouse_validation/export_localization_map.py writes both).
    amcl_map = os.path.join(share, 'maps', 'localization_map_amcl.yaml')
    return LaunchDescription([
        Node(package='nav2_map_server', executable='map_server',
             name='localization_map_server', output='screen',
             parameters=[{'use_sim_time': True, 'yaml_filename': localization_map,
                          'topic_name': 'localization_map', 'frame_id': 'map'}]),
        Node(package='nav2_map_server', executable='map_server',
             name='amcl_map_server', output='screen',
             parameters=[{'use_sim_time': True, 'yaml_filename': amcl_map,
                          'topic_name': 'localization_map_amcl', 'frame_id': 'map'}]),
        Node(package='nav2_amcl', executable='amcl', name='amcl', output='screen',
             parameters=[config]),
        # Publishes map -> odom (AMCL's tf_broadcast is off): AMCL's pose,
        # sharpened by the scan match when it is clearly good.
        Node(package='tugger_control', executable='refined_localization',
             name='tugger_refined_localization', output='screen'),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
             name='lifecycle_manager_localization', output='screen',
             parameters=[{'use_sim_time': True, 'autostart': True,
                          'node_names': ['localization_map_server', 'amcl_map_server', 'amcl'],
                          'bond_timeout': 0.0}]),
    ])
