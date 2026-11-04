"""Check cargo pose topic names without starting ROS nodes or commanding motion."""
from pathlib import Path
import re
from unittest.mock import Mock, patch

import pytest

pytest.importorskip('rclpy')
pytest.importorskip('python_qt_binding')
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.validate_topic_name import validate_topic_name

from tugger_control.operator_panel import OperatorBridge
from tugger_control.sim_motion_gate import SimMotionGate


ROOT = Path(__file__).resolve().parents[2]


def test_ground_truth_publisher_and_both_consumers_share_valid_ros_topics():
    """Exercise actual subscription constructors with native ROS validation."""
    captured = []

    def subscription(_self, message_type, topic, callback, qos):
        validate_topic_name(topic)
        if message_type is PoseStamped and topic.startswith('/tugger/cargo/'):
            captured.append(topic)
        return Mock()

    with (patch.object(Node, '__init__', return_value=None),
          patch.object(Node, 'set_parameters', return_value=[]),
          patch.object(Node, 'create_subscription', subscription),
          patch.object(Node, 'create_publisher', return_value=Mock()),
          patch.object(Node, 'create_client', return_value=Mock()),
          patch.object(Node, 'create_timer', return_value=Mock()),
          patch.object(Node, 'get_logger', return_value=Mock()),
          patch('tugger_control.operator_panel.Buffer', return_value=Mock()),
          patch('tugger_control.operator_panel.TransformListener', return_value=Mock()),
          patch('tugger_control.operator_panel.ActionClient', return_value=Mock()),
          patch('tugger_control.sim_motion_gate.Buffer', return_value=Mock()),
          patch('tugger_control.sim_motion_gate.TransformListener', return_value=Mock()),
          patch('tugger_control.sim_motion_gate.get_package_share_directory',
                side_effect=RuntimeError('not needed in topic contract test'))):
        OperatorBridge(Mock(), Mock(), Mock())
        SimMotionGate()

    source = (ROOT / 'tugger_ground_truth/src/ground_truth_bridge.cpp').read_text()
    published = re.findall(r'"(/tugger/cargo/[^\"]+/ground_truth_pose)"', source)
    assert len(published) == 2
    for topic in published:
        validate_topic_name(topic)
    assert set(published) == set(captured)
    assert len(captured) == 4  # Both station topics must reach both consumers.


def test_nav2_has_separate_normal_and_docking_follow_path_controllers():
    """The operator's two action clients must have distinct Nav2 servers."""
    launch = (ROOT / 'tugger_navigation/launch/nav2.launch.py').read_text()
    assert "namespace='docking'" in launch
    assert "'/tugger/dock_nav_cmd_vel'" in launch
    assert "'/tugger/nav_cmd_vel'" in launch
    assert 'lifecycle_manager_docking' in launch
