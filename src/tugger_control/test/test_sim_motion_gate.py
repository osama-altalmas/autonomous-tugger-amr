"""Unit tests use mocks only; no ROS commands are sent."""
from types import SimpleNamespace
from unittest.mock import Mock, patch
from pathlib import Path
import ast

import pytest

pytest.importorskip('rclpy')
from geometry_msgs.msg import TwistStamped
from rclpy.time import Time

from tugger_control.clearance_2_5d import Box
from tugger_control.sim_motion_gate import SimMotionGate


def gate(obstacles=()):
    request = TwistStamped()
    request.twist.linear.x = 0.4
    request.twist.angular.z = 0.2
    clock = Mock()
    clock.now.return_value = Time(seconds=10)
    return SimpleNamespace(
        _now=Mock(return_value=10.0), _last_time=9.9,
        get_clock=Mock(return_value=clock), _output=Mock(), _ready=Mock(), _dock_ready=Mock(),
        _status=Mock(), _last_reason=None, _states={0: 'attached', 3: 'detached'},
        _states_at={0: 9.9, 3: 9.9}, _owner='NAV', _owner_at=9.9,
        _requests={'NAV': request, 'PRECISION': None, 'DOCK_NAV': request},
        _request_at={'NAV': 9.9, 'PRECISION': None, 'DOCK_NAV': 9.9},
        _fresh=SimMotionGate._fresh,
        _snapshot=Mock(return_value=((0.0, 0.0, 0.0), obstacles)),
        _payloads={0: (0.0, 0.0, 0.065, 0.0), 3: (10, 10, 0, 0)},
        _lift=0.065, _speed=0.05, _wheel_speed=0.05,
    )


def test_loaded_output_is_speed_limited_and_keeps_curvature():
    item = gate()
    SimMotionGate._tick(item)
    result = item._output.publish.call_args.args[0]
    assert result.twist.linear.x == pytest.approx(0.1)
    assert result.twist.angular.z == pytest.approx(0.05)
    assert item._ready.publish.call_args.args[0].data is True


def test_loaded_reverse_keeps_signed_velocity_and_steering_curvature():
    item = gate()
    item._requests['NAV'].twist.linear.x = -0.4
    item._requests['NAV'].twist.angular.z = 0.2
    SimMotionGate._tick(item)
    result = item._output.publish.call_args.args[0]
    assert result.twist.linear.x == pytest.approx(-0.1)
    assert result.twist.angular.z == pytest.approx(0.05)


def test_loaded_obstacle_cuts_command_to_zero():
    obstacle = Box('overhead', 0.8, 0.0, 0.0, 0.05, 0.1, 0.34, 0.4)
    item = gate([('overhead_model', obstacle)])
    SimMotionGate._tick(item)
    result = item._output.publish.call_args.args[0]
    assert result.twist.linear.x == 0.0
    assert result.twist.angular.z == 0.0


def test_loaded_guard_uses_measured_wheel_speed_if_it_exceeds_commanded_speed():
    obstacle = Box('overhead', 0.85, 0.0, 0.0, 0.05, 0.1, 0.34, 0.4)
    item = gate([('overhead_model', obstacle)])
    item._speed = 0.0
    item._wheel_speed = 0.15
    SimMotionGate._tick(item)
    assert item._output.publish.call_args.args[0].twist.linear.x == 0.0


@pytest.mark.parametrize('field', ['_owner_at', '_request_at', '_states_at'])
def test_stale_owner_request_or_cargo_feedback_stops(field):
    item = gate()
    if field == '_owner_at':
        item._owner_at = 9.0
    elif field == '_request_at':
        item._request_at['NAV'] = 9.0
    else:
        item._states_at[0] = 1.0
    SimMotionGate._tick(item)
    assert item._output.publish.call_args.args[0].twist.linear.x == 0.0


def test_unavailable_scene_stops_loaded_output_and_reports_not_ready():
    item = gate()
    item._snapshot.side_effect = ValueError('scene feed stale')
    SimMotionGate._tick(item)
    assert item._output.publish.call_args.args[0].twist.linear.x == 0.0
    assert item._ready.publish.call_args.args[0].data is False


def test_owner_change_discards_previous_source_command():
    item = gate()
    item._request_at['PRECISION'] = 9.9
    SimMotionGate._on_owner(item, SimpleNamespace(data='PRECISION'))
    assert item._request_at['PRECISION'] is None
    assert item._owner == 'PRECISION'


def test_entering_docking_mode_requires_a_fresh_navigation_command():
    item = gate()
    SimMotionGate._on_owner(item, SimpleNamespace(data='DOCK_NAV'))
    assert item._owner == 'DOCK_NAV'
    assert item._request_at['DOCK_NAV'] is None


def test_unloaded_docking_is_speed_limited_and_height_aware():
    tabletop = Box('tabletop', 0, 0, 0, 0.65, 0.65, 0.28, 0.31)
    item = gate([('stand', tabletop)])
    item._states = {0: 'detached', 3: 'detached'}
    item._owner = 'DOCK_NAV'
    item._lift = 0.0
    item._speed = item._wheel_speed = 0.0
    item._requests['NAV'].twist.linear.x = -0.08
    item._requests['NAV'].twist.angular.z = 0.02
    SimMotionGate._tick(item)
    result = item._output.publish.call_args.args[0]
    assert result.twist.linear.x == pytest.approx(-0.05)
    assert result.twist.angular.z == pytest.approx(0.0125)


def test_unloaded_docking_stops_for_floor_obstacle_or_stale_scene():
    obstacle = Box('unexpected', 0.55, 0, 0, 0.02, 0.02, 0, 0.15)
    item = gate([('unexpected', obstacle)])
    item._states = {0: 'detached', 3: 'detached'}
    item._owner = 'DOCK_NAV'
    item._lift = item._speed = item._wheel_speed = 0.0
    item._requests['NAV'].twist.linear.x = 0.05
    item._requests['NAV'].twist.angular.z = 0.0
    SimMotionGate._tick(item)
    assert item._output.publish.call_args.args[0].twist.linear.x == 0.0
    item._snapshot.side_effect = ValueError('scene feed stale')
    SimMotionGate._tick(item)
    assert item._output.publish.call_args.args[0].twist.linear.x == 0.0


def test_docking_camera_curvature_is_clamped_to_4ws_limit():
    table = Box('tabletop', 0, 0, 0, .65, .65, .28, .31)
    item = gate([('stand', table)])
    item._states = {0: 'detached', 3: 'detached'}
    item._owner = 'DOCK_CAMERA'
    item._lift = item._speed = item._wheel_speed = 0.0
    item._requests['PRECISION'] = TwistStamped()
    item._requests['PRECISION'].twist.linear.x = -0.02
    item._requests['PRECISION'].twist.angular.z = 0.25
    item._request_at['PRECISION'] = 9.9
    SimMotionGate._tick(item)
    output = item._output.publish.call_args.args[0]
    assert output.twist.linear.x == pytest.approx(-0.02)
    assert output.twist.angular.z == pytest.approx(2 * .02 / .58)


def test_unloaded_docking_rejects_attached_cargo():
    item = gate()
    item._owner = 'DOCK_NAV'
    SimMotionGate._tick(item)
    assert item._output.publish.call_args.args[0].twist.linear.x == 0.0


def test_guard_processing_overrun_stops_and_withholds_readiness():
    item = gate()
    with patch('tugger_control.sim_motion_gate.time.monotonic', side_effect=(1.0, 1.2)):
        SimMotionGate._tick(item)
    assert item._output.publish.call_args.args[0].twist.linear.x == 0.0
    assert item._ready.publish.call_args.args[0].data is False


def test_snapshot_rejects_disagreement_between_cargo_pose_and_collision_scene():
    item = gate()
    item._model_valid = True
    item._expected_counts = {}
    item.count_subscribers = Mock(return_value=1)
    item.count_publishers = Mock(return_value=1)
    item._scene_at = 9.9
    item._scene = '{}'
    item._lift_at = item._speed_at = 9.9
    item._wheel_speed_at = 9.9
    item._payloads_at = {0: 9.9, 3: 9.9}
    item._tf = Mock()
    item._tf.lookup_transform.return_value = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=10, nanosec=0)),
        transform=SimpleNamespace(
            translation=SimpleNamespace(x=0, y=0, z=0),
            rotation=SimpleNamespace(x=0, y=0, z=0, w=1)))
    obstacles = [
        ('cargo_station_0_stand', Box('cargo0', 0, 0, 0, .65, .65, .345, .375)),
        ('cargo_station_3_stand', Box('cargo3', 10, 10, 0, .65, .65, .28, .31)),
    ]
    with patch('tugger_control.sim_motion_gate.decode_scene', return_value=obstacles):
        assert SimMotionGate._snapshot(item, 10.0)[0] == (0, 0, 0)
        item._payloads[0] = (1, 0, .065, 0)
        with pytest.raises(ValueError, match='disagrees'):
            SimMotionGate._snapshot(item, 10.0)


def test_simulation_launch_wires_nav_and_controller_through_gate():
    root = Path(__file__).resolve().parents[2]
    control = ast.parse((root / 'tugger_control/launch/control.launch.py').read_text())
    navigation = ast.parse((root / 'tugger_navigation/launch/nav2.launch.py').read_text())
    control_strings = {node.value for node in ast.walk(control)
                       if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    nav_strings = {node.value for node in ast.walk(navigation)
                   if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert {'sim_motion_gate', '/cmd_vel', '/tugger/guarded_cmd_vel'} <= control_strings
    assert {'controller_server', 'behavior_server', '/tugger/nav_cmd_vel'} <= nav_strings


def test_operator_bridge_uses_simulation_time_for_scene_freshness():
    root = Path(__file__).resolve().parents[2]
    source = (root / 'tugger_control/tugger_control/operator_panel.py').read_text()
    tree = ast.parse(source)
    bridge = next(node for node in tree.body
                  if isinstance(node, ast.ClassDef) and node.name == 'OperatorBridge')
    init = next(node for node in bridge.body
                if isinstance(node, ast.FunctionDef) and node.name == '__init__')
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id == 'Parameter' and node.args
               and isinstance(node.args[0], ast.Constant)
               and node.args[0].value == 'use_sim_time'
               and any(keyword.arg == 'value' and isinstance(keyword.value, ast.Constant)
                       and keyword.value.value is True for keyword in node.keywords)
               for node in ast.walk(init))
