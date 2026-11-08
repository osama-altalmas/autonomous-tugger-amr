"""Exercise real guard/GUI methods without requiring ROS or Gazebo in the test host.

The small AST harness only removes external imports and injects mock ROS message
classes. It executes the project's actual method bodies, not rewritten logic.
"""

import ast
import json
import math

from tugger_control.crab_motion import CRAB_FRAME, crab_command
from tugger_control.loaded_motion_guard import MEASURED_SPEED_LIMIT
from tugger_control.mission_planning import entrance_gate, routes_from_gate
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tugger_control.lift_geometry import LIFT_CONTACT_M, LIFT_UP_M, lift_is_raised

from tugger_control.cargo_policy import CARGO_IDS, confirmed_cargo
from tugger_control.precision_scene_clearance import FULL_LOCK_CURVATURE
from tugger_control.loaded_motion_guard import choose_guarded_speed
from tugger_control.mission_planning import plan_is_current
from tugger_control.station_departure import plan_station_departure

PACKAGE = Path(__file__).resolve().parents[1] / 'tugger_control'


class Message:
    def __init__(self):
        self.header = SimpleNamespace(stamp=None, frame_id='')
        self.twist = SimpleNamespace(
            linear=SimpleNamespace(x=0.0), angular=SimpleNamespace(z=0.0))
        self.data = None


class NavPath:
    def __init__(self):
        self.header = SimpleNamespace(frame_id='', stamp=None)
        self.poses = []


def methods_in_source(filename, class_name, names, context):
    tree = ast.parse((PACKAGE / filename).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == class_name)
    chosen = [node for node in cls.body if isinstance(node, ast.FunctionDef)
              and node.name in names]
    assert {node.name for node in chosen} == set(names)
    minimal = ast.Module(body=[ast.ClassDef(
        name='Isolated', bases=[], keywords=[], body=chosen, decorator_list=[])],
        type_ignores=[])
    ast.fix_missing_locations(minimal)
    exec(compile(minimal, str(PACKAGE / filename), 'exec'), context)
    return context['Isolated']


def time_value(seconds):
    class TimeValue:
        def __init__(self, value):
            self.value = value
            self.nanoseconds = int(value * 1e9)

        def __sub__(self, other):
            return TimeValue(self.value - other.value)

        def to_msg(self):
            return SimpleNamespace(sec=int(self.value), nanosec=0)
    return TimeValue(seconds)


def gate_test_class():
    return methods_in_source('sim_motion_gate.py', 'SimMotionGate',
                             ('_tick', '_on_owner', '_fresh', '_stopping_curvature',
                              '_apply_safety', '_loaded_arcs', '_rotate_command'), {
        'math': math, 'time': __import__('time'),
        'TwistStamped': Message, 'Bool': Message, 'String': Message,
        'confirmed_cargo': confirmed_cargo, 'CARGO_IDS': CARGO_IDS,
        'lift_is_raised': lift_is_raised,
        'FULL_LOCK_CURVATURE': FULL_LOCK_CURVATURE, 'STOPPING_CURVATURE_MARGIN': 0.3,
        'CAMERA_SPEED_LIMIT': 0.05, 'PRECISION_APPROACH_LIMIT': 0.4,
        'choose_guarded_speed': choose_guarded_speed,
        'precision_stopping_envelope': Mock(return_value=(True, None)),
        'CRAB_FRAME': CRAB_FRAME, 'crab_command': crab_command,
        'MEASURED_SPEED_LIMIT': MEASURED_SPEED_LIMIT,
    })


def gate_fixture():
    class UnderTest(gate_test_class()):
        pass
    gate = UnderTest()
    clock = Mock()
    clock.now.return_value = time_value(10.0)
    gate.get_clock = Mock(return_value=clock)
    gate._now = Mock(return_value=10.0)
    gate._last_time = 9.9
    gate._output = Mock()
    gate._ready = Mock()
    gate._dock_ready = Mock()
    gate._status = Mock()
    gate._last_reason = None
    gate._states = {0: 'attached', 3: 'detached'}
    gate._states_at = {0: 9.9, 3: 9.9}
    gate._owner = 'DOCK_DEPART'
    gate._owner_at = 9.9
    gate._safety = {'state': 'CLEAR', 'limit': 1.0, 'reason': ''}
    gate._safety_at = 9.95
    request = Message()
    request.twist.linear.x = 0.6
    request.twist.angular.z = 0.3
    gate._requests = {'NAV': None, 'PRECISION': None, 'DOCK_NAV': request}
    gate._request_at = {'NAV': None, 'PRECISION': None, 'DOCK_NAV': 9.9}
    gate._snapshot = Mock(return_value=((0, 0, 0), []))
    gate._payloads = {0: (0, 0, .065, 0), 3: (10, 10, 0, 0)}
    gate._lift = LIFT_UP_M
    gate._speed = 0.0
    gate._wheel_speed = 0.0
    return gate


def test_new_loaded_departure_owner_reports_ready_and_limits_speed():
    gate = gate_fixture()
    gate._tick()
    outgoing = gate._output.publish.call_args.args[0]
    assert outgoing.twist.linear.x == pytest.approx(0.3)  # transfer-station limit
    assert outgoing.twist.angular.z == pytest.approx(0.15)  # same curvature at 0.3 m/s
    assert gate._dock_ready.publish.call_args.args[0].data == 'DOCK_DEPART'


def test_loaded_docking_entry_still_rejects_cargo():
    gate = gate_fixture()
    gate._owner = 'DOCK_NAV'
    gate._tick()
    assert gate._output.publish.call_args.args[0].twist.linear.x == 0
    assert gate._dock_ready.publish.call_args.args[0].data == 'STOP'


def test_unloaded_robot_cannot_use_departure_owner():
    gate = gate_fixture()
    gate._states = {0: 'detached', 3: 'detached'}
    gate._tick()
    assert gate._output.publish.call_args.args[0].twist.linear.x == 0
    assert gate._dock_ready.publish.call_args.args[0].data == 'STOP'


def test_departure_owner_disallows_previous_navigation_command():
    gate = gate_fixture()
    gate._request_at['DOCK_NAV'] = 9.9
    gate._owner = 'NAV'
    gate._on_owner(SimpleNamespace(data='DOCK_DEPART'))
    assert gate._owner == 'DOCK_DEPART'
    assert gate._request_at['DOCK_NAV'] is None


def test_stale_departure_command_never_reaches_controller():
    gate = gate_fixture()
    gate._request_at['DOCK_NAV'] = 9.0
    gate._tick()
    assert gate._output.publish.call_args.args[0].twist.linear.x == 0


def panel_test_class():
    return methods_in_source('operator_panel.py', 'OperatorBridge',
                             ('_prepare_loaded_departure', '_same_gate_routes', '_station_approach', '_on_departure_guard_ready',
                              '_validate_departure_on_fresh_scene',
                              '_fail_loaded_departure', '_scene_update',
                              '_complete_loaded_departure', '_dock_wait_watchdog',
                              '_publish_motion_owner', 'run_selected_mission'), {
        'math': math, 'json': json,
        'NavPath': NavPath,
        'entrance_gate': entrance_gate, 'routes_from_gate': routes_from_gate,
        'String': Message, 'Bool': Message,
        'plan_station_departure': plan_station_departure,
        'plan_is_current': plan_is_current,
        'GoalStatus': SimpleNamespace(STATUS_ABORTED=6),
        'check_destination': Mock(),
    })


def panel_fixture():
    obj = panel_test_class()()
    obj._plan_generation = 5
    obj._robot_pose = Mock(return_value=(0.0, 0.0, 0.0))
    obj._confirmed_cargo = Mock(return_value=0)
    obj._check_destination_allowed = Mock()
    obj._loaded_guard_is_ready = Mock(return_value=True)
    obj._modeled_loaded_path_clearance = Mock(return_value=100)
    obj._approach_provider = Mock(return_value=(
        {'execution_approved': True}, [[2.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    obj._pose = Mock(side_effect=lambda xyz: list(xyz))
    obj._path_pub = Mock()
    obj._report = Mock()
    obj._plan_ready = Mock()
    obj.get_clock = Mock(return_value=SimpleNamespace(now=Mock(return_value=time_value(10))))
    obj._dock_follower = SimpleNamespace(server_is_ready=Mock(return_value=True))
    obj._follow = Mock()
    obj._execution_done = Mock()
    obj._pending_execution_approved = False
    obj._pending_departure = None
    obj._departure_pending = None
    obj._scene_sequence = 0
    obj._collision_scene = None
    obj._departure_target = None
    obj._departure_active = False
    obj._motion_armed = True
    obj._cancel_requested = False
    obj._precision_station = 'some_stale_station'
    obj._camera_handoff_station = 'some_stale_station'
    obj._queued_precision_poses = [[1, 2, 3]]
    obj._queued_precision_station = 'some_stale_station'
    obj._dock_precision_mode = True
    obj.docked_station = 'station'
    return obj


def test_departure_preview_preserves_working_pickup_and_requires_explicit_run():
    gui = panel_fixture()
    routes = [('station-to-home', {
        'source': 'station', 'source_entrance': 'east',
        'execution_approved': True, '_database_execution_approved': True}, [])]
    gui._prepare_loaded_departure('station', 0, (0, 0, 0), 'HOME', routes, [])
    assert gui._pending_departure is not None
    assert gui._follow.call_count == 0
    assert gui._pending_departure['path'].poses[-1][0] == 2.0
    assert gui._camera_handoff_station is None
    assert gui._precision_station is None
    assert gui._plan_ready.call_count == 1



def _test_pending():
    return {
        'generation': 5,
        'start_pose': (0, 0, 0),
        'cargo_id': 0,
        'poses': [[0, 0, 0], [1, 0, 0]],
        'path': object(),
        'started': time_value(10),
        'guard_seen_at': time_value(10),
    }


def test_departure_waits_for_new_fresh_scene():
    gui = panel_fixture()
    gui._departure_pending = _test_pending()

    gui._on_departure_guard_ready()
    gui._follow.assert_not_called()

    gui._scene_update(
        SimpleNamespace(data=json.dumps({'stamp': 9.4}))
    )
    gui._follow.assert_not_called()

    gui._scene_update(
        SimpleNamespace(data=json.dumps({'stamp': 9.9}))
    )
    gui._modeled_loaded_path_clearance.assert_called_once()
    gui._follow.assert_called_once()
    assert gui._departure_active


def test_departure_rejects_lost_cargo_on_new_scene():
    gui = panel_fixture()
    gui._departure_pending = _test_pending()
    gui._confirmed_cargo.return_value = None

    gui._on_departure_guard_ready()
    gui._scene_update(
        SimpleNamespace(data=json.dumps({'stamp': 9.9}))
    )

    gui._follow.assert_not_called()
    gui._execution_done.assert_called_once()


def test_departure_does_not_use_expired_guard_ack():
    gui = panel_fixture()
    gui._departure_pending = _test_pending()
    gui._departure_pending['guard_seen_at'] = time_value(9)

    gui._on_departure_guard_ready()
    gui._scene_update(
        SimpleNamespace(data=json.dumps({'stamp': 9.9}))
    )

    gui._follow.assert_not_called()


def test_departure_retries_when_scene_ages_during_validation():
    gui = panel_fixture()
    gui._departure_pending = _test_pending()

    gui._modeled_loaded_path_clearance.side_effect = [
        ValueError(
            'Simulator scene is stale or from a future clock epoch'
        ),
        259,
    ]

    gui._on_departure_guard_ready()
    gui._scene_update(
        SimpleNamespace(data=json.dumps({'stamp': 9.9}))
    )
    gui._follow.assert_not_called()

    gui._scene_update(
        SimpleNamespace(data=json.dumps({'stamp': 9.9}))
    )
    gui._follow.assert_called_once()


def test_completed_departure_replans_only_after_reaching_measured_staging():
    gui = panel_fixture()
    gui._departure_active = True
    gui._departure_exit_pose = (2, 0, 0)
    gui._departure_target = ('HOME', [], [])
    gui.execute_destination = Mock()
    gui._robot_pose.return_value = (2.02, 0, 0)
    gui._complete_loaded_departure()
    assert gui.docked_station is None
    gui.execute_destination.assert_called_once_with('HOME', [], [])

    gui = panel_fixture()
    gui._departure_active = True
    gui._departure_exit_pose = (2, 0, 0)
    gui._departure_target = ('HOME', [], [])
    gui.execute_destination = Mock()
    gui._robot_pose.return_value = (1.0, 0, 0)
    gui._complete_loaded_departure()
    gui.execute_destination.assert_not_called()
    assert gui._execution_done.call_count == 1


def test_departure_handshake_timeout_is_fail_closed():
    gui = panel_fixture()
    gui._departure_pending = {'started': time_value(1)}
    gui._dock_pending = None
    gui._dock_wait_watchdog()
    assert gui._departure_pending is None
    gui._execution_done.assert_called_once()


def test_owner_arbitration_favors_loaded_departure_and_stops_when_disarmed():
    gui = panel_fixture()
    gui._motion_owner_pub = Mock()
    gui._route_transit_pub = Mock()
    gui._departure_pending = {'started': time_value(10)}
    gui._qr_alignment_active = False
    gui._follow_goal_pending = False
    gui._active_follow = None
    gui._stage_validation_pending = False
    gui._publish_motion_owner()
    assert gui._motion_owner_pub.publish.call_args.args[0].data == 'DOCK_DEPART'
    gui._cancel_requested = True
    gui._publish_motion_owner()
    assert gui._motion_owner_pub.publish.call_args.args[0].data == 'STOP'


def test_run_requires_current_preview_then_waits_for_departure_guard():
    gui = panel_fixture()
    routes = [('station-to-home', {
        'source': 'station', 'source_entrance': 'east',
        'execution_approved': True, '_database_execution_approved': True}, [])]
    gui._prepare_loaded_departure('station', 0, (0, 0, 0), 'HOME', routes, [])
    gui._follow_goal_pending = False
    gui._stage_validation_pending = False
    gui._active_follow = None
    gui._qr_alignment_active = False
    gui._cargo_task = None
    gui._pending_path = None
    gui._pending_alignment_station = None
    gui.run_selected_mission(True)
    assert gui._departure_pending is not None
    assert gui._pending_departure is None
    gui._follow.assert_not_called()


def test_departure_preview_rejects_unapproved_source_and_preserves_no_motion():
    gui = panel_fixture()
    gui._approach_provider.return_value = (
        {'execution_approved': False}, [[2, 0, 0], [1, 0, 0], [0, 0, 0]])
    routes = [('station-to-home', {
        'source': 'station', 'source_entrance': 'east',
        'execution_approved': True, '_database_execution_approved': True}, [])]
    with pytest.raises(ValueError, match='not execution-approved'):
        gui._prepare_loaded_departure('station', 0, (0, 0, 0), 'HOME', routes, [])
    gui._plan_ready.assert_not_called()
    gui._follow.assert_not_called()


def test_loaded_departure_yaw_rate_respects_4ws_steering_limit():
    gate = gate_fixture()
    gate._requests['DOCK_NAV'].twist.angular.z = 1.0
    gate._tick()
    output = gate._output.publish.call_args.args[0].twist
    assert output.linear.x == pytest.approx(0.3)  # transfer-station limit
    assert abs(output.angular.z) <= 2 * abs(output.linear.x) / 0.58 + 1e-9


def test_departure_is_refused_when_no_route_leaves_by_the_entry_gate():
    gui = panel_fixture()
    routes = [('station-to-home', {
        'source': 'station', 'source_entrance': 'north',
        'execution_approved': True, '_database_execution_approved': True}, [])]
    with pytest.raises(ValueError, match='east gate'):
        gui._prepare_loaded_departure('station', 0, (0, 0, 0), 'HOME', routes, [])
    assert gui._pending_departure is None


def test_loaded_robot_leaves_home_through_a_fitting_gate():
    # Reported: Station 3 cargo -> HOME -> Station 4 was refused at HOME.
    gui = panel_fixture()
    gui._pass_through = ('HOME',)
    gui.docked_station = 'HOME'
    routes = [('HOME__TO__cargo_dropoff_southeast', {
        'source': 'HOME', 'source_entrance': 'east',
        'execution_approved': True, '_database_execution_approved': True}, [])]
    approach = [[2.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    gui._pass_through_exit = Mock(return_value=(approach, routes))
    gui._prepare_loaded_departure('HOME', 3, (0, 0, 0), 'cargo_dropoff_southeast', routes, [])
    gui._pass_through_exit.assert_called_once()
    gui._approach_provider.assert_not_called()
    assert gui._pending_departure['station'] == 'HOME'
    assert gui._pending_departure['cargo_id'] == 3
    gui._modeled_loaded_path_clearance.assert_called_once()
