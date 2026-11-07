"""HOME camera alignment must be previewed before any motion is enabled."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import math

import pytest

pytest.importorskip('rclpy')
pytest.importorskip('python_qt_binding')
from tugger_control.operator_panel import OperatorBridge
from tugger_control.lift_geometry import LIFT_CONTACT_M, LIFT_DOWN_M, LIFT_UP_M


def _bridge():
    bridge = SimpleNamespace(
        _begin_preflight=Mock(return_value=1),
        _confirmed_cargo=Mock(return_value=None),
        _robot_pose=Mock(return_value=(0.025, 0.025, 0.0)),
        _report=Mock(),
        _start_qr_alignment=Mock(),
        _pending_path=None,
        _pending_pose=None,
        _pending_alignment_station=None,
        _pending_destination=None,
        _pending_execution_approved=False,
        _stage_validation_pending=False,
        _dock_precision_mode=False,
        _dock_pending=None,
        _departure_pending=None,
        _departure_active=False,
        _pending_departure=None,
        _departure_target=None,
        _dock_follower=Mock(),
        _motion_armed=False,
        _cancel_requested=False,
        _active_follow=None,
        _follow_goal_pending=False,
        _qr_alignment_active=False,
        _cargo_task=None,
        _loaded_control_ready=True,
        _loaded_guard_is_ready=Mock(return_value=True),
        _start_cargo_operation=Mock(),
        _hold_visual_alignment=Mock(),
        _advance_cargo_operation=Mock(),
        _cargo_seen_at={0: None, 3: None},
        _cargo_poses={0: (0, 0, 0, 0), 3: (0, 0, 0, 0)},
        _cargo_pose_seen_at={0: None, 3: None},
        get_clock=Mock(return_value=SimpleNamespace(now=Mock(return_value=1))),
        docked_station=None,
        _pass_through=('HOME',),
        _station_centers={'HOME': [0.0, 0.0]},
        _infer_pass_through_dock=Mock(),
        _start_route_chain=Mock(return_value=False),
        _maybe_turn_at_dock=Mock(return_value=False),
    )
    bridge._report_run_block = lambda message: OperatorBridge._report_run_block(
        bridge, message)
    bridge._robot_pose_at = lambda stamp: bridge._robot_pose()
    # Simulation start: Stations 1 and 3 hold their cargo; drop-offs are empty.
    bridge._station_occupancy = Mock(return_value={
        'cargo_station_0': 0, 'cargo_station_3': 3,
        'cargo_dropoff_northwest': None, 'cargo_dropoff_southeast': None})
    for name in ('_assess_destination', '_check_destination_allowed',
                 '_deviation_confirmed', '_dock_only'):
        setattr(bridge, name, getattr(OperatorBridge, name).__get__(bridge))
    for name in ('_station_entry_approach', '_dispatch_decision', '_start_live_plan',
                 '_start_cargo_operation_now', '_start_qr_alignment_now'):
        setattr(bridge, name, getattr(OperatorBridge, name).__get__(bridge))
    bridge._live_planner = None
    return bridge


def test_alignment_requires_fresh_stationary_wheel_odometry():
    from rclpy.time import Time
    bridge = _bridge()
    now = Time(seconds=10)
    bridge._alignment_odom = None
    bridge._alignment_odom_stamp = None
    assert not OperatorBridge._alignment_is_stationary(bridge, now)
    bridge._alignment_odom = (0.001, 0.0, 0.005)
    bridge._alignment_odom_stamp = 9.9
    assert OperatorBridge._alignment_is_stationary(bridge, now)
    bridge._alignment_odom = (0.02, 0.0, 0.0)
    assert not OperatorBridge._alignment_is_stationary(bridge, now)
    bridge._alignment_odom = (0.0, 0.0, 0.0)
    bridge._alignment_odom_stamp = 9.0
    assert not OperatorBridge._alignment_is_stationary(bridge, now)


@pytest.mark.parametrize('stamp,previous', [(9.0, None), (10.1, None), (9.9, 9.9)])
def test_alignment_holds_for_stale_future_or_repeated_camera_frame(stamp, previous):
    from geometry_msgs.msg import PoseStamped
    from rclpy.time import Time
    bridge = _bridge()
    bridge._qr_alignment_active = True
    bridge._qr_alignment_station = 'HOME'
    bridge._red_frame_stamp = previous
    bridge.get_clock.return_value.now.return_value = Time(seconds=10)
    bridge._stop_qr_alignment = Mock()
    message = PoseStamped()
    message.header.frame_id = 'base_footprint'
    message.header.stamp = Time(nanoseconds=round(stamp * 1e9)).to_msg()
    OperatorBridge._red_frame_pose(bridge, message)
    bridge._stop_qr_alignment.assert_not_called()
    bridge._hold_visual_alignment.assert_called_once()


def test_alignment_waits_without_motion_for_expected_fresh_marker_identity():
    from geometry_msgs.msg import PoseStamped
    from rclpy.time import Time
    bridge = _bridge()
    bridge._qr_alignment_active = True
    bridge._qr_alignment_station = 'HOME'
    bridge._red_frame_stamp = None
    bridge._qr_aligned_frames = 2
    bridge._expected_qr_is_fresh = Mock(return_value=False)
    bridge._publish_precision_stop = Mock()
    bridge._precision_cmd_pub = Mock()
    bridge.get_clock.return_value.now.return_value = Time(seconds=10)
    message = PoseStamped()
    message.header.frame_id = 'base_footprint'
    message.header.stamp = Time(seconds=10).to_msg()
    OperatorBridge._red_frame_pose(bridge, message)
    bridge._hold_visual_alignment.assert_called_once()
    bridge._precision_cmd_pub.publish.assert_not_called()


@pytest.mark.parametrize('yaw,cancelled', [(0.105, False), (0.005, True)])
def test_heading_correction_is_not_cancelled_on_position_alone(yaw, cancelled):
    # Regression: a robot on the target position but 6 degrees off had every
    # heading correction cancelled after one camera frame, so it never turned.
    from geometry_msgs.msg import PoseStamped
    from rclpy.time import Time
    bridge = _bridge()
    bridge._qr_alignment_active = True
    bridge._qr_alignment_station = 'cargo_station_0'
    bridge._red_frame_stamp = None
    bridge._expected_qr_is_fresh = Mock(return_value=True)
    bridge._robot_pose = Mock(return_value=(0.0, 0.0, yaw))
    bridge._alignment_target_yaw = 0.0
    bridge._alignment_heading_error = (
        lambda pose=None: OperatorBridge._alignment_heading_error(bridge, pose))
    bridge._qr_heading_agrees = Mock(return_value=True)
    bridge._visual_marker_guard = Mock(marker=(0.0, 0.0))
    bridge._visual_marker_guard.check.return_value = ('OK', '')
    bridge._visual_camera_outliers = 0
    executor = object()
    bridge._visual_executor = executor
    bridge._visual_plan_marker = (0.001, 0.0)
    bridge._visual_plan_origin = (0.0, 0.0, yaw)
    bridge._visual_replan_requested = False
    bridge._qr_aligned_frames = 0
    bridge._publish_precision_stop = Mock()
    bridge._report_red_frame_state = Mock()
    bridge._visual_alignment_step = Mock()
    bridge._alignment_is_stationary = Mock(return_value=False)
    bridge.get_clock.return_value.now.return_value = Time(seconds=10)
    message = PoseStamped()
    message.header.frame_id = 'base_footprint'
    message.header.stamp = Time(seconds=10).to_msg()
    message.pose.position.x = 0.001
    OperatorBridge._red_frame_pose(bridge, message)
    assert (bridge._visual_executor is None) is cancelled


@pytest.mark.parametrize('map_shift,held', [(0.0, False), (0.01, True)])
def test_marker_change_during_a_maneuver_is_judged_in_the_map(map_shift, held):
    # Regression: the robot backing 3 cm moves the marker 3 cm in the robot
    # frame; that is not a target change and must not cancel the maneuver.
    from geometry_msgs.msg import PoseStamped
    from rclpy.time import Time
    bridge = _bridge()
    bridge._qr_alignment_active = True
    bridge._qr_alignment_station = 'cargo_station_0'
    bridge._red_frame_stamp = None
    bridge._expected_qr_is_fresh = Mock(return_value=True)
    bridge._robot_pose = Mock(return_value=(-0.03, 0.0, 0.0))
    bridge._alignment_target_yaw = 0.0
    bridge._alignment_heading_error = (
        lambda pose=None: OperatorBridge._alignment_heading_error(bridge, pose))
    bridge._qr_heading_agrees = Mock(return_value=True)
    bridge._visual_marker_guard = Mock(marker=(0.0, 0.0))
    bridge._visual_marker_guard.check.return_value = ('OK', '')
    bridge._visual_camera_outliers = 0
    bridge._visual_executor = object()
    bridge._visual_plan_origin = (0.0, 0.0, 0.0)
    bridge._visual_plan_marker = (0.0, 0.0)
    bridge._qr_aligned_frames = 0
    bridge._report_red_frame_state = Mock()
    bridge._visual_alignment_step = Mock()
    bridge._publish_precision_stop = Mock()
    bridge._alignment_is_stationary = Mock(return_value=False)
    bridge.get_clock.return_value.now.return_value = Time(seconds=10)
    message = PoseStamped()
    message.header.frame_id = 'base_footprint'
    message.header.stamp = Time(seconds=10).to_msg()
    message.pose.position.x = 0.03 + map_shift
    OperatorBridge._red_frame_pose(bridge, message)
    assert bridge._hold_visual_alignment.called is held


def test_near_home_is_a_preview_until_armed_run():
    bridge = _bridge()
    OperatorBridge.execute_destination(bridge, 'HOME', [], [[0.025, 0.025, 0.0]])
    assert bridge._pending_alignment_station == 'HOME'
    bridge._start_qr_alignment.assert_not_called()
    OperatorBridge.run_selected_mission(bridge, False)
    bridge._start_qr_alignment.assert_not_called()
    OperatorBridge.run_selected_mission(bridge, True)
    bridge._start_qr_alignment.assert_called_once_with('HOME')
    assert bridge._pending_alignment_station is None


def test_home_alignment_refuses_a_stale_preview():
    bridge = _bridge()
    OperatorBridge.execute_destination(bridge, 'HOME', [], [[0.025, 0.025, 0.0]])
    bridge._robot_pose.return_value = (1.0, 0.025, 0.0)
    OperatorBridge.run_selected_mission(bridge, True)
    bridge._start_qr_alignment.assert_not_called()


def test_run_cannot_start_another_mission_during_active_navigation():
    bridge = _bridge()
    OperatorBridge.execute_destination(bridge, 'HOME', [], [[0.025, 0.025, 0.0]])
    bridge._active_follow = object()
    OperatorBridge.run_selected_mission(bridge, True)
    bridge._start_qr_alignment.assert_not_called()


def test_motion_owner_follows_armed_active_stage_and_stops_on_cancel():
    bridge = _bridge()
    bridge._motion_owner_pub = Mock()
    bridge._motion_armed = True
    bridge._follow_goal_pending = True
    OperatorBridge._publish_motion_owner(bridge)
    assert bridge._motion_owner_pub.publish.call_args.args[0].data == 'NAV'
    bridge._follow_goal_pending = False
    bridge._qr_alignment_active = True
    OperatorBridge._publish_motion_owner(bridge)
    assert bridge._motion_owner_pub.publish.call_args.args[0].data == 'PRECISION'
    bridge._cancel_requested = True
    OperatorBridge._publish_motion_owner(bridge)
    assert bridge._motion_owner_pub.publish.call_args.args[0].data == 'STOP'


def test_loaded_guard_heartbeat_must_be_fresh():
    from rclpy.time import Time
    bridge = _bridge()
    bridge.get_clock.return_value.now.return_value = Time(seconds=10)
    bridge._loaded_guard_seen_at = Time(seconds=9)
    assert not OperatorBridge._loaded_guard_is_ready(bridge)
    bridge._loaded_guard_seen_at = Time(seconds=9, nanoseconds=900_000_000)
    assert OperatorBridge._loaded_guard_is_ready(bridge)
    bridge._loaded_control_ready = False
    assert not OperatorBridge._loaded_guard_is_ready(bridge)


def test_run_rechecks_the_chosen_cargo_action_after_preview():
    # ATTACH was chosen, but the robot is already carrying cargo at RUN time.
    bridge = _bridge()
    bridge._mission_action = 'attach'
    bridge._pending_alignment_station = 'cargo_dropoff_northwest'
    bridge._pending_destination = 'cargo_dropoff_northwest'
    bridge._pending_execution_approved = True
    bridge._pending_pose = (0.025, 0.025, 0.0)
    bridge._confirmed_cargo.return_value = 3
    OperatorBridge.run_selected_mission(bridge, True)
    bridge._start_qr_alignment.assert_not_called()
    assert any('Robot already carries Station 3 cargo: nothing more to attach' in call.args[0]
               for call in bridge._report.call_args_list)


def test_run_rejects_unapproved_saved_or_precision_path():
    bridge = _bridge()
    bridge._pending_alignment_station = 'cargo_station_0'
    bridge._pending_destination = 'cargo_station_0'
    bridge._pending_pose = (0.025, 0.025, 0.0)
    OperatorBridge.run_selected_mission(bridge, True)
    bridge._start_qr_alignment.assert_not_called()
    assert any('not execution-approved' in call.args[0]
               for call in bridge._report.call_args_list)


def test_no_cargo_action_selected_asks_for_a_second_run():
    bridge = _bridge()
    bridge._confirmed_cargo.return_value = None
    bridge._mission_confirm = None
    bridge._mission_action = None
    bridge._approach_provider = Mock(side_effect=ValueError('not needed for this check'))
    bridge._robot_pose.return_value = (0.0, 0.0, 0.0)
    OperatorBridge.execute_destination(bridge, 'cargo_dropoff_southeast', [], [[0, 0, 0]])
    assert bridge._mission_confirm['confirmed'] is False
    assert any('no cargo action selected' in call.args[0]
               for call in bridge._report.call_args_list)


def test_carrying_cargo_to_a_pickup_station_is_blocked():
    bridge = _bridge()
    bridge._confirmed_cargo.return_value = 0
    bridge._approach_provider = Mock()
    OperatorBridge.execute_destination(bridge, 'cargo_station_3', [], [[0, 0, 0]])
    bridge._approach_provider.assert_not_called()
    assert any('Robot is carrying Station 1 cargo and Station 3 holds Station 3 cargo: '
               'drop the carried cargo off first' in call.args[0]
               for call in bridge._report.call_args_list)


def test_first_run_warns_and_second_run_proceeds():
    bridge = _bridge()
    bridge._continue_attempt = Mock()
    bridge._mission_confirm = {'destination': 'cargo_dropoff_southeast',
                               'warnings': ['Station 1 cargo belongs at Station 2'],
                               'dock_only': False, 'confirmed': False}
    OperatorBridge.run_selected_mission_with_retry(bridge, True)
    bridge._continue_attempt.assert_not_called()
    assert 'Press RUN again to proceed' in bridge._report.call_args.args[0]
    OperatorBridge.run_selected_mission_with_retry(bridge, True)
    bridge._continue_attempt.assert_called_once_with(1)


def test_automatic_retries_cannot_confirm_a_warning():
    bridge = _bridge()
    bridge._mission_confirm = {'destination': 'cargo_dropoff_southeast',
                               'warnings': ['unusual'], 'dock_only': False,
                               'confirmed': False}
    bridge._confirmed_cargo.return_value = 0
    for _ in range(3):
        try:
            OperatorBridge._check_destination_allowed(bridge, 'cargo_dropoff_southeast', 0)
        except ValueError as exc:
            assert 'unconfirmed' in str(exc)
        else:
            raise AssertionError('an unconfirmed warning was allowed')
    assert bridge._mission_confirm['confirmed'] is False


def test_unknown_cargo_feedback_is_not_treated_as_unloaded():
    bridge = _bridge()
    bridge._confirmed_cargo.side_effect = ValueError('Cargo attachment state is unavailable or stale')
    bridge._approach_provider = Mock()
    OperatorBridge.execute_destination(bridge, 'cargo_station_0', [], [[0, 0, 0]])
    bridge._approach_provider.assert_not_called()
    assert any('unavailable or stale' in call.args[0]
               for call in bridge._report.call_args_list)


@pytest.mark.parametrize('success,valid', [(False, False), (True, False)])
def test_path_validation_failure_never_enables_run(success, valid):
    bridge = _bridge()
    bridge._plan_generation = 3
    bridge._plan_ready = Mock()
    future = Mock()
    future.result.return_value = SimpleNamespace(success=success, is_valid=valid)
    OperatorBridge._on_path_validated(bridge, future, 3, object(), (0, 0, 0), 'route')
    assert bridge._pending_path is None
    bridge._plan_ready.assert_not_called()


def test_successful_validation_from_old_preview_is_ignored():
    bridge = _bridge()
    bridge._plan_generation = 4
    bridge._plan_ready = Mock()
    future = Mock()
    future.result.return_value = SimpleNamespace(success=True, is_valid=True)
    OperatorBridge._on_path_validated(bridge, future, 3, object(), (0, 0, 0), 'route')
    assert bridge._pending_path is None
    bridge._plan_ready.assert_not_called()


def test_successful_current_validation_enables_exact_preview():
    bridge = _bridge()
    bridge._plan_generation = 4
    bridge._plan_ready = Mock()
    path = object()
    future = Mock()
    future.result.return_value = SimpleNamespace(success=True, is_valid=True)
    OperatorBridge._on_path_validated(bridge, future, 4, path, (0, 0, 0), 'route')
    assert bridge._pending_path is path
    bridge._plan_ready.assert_called_once()


def test_centering_at_cargo_station_is_not_reported_as_completed_pickup():
    bridge = _bridge()
    bridge._mission_action = 'attach'   # chosen by the operator
    bridge._qr_alignment_active = True
    bridge._qr_alignment_station = 'cargo_station_0'
    bridge._red_frame_error = (0.001, 0.002)
    bridge._publish_precision_stop = Mock()
    bridge._execution_done = Mock()
    OperatorBridge._complete_red_frame_alignment(bridge)
    assert bridge.docked_station == 'cargo_station_0'
    bridge._start_cargo_operation.assert_called_once_with('cargo_station_0')
    bridge._execution_done.assert_not_called()
    assert any('PRECISION POSITION REACHED' in call.args[0]
               for call in bridge._report.call_args_list)
    assert not any('CARGO ATTACHED' in call.args[0]
                   for call in bridge._report.call_args_list)


def test_cargo_attachment_report_requires_simulator_state_and_is_not_repeated():
    bridge = _bridge()
    bridge._cargo_states = {0: None, 3: None}
    OperatorBridge._cargo_state(bridge, 0, SimpleNamespace(data='attached'))
    bridge._report.assert_called_once_with(
        'CARGO JOINT ATTACHED: Station 1 cargo (simulator joint state).')
    OperatorBridge._cargo_state(bridge, 0, SimpleNamespace(data='attached'))
    OperatorBridge._cargo_state(bridge, 3, SimpleNamespace(data='unexpected'))
    assert bridge._report.call_count == 1


def test_cargo_detachment_is_reported_separately():
    bridge = _bridge()
    bridge._cargo_states = {0: 'attached', 3: None}
    OperatorBridge._cargo_state(bridge, 0, SimpleNamespace(data='detached'))
    bridge._report.assert_called_once_with(
        'CARGO JOINT DETACHED: Station 1 cargo (simulator joint state).')


def test_pickup_is_blocked_if_height_aware_clearance_is_unavailable():
    from rclpy.time import Time
    bridge = _bridge()
    now = Time(seconds=10)
    bridge.get_clock.return_value.now.return_value = now
    bridge._motion_armed = True
    bridge._expected_qr_is_fresh = Mock(return_value=True)
    bridge._lift_seen_at = now
    bridge._lift_position = 0.0
    bridge._cargo_clearance_guard = None
    bridge.cargo = Mock()
    bridge._execution_done = Mock()
    OperatorBridge._start_cargo_operation(bridge, 'cargo_station_0')
    bridge.cargo.assert_not_called()
    assert bridge._execution_done.call_args.args[0] is False



def test_pickup_sequence_waits_for_post_command_attachment_and_lift_feedback():
    from rclpy.time import Time
    bridge = _bridge()
    bridge._mission_action = 'attach'   # chosen by the operator
    clock = bridge.get_clock.return_value
    clock.now.return_value = Time(seconds=10)
    bridge._motion_armed = True
    bridge._expected_qr_is_fresh = Mock(return_value=True)
    bridge._cargo_clearance_guard = Mock(return_value=True)
    bridge._lift_seen_at = Time(seconds=10)
    bridge._lift_position = 0.0
    bridge._cargo_states = {0: 'detached', 3: 'detached'}
    bridge._cargo_seen_at = {0: Time(seconds=10), 3: Time(seconds=10)}
    bridge._cargo_pose_seen_at[0] = Time(seconds=10)
    bridge.cargo = Mock()
    bridge.lift = Mock()
    bridge._execution_done = Mock()
    OperatorBridge._start_cargo_operation(bridge, 'cargo_station_0')
    bridge.cargo.assert_not_called()
    clock.now.return_value = Time(seconds=11)
    OperatorBridge._advance_cargo_operation(bridge)
    # First the platform rises to touch the cargo; nothing is attached yet.
    bridge.lift.assert_called_once_with(LIFT_CONTACT_M)
    bridge.cargo.assert_not_called()
    clock.now.return_value = Time(seconds=12)
    for second in (12, 13):
        clock.now.return_value = Time(seconds=second)
        bridge._lift_seen_at = Time(seconds=second)
        bridge._lift_position = LIFT_CONTACT_M
        bridge._cargo_seen_at[0] = Time(seconds=second)
        bridge._cargo_pose_seen_at[0] = Time(seconds=second)
        OperatorBridge._advance_cargo_operation(bridge)
    bridge.cargo.assert_called_once_with(0, True)
    bridge._execution_done.assert_not_called()
    clock.now.return_value = Time(seconds=13.5)
    bridge._cargo_seen_at[0] = Time(seconds=13.5)
    bridge._cargo_states[0] = 'attached'
    OperatorBridge._advance_cargo_operation(bridge)
    assert bridge.lift.call_args.args[0] == LIFT_UP_M
    bridge._execution_done.assert_not_called()
    clock.now.return_value = Time(seconds=14)
    bridge._cargo_seen_at[0] = Time(seconds=14)
    bridge._lift_seen_at = Time(seconds=14)
    bridge._lift_position = LIFT_UP_M
    bridge._cargo_poses[0] = (0, 0, 0.065, 0)
    bridge._cargo_pose_seen_at[0] = Time(seconds=14)
    OperatorBridge._advance_cargo_operation(bridge)
    assert bridge._execution_done.call_args.args[0] is True


def test_lift_holds_and_mission_fails_if_clearance_is_lost():
    from rclpy.time import Time
    from tugger_control.cargo_operation import CargoOperation
    bridge = _bridge()
    bridge._motion_armed = True
    bridge.docked_station = 'cargo_station_0'
    bridge._cargo_task = CargoOperation('pickup', 0, 1.0, 0.0)
    bridge._cargo_task.phase = 'WAIT_RAISE'
    bridge._cargo_task.phase_started = 1.0
    bridge.get_clock.return_value.now.return_value = Time(seconds=2)
    bridge._cargo_states = {0: 'attached', 3: 'detached'}
    bridge._cargo_seen_at = {0: Time(seconds=2), 3: Time(seconds=2)}
    bridge._lift_seen_at = Time(seconds=2)
    bridge._lift_position = 0.03
    bridge._cargo_pose_seen_at[0] = Time(seconds=2)
    bridge._cargo_clearance_guard = Mock(return_value=False)
    bridge.lift = Mock()
    bridge._execution_done = Mock()
    OperatorBridge._advance_cargo_operation(bridge)
    bridge.lift.assert_called_once_with(0.03)
    assert bridge._execution_done.call_args.args[0] is False


def test_paired_dropoff_waits_for_lowering_then_detachment_feedback():
    from rclpy.time import Time
    bridge = _bridge()
    bridge._mission_action = 'detach'   # chosen by the operator
    clock = bridge.get_clock.return_value
    clock.now.return_value = Time(seconds=10)
    bridge._motion_armed = True
    bridge._confirmed_cargo.return_value = 0
    bridge._expected_qr_is_fresh = Mock(return_value=True)
    bridge._cargo_clearance_guard = Mock(return_value=True)
    bridge._lift_seen_at = Time(seconds=10)
    bridge._lift_position = LIFT_UP_M
    bridge._cargo_states = {0: 'attached', 3: 'detached'}
    bridge._cargo_seen_at = {0: Time(seconds=10), 3: Time(seconds=10)}
    bridge._cargo_poses[0] = (0, 0, 0.065, 0)
    bridge._cargo_pose_seen_at[0] = Time(seconds=10)
    bridge.cargo = Mock()
    bridge.lift = Mock()
    bridge._execution_done = Mock()
    OperatorBridge._start_cargo_operation(bridge, 'cargo_dropoff_northwest')
    # Lower only to contact: the cargo comes to rest on its support.
    bridge.lift.assert_called_once_with(LIFT_CONTACT_M)
    bridge.cargo.assert_not_called()
    for second in (11, 12):
        clock.now.return_value = Time(seconds=second)
        bridge._cargo_seen_at[0] = Time(seconds=second)
        bridge._lift_seen_at = Time(seconds=second)
        bridge._lift_position = LIFT_CONTACT_M
        bridge._cargo_poses[0] = (0, 0, 0, 0)
        bridge._cargo_pose_seen_at[0] = Time(seconds=second)
        OperatorBridge._advance_cargo_operation(bridge)
    bridge.cargo.assert_called_once_with(0, False)
    bridge._execution_done.assert_not_called()
    clock.now.return_value = Time(seconds=13)
    bridge._cargo_seen_at[0] = Time(seconds=13)
    bridge._cargo_states[0] = 'detached'
    bridge._cargo_pose_seen_at[0] = Time(seconds=13)
    OperatorBridge._advance_cargo_operation(bridge)
    # Detached at contact; only now does the platform lower fully.
    assert bridge.lift.call_args.args[0] == LIFT_DOWN_M
    bridge._execution_done.assert_not_called()
    clock.now.return_value = Time(seconds=14)
    bridge._cargo_seen_at[0] = Time(seconds=14)
    bridge._lift_seen_at = Time(seconds=14)
    bridge._lift_position = LIFT_DOWN_M
    bridge._cargo_pose_seen_at[0] = Time(seconds=14)
    OperatorBridge._advance_cargo_operation(bridge)
    assert bridge._execution_done.call_args.args[0] is True


def test_centering_home_completes_the_home_mission():
    bridge = _bridge()
    bridge._qr_alignment_active = True
    bridge._qr_alignment_station = 'HOME'
    bridge._red_frame_error = (0.001, 0.002)
    bridge._publish_precision_stop = Mock()
    bridge._execution_done = Mock()
    OperatorBridge._complete_red_frame_alignment(bridge)
    assert bridge._execution_done.call_args.args[0] is True


def test_station_mission_queues_matching_precision_segment():
    bridge = _bridge()
    bridge._approach_provider = Mock(return_value=({'execution_approved': True}, [[2, 0, 0], [3, 0, 0]]))
    bridge.execute_saved_route = Mock()
    bridge._robot_pose.return_value = (0.0, 0.0, 0.0)
    route = {'source': 'HOME', 'target': 'cargo_station_0'}
    poses = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    OperatorBridge.execute_destination(
        bridge, 'cargo_station_0', [('route', route, poses)], [[2, 0, 0]])
    assert bridge.execute_saved_route.call_args.kwargs['next_precision'] == [[2, 0, 0], [3, 0, 0]]


def test_database_rejection_cannot_be_overridden_by_route_file():
    bridge = _bridge()
    bridge._approach_provider = Mock(return_value=(
        {'execution_approved': True}, [[2, 0, 0], [3, 0, 0]]))
    bridge.execute_saved_route = Mock()
    bridge._robot_pose.return_value = (0.0, 0.0, 0.0)
    route = {'source': 'HOME', 'target': 'cargo_station_0',
             'execution_approved': True, '_database_execution_approved': False}
    poses = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    OperatorBridge.execute_destination(
        bridge, 'cargo_station_0', [('route', route, poses)], [[2, 0, 0]])
    assert bridge.execute_saved_route.call_args.kwargs['execution_approved'] is False


def test_unapproved_precision_segment_blocks_the_whole_mission():
    bridge = _bridge()
    bridge._approach_provider = Mock(return_value=(
        {'execution_approved': False}, [[2, 0, 0], [3, 0, 0]]))
    bridge.execute_saved_route = Mock()
    bridge._robot_pose.return_value = (0.0, 0.0, 0.0)
    route = {'source': 'HOME', 'target': 'cargo_station_0',
             'execution_approved': True, '_database_execution_approved': True}
    poses = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    OperatorBridge.execute_destination(
        bridge, 'cargo_station_0', [('route', route, poses)], [[2, 0, 0]])
    assert bridge.execute_saved_route.call_args.kwargs['execution_approved'] is False


def test_enabled_saved_and_precision_routes_are_passed_to_run_preflight():
    bridge = _bridge()
    bridge._approach_provider = Mock(return_value=(
        {'execution_approved': True}, [[2, 0, 0], [3, 0, 0]]))
    bridge.execute_saved_route = Mock()
    bridge._robot_pose.return_value = (0.0, 0.0, 0.0)
    route = {'source': 'HOME', 'target': 'cargo_station_0',
             'execution_approved': True, '_database_execution_approved': True}
    OperatorBridge.execute_destination(
        bridge, 'cargo_station_0', [('route', route, [[0, 0, 0], [1, 0, 0], [2, 0, 0]])],
        [[2, 0, 0]])
    assert bridge.execute_saved_route.call_args.kwargs['execution_approved'] is True


def test_station_mission_rejects_approach_from_a_different_entrance():
    bridge = _bridge()
    bridge._approach_provider = Mock(return_value=({'execution_approved': True}, [[2, 2, 0], [3, 2, 0]]))
    bridge.execute_saved_route = Mock()
    bridge._robot_pose.return_value = (0.0, 0.0, 0.0)
    route = {'source': 'HOME', 'target': 'cargo_station_0'}
    poses = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    OperatorBridge.execute_destination(
        bridge, 'cargo_station_0', [('route', route, poses)], [[2, 0, 0]])
    bridge.execute_saved_route.assert_not_called()
    assert any('without a matching' in call.args[0]
               for call in bridge._report.call_args_list)


def test_segment_one_success_starts_queued_precision_segment():
    bridge = _bridge()
    bridge._queued_precision_poses = [[2, 0, 0], [3, 0, 0]]
    bridge._start_precision_segment = Mock()
    bridge._execution_done = Mock()
    future = Mock()
    from action_msgs.msg import GoalStatus
    future.result.return_value = SimpleNamespace(status=GoalStatus.STATUS_SUCCEEDED)
    OperatorBridge._on_follow_result(bridge, future)
    bridge._start_precision_segment.assert_called_once()
    bridge._execution_done.assert_not_called()


def test_segment_two_not_dispatched_after_disarm():
    bridge = _bridge()
    bridge._plan_generation = 7
    bridge._stage_validation_pending = True
    bridge._motion_armed = False
    bridge._queued_precision_poses = [[2, 0, 0], [3, 0, 0]]
    bridge._follow = Mock()
    bridge._execution_done = Mock()
    future = Mock()
    future.result.return_value = SimpleNamespace(success=True, is_valid=True)
    OperatorBridge._on_precision_validated(bridge, future, 7, 'cargo_station_0', object(), (0, 0, 0))
    bridge._follow.assert_not_called()
    assert bridge._execution_done.call_args.args[0] is False


def test_segment_two_reports_first_rejected_map_pose_without_dispatch():
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Path
    bridge = _bridge()
    bridge._plan_generation = 7
    bridge._stage_validation_pending = True
    bridge._motion_armed = True
    bridge._queued_precision_poses = [[0, 0, 0], [1, 0, 0]]
    bridge._follow = Mock()
    bridge._execution_done = Mock()
    path = Path()
    point = PoseStamped()
    point.pose.position.x = -16.032
    point.pose.position.y = -6.990
    path.poses.append(point)
    future = Mock()
    future.result.return_value = SimpleNamespace(success=True, is_valid=False,
                                                  invalid_pose_indices=[0])
    OperatorBridge._on_precision_validated(bridge, future, 7, 'cargo_station_0', path, (0, 0, 0))
    bridge._follow.assert_not_called()
    assert bridge._execution_done.call_args.args[0] is False
    assert any('rejected pose 0 at map x=-16.032, y=-6.990' in call.args[0]
               for call in bridge._report.call_args_list)


def test_unloaded_precision_waits_for_docking_guard_before_following():
    from rclpy.time import Time
    from tugger_control.clearance_2_5d import Box
    bridge = _bridge()
    bridge._mission_action = 'attach'   # chosen by the operator
    bridge._motion_armed = True
    bridge._plan_generation = 7
    bridge._robot_pose.return_value = (0.0, 0.0, 0.0)
    bridge._queued_precision_station = 'cargo_station_0'
    bridge._queued_precision_poses = [[0, 0, 0], [1, 0, 0]]
    bridge._lift_position = 0.0
    bridge._lift_seen_at = Time(seconds=10)
    bridge.get_clock.return_value.now.return_value = Time(seconds=10)
    bridge._collision_scene = '{}'
    bridge._dock_follower.server_is_ready.return_value = True
    bridge._pose = lambda values: OperatorBridge._pose(bridge, values)
    bridge._follow = Mock()
    bridge._execution_done = Mock()
    obstacle = Box('far', 10, 10, 0, .1, .1, 0, .2)
    with patch('tugger_control.operator_panel.validate_modeled_geometry'), \
         patch('tugger_control.operator_panel.scene_expected_counts', return_value={}), \
         patch('tugger_control.operator_panel.decode_scene', return_value=[('other', obstacle)]):
        OperatorBridge._start_precision_segment(bridge)
    assert bridge._stage_validation_pending
    assert bridge._dock_precision_mode
    bridge._follow.assert_not_called()
    OperatorBridge._dock_guard_ready(bridge, SimpleNamespace(data='STOP'))
    bridge._follow.assert_not_called()
    OperatorBridge._dock_guard_ready(bridge, SimpleNamespace(data='DOCK_NAV'))
    bridge._follow.assert_called_once()
    assert bridge._precision_station == 'cargo_station_0'


def test_docking_guard_reply_after_cancel_cannot_start_motion():
    from rclpy.time import Time
    bridge = _bridge()
    bridge._plan_generation = 7
    bridge._motion_armed = False
    bridge._stage_validation_pending = True
    bridge._dock_precision_mode = True
    bridge._dock_pending = (7, 'cargo_station_0', object(), (0, 0, 0), Time(seconds=10))
    bridge._follow = Mock()
    bridge._execution_done = Mock()
    OperatorBridge._dock_guard_ready(bridge, SimpleNamespace(data='DOCK_NAV'))
    bridge._follow.assert_not_called()
    assert not bridge._dock_precision_mode


def test_missing_docking_guard_times_out_without_following():
    from rclpy.time import Time
    bridge = _bridge()
    bridge._dock_pending = (7, 'cargo_station_0', object(), (0, 0, 0), Time(seconds=10))
    bridge.get_clock.return_value.now.return_value = Time(seconds=16)
    bridge._stage_validation_pending = True
    bridge._dock_precision_mode = True
    bridge._follow = Mock()
    bridge._execution_done = Mock()
    OperatorBridge._dock_wait_watchdog(bridge)
    bridge._follow.assert_not_called()
    assert not bridge._dock_precision_mode
    assert not bridge._stage_validation_pending
    assert bridge._execution_done.call_args.args[0] is False


def test_precision_owner_uses_separate_guarded_channels():
    bridge = _bridge()
    bridge._motion_owner_pub = Mock()
    bridge._motion_armed = True
    bridge._dock_precision_mode = True
    bridge._stage_validation_pending = True
    OperatorBridge._publish_motion_owner(bridge)
    assert bridge._motion_owner_pub.publish.call_args.args[0].data == 'DOCK_NAV'
    bridge._stage_validation_pending = False
    bridge._qr_alignment_active = True
    OperatorBridge._publish_motion_owner(bridge)
    assert bridge._motion_owner_pub.publish.call_args.args[0].data == 'DOCK_CAMERA'


def test_late_success_after_cancellation_does_not_start_segment_two():
    bridge = _bridge()
    bridge._cancel_requested = True
    bridge._queued_precision_poses = [[2, 0, 0], [3, 0, 0]]
    bridge._start_precision_segment = Mock()
    bridge._execution_done = Mock()
    future = Mock()
    from action_msgs.msg import GoalStatus
    future.result.return_value = SimpleNamespace(status=GoalStatus.STATUS_SUCCEEDED)
    OperatorBridge._on_follow_result(bridge, future)
    bridge._start_precision_segment.assert_not_called()
    assert bridge._execution_done.call_args.args == (False, GoalStatus.STATUS_CANCELED)


def _exit_bridge():
    bridge = _bridge()
    bridge._lift_position = 0.0
    bridge._collision_scene = 'scene'
    bridge._path_pub = Mock()
    bridge._plan_ready = Mock()
    bridge._pose = Mock(return_value=Mock())
    bridge._approach_provider = Mock(return_value=({}, []))
    bridge._station_approach = Mock(return_value=[])
    bridge._same_gate_routes = Mock(return_value=[])
    bridge._execution_done = Mock()
    bridge.execute_destination = Mock()
    bridge._station_exit_active = True
    bridge._station_exit_target = ('HOME', [], [])
    bridge._departure_exit_pose = (0.025, 0.025, 0.0)
    bridge._dock_precision_mode = True
    bridge._motion_armed = True
    bridge.docked_station = 'cargo_dropoff_northwest'
    return bridge


def test_station_exit_completion_requires_a_new_plan_from_the_measured_pose():
    bridge = _exit_bridge()
    OperatorBridge._complete_station_exit(bridge, True, 4)
    bridge.execute_destination.assert_called_once_with('HOME', [], [])
    bridge._execution_done.assert_not_called()
    assert bridge.docked_station is None
    assert not bridge._station_exit_active and not bridge._dock_precision_mode


def test_failed_station_exit_stops_and_keeps_the_docked_state():
    bridge = _exit_bridge()
    OperatorBridge._complete_station_exit(bridge, False, 6)
    bridge._execution_done.assert_called_once_with(False, 6)
    bridge.execute_destination.assert_not_called()
    assert bridge.docked_station == 'cargo_dropoff_northwest'


def test_station_exit_is_refused_with_the_lift_raised():
    bridge = _exit_bridge()
    bridge._lift_position = 0.065
    with pytest.raises(ValueError, match='lift'):
        OperatorBridge._prepare_station_exit(
            bridge, 'cargo_dropoff_northwest', (0.0, 0.0, 0.0), 'HOME', [], [])
    assert bridge._pending_path is None


def _continue_bridge(started_after):
    bridge = _bridge()
    calls = []

    def run(_armed):
        calls.append(1)
        if len(calls) >= started_after:
            bridge._active_follow = object()
    bridge.run_selected_mission = run
    bridge._motion_armed = True
    bridge._mission_started = lambda: OperatorBridge._mission_started(bridge)
    timers = []
    bridge.create_timer = lambda _period, callback: timers.append(callback) or Mock()
    bridge._continue_attempt = lambda attempt: OperatorBridge._continue_attempt(bridge, attempt)
    return bridge, calls, timers


def test_automatic_continuation_retries_a_momentary_block():
    bridge, calls, timers = _continue_bridge(started_after=2)
    bridge._continue_attempt(1)
    assert len(calls) == 1 and len(timers) == 1
    timers[0]()
    assert len(calls) == 2 and bridge._active_follow is not None


def test_automatic_continuation_gives_up_and_says_so():
    bridge, calls, timers = _continue_bridge(started_after=99)
    bridge._continue_attempt(1)
    while timers:
        timers.pop(0)()
    assert len(calls) == 5
    assert 'press RUN' in bridge._report.call_args.args[0]


def test_robot_on_home_centre_counts_as_docked_there():
    bridge = _bridge()
    OperatorBridge._infer_pass_through_dock(bridge, (0.005, -0.004, 3.14))
    assert bridge.docked_station == 'HOME'
    bridge.docked_station = None
    OperatorBridge._infer_pass_through_dock(bridge, (0.04, 0.0, 0.0))
    assert bridge.docked_station is None


def test_lift_never_commands_the_latching_lower_limit():
    from tugger_control.lift_geometry import LIFT_DOWN_M
    bridge = _bridge()
    bridge._lift_pub = Mock()
    for requested in (0.0, -0.01, 0.5):
        OperatorBridge.lift(bridge, requested)
        sent = bridge._lift_pub.publish.call_args.args[0].data[0]
        assert LIFT_DOWN_M <= sent <= LIFT_UP_M
    assert LIFT_DOWN_M > 0.0


def test_routine_guard_states_are_quiet_and_proximity_is_summarised():
    from tugger_control.operator_panel import guard_status_text
    assert guard_status_text('precision guarded') is None
    assert guard_status_text('loaded clearance reducing speed') is None
    text = guard_status_text(
        "loaded stopping envelope intersects ('chassis', 'cargo_dropoff_southeast_legs', 'x')")
    assert text == ('MOTION GUARD: slowing or holding near cargo_dropoff_southeast_legs '
                    '(chassis).')
    scoped = guard_status_text(
        "precision stopping envelope intersects ('chassis', "
        "'tugger_test::cargo_station_3_legs::legs_link::leg_0_0_collision')")
    assert scoped == 'MOTION GUARD: slowing or holding near cargo_station_3_legs (chassis).'
    assert guard_status_text('scene feed stale') == 'MOTION GUARD: scene feed stale.'


def test_automatic_retries_report_only_the_final_block():
    bridge, calls, timers = _continue_bridge(started_after=99)
    reports = []
    bridge._report = lambda message: reports.append(message)

    def run(_armed):
        calls.append(1)
        OperatorBridge._report_run_block(bridge, 'RUN blocked: scene stale.')
    bridge.run_selected_mission = run
    bridge._continue_attempt(1)
    while timers:
        timers.pop(0)()
    blocked = [line for line in reports if line.startswith('RUN blocked')]
    assert len(blocked) == 1


def test_first_lift_reading_sends_one_hold_command_never_below_down():
    from tugger_control.lift_geometry import LIFT_DOWN_M
    bridge = _bridge()
    bridge.lift = Mock()
    bridge._advance_cargo_operation = Mock()
    message = SimpleNamespace(name=['lift_joint'], position=[0.0])
    OperatorBridge._joint_state(bridge, message)
    OperatorBridge._joint_state(bridge, message)
    bridge.lift.assert_called_once_with(LIFT_DOWN_M)


def _recovery_bridge(phase, attempts=0):
    from action_msgs.msg import GoalStatus
    bridge = _bridge()
    bridge._motion_armed = True
    bridge._cancel_requested = False
    bridge._mission_phase = phase
    bridge._mission_retry = {'destination': 'cargo_station_0', 'candidates': [],
                             'targets': [], 'attempts': attempts}
    timers = []
    bridge.create_timer = lambda _period, callback: timers.append(callback) or Mock()
    bridge._recover = Mock()
    bridge._recoverable = lambda phase: OperatorBridge._recoverable(bridge, phase)
    done = Mock()
    return bridge, timers, done, GoalStatus


def test_recoverable_failure_retries_within_the_budget():
    bridge, timers, done, status = _recovery_bridge('approach')
    OperatorBridge._finish_mission(bridge, done, False, status.STATUS_ABORTED)
    done.assert_not_called()
    assert bridge._mission_retry['attempts'] == 1 and len(timers) == 1
    timers[0]()
    bridge._recover.assert_called_once()


def test_failure_after_the_budget_is_reported_with_a_clear_reason():
    bridge, timers, done, status = _recovery_bridge('camera', attempts=2)
    OperatorBridge._finish_mission(bridge, done, False, status.STATUS_ABORTED)
    done.assert_called_once_with(False, status.STATUS_ABORTED)
    assert 'after 2 recovery attempts' in bridge._report.call_args.args[0]
    assert not timers


def test_cancel_and_latched_lift_are_never_retried():
    bridge, timers, done, status = _recovery_bridge('approach')
    OperatorBridge._finish_mission(bridge, done, False, status.STATUS_CANCELED)
    done.assert_called_once()
    bridge, timers, done, status = _recovery_bridge('cargo')
    bridge._last_cargo_failure = ('pickup', 'lift_latched_in_simulator_restart_gazebo')
    OperatorBridge._finish_mission(bridge, done, False, status.STATUS_ABORTED)
    done.assert_called_once()
    assert not timers


def test_success_clears_the_recovery_budget():
    bridge, timers, done, status = _recovery_bridge('cargo', attempts=1)
    OperatorBridge._finish_mission(bridge, done, True, status.STATUS_SUCCEEDED)
    done.assert_called_once_with(True, status.STATUS_SUCCEEDED)
    assert bridge._mission_retry is None


def test_rejected_pickup_reverifies_the_dock_with_the_camera():
    # The simulator rejects an attach when the robot is misaligned with the
    # cargo, so recovery must re-check the dock, not just retry the lift.
    bridge, timers, _done, _status = _recovery_bridge('cargo')
    bridge.docked_station = 'cargo_station_0'
    bridge.lift = Mock()
    bridge._start_qr_alignment = Mock()
    bridge._start_cargo_operation = Mock()
    OperatorBridge._recover(bridge, 'cargo', bridge._mission_retry)
    bridge.lift.assert_called_once_with(LIFT_DOWN_M)
    timers[0]()
    bridge._start_qr_alignment.assert_called_once_with('cargo_station_0')
    bridge._start_cargo_operation.assert_not_called()


def test_camera_stage_gives_up_when_corrections_stop_helping():
    bridge = _bridge()
    bridge._alignment_heading_error = lambda pose: 0.105
    bridge._red_frame_error = (0.001, 0.0)
    stalled = [OperatorBridge._visual_progress_stalled(bridge, (0, 0, 0)) for _ in range(7)]
    assert stalled[:6] == [False] * 6 and stalled[6] is True


def test_camera_stage_keeps_going_while_the_error_shrinks():
    bridge = _bridge()
    bridge._alignment_heading_error = lambda pose: 0.0
    for step in range(10):
        bridge._red_frame_error = (0.03 - 0.003 * step, 0.0)
        assert OperatorBridge._visual_progress_stalled(bridge, (0, 0, 0)) is False


def test_camera_frame_waits_for_tf_at_its_own_stamp():
    from geometry_msgs.msg import PoseStamped
    from rclpy.time import Time
    from rclpy.clock_type import ClockType
    ROS = ClockType.ROS_TIME
    bridge = _bridge()
    bridge._qr_alignment_active = True
    bridge._red_frame_waiting = None
    bridge._red_frame_pose = Mock()
    bridge._release_red_frame = lambda: OperatorBridge._release_red_frame(bridge)
    bridge._tf = Mock()
    bridge._tf.can_transform.return_value = False
    message = PoseStamped()
    message.header.stamp = Time(seconds=10).to_msg()
    bridge.get_clock.return_value.now.return_value = Time(seconds=10.03, clock_type=ROS)
    OperatorBridge._red_frame_received(bridge, message)
    bridge._red_frame_pose.assert_not_called()
    bridge._tf.can_transform.return_value = True
    OperatorBridge._release_red_frame(bridge)
    bridge._red_frame_pose.assert_called_once_with(message)
    # A frame whose TF never arrives goes to the handler's staleness checks.
    bridge._red_frame_pose.reset_mock()
    bridge._tf.can_transform.return_value = False
    OperatorBridge._red_frame_received(bridge, message)
    bridge.get_clock.return_value.now.return_value = Time(seconds=10.25, clock_type=ROS)
    OperatorBridge._release_red_frame(bridge)
    bridge._red_frame_pose.assert_called_once_with(message)


def test_failed_short_connector_falls_through_to_a_live_plan():
    bridge = _bridge()
    bridge._recovery_only = False
    bridge._live_planner = Mock()
    bridge._start_live_plan = Mock()
    context = ('cargo_station_0', (0.04, -0.03, 0.0), [[1, 1, 0]], 3, [[0, 0, 0]], True)
    bridge._connector_fallback = context
    OperatorBridge._connector_failed(bridge, 'planner path changes direction', 5)
    bridge._start_live_plan.assert_called_once_with(
        'cargo_station_0', (0.04, -0.03, 0.0), [[1, 1, 0]], 5, [[0, 0, 0]], True)
    assert bridge._connector_fallback is None


def test_last_resort_recovery_connector_failure_stops_with_a_reason():
    bridge = _bridge()
    bridge._recovery_only = True
    bridge._live_planner = Mock()
    bridge._start_live_plan = Mock()
    bridge._connector_fallback = None
    OperatorBridge._connector_failed(bridge, 'no safe short connector', 5)
    bridge._start_live_plan.assert_not_called()
    assert 'No plan was sent' in bridge._report.call_args.args[0]


def _lifting_bridge(seconds):
    from rclpy.time import Time
    from tugger_control.cargo_operation import CargoOperation
    bridge = _bridge()
    bridge._motion_armed = True
    bridge.docked_station = 'cargo_station_3'
    bridge._cargo_task = CargoOperation('pickup', 3, 1.0, 0.0)
    bridge._cargo_task.phase = 'WAIT_RAISE'
    bridge._cargo_task.phase_started = 1.0
    bridge._cargo_states = {0: 'detached', 3: 'attached'}
    bridge._lift_target = LIFT_UP_M
    bridge._lift_gap = None
    bridge.lift = Mock()
    bridge._execution_done = Mock()

    def at(second):
        bridge.get_clock.return_value.now.return_value = Time(seconds=second)
        bridge._cargo_seen_at = {0: Time(seconds=second), 3: Time(seconds=second)}
        bridge._lift_seen_at = Time(seconds=second)
        bridge._cargo_pose_seen_at[3] = Time(seconds=second)
        bridge._lift_position = 0.06
    at(seconds)
    return bridge, at


def test_a_momentary_scene_gap_pauses_the_lift_then_resumes():
    # Seen in Gazebo: one stale scene during the raise aborted a good pickup.
    from tugger_control.operator_panel import ClearanceUnavailable
    bridge, at = _lifting_bridge(2.0)
    bridge._cargo_clearance_guard = Mock(side_effect=ClearanceUnavailable('scene stale'))
    OperatorBridge._advance_cargo_operation(bridge)
    bridge.lift.assert_called_once_with(0.06, remember=False)
    at(2.5)
    OperatorBridge._advance_cargo_operation(bridge)
    bridge._execution_done.assert_not_called()
    bridge._cargo_clearance_guard = Mock(return_value=True)
    at(2.7)
    bridge.lift.reset_mock()
    OperatorBridge._advance_cargo_operation(bridge)
    bridge.lift.assert_any_call(LIFT_UP_M)
    assert bridge._lift_gap is None
    bridge._execution_done.assert_not_called()


def test_a_lasting_scene_gap_still_fails_the_lift():
    from tugger_control.operator_panel import ClearanceUnavailable
    bridge, at = _lifting_bridge(2.0)
    bridge._cargo_clearance_guard = Mock(side_effect=ClearanceUnavailable('scene stale'))
    OperatorBridge._advance_cargo_operation(bridge)
    at(3.2)
    OperatorBridge._advance_cargo_operation(bridge)
    assert bridge._execution_done.call_args.args[0] is False
    assert bridge._cargo_task is None or bridge._cargo_task.result == (
        'clearance_unavailable_during_lift')


def test_a_real_collision_during_the_lift_fails_at_once():
    bridge, _at = _lifting_bridge(2.0)
    bridge._cargo_clearance_guard = Mock(return_value=False)
    OperatorBridge._advance_cargo_operation(bridge)
    assert bridge._execution_done.call_args.args[0] is False


def _approach_to(dock_x):
    # Gate 2 m out along +x, dock at dock_x, as at Station 1.
    return [[dock_x + 2.0 - 0.01 * i, -7.0, math.pi] for i in range(201)]


def test_partway_out_of_a_station_is_recognised():
    bridge = _bridge()
    bridge._station_approach = lambda station: (
        _approach_to(-17.5) if station == 'cargo_station_0' else _approach_to(100.0))
    find = lambda pose: OperatorBridge._station_partway(bridge, pose)
    assert find((-17.03, -7.0, 0.0)) == 'cargo_station_0'     # reported: 47 cm out
    assert find((-17.5, -7.0, 0.0)) is None                   # docked, not partway
    assert find((-15.5, -7.0, 0.0)) is None                   # at the gate: out
    assert find((-17.0, -6.5, 0.0)) is None                   # off the approach


def test_loaded_robot_partway_out_backs_out_then_continues():
    # Reported: cancelled loaded departure left the robot under the stand and
    # every later request failed to plan.
    bridge = _bridge()
    bridge._confirmed_cargo.return_value = 0
    bridge._robot_pose.return_value = (-17.03, -7.0, 0.0)
    bridge.docked_station = None
    bridge._station_partway = Mock(return_value='cargo_station_0')
    bridge._prepare_backout = Mock()
    bridge._recover_loaded_dock_station = Mock()
    OperatorBridge.execute_destination(bridge, 'cargo_dropoff_northwest', [], [[0, 0, 0]])
    bridge._prepare_backout.assert_called_once_with(
        'cargo_dropoff_northwest', [], [[0, 0, 0]], station='cargo_station_0')


def test_no_saved_route_from_the_entry_gate_falls_back_to_live_planning():
    # Reported: a shuffled cycle entered Station 4 by its west gate and every
    # destination without a west-gate route was rejected.
    approach = [[19.5, -7.0, 3.14], [17.5, -7.0, 3.14]]       # a gate other than north
    north_only = [('cargo_dropoff_southeast__TO__cargo_station_3',
                   {'source': 'cargo_dropoff_southeast', 'source_entrance': 'north'}, [])]
    bridge = _bridge()
    bridge._live_planner = Mock()
    routes = OperatorBridge._same_gate_routes(
        bridge, 'cargo_dropoff_southeast', approach, north_only, 'cargo_station_3')
    assert routes == []
    assert 'planned live' in bridge._report.call_args.args[0]
    bridge._live_planner = None
    with pytest.raises(ValueError, match='leaves by the'):
        OperatorBridge._same_gate_routes(
            bridge, 'cargo_dropoff_southeast', approach, north_only, 'cargo_station_3')


def test_mission_goes_via_home_when_no_path_leaves_the_entry_gate():
    bridge = _bridge()
    bridge.docked_station = 'cargo_station_3'
    bridge._pass_through = ('HOME',)
    bridge._station_centers = {'HOME': [0, 0], 'cargo_station_3': [17.5, 7.0],
                               'cargo_dropoff_southeast': [17.5, -7.0]}
    bridge._mission_via = None
    bridge._mission_retry = {'destination': 'cargo_dropoff_southeast', 'candidates': [],
                             'targets': [], 'attempts': 0}
    west_exit = [[15.5, 7.0, 0.0], [17.5, 7.0, 0.0]]          # gate west of the dock
    bridge._station_approach = Mock(return_value=west_exit)
    meta = lambda s, go, t, gi: {'source': s, 'target': t, 'source_entrance': go,
                                 'target_entrance': gi, 'length_m': 20.0,
                                 'execution_approved': True}
    paths = {'HOME': [('a', meta('cargo_station_3', 'west', 'HOME', 'east'), [])],
             'cargo_dropoff_southeast': [('b', meta('HOME', 'east', 'cargo_dropoff_southeast',
                                                    'north'), []),
                                         ('c', meta('cargo_station_3', 'south',
                                                    'cargo_dropoff_southeast', 'north'), [])],
             'cargo_station_3': []}
    bridge._candidate_provider = lambda d: (paths[d], [[0, 0, 0]])
    bridge.execute_destination = Mock()
    started = OperatorBridge._start_route_chain(
        bridge, 'cargo_dropoff_southeast', paths['cargo_dropoff_southeast'], [[1, 1, 0]])
    assert started
    bridge.execute_destination.assert_called_once_with('HOME', paths['HOME'], [[0, 0, 0]])
    assert bridge._mission_via['final'] == 'cargo_dropoff_southeast'
    assert bridge._mission_retry['destination'] == 'HOME'
    assert 'going via HOME -> cargo_dropoff_southeast' in bridge._report.call_args.args[0]


def test_a_waypoint_station_has_no_cargo_rules_and_no_docking():
    # A loaded robot passing a pickup station's gate is not going to it.
    bridge = _bridge()
    bridge._mission_via = {'final': 'cargo_dropoff_southeast', 'hop': 'cargo_station_0',
                           'candidates': [], 'targets': []}
    OperatorBridge._check_destination_allowed(bridge, 'cargo_station_0', 3)
    with pytest.raises(ValueError, match='drop the carried cargo off first'):
        bridge._mission_via = None
        OperatorBridge._check_destination_allowed(bridge, 'cargo_station_0', 3)


def test_robot_at_a_station_gate_is_not_partway_inside():
    bridge = _bridge()
    bridge._station_approach = lambda station: (
        _approach_to(-17.5) if station == 'cargo_station_0' else _approach_to(100.0))
    find = lambda pose: OperatorBridge._station_partway(bridge, pose)
    assert find((-15.75, -7.0, 0.0)) is None      # stopped 25 cm short of the gate
    assert find((-17.03, -7.0, 0.0)) == 'cargo_station_0'


def test_attach_in_place_with_no_cargo_at_the_station_never_moves_the_lift():
    bridge = _bridge()
    bridge.docked_station = 'cargo_dropoff_northwest'          # holds no cargo
    bridge._pending_destination = None
    bridge._mission_retry = None
    bridge._motion_armed = True
    bridge._confirmed_cargo.return_value = None
    bridge._start_cargo_operation = Mock()
    bridge.lift = Mock()
    bridge._manual_cargo_action = lambda action, cargo: OperatorBridge._manual_cargo_action(
        bridge, action, cargo)
    OperatorBridge.select_cargo_action(bridge, 'attach')
    bridge._start_cargo_operation.assert_not_called()
    bridge.lift.assert_not_called()
    assert 'Station 2 has no cargo to attach' in bridge._report.call_args.args[0]


def test_attach_in_place_where_cargo_rests_starts_the_pickup():
    bridge = _bridge()
    bridge.docked_station = 'cargo_station_3'                  # holds Station 3 cargo
    bridge._pending_destination = None
    bridge._mission_retry = None
    bridge._motion_armed = True
    bridge._confirmed_cargo.return_value = None
    bridge._start_cargo_operation = Mock()
    bridge._manual_cargo_action = lambda action, cargo: OperatorBridge._manual_cargo_action(
        bridge, action, cargo)
    OperatorBridge.select_cargo_action(bridge, 'attach')
    bridge._start_cargo_operation.assert_called_once_with('cargo_station_3')
    assert bridge._mission_action == 'attach'


def test_selecting_an_action_for_a_planned_mission_is_checked_first():
    bridge = _bridge()
    bridge._pending_destination = 'cargo_dropoff_northwest'
    bridge._mission_retry = {'destination': 'cargo_dropoff_northwest', 'candidates': [],
                             'targets': [], 'attempts': 0}
    bridge._mission_confirm = {'destination': 'cargo_dropoff_northwest',
                               'warnings': ['no cargo action selected'], 'dock_only': True,
                               'confirmed': False}
    bridge._confirmed_cargo.return_value = None
    OperatorBridge.select_cargo_action(bridge, 'detach')       # empty robot
    assert 'nothing to detach' in bridge._report.call_args.args[0]
    bridge._confirmed_cargo.return_value = 0
    OperatorBridge.select_cargo_action(bridge, 'detach')
    assert bridge._mission_action == 'detach' and bridge._mission_confirm is None


def test_cargo_action_during_a_chained_mission_is_for_the_final_station():
    # Reported in validation: DETACH was checked against the HOME waypoint.
    bridge = _bridge()
    bridge._mission_via = {'final': 'cargo_dropoff_southeast', 'hop': 'HOME',
                           'candidates': [], 'targets': []}
    bridge._mission_retry = {'destination': 'HOME', 'candidates': [], 'targets': [],
                             'attempts': 0}
    bridge._pending_destination = 'HOME'
    bridge._confirmed_cargo.return_value = 3
    OperatorBridge.select_cargo_action(bridge, 'detach')
    assert bridge._mission_action == 'detach'
    assert 'DETACH selected for Station 4' in bridge._report.call_args.args[0]


def test_loaded_approach_checks_only_the_open_part_against_the_2d_costmap():
    # Validation 2026-11-07: the flat costmap rejected every loaded approach
    # between real stand legs; under the stand the 3D loaded check decides.
    from rclpy.time import Time
    bridge = _bridge()
    bridge._motion_armed = True
    bridge._mission_action = 'detach'
    bridge._confirmed_cargo.return_value = 3
    bridge._station_centers = {'cargo_dropoff_southeast': [17.5, -7.0]}
    bridge._plan_generation = 4
    approach = [[17.5, -5.0 - 0.1 * i, -math.pi / 2] for i in range(21)]
    bridge._queued_precision_station = 'cargo_dropoff_southeast'
    bridge._queued_precision_poses = approach
    bridge._robot_pose.return_value = (17.5, -5.0, -math.pi / 2)
    bridge.get_clock.return_value.now.return_value = Time(seconds=10)
    bridge._modeled_loaded_path_clearance = Mock(return_value=100)
    bridge._pose = lambda values: OperatorBridge._pose(bridge, values)
    bridge._path_validator = Mock()
    bridge._path_validator.service_is_ready.return_value = True
    bridge._follow = Mock()
    bridge._execution_done = Mock()
    OperatorBridge._start_precision_segment(bridge)
    request = bridge._path_validator.call_async.call_args.args[0]
    checked = [(p.pose.position.x, p.pose.position.y) for p in request.path.poses]
    assert checked and all(math.hypot(x - 17.5, y + 7.0) > 1.5 for x, y in checked)
    bridge._modeled_loaded_path_clearance.assert_called_once()   # full path in 3D
