"""HOME camera alignment must be previewed before any motion is enabled."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

pytest.importorskip('rclpy')
pytest.importorskip('python_qt_binding')
from tugger_control.operator_panel import OperatorBridge


def _bridge():
    return SimpleNamespace(
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
    )


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


def test_run_rechecks_cargo_pairing_after_preview():
    bridge = _bridge()
    bridge._pending_alignment_station = 'cargo_dropoff_northwest'
    bridge._pending_destination = 'cargo_dropoff_northwest'
    bridge._pending_execution_approved = True
    bridge._pending_pose = (0.025, 0.025, 0.0)
    bridge._confirmed_cargo.return_value = 3
    OperatorBridge.run_selected_mission(bridge, True)
    bridge._start_qr_alignment.assert_not_called()
    assert any('does not match the confirmed cargo' in call.args[0]
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


def test_wrong_dropoff_is_rejected_before_planning():
    bridge = _bridge()
    bridge._confirmed_cargo.return_value = 0
    bridge._approach_provider = Mock()
    OperatorBridge.execute_destination(bridge, 'cargo_dropoff_southeast', [], [[0, 0, 0]])
    bridge._approach_provider.assert_not_called()
    assert any('does not match the confirmed cargo' in call.args[0]
               for call in bridge._report.call_args_list)


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
    bridge.cargo.assert_called_once_with(0, True)
    bridge._execution_done.assert_not_called()
    clock.now.return_value = Time(seconds=12)
    bridge._cargo_seen_at[0] = Time(seconds=12)
    bridge._cargo_states[0] = 'attached'
    OperatorBridge._advance_cargo_operation(bridge)
    bridge.lift.assert_called_once_with(0.065)
    bridge._execution_done.assert_not_called()
    clock.now.return_value = Time(seconds=13)
    bridge._cargo_seen_at[0] = Time(seconds=13)
    bridge._lift_seen_at = Time(seconds=13)
    bridge._lift_position = 0.065
    bridge._cargo_poses[0] = (0, 0, 0.065, 0)
    bridge._cargo_pose_seen_at[0] = Time(seconds=13)
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
    clock = bridge.get_clock.return_value
    clock.now.return_value = Time(seconds=10)
    bridge._motion_armed = True
    bridge._confirmed_cargo.return_value = 0
    bridge._expected_qr_is_fresh = Mock(return_value=True)
    bridge._cargo_clearance_guard = Mock(return_value=True)
    bridge._lift_seen_at = Time(seconds=10)
    bridge._lift_position = 0.065
    bridge._cargo_states = {0: 'attached', 3: 'detached'}
    bridge._cargo_seen_at = {0: Time(seconds=10), 3: Time(seconds=10)}
    bridge._cargo_poses[0] = (0, 0, 0.065, 0)
    bridge._cargo_pose_seen_at[0] = Time(seconds=10)
    bridge.cargo = Mock()
    bridge.lift = Mock()
    bridge._execution_done = Mock()
    OperatorBridge._start_cargo_operation(bridge, 'cargo_dropoff_northwest')
    bridge.lift.assert_called_once_with(0.0)
    bridge.cargo.assert_not_called()
    clock.now.return_value = Time(seconds=11)
    bridge._cargo_seen_at[0] = Time(seconds=11)
    bridge._lift_seen_at = Time(seconds=11)
    bridge._lift_position = 0.0
    bridge._cargo_poses[0] = (0, 0, 0, 0)
    bridge._cargo_pose_seen_at[0] = Time(seconds=11)
    OperatorBridge._advance_cargo_operation(bridge)
    bridge.cargo.assert_not_called()
    clock.now.return_value = Time(seconds=12)
    bridge._cargo_seen_at[0] = Time(seconds=12)
    bridge._lift_seen_at = Time(seconds=12)
    bridge._cargo_pose_seen_at[0] = Time(seconds=12)
    OperatorBridge._advance_cargo_operation(bridge)
    bridge.cargo.assert_called_once_with(0, False)
    bridge._execution_done.assert_not_called()
    clock.now.return_value = Time(seconds=13)
    bridge._cargo_seen_at[0] = Time(seconds=13)
    bridge._cargo_states[0] = 'detached'
    bridge._cargo_pose_seen_at[0] = Time(seconds=13)
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
