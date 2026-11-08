"""Crab steering and the camera-stage / guard fixes found in the campaign."""

import math

from geometry_msgs.msg import TwistStamped
import pytest
import rclpy

from tugger_control.crab_motion import (CRAB_FRAME, CRAB_MAX_ANGLE_RAD, crab_command,
                                        crab_propagate, crab_steering)
from tugger_control.four_ws_controller import FourWSController
from tugger_control.loaded_motion_guard import MEASURED_SPEED_LIMIT, loaded_envelope_clear
from tugger_control.safety_fields import CLEAR, PROTECTIVE_STOP, WARNING, classify_crab
from tugger_control.turn_in_place import pivot_odometry
from tugger_control.visual_maneuver_executor import VisualManeuverExecutor
from tugger_control.visual_maneuver_planner import advance, replan_visual_maneuver
from tugger_control.visual_marker_guard import FixedMarkerGuard


def _crab(vx, vy):
    message = TwistStamped()
    message.header.frame_id = CRAB_FRAME
    message.twist.linear.x = vx
    message.twist.linear.y = vy
    return message


def _drive(v, omega=0.0):
    message = TwistStamped()
    message.header.frame_id = 'base_footprint'
    message.twist.linear.x = v
    message.twist.angular.z = omega
    return message


# --- crab geometry -------------------------------------------------------

def test_crab_command_folds_backward_diagonals_onto_the_wheel_limit():
    speed, beta = crab_command(0.01, 0.01)
    assert speed > 0 and math.isclose(beta, math.pi / 4)
    speed, beta = crab_command(-0.01, 0.01)
    # Backwards and to the left: wheels point forward-right, roll backwards.
    assert speed < 0 and math.isclose(beta, -math.pi / 4)
    assert math.isclose(crab_propagate((0, 0, 0), speed / abs(speed) * 0.01414, beta)[1],
                        0.01, abs_tol=1e-4)
    with pytest.raises(ValueError):
        crab_command(0.0, 0.01)   # straight sideways needs 90 degree wheels


def test_wheel_odometry_measures_a_crab_move():
    beta = math.radians(40)
    wheel = 0.01 / 0.055
    vx, vy, w = pivot_odometry(crab_steering(beta), (wheel,) * 4)
    assert math.isclose(vx, 0.01 * math.cos(beta), abs_tol=1e-9)
    assert math.isclose(vy, 0.01 * math.sin(beta), abs_tol=1e-9)
    assert abs(w) < 1e-9


# --- controller -----------------------------------------------------------

def _run(node, ticks):
    for _ in range(ticks):
        node._crab_tick(0.05)


def test_controller_stops_steers_settles_then_crabs_without_turning():
    rclpy.init()
    try:
        node = FourWSController()
        node.v = 0.05
        node.cmd_vel_callback(_crab(0.01, 0.008))
        node._crab_tick(0.05)
        # Car-like motion stops first; the wheels do not move meanwhile.
        assert node.crab_v == 0.0 and node._in_drive_pattern()
        _run(node, 200)
        beta = math.atan2(0.008, 0.01)
        assert list(node.wheel_angles) == list(crab_steering(beta))
        assert node.v == 0.0 and node.yaw_rate == 0.0
        assert math.isclose(node.crab_v, math.hypot(0.01, 0.008))
        node.destroy_node()
    finally:
        rclpy.shutdown()


def test_a_stop_holds_the_crab_wheels_and_driving_steers_back_first():
    rclpy.init()
    try:
        node = FourWSController()
        node.cmd_vel_callback(_crab(0.01, 0.008))
        _run(node, 200)
        angles = list(node.wheel_angles)
        node.cmd_vel_callback(_drive(0.0))
        assert node.want_crab
        _run(node, 100)
        assert node.crab_v == 0.0 and list(node.wheel_angles) == angles
        # A moving car-like command: stop, steer back, then drive.
        node.cmd_vel_callback(_drive(0.05))
        assert not node.want_crab
        node.tick_count = 0
        for _ in range(200):
            if abs(node.crab_v) > 1e-6:
                node.crab_v = 0.0
            if node._in_drive_pattern():
                break
            node._pivot_tick(0.05)
        assert node._in_drive_pattern() and node.v == 0.0
        node.destroy_node()
    finally:
        rclpy.shutdown()


def test_wheels_never_change_angle_while_crabbing():
    rclpy.init()
    try:
        node = FourWSController()
        node.cmd_vel_callback(_crab(0.01, 0.008))
        _run(node, 200)
        node.cmd_vel_callback(_crab(0.01, -0.008))
        before = list(node.wheel_angles)
        node._crab_tick(0.05)
        # Still rolling: it slows first, wheels unchanged.
        assert list(node.wheel_angles) == before and node.crab_v < math.hypot(0.01, 0.008)
        _run(node, 200)
        assert list(node.wheel_angles) == list(crab_steering(math.atan2(-0.008, 0.01)))
        node.destroy_node()
    finally:
        rclpy.shutdown()


def test_controller_refuses_a_crab_direction_beyond_the_wheel_limit():
    rclpy.init()
    try:
        node = FourWSController()
        node.cmd_vel_callback(_crab(0.001, 0.01))
        assert node.timed_out and node.crab_v == 0.0
        node.destroy_node()
    finally:
        rclpy.shutdown()


# --- planner and executor -------------------------------------------------

@pytest.mark.parametrize('target', [(0.0, 0.025, 0.0), (0.013, 0.022, 0.0),
                                    (-0.013, 0.03, 0.01), (0.004, 0.038, 0.0),
                                    (0.02, 0.05, -0.01)])
def test_planner_removes_centimetres_of_lateral_error(target):
    """These camera-stage errors had no plan before crab steering."""
    plan = replan_visual_maneuver((0, 0, 0), target, lambda pose: True,
                                  max_depth=8, beam_width=400, time_budget=3.0,
                                  position_tolerance=0.010,
                                  heading_tolerance=math.radians(1.0),
                                  require_full_visibility=False)
    assert plan is not None and len(plan) <= 4
    pose = (0.0, 0.0, 0.0)
    for primitive in plan:
        pose = advance(pose, primitive)
    # Lands well inside the 10 mm tolerance, not on its edge.
    assert math.hypot(pose[0] - target[0], pose[1] - target[1]) <= 0.005
    for primitive in plan:
        if len(primitive) > 2:
            assert abs(primitive[2]) <= CRAB_MAX_ANGLE_RAD + 1e-9


def test_planner_respects_clearance_for_crab_moves():
    def clearance(pose):
        return pose[1] <= 0.01   # a wall 1 cm to the left
    plan = replan_visual_maneuver((0, 0, 0), (0.0, 0.03, 0.0), clearance,
                                  max_depth=8, beam_width=400, time_budget=1.0,
                                  position_tolerance=0.010,
                                  heading_tolerance=math.radians(1.0),
                                  require_full_visibility=False)
    assert plan is None


def test_executor_tracks_a_crab_primitive():
    beta = math.radians(50)
    executor = VisualManeuverExecutor([(0.03, 0.0, beta)], speed=0.005, crab_speed=0.01)
    t = 0.0
    result = None
    for _ in range(10):   # settle before the move
        result = executor.update((0, 0, 0), 0.0, t, t, t, identity_ok=True,
                                 clearance_ok=True, visible=True)
        t += 0.05
    assert result['state'] == 'MOVING' and result['frame'] == CRAB_FRAME
    assert math.isclose(result['linear_x'], 0.01 * math.cos(beta))
    assert math.isclose(result['linear_y'], 0.01 * math.sin(beta))
    assert result['angular_z'] == 0.0
    # Halfway along the crab direction: still moving, no divergence.
    half = crab_propagate((0, 0, 0), 0.015, beta)
    result = executor.update(half, 0.01, t, t, t, identity_ok=True,
                             clearance_ok=True, visible=True)
    assert result['state'] == 'MOVING'
    # Sliding off the crab line is a divergence stop.
    t += 0.05
    result = executor.update((half[0] + 0.02, half[1] - 0.02, 0.0), 0.01, t, t, t,
                             identity_ok=True, clearance_ok=True, visible=True)
    assert result['state'] == 'STOP'


def test_executor_rejects_crab_beyond_the_wheel_limit():
    with pytest.raises(ValueError):
        VisualManeuverExecutor([(0.03, 0.0, math.radians(60))])
    with pytest.raises(ValueError):
        VisualManeuverExecutor([(0.03, 1.0, math.radians(30))])


# --- safety field ---------------------------------------------------------

def test_crab_field_surrounds_the_footprint():
    assert classify_crab([(0.0, 0.0)], 0.01)[0] == CLEAR   # own body ignored
    assert classify_crab([(0.0, 0.40)], 0.01)[0] == PROTECTIVE_STOP
    assert classify_crab([(0.0, 0.62)], 0.01)[0] == WARNING
    assert classify_crab([(0.0, 1.20)], 0.01)[0] == CLEAR
    # Station legs about 0.95 m from the centre stay clear.
    assert classify_crab([(0.7, 0.66)], 0.01)[0] != PROTECTIVE_STOP


# --- marker guard: drift is not a fault, a jump is -----------------------

def test_marker_guard_accepts_slow_odometry_drift_but_holds_on_a_jump():
    guard = FixedMarkerGuard((0.0, 0.0, 0.0), (0.02, 0.0))
    # Odometry drifts 2 mm per correction against the camera, 10 times.
    for step in range(1, 11):
        state, _ = guard.check((0.0, -0.002 * step, 0.0), (0.02, 0.0), 0.0)
        assert state == 'OK'
    # A 3 cm jump (misdetection) is still caught.
    state, _ = guard.check((0.0, -0.02, 0.0), (0.05, 0.0), 0.0)
    assert state == 'HOLD'
    # The capture boundary stays where the marker was first seen.
    state, _ = guard.check((0.2, 0.0, 0.0), (0.0, 0.0), 0.0)
    assert state == 'ABORT'


# --- loaded guard: the speed ceiling dropping is not a fault ---------------

def test_loaded_guard_accepts_a_robot_still_above_a_lower_new_ceiling():
    from tugger_control.lift_geometry import LIFT_UP_M
    pose = (0.0, 0.0, 0.0)
    cargo = (0.0, 0.0, LIFT_UP_M, 0.0)
    # Departure ceiling 0.3 m/s while the robot still rolls at 0.31 m/s.
    clear, _ = loaded_envelope_clear(pose, cargo, LIFT_UP_M, 0, [], 0.31, 0.3,
                                     max_speed=0.3)
    assert clear
    with pytest.raises(ValueError):
        loaded_envelope_clear(pose, cargo, LIFT_UP_M, 0, [], MEASURED_SPEED_LIMIT + 0.1,
                              0.3, max_speed=0.3)


# --- camera-docking relay -------------------------------------------------

def test_docking_relay_forwards_crab_commands_intact():
    from unittest.mock import Mock
    from std_msgs.msg import String
    from tugger_control.docking_precision_controller import DockingPrecisionController
    rclpy.init()
    try:
        node = DockingPrecisionController()
        node._output = Mock()
        node._on_owner(String(data='DOCK_CAMERA'))
        node._on_command(_crab(0.006, 0.008))
        node._tick()
        sent = node._output.publish.call_args.args[0]
        assert sent.header.frame_id == CRAB_FRAME
        assert sent.twist.linear.x == pytest.approx(0.006)
        assert sent.twist.linear.y == pytest.approx(0.008)
        # A car-like command still goes out car-like, without sideways speed.
        node._on_command(_drive(0.005, 0.01))
        node._tick()
        sent = node._output.publish.call_args.args[0]
        assert sent.header.frame_id == 'base_footprint' and sent.twist.linear.y == 0.0
        node.destroy_node()
    finally:
        rclpy.shutdown()


# --- camera stage: no early-stop trap between 10 and 12 mm ---------------

def test_a_correction_is_not_cut_short_just_outside_the_target():
    import math as m
    from tugger_control.operator_panel import early_stop_inside_target
    one_degree = m.radians(1.0)
    # 11 mm: outside the 10 mm limit. The correction must run to the end.
    assert not early_stop_inside_target(0.011, 0.011, 0.0)
    assert not early_stop_inside_target(0.0095, 0.0095, 0.0)
    # Well inside and aligned: stop and settle for acceptance.
    assert early_stop_inside_target(0.004, 0.005, 0.5 * one_degree)
    # A heading correction passing through the centre is not cut short.
    assert not early_stop_inside_target(0.002, 0.002, 3 * one_degree)


# --- lift clearance: one timing source for the cargo height ---------------

def test_lift_sweep_needs_a_resting_height_consistent_with_the_lift():
    """Drop-off at the north-west stand: cargo lowered onto its legs.

    With the cargo height taken from the same lift value the sweep starts
    from, lowering is clear. A cargo estimate made from a lift reading 8 mm
    lower (a few ms apart while the lift moves) looked like the cargo sinking
    into a leg; the panel now latches the resting height with the lift still.
    """
    import math as m
    from pathlib import Path
    from tugger_control.lift_geometry import LIFT_DOWN_M, LIFT_UP_M, cargo_lift_offset
    from tugger_control.warehouse_clearance import lift_sweep_clear, static_world_boxes
    world = Path(__file__).resolve().parents[2] / 'tugger_gazebo/worlds/tugger_warehouse.sdf'
    obstacles = [box for box in static_world_boxes(world)
                 if not box.name.startswith('cargo_station_')]
    robot = (-17.5, 7.0, m.pi / 2)
    other = (17.5, 7.0, 0.0, 0.0)          # the other cargo, far away
    rest = 0.0                              # cargo bottom on the 0.28 m leg tops
    lift = LIFT_UP_M
    consistent = (-17.5, 7.0, rest + cargo_lift_offset(lift), m.pi / 2)
    clear, hit = lift_sweep_clear(robot, consistent, other, obstacles, lift, LIFT_DOWN_M)
    assert clear, hit
    mixed = (-17.5, 7.0, rest + cargo_lift_offset(lift - 0.008), m.pi / 2)
    clear, hit = lift_sweep_clear(robot, mixed, other, obstacles, lift, LIFT_DOWN_M)
    assert not clear and 'legs' in hit[1]
