"""Turn in place: geometry, odometry, clearance, gate choice and controller."""

import math

from geometry_msgs.msg import TwistStamped
import rclpy

from tugger_control.clearance_2_5d import Box
from tugger_control.four_ws_controller import FourWSController
from tugger_control.rotation_clearance import rotation_clear, stopping_turn
from tugger_control.route_network import plan_route_chain_from
from tugger_control.turn_in_place import (
    approach_axis, drive_steering, pivot_odometry, PIVOT_STEERING, PIVOT_STEERING_RAD,
    pivot_wheel_speeds, rotation_time_s, rotation_to_axis, STEERING_LIMIT_RAD)


def test_pivot_angle_fits_the_54_degree_steering_limit():
    assert math.isclose(math.degrees(PIVOT_STEERING_RAD), 52.8153, abs_tol=1e-3)
    assert PIVOT_STEERING_RAD < STEERING_LIMIT_RAD <= math.radians(54.0) + 1e-12
    # Car-like driving keeps its 45-degree limit, so no existing motion changes.
    assert all(abs(a) <= math.pi / 4 for a in drive_steering(math.pi / 4))


def test_pivot_wheels_turn_the_robot_without_moving_its_centre():
    speeds = pivot_wheel_speeds(0.4)
    vx, vy, w = pivot_odometry(PIVOT_STEERING, speeds)
    assert abs(vx) < 1e-9 and abs(vy) < 1e-9 and math.isclose(w, 0.4)
    # The rear wheels alone (the driven pair) measure the same motion.
    vx, vy, w = pivot_odometry(PIVOT_STEERING[2:], speeds[2:],
                               wheel_positions=((-0.29, 0.22), (-0.29, -0.22)))
    assert abs(vx) < 1e-9 and abs(vy) < 1e-9 and math.isclose(w, 0.4)


def test_odometry_fit_agrees_with_car_like_driving():
    # Straight: all wheels at 0 rad rolling at the same speed.
    vx, vy, w = pivot_odometry((0, 0, 0, 0), (10, 10, 10, 10))
    assert math.isclose(vx, 0.55) and abs(vy) < 1e-9 and abs(w) < 1e-9


def test_rotation_takes_the_shorter_way_to_the_gate_axis():
    assert math.isclose(rotation_to_axis(0.0, math.pi / 2), math.pi / 2)
    # Facing west (pi) and the gate axis is east-west: no turn, drive the other end first.
    assert abs(rotation_to_axis(math.pi, 0.0)) < 1e-9
    assert math.isclose(rotation_to_axis(math.radians(170), math.radians(-90)),
                        math.radians(-80))
    approach = [(-15.5, 7.0, 0), (-16.5, 7.0, 0), (-17.5, 7.0, 0)]
    assert math.isclose(approach_axis(approach), math.pi)


def test_rotation_time_counts_steering_turning_and_settling():
    quarter = rotation_time_s(math.pi / 2, loaded=False)
    assert rotation_time_s(0.0, loaded=False) == 0.0
    assert 5.0 < quarter < 10.0
    assert rotation_time_s(math.pi / 2, loaded=True) > quarter


def _legs(cx, cy, spacing=0.56):
    """Four 8 cm station legs, 28 cm tall, at +-spacing from the centre."""
    return [('legs', Box(f'leg_{i}', cx + dx, cy + dy, 0.0, 0.04, 0.04, 0.0, 0.28))
            for i, (dx, dy) in enumerate([(-spacing, -spacing), (-spacing, spacing),
                                          (spacing, -spacing), (spacing, spacing)])]


def test_unloaded_turn_clears_the_station_legs():
    clear, hit = rotation_clear((0.0, 0.0, 0.0), 0.002, _legs(0, 0), 2 * math.pi, margin=0.05)
    assert clear and hit is None


def test_turn_is_refused_when_a_leg_is_inside_the_turning_circle():
    clear, hit = rotation_clear((0.0, 0.0, 0.0), 0.002, _legs(0, 0, spacing=0.40),
                                2 * math.pi, margin=0.05)
    assert not clear and hit[1][1].startswith('leg_')


def test_loaded_turn_passes_above_the_legs_but_not_past_a_wall():
    from tugger_control.lift_geometry import LIFT_UP_M
    cargo = (0.0, 0.0, LIFT_UP_M - 0.0425, 0.0)
    scene = _legs(0, 0) + [('cargo_station_0_stand',
                            Box('stand', 0, 0, 0, 0.65, 0.65, cargo[2] + 0.28, cargo[2] + 0.31))]
    clear, _ = rotation_clear((0.0, 0.0, 0.0), LIFT_UP_M, scene, 2 * math.pi,
                              cargo_id=0, cargo_pose=cargo, margin=0.05)
    assert clear
    # A wall 0.85 m away: the unrotated cargo clears it, its corners do not.
    wall = [('east_wall', Box('wall', 0.85 + 0.1, 0, 0, 0.1, 3.0, 0.0, 2.0))]
    clear, hit = rotation_clear((0.0, 0.0, 0.0), LIFT_UP_M, scene + wall, 2 * math.pi,
                                cargo_id=0, cargo_pose=cargo, margin=0.05)
    assert not clear and hit[1] == ('cargo', 'wall')


def test_stopping_turn_grows_with_yaw_rate():
    assert stopping_turn(0.5, 0.5) > stopping_turn(0.3, 0.5) > stopping_turn(0.0, 0.5)


def _edge(source, gate_out, target, gate_in, length):
    return {'key': f'{source}-{gate_out}-{target}', 'source': source, 'target': target,
            'source_entrance': gate_out, 'target_entrance': gate_in,
            'length_m': length, 'approved': True}


def test_another_gate_is_chosen_only_when_the_turn_pays_off():
    edges = [_edge('S3', 'west', 'S4', 'north', 38.7), _edge('S3', 'south', 'S4', 'north', 14.0)]
    # Entered by the west gate; a 7 s turn makes the 14 m south route faster.
    stops, gate, cost = plan_route_chain_from(edges, 'S3', {'west': 0.0, 'south': 7.0},
                                              'S4', ['HOME'])
    assert (stops, gate) == (['S4'], 'south') and math.isclose(cost, 21.0)
    # A turn slower than the saving keeps the entry gate.
    _, gate, _ = plan_route_chain_from(edges, 'S3', {'west': 0.0, 'south': 30.0},
                                       'S4', ['HOME'])
    assert gate == 'west'


def _pivot(omega):
    message = TwistStamped()
    message.header.frame_id = 'pivot'
    message.twist.angular.z = omega
    return message


def _drive(v, omega):
    message = TwistStamped()
    message.header.frame_id = 'base_footprint'
    message.twist.linear.x = v
    message.twist.angular.z = omega
    return message


def test_controller_steers_to_pivot_turns_and_returns_before_driving():
    rclpy.init()
    try:
        node = FourWSController()
        node.cmd_vel_callback(_pivot(0.4))
        for _ in range(200):
            node._pivot_tick(0.05)
        assert list(node.wheel_angles) == list(PIVOT_STEERING)
        assert math.isclose(node.yaw_rate, 0.4) and node.v == 0.0
        # Back to driving: the yaw rate stops and the wheels straighten before travel.
        node.cmd_vel_callback(_drive(0.3, 0.0))
        node._pivot_tick(0.05)
        assert node.v == 0.0 and node.yaw_rate < 0.4
        for _ in range(200):
            if node._in_drive_pattern():
                break
            node._pivot_tick(0.05)
        assert node._in_drive_pattern() and node.yaw_rate == 0.0 and node.v == 0.0
        node.destroy_node()
    finally:
        rclpy.shutdown()


def test_a_pivot_command_never_moves_a_driving_robot_sideways():
    rclpy.init()
    try:
        node = FourWSController()
        node.v = 0.5
        node.cmd_vel_callback(_pivot(0.4))
        node._pivot_tick(0.05)
        # Stops with its car-like steering first; no pivot steering while moving.
        assert 0.0 < node.v < 0.5
        assert node._in_drive_pattern() and node.yaw_rate == 0.0
        node.destroy_node()
    finally:
        rclpy.shutdown()


def test_unloaded_robot_turns_under_a_loaded_stand_it_fits_beneath():
    """The lift top is 4.25 cm below the stand: clear vertically, as when driving under it."""
    stand = [('cargo_station_0_stand', Box('stand', 0, 0, 0, 0.65, 0.65, 0.28, 0.31))]
    clear, _ = rotation_clear((0.0, 0.0, 0.0), 0.002, _legs(0, 0) + stand, 2 * math.pi,
                              margin=0.05)
    assert clear
