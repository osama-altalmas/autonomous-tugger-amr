"""4WS controller command handling (node constructed, nothing published to a robot)."""

import math

import rclpy
from geometry_msgs.msg import TwistStamped

from tugger_control.four_ws_controller import FourWSController


def _twist(v, omega):
    message = TwistStamped()
    message.twist.linear.x = v
    message.twist.angular.z = omega
    return message


def test_a_stop_holds_the_wheel_angle_of_the_arc():
    rclpy.init()
    try:
        node = FourWSController()
        node.cmd_vel_callback(_twist(-0.005, 0.0075))     # curvature -1.5 arc
        arc = node.target_delta
        assert abs(abs(arc) - math.atan(0.58 * 1.5 / 2)) < 1e-6
        node.cmd_vel_callback(_twist(0.0, 0.0))          # the gate pauses it
        assert node.target_v == 0.0 and node.target_delta == arc
        node.cmd_vel_callback(_twist(0.2, 0.0))          # a straight command
        assert node.target_delta == 0.0
        node.destroy_node()
    finally:
        rclpy.shutdown()


def test_full_lock_is_the_ackermann_limit_and_each_wheel_rolls_its_own_circle():
    from tugger_control.steering_geometry import (
        MAX_VIRTUAL_STEERING_RAD, MAX_WHEEL_STEERING_RAD, ackermann_steering,
        curvature_from_virtual)
    rclpy.init()
    try:
        node = FourWSController()
        node.cmd_vel_callback(_twist(0.1, 0.3))          # curvature 3: beyond the limit
        assert abs(node.target_delta - MAX_VIRTUAL_STEERING_RAD) < 1e-9
        node.delta = node.target_delta
        node.wheel_angles = list(ackermann_steering(curvature_from_virtual(node.delta)))
        assert max(abs(a) for a in node.wheel_angles) <= MAX_WHEEL_STEERING_RAD + 1e-9
        # Inner (left) wheels steer more and roll slower than the outer ones.
        fl, fr, rl, rr = node.wheel_angles
        assert fl > fr > 0.0
        node.destroy_node()
    finally:
        rclpy.shutdown()


def test_steering_steps_respect_the_rate_of_the_fastest_wheel():
    from tugger_control.turn_in_place import drive_steering
    rclpy.init()
    try:
        node = FourWSController()
        node.delta = 0.2
        dt = 0.05
        proposed = node._wheel_rate_limited(0.5, dt)
        moved = max(abs(a - b) for a, b in zip(drive_steering(0.2), drive_steering(proposed)))
        assert moved <= node.max_steering_rate * dt + 1e-9
        assert 0.2 < proposed < 0.5
        node.destroy_node()
    finally:
        rclpy.shutdown()
