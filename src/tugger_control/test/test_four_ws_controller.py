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
        node.cmd_vel_callback(_twist(-0.005, 0.015))      # curvature -3 arc
        arc = node.target_delta
        assert abs(abs(arc) - math.atan(0.58 * 3 / 2)) < 1e-6
        node.cmd_vel_callback(_twist(0.0, 0.0))          # the gate pauses it
        assert node.target_v == 0.0 and node.target_delta == arc
        node.cmd_vel_callback(_twist(0.2, 0.0))          # a straight command
        assert node.target_delta == 0.0
        node.destroy_node()
    finally:
        rclpy.shutdown()
