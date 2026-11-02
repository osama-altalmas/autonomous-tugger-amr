#!/usr/bin/env python3

"""Offline tests of FRONT/REAR path-tracking geometry.

No ROS nodes.
No Gazebo commands.
No robot movement.
"""

import math
import sys
import unittest
from pathlib import Path

CONTROL = (
    Path(__file__).resolve().parents[2]
    / "src/tugger_control/tugger_control"
)

sys.path.insert(0, str(CONTROL))

from motion_math import steering_for_twist, wheel_speeds


WHEELBASE = 0.58
TRACK_WIDTH = 0.44
WHEEL_RADIUS = 0.055
MAX_STEERING = math.pi / 4


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def reverse_command(robot_yaw, target_x, target_y, speed=0.08):
    """Test the Phase 15 rear-leading pursuit convention."""

    if speed <= 0:
        raise ValueError("Speed magnitude must be positive")

    target_heading = math.atan2(target_y, target_x)

    alpha = wrap(target_heading - robot_yaw - math.pi)

    lookahead = max(
        0.65,
        math.hypot(target_x, target_y),
    )

    curvature = 2.0 * math.sin(alpha) / lookahead

    signed_velocity = -speed

    angular_velocity = abs(signed_velocity) * curvature

    steering = steering_for_twist(
        signed_velocity,
        angular_velocity,
        WHEELBASE,
        MAX_STEERING,
    )

    rear_left, rear_right, actual_omega = wheel_speeds(
        signed_velocity,
        steering,
        WHEELBASE,
        TRACK_WIDTH,
        WHEEL_RADIUS,
    )

    return {
        "v": signed_velocity,
        "curvature": curvature,
        "omega": angular_velocity,
        "steering": steering,
        "actual_omega": actual_omega,
        "rear_left_wheel_speed": rear_left,
        "rear_right_wheel_speed": rear_right,
    }


class ReverseKinematicsTests(unittest.TestCase):

    def test_reverse_straight(self):
        result = reverse_command(0, -1, 0)

        self.assertLess(result["v"], 0)
        self.assertAlmostEqual(result["omega"], 0, places=10)
        self.assertAlmostEqual(result["steering"], 0, places=10)

    def test_reverse_left_travel_turn(self):
        # For westbound travel, south is to the LEFT.
        result = reverse_command(0, -1, -0.2)

        self.assertGreater(result["curvature"], 0)
        self.assertGreater(result["omega"], 0)
        self.assertLess(result["steering"], 0)

    def test_reverse_right_travel_turn(self):
        # For westbound travel, north is to the RIGHT.
        result = reverse_command(0, -1, 0.2)

        self.assertLess(result["curvature"], 0)
        self.assertLess(result["omega"], 0)
        self.assertGreater(result["steering"], 0)

    def test_reverse_wheels_turn_backward(self):
        result = reverse_command(0, -1, 0)

        self.assertLess(result["rear_left_wheel_speed"], 0)
        self.assertLess(result["rear_right_wheel_speed"], 0)

    def test_controller_omega_matches_requested_omega(self):
        for lateral in (-0.2, -0.1, 0, 0.1, 0.2):
            with self.subTest(lateral=lateral):
                result = reverse_command(0, -1, lateral)

                self.assertAlmostEqual(
                    result["actual_omega"],
                    result["omega"],
                    places=8,
                )

    def test_reverse_velocity_is_negative(self):
        for lateral in (-0.3, 0, 0.3):
            with self.subTest(lateral=lateral):
                result = reverse_command(0, -1, lateral)

                self.assertLess(result["v"], 0)

    def test_reverse_heading_wrap(self):
        yaw = math.radians(179)

        result = reverse_command(
            yaw,
            1.0,
            -0.01,
        )

        self.assertTrue(math.isfinite(result["omega"]))
        self.assertTrue(math.isfinite(result["steering"]))

    def test_forward_kinematics_unchanged(self):
        v = 0.08
        curvature = 0.2
        omega = v * curvature

        steering = steering_for_twist(
            v,
            omega,
            WHEELBASE,
            MAX_STEERING,
        )

        self.assertGreater(steering, 0)

        _, _, actual_omega = wheel_speeds(
            v,
            steering,
            WHEELBASE,
            TRACK_WIDTH,
            WHEEL_RADIUS,
        )

        self.assertAlmostEqual(
            actual_omega,
            omega,
            places=8,
        )

    def test_reject_zero_speed_magnitude(self):
        with self.assertRaises(ValueError):
            reverse_command(0, -1, 0, speed=0)


if __name__ == "__main__":
    unittest.main()
