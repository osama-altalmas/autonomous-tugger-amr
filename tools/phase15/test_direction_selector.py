#!/usr/bin/env python3

import math
import json
import tempfile
import unittest
from pathlib import Path

from direction_selector import (
    choose_direction,
    initial_travel_heading,
)


class DirectionSelectorTests(unittest.TestCase):

    @staticmethod
    def route(degrees):
        a = math.radians(degrees)

        return [
            [0.0, 0.0, a],
            [0.025 * math.cos(a), 0.025 * math.sin(a), a],
            [0.050 * math.cos(a), 0.050 * math.sin(a), a],
        ]

    def check(self, robot_heading, travel_heading, expected):
        result = choose_direction(
            self.route(travel_heading),
            math.radians(robot_heading),
        )

        self.assertEqual(result["leading_end"], expected)

        expected_body_motion = (
            "FORWARD" if expected == "FRONT" else "REVERSE"
        )

        self.assertEqual(
            result["route_progress_direction"],
            "FORWARD",
        )
        self.assertEqual(
            result["body_motion_direction"],
            expected_body_motion,
        )
        self.assertEqual(
            result["body_velocity_sign"],
            1 if expected == "FRONT" else -1,
        )

    def test_east_front(self):
        self.check(0, 0, "FRONT")

    def test_west_rear(self):
        self.check(0, 180, "REAR")

    def test_north_front(self):
        self.check(90, 90, "FRONT")

    def test_south_rear(self):
        self.check(90, -90, "REAR")

    def test_west_facing_east_rear(self):
        self.check(180, 0, "REAR")

    def test_south_facing_north_rear(self):
        self.check(-90, 90, "REAR")

    def test_wrapped_front(self):
        self.check(-179, 179, "FRONT")

    def test_wrapped_rear(self):
        self.check(-1, 179, "REAR")

    def test_reject_perpendicular(self):
        with self.assertRaises(ValueError):
            choose_direction(self.route(90), 0)

    def test_reject_diagonal(self):
        with self.assertRaises(ValueError):
            choose_direction(self.route(45), 0)

    def test_skip_duplicate_start_points(self):
        poses = [
            [0, 0, 0],
            [0, 0, 0],
            [0.001, 0, 0],
            [0.025, 0, 0],
        ]

        self.assertAlmostEqual(
            initial_travel_heading(poses),
            0,
        )

    def test_reject_zero_length_route(self):
        with self.assertRaises(ValueError):
            initial_travel_heading([[0, 0, 0], [0, 0, 0]])

    def test_reject_nonfinite_heading(self):
        with self.assertRaises(ValueError):
            choose_direction(self.route(0), float("nan"))

    def test_standard_saved_route_is_accepted(self):
        from direction_selector import load_segment_one

        route = {
            "version": "AMR_V1_SAVED_ROUTE",
            "source": "A",
            "target": "B",
            "execution_approved": False,
            "poses": self.route(0),
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "route.json"
            path.write_text(json.dumps(route))
            loaded, poses = load_segment_one(path)

        self.assertEqual(loaded["source"], "A")
        self.assertEqual(poses, route["poses"])


if __name__ == "__main__":
    unittest.main()
