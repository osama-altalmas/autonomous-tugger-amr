#!/usr/bin/env python3

import math
import unittest

from route_adapter import (
    adapt_poses,
    validate_motion_headings,
    build_preview,
    wrap,
)


class RouteAdapterTests(unittest.TestCase):

    def test_forward_keeps_heading(self):
        poses = [[0, 0, 0], [0.1, 0, 0]]
        result = adapt_poses(poses, "FORWARD")
        self.assertAlmostEqual(result[0][2], 0)

    def test_reverse_rotates_body_heading(self):
        poses = [[0, 0, math.pi], [-0.1, 0, math.pi]]
        result = adapt_poses(poses, "REVERSE")
        self.assertAlmostEqual(result[0][2], 0, places=6)

    def test_reverse_preserves_xy(self):
        poses = [[0, 0, math.pi], [-0.1, 0, math.pi]]
        result = adapt_poses(poses, "REVERSE")
        self.assertEqual(
            [(p[0], p[1]) for p in poses],
            [(p[0], p[1]) for p in result],
        )

    def test_reverse_body_heading_matches_westward_travel(self):
        poses = [[0, 0, 0], [-0.1, 0, 0]]
        self.assertLess(
            validate_motion_headings(poses, "REVERSE"), 0.01
        )

    def test_reject_inconsistent_heading(self):
        poses = [[0, 0, 0], [0, 0.1, 0]]
        with self.assertRaises(ValueError):
            validate_motion_headings(poses, "FORWARD")

    def test_reject_invalid_direction(self):
        with self.assertRaises(ValueError):
            adapt_poses([[0, 0, 0], [1, 0, 0]], "SIDEWAYS")

    def test_rear_preview(self):
        poses = [[0, 0, math.pi], [-0.1, 0, math.pi]]
        result = build_preview(poses, 0)
        self.assertEqual(
            result["selection"]["motion_direction"], "REVERSE"
        )
        self.assertAlmostEqual(result["adapted_poses"][0][2], 0)

    def test_existing_reverse_route_keeps_body_headings(self):
        # Stored reverse routes already contain body headings.  A robot facing
        # east while progressing west is rear-leading without another pi turn.
        poses = [[0, 0, 0], [-0.1, 0, 0]]
        result = build_preview(poses, 0, "REVERSE")
        self.assertEqual(result["selection"]["leading_end"], "REAR")
        self.assertEqual(result["body_heading_rotation_deg"], 0)
        self.assertAlmostEqual(result["adapted_poses"][0][2], 0)

    def test_switching_leading_end_rotates_body_headings_only(self):
        poses = [[0, 0, 0], [0.1, 0, 0]]
        result = build_preview(poses, math.pi, "FORWARD")
        self.assertEqual(result["selection"]["leading_end"], "REAR")
        self.assertAlmostEqual(result["body_heading_rotation_deg"], 180)
        self.assertEqual(
            [p[:2] for p in result["adapted_poses"]],
            [p[:2] for p in poses],
        )

    def test_front_preview(self):
        poses = [[0, 0, 0], [0.1, 0, 0]]
        result = build_preview(poses, 0)
        self.assertEqual(
            result["selection"]["motion_direction"], "FORWARD"
        )

    def test_reject_incompatible_start(self):
        poses = [[0, 0, 0], [0, 0.1, 0]]
        with self.assertRaises(ValueError):
            build_preview(poses, 0)

    def test_angle_wrap(self):
        self.assertAlmostEqual(wrap(2 * math.pi), 0)


if __name__ == "__main__":
    unittest.main()
