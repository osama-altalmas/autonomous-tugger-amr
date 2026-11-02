#!/usr/bin/env python3
"""Phase 15A.1: read-only front/rear leading-end selection.

Decides the leading end from:
  1. The first nonzero movement of the selected path.
  2. The robot's CURRENT body heading.

Mission progress is always FORWARD along the selected path.  A rear-leading
start is converted to a negative body-frame velocity only at the controller
boundary; it is not a reverse mission.

This module NEVER sends motion commands.
"""

import argparse
import json
import math
from pathlib import Path


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def initial_travel_heading(poses, min_displacement=0.01):
    """Heading from first pose to first sufficiently distinct pose."""
    if len(poses) < 2:
        raise ValueError("Route has fewer than two poses")

    x0 = float(poses[0][0])
    y0 = float(poses[0][1])

    for pose in poses[1:]:
        dx = float(pose[0]) - x0
        dy = float(pose[1]) - y0

        if math.hypot(dx, dy) >= min_displacement:
            return math.atan2(dy, dx)

    raise ValueError("Route has no measurable initial displacement")


def choose_direction(
    poses,
    robot_yaw,
    angular_tolerance_deg=20.0,
):
    if not (0 < angular_tolerance_deg < 90):
        raise ValueError("Angular tolerance must be between 0 and 90 degrees")

    if not math.isfinite(robot_yaw):
        raise ValueError("Robot heading must be finite")

    travel_yaw = initial_travel_heading(poses)

    difference = abs(wrap(travel_yaw - robot_yaw))

    tolerance = math.radians(angular_tolerance_deg)

    front_error = difference
    rear_error = abs(math.pi - difference)

    if front_error <= tolerance:
        leading_end = "FRONT"
        body_motion_direction = "FORWARD"
        body_velocity_sign = 1

    elif rear_error <= tolerance:
        leading_end = "REAR"
        body_motion_direction = "REVERSE"
        body_velocity_sign = -1

    else:
        raise ValueError(
            "AMBIGUOUS START: robot and path headings differ "
            f"by {math.degrees(difference):.2f} degrees. "
            "No direction selected."
        )

    return {
        "leading_end": leading_end,
        "route_progress_direction": "FORWARD",
        "body_motion_direction": body_motion_direction,
        "body_velocity_sign": body_velocity_sign,
        # Compatibility aliases for the existing Phase 15 prototype.
        "motion_direction": body_motion_direction,
        "speed_sign": body_velocity_sign,
        "robot_heading_deg": math.degrees(wrap(robot_yaw)),
        "travel_heading_deg": math.degrees(wrap(travel_yaw)),
        "heading_difference_deg": math.degrees(difference),
    }


def load_segment_one(filename):
    """Load either a Phase 14 segment or a standard saved-route record.

    The leading-end decision is geometry based and therefore applies to every
    map-validated saved route, not only to the original Phase 14 refined
    approach files.
    """
    data = json.loads(Path(filename).read_text())

    if data.get("execution_approved") is not False:
        raise ValueError(
            "Unexpected route approval state; refusing selection"
        )

    if data.get("schema") == "amr_phase14_two_segment_refined_v1":
        segment = data.get("segment_1")

        if not isinstance(segment, dict):
            raise ValueError("Missing segment_1")

        poses = segment.get("poses")

    elif data.get("version") == "AMR_V1_SAVED_ROUTE":
        poses = data.get("poses")

    else:
        raise ValueError("Unsupported saved-route format")

    if not isinstance(poses, list) or len(poses) < 2:
        raise ValueError("Invalid segment_1 poses")

    return data, poses


def main():
    parser = argparse.ArgumentParser(
        description="Select FRONT/REAR after loading a saved route"
    )

    parser.add_argument("--route", required=True)

    parser.add_argument(
        "--robot-yaw-deg",
        type=float,
        required=True,
        help="Current measured ROBOT BODY heading, in degrees",
    )

    parser.add_argument(
        "--tolerance-deg",
        type=float,
        default=20.0,
    )

    args = parser.parse_args()

    try:
        data, poses = load_segment_one(args.route)

        result = choose_direction(
            poses,
            math.radians(args.robot_yaw_deg),
            args.tolerance_deg,
        )

    except (ValueError, OSError, KeyError) as exc:
        print("DIRECTION SELECTION: REJECTED")
        print("Reason:", exc)
        print("Motion authorized: NO")
        return 2

    print("=" * 55)
    print("PHASE 15A — DIRECTION SELECTED")
    print("=" * 55)

    print("Route:", data.get("source", "HOME"), "->",
          data.get("destination", data.get("target")))
    print("Segment 1 poses:", len(poses))

    for key, value in result.items():
        print(f"{key}: {value}")

    print()
    print("Motion authorized: NO")
    print("Route execution: NOT STARTED")
    print("Phase 14 source: UNCHANGED")
    print("=" * 55)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
