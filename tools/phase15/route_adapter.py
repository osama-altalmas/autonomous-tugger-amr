#!/usr/bin/env python3
"""Phase 15A: offline, motion-disabled front/rear route adapter."""

import argparse
import json
import math
from pathlib import Path

from direction_selector import choose_direction, load_segment_one, wrap


def adapt_poses(poses, motion_direction):
    if motion_direction not in ("FORWARD", "REVERSE"):
        raise ValueError("Invalid motion direction")

    offset = math.pi if motion_direction == "REVERSE" else 0.0

    adapted = [
        [float(p[0]), float(p[1]), wrap(float(p[2]) + offset)]
        for p in poses
    ]

    if len(adapted) < 2:
        raise ValueError("Insufficient route poses")

    for p in adapted:
        if not all(math.isfinite(v) for v in p):
            raise ValueError("Nonfinite route pose")

    return adapted


def rotate_body_headings(poses, rotation):
    """Return execution-only poses with body headings rotated in place.

    XY geometry is intentionally preserved.  ``rotation`` is normally pi
    when changing which end leads, and zero when saved body headings already
    describe the selected leading end.
    """
    adapted = [
        [float(p[0]), float(p[1]), wrap(float(p[2]) + rotation)]
        for p in poses
    ]

    if len(adapted) < 2:
        raise ValueError("Insufficient route poses")

    if any(not all(math.isfinite(value) for value in pose)
           for pose in adapted):
        raise ValueError("Nonfinite route pose")

    return adapted


def validate_motion_headings(poses, motion_direction, tolerance_deg=12):
    if motion_direction not in ("FORWARD", "REVERSE"):
        raise ValueError("Invalid motion direction")

    offset = math.pi if motion_direction == "REVERSE" else 0.0
    worst = 0.0
    checked = 0

    for first, second in zip(poses, poses[1:]):
        dx = second[0] - first[0]
        dy = second[1] - first[1]

        if math.hypot(dx, dy) < 1e-7:
            continue

        travel_yaw = math.atan2(dy, dx)
        expected_travel_yaw = first[2] + offset
        error = abs(wrap(travel_yaw - expected_travel_yaw))

        worst = max(worst, error)
        checked += 1

    if checked == 0:
        raise ValueError("Route has no nonzero displacement")

    if math.degrees(worst) > tolerance_deg:
        raise ValueError(
            f"Heading disagreement: {math.degrees(worst):.3f} degrees "
            f"(limit {tolerance_deg} degrees)"
        )

    return math.degrees(worst)


def build_preview(poses, robot_yaw, stored_motion_direction="FORWARD"):
    """Prepare an in-memory logical-forward execution preview.

    Saved routes record body headings plus their body-frame direction.  If
    the live start selects the other leading end, rotate body headings by pi;
    the route's XY geometry and mission progress remain unchanged.
    """
    if stored_motion_direction not in ("FORWARD", "REVERSE"):
        raise ValueError("Invalid stored motion direction")

    selection = choose_direction(poses, robot_yaw)
    direction = selection["body_motion_direction"]

    rotation = math.pi if direction != stored_motion_direction else 0.0
    adapted = rotate_body_headings(poses, rotation)

    worst = validate_motion_headings(adapted, direction)

    start_error = abs(wrap(adapted[0][2] - robot_yaw))

    if start_error > math.radians(20):
        raise ValueError(
            "Adapted starting body heading differs from robot heading"
        )

    for original, transformed in zip(poses, adapted):
        if (
            float(original[0]) != transformed[0]
            or float(original[1]) != transformed[1]
        ):
            raise ValueError("Adapter changed route positions")

    return {
        "selection": selection,
        "adapted_poses": adapted,
        "stored_body_motion_direction": stored_motion_direction,
        "execution_body_motion_direction": direction,
        "body_heading_rotation_deg": math.degrees(rotation),
        "worst_heading_error_deg": worst,
        "starting_body_heading_error_deg": math.degrees(start_error),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", required=True)
    parser.add_argument("--robot-yaw-deg", type=float, required=True)
    args = parser.parse_args()

    try:
        route, poses = load_segment_one(args.route)
        result = build_preview(poses, math.radians(args.robot_yaw_deg))
    except (ValueError, OSError, KeyError) as exc:
        print("ROUTE ADAPTER: REJECTED")
        print("Reason:", exc)
        print("Motion authorized: NO")
        return 2

    selection = result["selection"]
    adapted = result["adapted_poses"]

    print("=" * 60)
    print("PHASE 15A — ROUTE ADAPTER OFFLINE PREVIEW")
    print("=" * 60)
    print("Destination:", route.get("destination"))
    print("Leading end:", selection["leading_end"])
    print("Motion direction:", selection["motion_direction"])
    print("Speed sign:", selection["speed_sign"])
    print("Route poses:", len(adapted))
    print("Original first pose:", poses[0])
    print("Adapted first pose:", adapted[0])
    print("Original last pose:", poses[-1])
    print("Adapted last pose:", adapted[-1])
    print(
        "Worst heading disagreement:",
        f"{result['worst_heading_error_deg']:.3f} degrees",
    )
    print(
        "Starting body-heading error:",
        f"{result['starting_body_heading_error_deg']:.3f} degrees",
    )
    print("Route XY coordinates: PRESERVED")
    print("Motion authorized: NO")
    print("Saved routes modified: NO")
    print("Phase 14 code modified: NO")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
