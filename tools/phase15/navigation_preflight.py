#!/usr/bin/env python3
"""Phase 15A.4: read-only saved-route direction integration.

No ROS node is created.
No motion command is published.
No route or database is modified.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import yaml

from route_adapter import build_preview, validate_motion_headings


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "v1_manhattan_bezier"))

from build_network import load_occupancy, validate_trajectory


def map_fingerprint(map_file):
    map_file = Path(map_file)
    metadata = yaml.safe_load(map_file.read_text())

    image_file = map_file.parent / metadata["image"]

    return hashlib.sha256(
        map_file.read_bytes() + image_file.read_bytes()
    ).hexdigest()


def check_route(database_file, map_file, source, target, robot_yaw):
    database_file = Path(database_file)
    map_file = Path(map_file)

    database = json.loads(database_file.read_text())
    fingerprint = map_fingerprint(map_file)

    if database["map_sha256"] != fingerprint:
        raise ValueError("Map fingerprint does not match database")

    route_key = f"{source}__TO__{target}"
    entry = database["routes"].get(route_key)

    if entry is None:
        raise ValueError(f"Route not found: {route_key}")

    if entry["status"] != "VALIDATED_STATIC_CANDIDATE":
        raise ValueError("Route lacks validated static-candidate status")

    route_path = database_file.parent / entry["route_file"]
    route = json.loads(route_path.read_text())

    if (
        route["source"] != source
        or route["target"] != target
        or route["map_sha256"] != fingerprint
    ):
        raise ValueError("Route identity or map fingerprint mismatch")

    poses = route["poses"]

    if len(poses) < 2:
        raise ValueError("Route has insufficient poses")

    if any(
        len(p) != 3
        or not all(math.isfinite(float(value)) for value in p)
        for p in poses
    ):
        raise ValueError("Invalid or nonfinite route pose")

    # Saved poses describe robot BODY heading.  Existing route metadata tells
    # whether that body travels along or opposite the route tangent.
    stored_motion_direction = route.get("motion_direction", "FORWARD")

    original_error = validate_motion_headings(
        poses, stored_motion_direction
    )

    preview = build_preview(
        poses,
        robot_yaw,
        stored_motion_direction,
    )

    adapted = preview["adapted_poses"]
    direction = preview["selection"]["body_motion_direction"]

    # Independently validate the adapted route's headings.
    adapted_error = validate_motion_headings(
        adapted, direction
    )

    # Preserve original XY positions exactly.
    if any(
        (float(a[0]), float(a[1]))
        != (float(b[0]), float(b[1]))
        for a, b in zip(poses, adapted)
    ):
        raise ValueError("Route geometry was modified")

    # Re-run the existing footprint validator.
    map_spec, blocked = load_occupancy(map_file)

    collision = validate_trajectory(
        adapted, map_spec, blocked
    )

    if not collision["valid"]:
        raise ValueError(
            "Adapted route failed static footprint validation"
        )

    return {
        "source": source,
        "target": target,
        "route_file": str(route_path),
        "poses": len(adapted),
        "selection": preview["selection"],
        "stored_body_motion_direction": stored_motion_direction,
        "execution_body_motion_direction": direction,
        "body_heading_rotation_deg": preview["body_heading_rotation_deg"],
        "original_heading_error_deg": original_error,
        "adapted_heading_error_deg": adapted_error,
        "route_approved": route.get("execution_approved"),
        "static_validation": "PASSED",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Offline direction selection with full route checks"
    )

    parser.add_argument("--database", required=True)
    parser.add_argument("--map", required=True)
    parser.add_argument("--from", dest="source", default="HOME")
    parser.add_argument("--to", dest="target", required=True)

    parser.add_argument(
        "--robot-yaw-deg",
        type=float,
        required=True,
        help="Hypothetical robot body heading for offline testing",
    )

    args = parser.parse_args()

    try:
        result = check_route(
            args.database,
            args.map,
            args.source,
            args.target,
            math.radians(args.robot_yaw_deg),
        )
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print()
        print("PHASE 15A.4: PREFLIGHT REJECTED")
        print("Reason:", exc)
        print("Motion authorized: NO")
        return 2

    selection = result["selection"]

    print()
    print("=" * 60)
    print("PHASE 15A.4 — NAVIGATION INTEGRATION PREFLIGHT")
    print("=" * 60)

    print("Route:", result["source"], "->", result["target"])
    print("Saved route:", result["route_file"])
    print("Route poses:", result["poses"])

    print()
    print("LEADING END:", selection["leading_end"])
    print("ROUTE PROGRESS:", selection["route_progress_direction"])
    print("BODY MOTION:", result["execution_body_motion_direction"])
    print("BODY VELOCITY SIGN:", selection["body_velocity_sign"])
    print("BODY HEADING ROTATION:",
          f"{result['body_heading_rotation_deg']:.1f} deg")

    print()
    print(
        "Original heading error:",
        f"{result['original_heading_error_deg']:.3f} deg",
    )
    print(
        "Adapted heading error:",
        f"{result['adapted_heading_error_deg']:.3f} deg",
    )

    print("Static footprint validation:", result["static_validation"])
    print("Saved route execution approval:", result["route_approved"])

    print()
    print("Robot live pose checked: NO")
    print("Motion authorized: NO")
    print("Navigation started: NO")
    print("Route files changed: NO")
    print("Phase 14 modified: NO")

    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
