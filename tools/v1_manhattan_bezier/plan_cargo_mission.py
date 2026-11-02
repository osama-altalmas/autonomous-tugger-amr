#!/usr/bin/env python3
"""
AMR V1 — Offline cargo approach planner.

For every cargo station and entry direction:
  1. Verify the geometry audit approved chassis/leg clearance.
  2. Find a Manhattan-Bezier path from HOME to pre-docking.
  3. Validate the entire path with the rectangular footprint.
  4. Check final arrival heading.
  5. Select the shortest candidate passing all checks.

This tool DOES NOT command movement or lift cargo.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import plan_routes as planner

from build_network import (
    load_occupancy,
    validate_trajectory,
    footprint_is_free,
)


def angle_error(a, b):
    return abs(math.atan2(
        math.sin(a - b),
        math.cos(a - b)
    ))


def hash_map(map_file):
    import yaml

    map_file = Path(map_file)
    spec = yaml.safe_load(map_file.read_text())
    image = map_file.parent / spec["image"]

    return hashlib.sha256(
        map_file.read_bytes() + image.read_bytes()
    ).hexdigest()


def snap_to_cell_center(xy, spec, shape):
    """
    The V1 planner operates on cardinal grid centerlines.
    Snap only within half a grid cell, never silently move
    a station a large distance.
    """
    q = planner.cell(xy, spec, shape)
    snapped = planner.world(q, spec)

    error = math.dist(xy, snapped)

    max_error = float(spec["resolution"]) * 0.72

    if error > max_error:
        raise ValueError(
            f"Station alignment error {error:.3f} m"
        )

    return snapped, error


def evaluate_candidate(
    start_xy,
    target_pose,
    audit_entry,
    spec,
    safe,
    occupancy_spec,
    blocked,
    args
):
    result = {
        "status": "REJECTED",
        "reason": None,
        "predock_pose": target_pose,
        "execution_approved": False
    }

    if not audit_entry.get("leg_clearance_pass"):
        result["reason"] = "Cargo leg collision"
        return result

    if not audit_entry.get(
        "checked_against"
    ) == "base_link_chassis_and_stand_legs":
        result["reason"] = "Unexpected geometry audit"
        return result

    try:
        start, start_shift = snap_to_cell_center(
            start_xy, spec, safe.shape
        )

        target, target_shift = snap_to_cell_center(
            target_pose[:2], spec, safe.shape
        )

        # Both the exact requested destination and its
        # snapped counterpart must be footprint-clear.
        if not footprint_is_free(
            target_pose,
            occupancy_spec,
            blocked
        ):
            raise ValueError(
                "Exact pre-docking footprint blocked"
            )

        if not footprint_is_free(
            [target[0], target[1], target_pose[2]],
            occupancy_spec,
            blocked
        ):
            raise ValueError(
                "Snapped pre-docking footprint blocked"
            )

        route = planner.route(
            start,
            target,
            spec,
            safe,
            args.corner_trim,
            args.max_curvature,
            args.max_alternatives
        )

        if route.get("status") != \
                "VALIDATED_STATIC_CLEARANCE":
            raise ValueError(
                "Planner did not validate route"
            )

        poses = route["poses"]

        if len(poses) < 2:
            raise ValueError("Insufficient route poses")

        swept = validate_trajectory(
            poses,
            occupancy_spec,
            blocked
        )

        if not swept["valid"]:
            raise ValueError(
                "Swept footprint collision at "
                f"{swept['failed_pose']}"
            )

        # The original planner does not constrain arrival yaw.
        # Do not silently claim that a route meets docking
        # alignment requirements.
        final_yaw = poses[-1][2]
        required_yaw = target_pose[2]

        yaw_error = angle_error(final_yaw, required_yaw)

        if yaw_error > math.radians(
            args.yaw_tolerance_deg
        ):
            raise ValueError(
                f"Arrival yaw mismatch "
                f"{math.degrees(yaw_error):.1f} degrees"
            )

        # No zero-length "teleport" to the exact station.
        # Snapping is a bounded approximation, not
        # a verified physical final positioning maneuver.
        result.update({
            "status": "OFFLINE_CANDIDATE",
            "reason": None,
            "route": route,
            "swept_validation": swept,
            "start_snap_error_m": round(
                start_shift, 4
            ),
            "target_snap_error_m": round(
                target_shift, 4
            ),
            "arrival_yaw_error_deg": round(
                math.degrees(yaw_error), 3
            ),
            "path_length_m": route["path_length_m"],
            "execution_approved": False
        })

    except Exception as exc:
        result["reason"] = str(exc)

    return result


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--map", required=True)
    p.add_argument("--geometry", required=True)
    p.add_argument("--start", nargs=2, type=float,
                   default=[0.025, 0.025])
    p.add_argument("--output", required=True)

    p.add_argument("--clearance", type=float, default=1.05)
    p.add_argument("--corner-trim", type=float, default=1.0)
    p.add_argument("--max-curvature", type=float, default=2.5)
    p.add_argument("--max-alternatives", type=int, default=8)
    p.add_argument("--yaw-tolerance-deg", type=float, default=10.0)

    args = p.parse_args()

    geometry = json.loads(
        Path(args.geometry).read_text()
    )

    if not geometry.get("stations"):
        raise RuntimeError("No cargo stations in audit")

    spec, safe = planner.load_map(
        args.map,
        args.clearance
    )

    occupancy_spec, blocked = load_occupancy(
        args.map
    )

    map_sha = hash_map(args.map)

    database = {
        "version": "AMR_V1_PHASE5",
        "mode": "STATIC_MAP_ONLY",
        "map_sha256": map_sha,
        "source_geometry": str(args.geometry),
        "start": args.start,
        "clearance_m": args.clearance,
        "max_curvature": args.max_curvature,
        "yaw_tolerance_deg": args.yaw_tolerance_deg,
        "stations": {},
        "execution_approved": False,
        "limitations": [
            "No steering-dynamics validation",
            "No full 3D robot/seat collision validation",
            "No final-centimeter localization validation",
            "No verified lift or cargo attachment",
            "No motion controller integration"
        ]
    }

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    print()
    print("========= AMR V1 CARGO ROUTING =========")
    print("Start:", args.start)
    print("Cargo stations:", len(geometry["stations"]))
    print("Static map collision checking: ON")
    print("Swept rectangular footprint check: ON")
    print("Arrival orientation check: ON")
    print("========================================")

    total_accepted = 0

    for station_name, station in geometry["stations"].items():

        print(f"\nSTATION: {station_name}")

        candidates = {}
        accepted = []

        for direction, entry in station["entries"].items():

            target = entry["predock_pose"]

            candidate = evaluate_candidate(
                args.start,
                target,
                entry,
                spec,
                safe,
                occupancy_spec,
                blocked,
                args
            )

            candidates[direction] = candidate

            if candidate["status"] == "OFFLINE_CANDIDATE":
                accepted.append((
                    candidate["path_length_m"],
                    direction
                ))

                print(
                    f"  {direction:5s}: CANDIDATE | "
                    f"{candidate['path_length_m']:.2f} m"
                )
            else:
                print(
                    f"  {direction:5s}: REJECTED | "
                    f"{candidate['reason']}"
                )

        selected = min(accepted) if accepted else None

        station_result = {
            "center": station["center"],
            "candidates": candidates,
            "selected_entry": (
                selected[1] if selected else None
            ),
            "selection_status": (
                "OFFLINE_CANDIDATE"
                if selected else "NO_VALIDATED_ROUTE"
            ),
            "execution_approved": False
        }

        database["stations"][station_name] = station_result

        if selected:
            total_accepted += 1

            direction = selected[1]

            path_file = output / (
                f"{station_name}_{direction}_path.json"
            )

            path_file.write_text(
                json.dumps(
                    candidates[direction]["route"],
                    indent=2
                ) + "\n"
            )

            station_result["selected_path_file"] = (
                path_file.name
            )

            print(
                f"  SELECTED: {direction} "
                f"({selected[0]:.2f} m)"
            )
        else:
            print("  SELECTED: NONE")

    database_path = output / "cargo_route_database.json"

    database_path.write_text(
        json.dumps(database, indent=2) + "\n"
    )

    print()
    print("========== FINAL SUMMARY ==========")
    print("Cargo stations:", len(geometry["stations"]))
    print("Stations with candidates:", total_accepted)
    print("Database:", database_path)
    print("Robot execution: NOT APPROVED")
    print("===================================")


if __name__ == "__main__":
    main()
