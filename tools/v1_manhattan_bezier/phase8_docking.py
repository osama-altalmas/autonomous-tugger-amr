#!/usr/bin/env python3
"""
AMR V1 Phase 8 — Experimental layered docking planner.

NORMAL ROUTING:
    Uses the original, unchanged occupancy map.

LOCAL DOCKING:
    Uses an experimental copy of occupancy data in which
    cells overlapping the modeled tabletop are reclassified.
    Leg cells remain blocked.

IMPORTANT:
    Overlap does not prove obstacle identity.
    Results are OFFLINE EXPERIMENTAL CANDIDATES ONLY.
    No controller or lifting commands are produced.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

import plan_routes as planner

from build_network import (
    load_occupancy,
    footprint_is_free,
    validate_trajectory,
)

from audit_map_cargo_overlap import collision_cells


DIRECTIONS = ("east", "north", "west", "south")


def angle_difference(a, b):
    return abs(math.atan2(
        math.sin(a - b),
        math.cos(a - b)
    ))


def get_map_hash(map_file):
    import yaml

    p = Path(map_file)
    spec = yaml.safe_load(p.read_text())
    image = p.parent / spec["image"]

    return hashlib.sha256(
        p.read_bytes() + image.read_bytes()
    ).hexdigest()


def grid_center(position, spec, shape):
    cell = planner.cell(position, spec, shape)
    xy = planner.world(cell, spec)

    error = math.hypot(
        xy[0] - position[0],
        xy[1] - position[1]
    )

    if error > float(spec["resolution"]) * 0.72:
        raise ValueError("Grid alignment error")

    return [float(xy[0]), float(xy[1])], error



def make_local_collision_layer(station, spec, original):
    """
    Create an experimental docking-only collision layer.

    Pickup stand:
        Reclassify tabletop-overlap cells in a copy,
        but keep leg cells occupied.

    Drop-off:
        No elevated tabletop exists in the world.
        Keep the original map and add leg obstacles.

    Never modify the original map.

    WARNING:
        Occupancy overlap does not prove obstacle identity.
        Any resulting route is experimental, not executable.
    """
    modified = original.copy()

    table_cells = set()
    leg_cells = set()

    for box in station.get("stand_collisions", []):
        table_cells.update(
            collision_cells(box, spec, original)
        )

    for box in station.get("leg_collisions", []):
        leg_cells.update(
            collision_cells(box, spec, original)
        )

    if not leg_cells:
        raise ValueError(
            "Cargo station has no modeled leg collisions"
        )

    height, width = original.shape
    reclassified = 0

    # Only elevated tabletop cells are considered for
    # experimental reclassification.
    # A proper 3D collision check is still required.
    if table_cells:
        for x, y in table_cells:
            if not (0 <= x < width and 0 <= y < height):
                continue

            if (x, y) in leg_cells:
                continue

            if modified[y, x]:
                modified[y, x] = False
                reclassified += 1

    # Always preserve modeled leg obstacles.
    for x, y in leg_cells:
        if 0 <= x < width and 0 <= y < height:
            modified[y, x] = True

    return modified, {
        "tabletop_model_present": bool(table_cells),
        "tabletop_grid_cells": len(table_cells),
        "leg_grid_cells": len(leg_cells),
        "occupied_cells_reclassified": reclassified,
        "original_map_modified": False,
        "wall_overlap_excluded_by_evidence": False,
        "full_3d_collision_verified": False,
        "physical_validation_complete": False
    }


def build_straight(a, b, yaw, spacing=0.025):
    distance = math.hypot(
        b[0] - a[0],
        b[1] - a[1]
    )

    count = max(1, math.ceil(distance / spacing))

    return [
        [
            a[0] + (b[0] - a[0]) * i / count,
            a[1] + (b[1] - a[1]) * i / count,
            yaw
        ]
        for i in range(count + 1)
    ]


def validate_original_approach(
    poses, original_spec, original_blocked
):
    return validate_trajectory(
        poses,
        original_spec,
        original_blocked
    )


def evaluate_direction(
    start,
    entry,
    station,
    spec,
    original_blocked,
    local_blocked,
    safe,
    args,
):
    target = entry["predock_pose"]
    yaw = target[2]

    base = {
        "status": "REJECTED",
        "reason": None,
        "requested_predock_pose": target,
        "execution_approved": False,
    }

    if not entry.get("leg_clearance_pass"):
        base["reason"] = "Cargo leg geometry collision"
        return base

    if not footprint_is_free(
        target, spec, local_blocked
    ):
        base["reason"] = (
            "Pre-docking footprint blocked "
            "even in experimental local layer"
        )
        return base

    try:
        start_grid, start_error = grid_center(
            start, spec, safe.shape
        )

        target_grid, target_error = grid_center(
            target[:2], spec, safe.shape
        )

        if not footprint_is_free(
            [target_grid[0], target_grid[1], yaw],
            spec,
            local_blocked
        ):
            raise ValueError(
                "Grid-aligned pre-dock footprint blocked"
            )

        options = []

        # Try different straight-approach lengths.
        # The outer route must remain valid on the
        # ORIGINAL occupancy map.
        for approach_length in (1.0, 1.5, 2.0):

            lead_xy = [
                target_grid[0] -
                approach_length * math.cos(yaw),

                target_grid[1] -
                approach_length * math.sin(yaw)
            ]

            try:
                lead, _ = grid_center(
                    lead_xy, spec, safe.shape
                )

                lead_pose = [lead[0], lead[1], yaw]

                if not footprint_is_free(
                    lead_pose, spec, original_blocked
                ):
                    continue

                path = planner.route(
                    start_grid,
                    lead,
                    spec,
                    safe,
                    args.corner_trim,
                    args.max_curvature,
                    args.max_alternatives
                )

                if path.get("status") != \
                        "VALIDATED_STATIC_CLEARANCE":
                    continue

                poses = path["poses"]

                if len(poses) < 2:
                    continue

                # No instantaneous rotation at the join.
                if angle_difference(
                    poses[-1][2], yaw
                ) > 1e-3:
                    continue

                if math.hypot(
                    poses[-1][0] - lead[0],
                    poses[-1][1] - lead[1]
                ) > 1e-4:
                    continue

                # The final segment must be forward and
                # exactly on the requested cardinal axis.
                dx = target_grid[0] - lead[0]
                dy = target_grid[1] - lead[1]

                forward = (
                    dx * math.cos(yaw) +
                    dy * math.sin(yaw)
                )

                lateral = abs(
                    -dx * math.sin(yaw) +
                    dy * math.cos(yaw)
                )

                if forward <= 0 or lateral > 1e-3:
                    continue

                # Travel to the outer approach lead
                # uses the unchanged warehouse map.
                outer_validation = validate_original_approach(
                    poses, spec, original_blocked
                )

                if not outer_validation["valid"]:
                    continue

                straight = build_straight(
                    lead, target_grid, yaw
                )

                # Local approach must avoid all remaining
                # occupied cells, including modeled legs.
                local_validation = validate_trajectory(
                    straight, spec, local_blocked
                )

                if not local_validation["valid"]:
                    continue

                combined = poses + straight[1:]

                length = (
                    float(path["path_length_m"]) +
                    math.hypot(dx, dy)
                )

                options.append({
                    "status": "EXPERIMENTAL_OFFLINE_CANDIDATE",
                    "poses": combined,
                    "path_length_m": round(length, 4),
                    "approach_length_m": approach_length,
                    "approach_lead": lead_pose,
                    "grid_predock_pose": [
                        target_grid[0],
                        target_grid[1],
                        yaw
                    ],
                    "predock_position_error_m": round(
                        target_error, 4
                    ),
                    "start_position_error_m": round(
                        start_error, 4
                    ),
                    "original_map_outer_path_valid": True,
                    "experimental_local_path_valid": True,
                    "wall_overlap_excluded_by_evidence": False,
                    "full_3d_collision_verified": False,
                    "execution_approved": False
                })

            except Exception:
                # An unsuccessful outer-route attempt
                # cannot invalidate other alternatives.
                continue

        if not options:
            raise ValueError(
                "No correctly oriented, collision-checked "
                "outer-route plus straight local approach"
            )

        return min(
            options,
            key=lambda item: item["path_length_m"]
        )

    except Exception as exc:
        base["reason"] = str(exc)
        return base


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--map", required=True)
    ap.add_argument("--geometry", required=True)
    ap.add_argument("--output", required=True)

    ap.add_argument(
        "--start",
        nargs=2,
        type=float,
        default=[0.025, 0.025]
    )

    ap.add_argument(
        "--clearance",
        type=float,
        default=1.05
    )

    ap.add_argument(
        "--corner-trim",
        type=float,
        default=1.0
    )

    ap.add_argument(
        "--max-curvature",
        type=float,
        default=2.5
    )

    ap.add_argument(
        "--max-alternatives",
        type=int,
        default=8
    )

    args = ap.parse_args()

    geometry = json.loads(
        Path(args.geometry).read_text()
    )

    spec, original_blocked = load_occupancy(
        args.map
    )

    _, safe = planner.load_map(
        Path(args.map),
        args.clearance
    )

    output = Path(args.output)
    output.mkdir(
        parents=True,
        exist_ok=True
    )

    report = {
        "version": "AMR_V1_PHASE8_EXPERIMENTAL",
        "map_sha256": get_map_hash(args.map),
        "mode": "OFFLINE_LAYERED_DOCKING_DIAGNOSTIC",
        "original_warehouse_map_modified": False,
        "all_four_directions_evaluated": True,
        "physical_execution_approved": False,
        "stations": {},
    }

    print()
    print("========== AMR V1 PHASE 8 ==========")
    print("Cargo stations:", len(geometry["stations"]))
    print("Normal routes: ORIGINAL STATIC MAP")
    print("Local docking: EXPERIMENTAL GEOMETRY LAYER")
    print("Robot movement: DISABLED")
    print("====================================")

    for name, station in geometry["stations"].items():
        print("\nSTATION:", name, flush=True)

        local_blocked, layer_info = (
            make_local_collision_layer(
                station,
                spec,
                original_blocked
            )
        )

        entries = {}
        accepted = []

        print(
            "  Reclassified tabletop-overlap cells:",
            layer_info["occupied_cells_reclassified"]
        )

        for direction in DIRECTIONS:
            entry = station["entries"][direction]

            result = evaluate_direction(
                args.start,
                entry,
                station,
                spec,
                original_blocked,
                local_blocked,
                safe,
                args
            )

            entries[direction] = {
                k: v
                for k, v in result.items()
                if k != "poses"
            }

            if result["status"] == \
                    "EXPERIMENTAL_OFFLINE_CANDIDATE":

                length = result["path_length_m"]

                accepted.append(
                    (length, direction)
                )

                route_file = output / (
                    f"{name}_{direction}.json"
                )

                route_file.write_text(
                    json.dumps(result, indent=2) + "\n"
                )

                entries[direction]["route_file"] = (
                    route_file.name
                )

                print(
                    f"  {direction:5s}: "
                    f"EXPERIMENTAL CANDIDATE "
                    f"{length:.2f} m",
                    flush=True
                )

            else:
                print(
                    f"  {direction:5s}: "
                    f"REJECTED — {result['reason']}",
                    flush=True
                )

        selected = min(accepted) if accepted else None

        report["stations"][name] = {
            "collision_layer": layer_info,
            "entries": entries,
            "selected_entry": (
                selected[1] if selected else None
            ),
            "selected_distance_m": (
                selected[0] if selected else None
            ),
            "physical_execution_approved": False
        }

        print(
            "  BEST EXPERIMENTAL ENTRY:",
            selected[1] if selected else "NONE",
            flush=True
        )

    report_file = output / "phase8_report.json"

    report_file.write_text(
        json.dumps(report, indent=2) + "\n"
    )

    print()
    print("========== PHASE 8 SUMMARY ==========")

    for name, station in report["stations"].items():
        valid = [
            direction
            for direction, entry
            in station["entries"].items()
            if entry["status"] ==
                "EXPERIMENTAL_OFFLINE_CANDIDATE"
        ]

        print(
            f"{name}: {len(valid)}/4 "
            f"experimental candidates {valid}"
        )

    print("Report:", report_file)
    print("Original map changed: NO")
    print("Physical execution approved: NO")
    print("====================================")


if __name__ == "__main__":
    main()
