#!/usr/bin/env python3
"""
AMR V1 — Cargo access diagnostics.

Read-only:
  - Four entrances per cargo station
  - Static-map footprint clearance
  - Nearest occupied map cells
  - Obstructions inside/outside stand footprint
  - Available clearance behind each pre-docking pose

Does not modify the map or command the robot.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from build_network import (
    load_occupancy,
    footprint_is_free,
)


def map_to_world(ix, iy, spec):
    res = float(spec["resolution"])
    ox, oy, theta = spec["origin"]

    gx = (ix + 0.5) * res
    gy = (iy + 0.5) * res

    c = math.cos(theta)
    s = math.sin(theta)

    return (
        ox + c * gx - s * gy,
        oy + s * gx + c * gy
    )


def world_to_grid(x, y, spec):
    res = float(spec["resolution"])
    ox, oy, theta = spec["origin"]

    dx = x - ox
    dy = y - oy

    c = math.cos(theta)
    s = math.sin(theta)

    return (
        (c * dx + s * dy) / res,
        (-s * dx + c * dy) / res
    )


def nearest_blockers(x, y, spec, blocked, radius=2.0):
    gx, gy = world_to_grid(x, y, spec)

    res = float(spec["resolution"])
    r = math.ceil(radius / res)

    h, w = blocked.shape

    x0 = max(0, math.floor(gx) - r)
    x1 = min(w, math.floor(gx) + r + 1)
    y0 = max(0, math.floor(gy) - r)
    y1 = min(h, math.floor(gy) + r + 1)

    ys, xs = np.nonzero(blocked[y0:y1, x0:x1])

    results = []

    for ix, iy in zip(xs + x0, ys + y0):
        wx, wy = map_to_world(int(ix), int(iy), spec)

        distance = math.hypot(wx - x, wy - y)

        if distance <= radius:
            results.append({
                "distance_m": round(distance, 3),
                "world_xy": [
                    round(wx, 3),
                    round(wy, 3)
                ],
                "grid_cell": [int(ix), int(iy)]
            })

    results.sort(key=lambda item: item["distance_m"])

    return results[:10]


def stand_bounds(station):
    """Bounding box of all recorded stand and leg collisions."""
    boxes = (
        station.get("stand_collisions", []) +
        station.get("leg_collisions", [])
    )

    if not boxes:
        return None

    points = []

    for box in boxes:
        x, y, _, _, _, yaw = box["pose"]
        sx, sy, _ = box["size"]

        c = math.cos(yaw)
        s = math.sin(yaw)

        for dx in (-sx / 2, sx / 2):
            for dy in (-sy / 2, sy / 2):
                points.append((
                    x + c * dx - s * dy,
                    y + s * dx + c * dy
                ))

    return [
        min(p[0] for p in points),
        min(p[1] for p in points),
        max(p[0] for p in points),
        max(p[1] for p in points)
    ]


def inside_bounds(point, bounds, margin=0.05):
    if bounds is None:
        return False

    x, y = point

    return (
        bounds[0] - margin <= x <= bounds[2] + margin
        and
        bounds[1] - margin <= y <= bounds[3] + margin
    )


def inspect_entry(entry, bounds, spec, blocked, distances):
    pose = entry["predock_pose"]

    x, y, yaw = pose

    clear = bool(
        footprint_is_free(pose, spec, blocked)
    )

    gx, gy = world_to_grid(x, y, spec)
    ix, iy = math.floor(gx), math.floor(gy)

    h, w = blocked.shape

    if 0 <= ix < w and 0 <= iy < h:
        center_occupied = bool(blocked[iy, ix])
        center_clearance = float(
            distances[iy, ix]
        )
    else:
        center_occupied = True
        center_clearance = 0.0

    blockers = nearest_blockers(
        x, y, spec, blocked
    )

    for item in blockers:
        item["inside_stand_bounds"] = inside_bounds(
            item["world_xy"], bounds
        )

    # Test locations farther away from the cargo seat,
    # along the same docking axis.
    lead_checks = []

    for distance in (0.5, 1.0, 1.5, 2.0):
        lx = x - distance * math.cos(yaw)
        ly = y - distance * math.sin(yaw)

        lead_checks.append({
            "distance_behind_predock_m": distance,
            "pose": [
                round(lx, 3),
                round(ly, 3),
                round(yaw, 5)
            ],
            "full_footprint_clear": bool(
                footprint_is_free(
                    [lx, ly, yaw],
                    spec,
                    blocked
                )
            )
        })

    return {
        "predock_pose": pose,
        "leg_clear": bool(
            entry["leg_clearance_pass"]
        ),
        "static_map_footprint_clear": clear,
        "center_cell_occupied": center_occupied,
        "center_clearance_m": round(
            center_clearance, 3
        ),
        "nearest_blockers": blockers,
        "approach_lead_checks": lead_checks,
        "execution_approved": False
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--geometry", required=True)
    ap.add_argument("--output", required=True)

    args = ap.parse_args()

    spec, blocked = load_occupancy(args.map)

    geometry = json.loads(
        Path(args.geometry).read_text()
    )

    distance = distance_transform_edt(
        ~blocked
    ) * float(spec["resolution"])

    results = {
        "version": "AMR_V1_CARGO_ACCESS_DIAGNOSTICS",
        "static_map_only": True,
        "execution_approved": False,
        "stations": {}
    }

    print("\n========== CARGO ACCESS DIAGNOSTICS ==========")
    print("Map dimensions:", blocked.shape)
    print("Map resolution:", spec["resolution"])
    print("Blocked cells:", int(blocked.sum()))
    print("Stations:", len(geometry["stations"]))

    for name, station in geometry["stations"].items():
        bounds = stand_bounds(station)

        print("\n" + "=" * 60)
        print("STATION:", name)
        print("Center:", station["center"][:2])
        print("Stand world bounds:", bounds)

        report = {
            "center": station["center"],
            "stand_world_bounds": bounds,
            "entries": {}
        }

        for direction in (
            "east", "north", "west", "south"
        ):
            entry = station["entries"][direction]

            analysis = inspect_entry(
                entry, bounds, spec,
                blocked, distance
            )

            report["entries"][direction] = analysis

            print(
                f"\n  {direction.upper()}"
                f"  pre-dock={analysis['predock_pose']}"
            )

            print(
                "    chassis vs legs:",
                "CLEAR" if analysis["leg_clear"]
                else "BLOCKED"
            )

            print(
                "    static-map footprint:",
                "CLEAR"
                if analysis["static_map_footprint_clear"]
                else "BLOCKED"
            )

            print(
                "    robot center cell occupied:",
                analysis["center_cell_occupied"]
            )

            print(
                "    distance to occupied map cell:",
                analysis["center_clearance_m"],
                "m"
            )

            print(
                "    approach lead:",
                [
                    (
                        item["distance_behind_predock_m"],
                        item["full_footprint_clear"]
                    )
                    for item in analysis[
                        "approach_lead_checks"
                    ]
                ]
            )

            if analysis["nearest_blockers"]:
                print(
                    "    nearest blocker:",
                    analysis["nearest_blockers"][0]
                )

        results["stations"][name] = report

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output.write_text(
        json.dumps(results, indent=2) + "\n"
    )

    print("\n========== SUMMARY ==========")

    for name, station in results["stations"].items():
        clear = [
            direction
            for direction, entry
            in station["entries"].items()
            if entry["static_map_footprint_clear"]
            and entry["leg_clear"]
        ]

        print(
            f"{name}: "
            f"{len(clear)}/4 map-and-leg clear: {clear}"
        )

    print("Report:", output)
    print("Robot execution: NOT APPROVED")
    print("=============================")


if __name__ == "__main__":
    main()
