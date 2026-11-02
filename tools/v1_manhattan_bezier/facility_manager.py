#!/usr/bin/env python3
"""
AMR V1 Facility Manager

Functions:
- Load named stations and saved routes
- Find station-to-station routes through a directed graph
- Generate four cargo-seat approach candidates
- Validate pre-docking poses and straight approach segments
- Save docking feasibility reports

Offline planning only. No robot motion.
"""

import argparse
import heapq
import json
import math
import sys
from pathlib import Path

import numpy as np

from build_network import (
    load_occupancy,
    footprint_is_free,
    validate_trajectory,
)


DIRECTIONS = {
    "east": 0.0,
    "north": math.pi / 2,
    "west": math.pi,
    "south": -math.pi / 2,
}


def load_graph(path):
    graph = json.loads(Path(path).read_text())

    if graph.get("mode") != "STATIC_MAP_ONLY":
        raise ValueError("Not a V1 static route graph")

    return graph


def shortest_station_route(graph, start, goal):
    """Find the lowest-distance route through saved station links."""

    stations = graph["stations"]
    routes = graph["routes"]

    if start not in stations:
        raise ValueError(f"Unknown start station: {start}")

    if goal not in stations:
        raise ValueError(f"Unknown destination: {goal}")

    adjacency = {name: [] for name in stations}

    for route_id, route in routes.items():
        if not route.get("validated", False):
            continue

        a = route["from"]
        b = route["to"]

        adjacency.setdefault(a, []).append((
            float(route["length_m"]),
            b,
            route_id
        ))

    queue = [(0.0, start, [], [start])]
    best = {}

    while queue:
        cost, current, links, visited = heapq.heappop(queue)

        if current in best and best[current] <= cost:
            continue

        best[current] = cost

        if current == goal:
            return {
                "status": "ROUTE_FOUND",
                "start": start,
                "goal": goal,
                "stations": visited,
                "route_ids": links,
                "length_m": round(cost, 4),
                "execution_approved": False
            }

        for distance, neighbor, route_id in adjacency.get(current, []):
            if neighbor in visited:
                continue

            heapq.heappush(
                queue,
                (
                    cost + distance,
                    neighbor,
                    links + [route_id],
                    visited + [neighbor]
                )
            )

    return {
        "status": "NO_ROUTE",
        "start": start,
        "goal": goal,
        "execution_approved": False
    }


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def calculate_entries(center, seat_yaw, approach_distance):
    """
    Return four pre-docking poses.

    Each pose faces the seat center.
    Directions are relative to the seat orientation.
    """

    cx, cy = center

    entries = {}

    for name, local_angle in DIRECTIONS.items():
        outward_angle = seat_yaw + local_angle

        ux = math.cos(outward_angle)
        uy = math.sin(outward_angle)

        x = cx + approach_distance * ux
        y = cy + approach_distance * uy

        heading = wrap_angle(outward_angle + math.pi)

        entries[name] = {
            "position": [x, y],
            "heading": heading,
            "pose": [x, y, heading],
            "approach_distance_m": approach_distance
        }

    return entries


def straight_docking_samples(start, center, step=0.025):
    """
    Generate a straight geometric approach toward the seat center.

    This is checked as a provisional geometric path only.
    It does not establish underside or leg clearance.
    """

    x0, y0, yaw = start
    x1, y1 = center

    distance = math.hypot(x1 - x0, y1 - y0)

    count = max(1, math.ceil(distance / step))

    return [
        [
            x0 + (x1 - x0) * i / count,
            y0 + (y1 - y0) * i / count,
            yaw
        ]
        for i in range(count + 1)
    ]


def inspect_seat(seat, spec, blocked):
    center = seat["center"]
    seat_yaw = float(seat.get("yaw", 0.0))

    approach_distance = float(
        seat.get("approach_distance_m", 1.0)
    )

    if approach_distance <= 0:
        raise ValueError("Approach distance must be positive")

    entries = calculate_entries(
        center,
        seat_yaw,
        approach_distance
    )

    allowed = set(
        seat.get(
            "allowed_entries",
            list(DIRECTIONS)
        )
    )

    unknown = allowed - set(DIRECTIONS)
    if unknown:
        raise ValueError(f"Unknown entries: {sorted(unknown)}")

    results = {}

    for name, entry in entries.items():
        pose = entry["pose"]

        entry_allowed = name in allowed

        pre_dock_clear = (
            entry_allowed and
            footprint_is_free(pose, spec, blocked)
        )

        # Only a diagnostic; this does not authorize docking.
        approach = straight_docking_samples(
            pose,
            center
        )

        validation = validate_trajectory(
            approach,
            spec,
            blocked
        )

        results[name] = {
            **entry,
            "enabled": entry_allowed,
            "pre_dock_footprint_clear": bool(pre_dock_clear),
            "centerline_approach_clear": bool(
                validation["valid"]
            ),
            "failed_pose": validation["failed_pose"],
            "physical_seat_clearance_verified": False,
            "execution_approved": False
        }

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--map", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--seats", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--from-station")
    parser.add_argument("--to-station")

    args = parser.parse_args()

    graph = load_graph(args.graph)

    spec, blocked = load_occupancy(args.map)

    seats = json.loads(Path(args.seats).read_text())

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # Verify the saved route graph belongs to the current map.
    import hashlib
    import yaml

    map_path = Path(args.map)
    map_spec = yaml.safe_load(map_path.read_text())
    image_path = map_path.parent / map_spec["image"]

    current_hash = hashlib.sha256(
        map_path.read_bytes() + image_path.read_bytes()
    ).hexdigest()

    if graph["map_sha256"] != current_hash:
        raise RuntimeError(
            "Saved route graph belongs to a different map. "
            "Regenerate it before continuing."
        )

    print("\n========== AMR V1 FACILITY ==========")
    print("Registered stations:", len(graph["stations"]))
    print("Saved routes:", len(graph["routes"]))
    print("Cargo seats:", len(seats))

    if args.from_station and args.to_station:
        result = shortest_station_route(
            graph,
            args.from_station,
            args.to_station
        )

        print("\nSTATION ROUTE:")
        print(json.dumps(result, indent=2))

        (output / "selected_route.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )

    docking_report = {}

    for seat_id, seat in seats.items():
        print(f"\nInspecting cargo seat: {seat_id}")

        entries = inspect_seat(
            seat,
            spec,
            blocked
        )

        docking_report[seat_id] = {
            "center": seat["center"],
            "entries": entries,
            "execution_approved": False
        }

        for name, entry in entries.items():
            print(
                f"  {name:5s} | "
                f"pre-dock clear: "
                f"{entry['pre_dock_footprint_clear']} | "
                f"approach clear: "
                f"{entry['centerline_approach_clear']}"
            )

    report = {
        "version": "AMR_V1_PHASE4",
        "mode": "STATIC_MAP_ONLY",
        "map_sha256": current_hash,
        "seats": docking_report,
        "physical_docking_validated": False,
        "execution_approved": False
    }

    report_path = output / "docking_report.json"

    report_path.write_text(
        json.dumps(report, indent=2) + "\n"
    )

    print("\nDocking report:", report_path)
    print("Robot execution: NOT APPROVED")
    print("====================================")


if __name__ == "__main__":
    main()
