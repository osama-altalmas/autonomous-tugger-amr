#!/usr/bin/env python3
"""
AMR V1 — Map-independent, multi-stop Manhattan-Bezier mission planner.

Input:
    - Nav2-compatible occupancy-map YAML
    - JSON file of named stations
    - Ordered sequence of station names

Output:
    - Complete mission path
    - Individual route legs
    - Validation results
    - Map fingerprint

OFFLINE ONLY. No robot commands.
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


def angular_error(a, b):
    return abs(math.atan2(
        math.sin(a - b),
        math.cos(a - b)
    ))


def normalize_station(name, value):
    """
    Accept either:
        "HOME": [x, y]

    or:
        "HOME": {
            "position": [x, y],
            "type": "home",
            "yaw": 1.570796
        }

    A station yaw is optional.
    If specified, it becomes a mandatory arrival constraint.
    """

    if isinstance(value, list):
        value = {"position": value}

    if not isinstance(value, dict):
        raise ValueError(
            f"{name}: station must be an object or [x,y]"
        )

    position = value.get("position")

    if not isinstance(position, list) or len(position) != 2:
        raise ValueError(
            f"{name}: position must contain [x,y]"
        )

    xy = [float(position[0]), float(position[1])]

    if not all(math.isfinite(v) for v in xy):
        raise ValueError(f"{name}: nonfinite coordinates")

    yaw = value.get("yaw")

    if yaw is not None:
        yaw = float(yaw)
        if not math.isfinite(yaw):
            raise ValueError(f"{name}: nonfinite yaw")

    return {
        "position": xy,
        "type": value.get("type", "waypoint"),
        "yaw": yaw,
    }


def map_fingerprint(map_path):
    import yaml

    map_path = Path(map_path)
    spec = yaml.safe_load(map_path.read_text())
    image = map_path.parent / spec["image"]

    return hashlib.sha256(
        map_path.read_bytes() + image.read_bytes()
    ).hexdigest()


def path_length(poses):
    return sum(
        math.hypot(
            b[0] - a[0],
            b[1] - a[1]
        )
        for a, b in zip(poses, poses[1:])
    )


def validate_station(
    name, station, map_spec, blocked
):
    x, y = station["position"]

    # A requested docking heading is tested exactly.
    # Without a requested heading, check conservative
    # cardinal orientations at the station.
    yaw = station["yaw"]

    if yaw is not None:
        orientations = [yaw]
    else:
        orientations = [
            0.0,
            math.pi / 2,
            math.pi,
            -math.pi / 2,
        ]

    valid = [
        angle
        for angle in orientations
        if footprint_is_free(
            [x, y, angle],
            map_spec,
            blocked
        )
    ]

    if not valid:
        raise ValueError(
            f"Station {name} has no verified "
            "footprint-clear orientation"
        )


def plan_leg(
    start_name,
    goal_name,
    stations,
    spec,
    safe,
    occupancy_spec,
    blocked,
    args
):
    start = stations[start_name]
    goal = stations[goal_name]

    result = planner.route(
        start["position"],
        goal["position"],
        spec,
        safe,
        args.corner_trim,
        args.max_curvature,
        args.max_alternatives,
    )

    if result.get("status") != "VALIDATED_STATIC_CLEARANCE":
        raise ValueError("Underlying planner rejected path")

    poses = result["poses"]

    if len(poses) < 2:
        raise ValueError("Route has fewer than two poses")

    # Reject XY endpoint discrepancies.
    for actual, expected in (
        (poses[0], start["position"]),
        (poses[-1], goal["position"]),
    ):
        if math.dist(actual[:2], expected) > 0.005:
            raise ValueError("Route endpoint discrepancy")

    # Enforce requested departure/arrival heading.
    # No in-place rotation is silently inserted.
    tolerance = math.radians(
        args.yaw_tolerance_deg
    )

    if start["yaw"] is not None:
        error = angular_error(
            poses[0][2], start["yaw"]
        )

        if error > tolerance:
            raise ValueError(
                "Departure heading mismatch: "
                f"{math.degrees(error):.1f} degrees"
            )

    if goal["yaw"] is not None:
        error = angular_error(
            poses[-1][2], goal["yaw"]
        )

        if error > tolerance:
            raise ValueError(
                "Arrival heading mismatch: "
                f"{math.degrees(error):.1f} degrees"
            )

    # Verify the complete sampled rectangular footprint.
    validation = validate_trajectory(
        poses,
        occupancy_spec,
        blocked
    )

    if not validation["valid"]:
        raise ValueError(
            "Rectangular footprint collision at "
            f"{validation['failed_pose']}"
        )

    return {
        "from": start_name,
        "to": goal_name,
        "status": "STATIC_ROUTE_VALIDATED",
        "poses": poses,
        "waypoints": result["waypoints"],
        "length_m": round(path_length(poses), 4),
        "maximum_curvature": result[
            "max_curvature_1_per_m"
        ],
        "search_attempts": result.get(
            "search_attempts", 1
        ),
        "swept_footprint": validation,
        "execution_approved": False,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__
    )

    ap.add_argument("--map", required=True)
    ap.add_argument("--stations", required=True)

    ap.add_argument(
        "--stops",
        nargs="+",
        required=True,
        help="Ordered station IDs"
    )

    ap.add_argument("--output", required=True)

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

    ap.add_argument(
        "--yaw-tolerance-deg",
        type=float,
        default=5.0
    )

    args = ap.parse_args()

    if len(args.stops) < 2:
        ap.error("Specify at least two stops")

    if (
        args.clearance <= 0 or
        args.corner_trim <= 0 or
        args.max_curvature <= 0 or
        args.yaw_tolerance_deg < 0
    ):
        ap.error("Invalid planning parameters")

    raw_stations = json.loads(
        Path(args.stations).read_text()
    )

    stations = {
        name: normalize_station(name, value)
        for name, value in raw_stations.items()
    }

    for name in args.stops:
        if name not in stations:
            ap.error(f"Unknown station: {name}")

    print("\n========== AMR V1 MISSION ==========")
    print("Map:", args.map)
    print("Stops:", " -> ".join(args.stops))
    print("Stations registered:", len(stations))
    print("Static collision checking: ENABLED")
    print("Robot motion: DISABLED")
    print("====================================")

    spec, safe = planner.load_map(
        Path(args.map),
        args.clearance
    )

    occupancy_spec, blocked = load_occupancy(
        args.map
    )

    # Verify all named locations before route generation.
    for name in set(args.stops):
        validate_station(
            name,
            stations[name],
            occupancy_spec,
            blocked
        )

    legs = []
    combined = []

    for index, (start, goal) in enumerate(
        zip(args.stops, args.stops[1:]),
        start=1
    ):
        print(
            f"\nPlanning leg {index}: "
            f"{start} -> {goal}",
            flush=True
        )

        if start == goal:
            raise ValueError(
                "Consecutive identical stations are "
                "not supported"
            )

        leg = plan_leg(
            start,
            goal,
            stations,
            spec,
            safe,
            occupancy_spec,
            blocked,
            args
        )

        # Preserve each leg's boundary pose.
        # Do not assume that changing heading between
        # mission legs is physically possible.
        if combined:
            previous = combined[-1]
            following = leg["poses"][0]

            if math.dist(
                previous[:2],
                following[:2]
            ) > 0.005:
                raise ValueError(
                    "Discontinuous mission position"
                )

            difference = angular_error(
                previous[2], following[2]
            )

            if difference > math.radians(
                args.yaw_tolerance_deg
            ):
                print(
                    "  NOTE: Heading changes between "
                    "station visits; mission requires a "
                    "separate verified transition."
                )

        combined.extend(leg["poses"])

        legs.append(leg)

        print(
            f"  PASS: {leg['length_m']:.2f} m | "
            f"{len(leg['poses'])} poses | "
            f"curvature {leg['maximum_curvature']:.3f}"
        )

    # Detect heading discontinuities between legs.
    heading_discontinuities = []

    offset = 0

    for i, leg in enumerate(legs[:-1]):
        offset += len(leg["poses"])

        before = combined[offset - 1]
        after = combined[offset]

        error = angular_error(
            before[2], after[2]
        )

        if error > math.radians(
            args.yaw_tolerance_deg
        ):
            heading_discontinuities.append({
                "after_leg": i + 1,
                "station": args.stops[i + 1],
                "heading_change_deg": round(
                    math.degrees(error), 3
                )
            })

    # Each leg is geometrically validated, but a
    # multi-leg mission with heading discontinuities
    # is not a continuous driveable trajectory.
    continuous = not heading_discontinuities

    output = {
        "version": "AMR_V1_MULTI_STOP_1",
        "mode": "OFFLINE_STATIC_MAP",
        "map_sha256": map_fingerprint(args.map),
        "map_file": str(Path(args.map).resolve()),
        "stops": args.stops,
        "stations": {
            name: stations[name]
            for name in set(args.stops)
        },
        "legs": legs,
        "poses": combined,
        "total_length_m": round(
            sum(leg["length_m"] for leg in legs), 4
        ),
        "individual_legs_validated": True,
        "continuous_heading": continuous,
        "heading_discontinuities": heading_discontinuities,
        "complete_mission_executable": False,
        "execution_approved": False,
        "limitations": [
            "Offline static map only",
            "Bounded alternative-corridor search",
            "Sampled footprint validation",
            "No dynamic obstacle handling",
            "No controller feasibility proof",
            "No automatic station turn-around maneuver",
            "No cargo lifting or docking execution"
        ]
    }

    destination = Path(args.output)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Atomic replacement: only save a complete
    # successfully generated mission.
    temporary = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(output, indent=2) + "\n"
    )

    temporary.replace(destination)

    print("\n========== MISSION SUMMARY ==========")
    print("Legs generated:", len(legs))
    print("Total length:", output["total_length_m"], "m")
    print("Individual legs validated: YES")
    print("Continuous heading:", continuous)
    print(
        "Heading discontinuities:",
        len(heading_discontinuities)
    )
    print("Saved:", destination)
    print("Execution approved: NO")
    print("=====================================")


if __name__ == "__main__":
    main()
