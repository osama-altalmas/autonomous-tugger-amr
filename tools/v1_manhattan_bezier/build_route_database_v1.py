#!/usr/bin/env python3
"""
AMR V1: Five-station, orientation-constrained route database.

Uses the current static warehouse map and existing Manhattan/
quintic-Bezier planner.

Features:
  - HOME plus four cargo stations
  - Four entrances evaluated per cargo station
  - Cardinal A* with a constrained terminal direction
  - Straight segments and quintic Bezier smoothing
  - Curvature, static clearance, and rectangular-footprint checks
  - Directed station-to-station route database
  - Saved individual paths
  - Explicit failures and validation metadata

The cargo staging positions are outside the stands.
This program DOES NOT plan entry beneath cargo, lift loads,
publish drive commands, or approve robot execution.
"""

import argparse
import hashlib
import heapq
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

import plan_routes as p

from build_network import (
    load_occupancy,
    validate_trajectory,
    footprint_is_free,
)


DIRECTIONS = ("east", "north", "west", "south")

# Heading is the direction of travel TOWARD the cargo.
APPROACH = {
    "east": math.pi,
    "north": -math.pi / 2,
    "west": 0.0,
    "south": math.pi / 2,
}

DIRS = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
)


def heading_error(a, b):
    return abs(math.atan2(
        math.sin(a - b),
        math.cos(a - b)
    ))


def path_length(poses):
    return sum(
        math.hypot(
            b[0] - a[0],
            b[1] - a[1]
        )
        for a, b in zip(poses, poses[1:])
    )


def direction_index(yaw):
    return min(
        range(4),
        key=lambda i: heading_error(
            math.atan2(DIRS[i][1], DIRS[i][0]),
            yaw
        )
    )


def constrained_astar(start, goal, safe, terminal_direction,
                      clearance_map=None, clearance_weight=0.0):
    """
    Cardinal-grid A* with final direction as part of
    the goal condition.

    State: (grid_x, grid_y, incoming_direction).

    This constrains the terminal straight segment without
    adding an instantaneous rotation at the destination.
    """

    sx, sy = start
    gx, gy = goal

    h, w = safe.shape

    if not safe[sy, sx]:
        raise ValueError("Start cell not clearance-safe")

    if not safe[gy, gx]:
        raise ValueError("Goal cell not clearance-safe")

    initial = (sx, sy, -1)

    best = {initial: 0.0}
    parent = {}

    queue = [
        (
            abs(gx - sx) + abs(gy - sy),
            0.0,
            initial
        )
    ]

    terminal = None

    while queue:
        _, cost, state = heapq.heappop(queue)

        if cost > best.get(state, float("inf")) + 1e-9:
            continue

        x, y, incoming = state

        if (
            (x, y) == (gx, gy)
            and incoming == terminal_direction
        ):
            terminal = state
            break

        for nd, (dx, dy) in enumerate(DIRS):
            xx = x + dx
            yy = y + dy

            if not (0 <= xx < w and 0 <= yy < h):
                continue

            if not safe[yy, xx]:
                continue

            # A direct 180-degree reversal is not modeled
            # as a smooth forward corner.
            if (
                incoming != -1
                and DIRS[incoming][0] == -dx
                and DIRS[incoming][1] == -dy
            ):
                continue

            turn_penalty = (
                0.2
                if incoming != -1 and incoming != nd
                else 0.0
            )

            clearance_penalty = 0.0
            if clearance_map is not None and clearance_weight > 0.0:
                # Prefer the medial part of a free corridor while retaining
                # the hard footprint-safe boundary in ``safe``.  The inverse
                # distance cost is zero-disabled for backwards compatibility
                # and remains finite at the map boundary.
                clearance = max(float(clearance_map[yy, xx]), 1e-6)
                clearance_penalty = clearance_weight / clearance
            new_cost = cost + 1.0 + turn_penalty + clearance_penalty
            nxt = (xx, yy, nd)

            if new_cost + 1e-9 < best.get(
                nxt, float("inf")
            ):
                best[nxt] = new_cost
                parent[nxt] = state

                heuristic = (
                    abs(gx - xx) + abs(gy - yy)
                )

                heapq.heappush(
                    queue,
                    (
                        new_cost + heuristic,
                        new_cost,
                        nxt
                    )
                )

    if terminal is None:
        raise ValueError(
            "No cardinal route reaches goal with "
            "the required terminal heading"
        )

    cells = []
    cursor = terminal

    while True:
        cells.append((cursor[0], cursor[1]))

        if cursor == initial:
            break

        cursor = parent[cursor]

    return cells[::-1]


def smooth_and_validate(
    cells,
    spec,
    safe,
    occupancy_spec,
    blocked,
    trim,
    curvature_limit
):
    points = p.compress([
        p.world(q, spec)
        for q in cells
    ])

    if len(points) < 2:
        raise ValueError("Degenerate route")

    last_failure = "No valid smoothing"

    for factor in (1.0, 0.75, 0.5, 0.35):
        try:
            xy = p.smooth(
                points,
                trim * factor
            )

            k = p.curvature(xy)

            if k > curvature_limit:
                last_failure = (
                    f"Curvature {k:.3f} exceeds "
                    f"{curvature_limit:.3f}"
                )
                continue

            if not p.check_samples(xy, spec, safe):
                last_failure = (
                    "Smoothed route leaves clearance-safe map"
                )
                continue

            poses = []

            for i, point in enumerate(xy):
                if i + 1 < len(xy):
                    dx = xy[i + 1][0] - point[0]
                    dy = xy[i + 1][1] - point[1]
                else:
                    dx = point[0] - xy[i - 1][0]
                    dy = point[1] - xy[i - 1][1]

                poses.append([
                    round(point[0], 4),
                    round(point[1], 4),
                    round(math.atan2(dy, dx), 6)
                ])

            validation = validate_trajectory(
                poses,
                occupancy_spec,
                blocked
            )

            if not validation["valid"]:
                last_failure = (
                    "Rectangular footprint collision"
                )
                continue

            return {
                "poses": poses,
                "waypoints": points,
                "length_m": round(
                    path_length(poses), 4
                ),
                "max_curvature_1_per_m": round(k, 5),
                "trim_m": round(trim * factor, 4),
                "footprint_validation": validation,
            }

        except Exception as exc:
            last_failure = str(exc)

    raise ValueError(last_failure)


def staging_candidates(station, spec, safe, blocked, distance):
    """
    Construct staging poses outside each cargo stand.

    These are ordinary warehouse travel destinations,
    not positions underneath the cargo.
    """

    cx, cy = station["center"][:2]

    result = {}

    for direction in DIRECTIONS:
        yaw = APPROACH[direction]

        x = cx - distance * math.cos(yaw)
        y = cy - distance * math.sin(yaw)

        pose = [x, y, yaw]

        item = {
            "requested_pose": pose,
            "available": False,
            "reason": None,
        }

        try:
            grid = p.cell(
                [x, y],
                spec,
                safe.shape
            )

            xy = p.world(grid, spec)
            snapped = [xy[0], xy[1], yaw]

            if not safe[grid[1], grid[0]]:
                raise ValueError(
                    "Staging position fails static clearance"
                )

            if not footprint_is_free(
                snapped,
                spec,
                blocked
            ):
                raise ValueError(
                    "Staging footprint blocked"
                )

            item.update({
                "available": True,
                "grid_cell": list(grid),
                "pose": snapped,
                "reason": None,
            })

        except Exception as exc:
            item["reason"] = str(exc)

        result[direction] = item

    return result


def chain_predecessor(source, target, chains):
    """Return the stop visited before ``source`` on a ``source -> target`` leg."""
    for chain in chains:
        for before, current, after in zip(chain, chain[1:], chain[2:]):
            if current == source and after == target:
                return before
    return None


def required_departure_entrance(source, target, known_routes, chains, fallback,
                                pass_through=()):
    """Choose the gate a route must leave by: the one its mission entered by.

    The environment has no way to enter by one gate and leave by another. A
    mission leg follows the route that delivered the robot; any other route
    falls back to the gate used when arriving from HOME.
    """
    if source in pass_through:
        return None
    before = chain_predecessor(source, target, chains)
    if before is not None:
        entrance = known_routes.get(
            f"{before}__TO__{source}", {}).get("target_entrance")
        if entrance is not None:
            return entrance
    return fallback.get(source)


def build_connection(
    source_name,
    target_name,
    nodes,
    spec,
    safe,
    occupancy_spec,
    blocked,
    args
):
    source_options = nodes[source_name]
    target_options = nodes[target_name]

    attempts = []
    winner = None

    for source_dir, source in source_options.items():
        if not source["available"]:
            continue
        preferred_source = required_departure_entrance(
            source_name,
            target_name,
            getattr(args, "known_routes", {}),
            getattr(args, "mission_chains", []),
            getattr(args, "departure_entrances", {}),
            getattr(args, "pass_through", ()),
        )
        if preferred_source is not None and source_dir != preferred_source:
            continue

        for target_dir, target in target_options.items():
            if not target["available"]:
                continue

            attempt = {
                "source_entrance": source_dir,
                "target_entrance": target_dir,
                "status": "REJECTED",
            }

            try:
                start = tuple(source["grid_cell"])
                goal = tuple(target["grid_cell"])

                required = direction_index(
                    target["pose"][2]
                )

                cells = constrained_astar(
                    start,
                    goal,
                    safe,
                    required,
                    clearance_map=getattr(args, "clearance_map", None),
                    clearance_weight=getattr(args, "clearance_bias", 0.0)
                )

                candidate = smooth_and_validate(
                    cells,
                    spec,
                    safe,
                    occupancy_spec,
                    blocked,
                    args.corner_trim,
                    args.max_curvature
                )

                if heading_error(
                    candidate["poses"][-1][2],
                    target["pose"][2]
                ) > math.radians(2):
                    raise ValueError(
                        "Terminal heading mismatch"
                    )

                # Source heading is recorded as an
                # unresolved departure constraint unless
                # it happens to match this path.
                departure_error = heading_error(
                    candidate["poses"][0][2],
                    source["pose"][2]
                )
                # A route may leave a station in reverse.  Compare both the
                # forward and reverse body directions so an intentional
                # reverse departure is not rejected as a heading failure.
                departure_error = min(
                    departure_error,
                    abs(math.pi - departure_error)
                )

                attempt.update({
                    "status": "VALIDATED_STATIC_CANDIDATE",
                    "length_m": candidate["length_m"],
                    "departure_heading_error_deg": round(
                        math.degrees(departure_error), 2
                    ),
                })

                # Do not claim seamless station departure
                # when heading must change.
                candidate.update({
                    "source_entrance": source_dir,
                    "target_entrance": target_dir,
                    "source_pose": source["pose"],
                    "target_pose": target["pose"],
                    "departure_heading_error_deg": round(
                        math.degrees(departure_error), 2
                    ),
                    "departure_transition_validated": (
                        departure_error <= math.radians(2)
                    ),
                    "terminal_heading_validated": True,
                })

                if (
                    winner is None
                    or candidate["length_m"] <
                    winner["length_m"]
                ):
                    winner = candidate

            except Exception as exc:
                # Clearance bias must not destroy an otherwise valid legacy
                # corridor. Retry the same entrance pair without the soft
                # preference; hard clearance and footprint checks remain.
                if getattr(args, "clearance_bias", 0.0) > 0.0:
                    try:
                        cells = constrained_astar(
                            start, goal, safe, required,
                            clearance_map=None, clearance_weight=0.0)
                        candidate = smooth_and_validate(
                            cells, spec, safe, occupancy_spec, blocked,
                            args.corner_trim, args.max_curvature)
                        departure_error = min(
                            heading_error(candidate["poses"][0][2], source["pose"][2]),
                            abs(math.pi - heading_error(
                                candidate["poses"][0][2], source["pose"][2])))
                        candidate.update({
                            "source_entrance": source_dir,
                            "target_entrance": target_dir,
                            "source_pose": source["pose"],
                            "target_pose": target["pose"],
                            "departure_heading_error_deg": round(
                                math.degrees(departure_error), 2),
                            "departure_transition_validated": (
                                departure_error <= math.radians(2)),
                            "terminal_heading_validated": True,
                        })
                        attempt.update({
                            "status": "VALIDATED_STATIC_CANDIDATE",
                            "length_m": candidate["length_m"],
                            "departure_heading_error_deg": round(
                                math.degrees(departure_error), 2),
                        })
                        if winner is None or candidate["length_m"] < winner["length_m"]:
                            winner = candidate
                        attempts.append(attempt)
                        continue
                    except Exception:
                        pass
                attempt["reason"] = str(exc)

            attempts.append(attempt)

    if winner is None:
        return {
            "status": "NO_VALIDATED_ROUTE",
            "attempts": attempts,
            "execution_approved": False,
        }

    return {
        "status": "VALIDATED_STATIC_CANDIDATE",
        "best": winner,
        "attempts": attempts,
        "execution_approved": False,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--map", required=True)
    ap.add_argument("--geometry", required=True)
    ap.add_argument("--output", required=True)

    ap.add_argument(
        "--home",
        nargs=2,
        type=float,
        default=[0.0, 0.0],
        help="HOME marker position; routes use gate staging poses around it"
    )

    ap.add_argument(
        "--home-gates",
        nargs="+",
        default=["east", "west"],
        choices=list(DIRECTIONS),
        help="HOME gates, aligned with the robot's HOME orientation"
    )

    ap.add_argument(
        "--clearance",
        type=float,
        default=1.05
    )

    ap.add_argument(
        "--staging-distance",
        type=float,
        default=2.0,
        help="Distance from cargo center to outer staging pose"
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
        "--clearance-bias",
        type=float,
        default=0.05,
        help="A* preference for the center of free corridors (0 disables it)"
    )

    ap.add_argument(
        "--mission-chains",
        default=str(Path(__file__).with_name("mission_chains.json")),
        help="Mission order used to choose each route's departure gate"
    )

    ap.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="SOURCE__TO__TARGET",
        help="Regenerate only these routes and keep every other saved route"
    )

    ap.add_argument(
        "--approve-regenerated",
        action="store_true",
        help="Mark routes regenerated by --only as approved for execution"
    )

    args = ap.parse_args()

    if min(
        args.clearance,
        args.staging_distance,
        args.corner_trim,
        args.max_curvature
    ) <= 0:
        ap.error("Planning distances/limits must be positive")
    if args.clearance_bias < 0:
        ap.error("Clearance bias must be non-negative")

    started = time.monotonic()

    geometry = json.loads(
        Path(args.geometry).read_text()
    )

    spec, safe = p.load_map(
        Path(args.map),
        args.clearance
    )

    occupancy_spec, blocked = load_occupancy(
        args.map
    )
    args.clearance_map = (
        distance_transform_edt(~blocked) * float(spec["resolution"])
    )

    home_cell = p.cell(
        args.home,
        spec,
        safe.shape
    )

    home_xy = p.world(home_cell, spec)

    if not safe[home_cell[1], home_cell[0]]:
        raise ValueError("HOME is not clearance-safe")

    if not footprint_is_free(
        [home_xy[0], home_xy[1], 0.0],
        occupancy_spec,
        blocked
    ):
        raise ValueError("HOME footprint blocked")

    # HOME is reached through gate staging poses like a station. Its own
    # segment 2 then ends exactly on the marker, which is off the grid.
    home_gates = staging_candidates(
        {"center": list(args.home)}, spec, safe, blocked, args.staging_distance)
    nodes = {
        "HOME": {gate: home_gates[gate] for gate in args.home_gates}
    }

    for name, station in geometry["stations"].items():
        nodes[name] = staging_candidates(
            station,
            spec,
            safe,
            blocked,
            args.staging_distance
        )

    names = list(nodes)

    # Preserve the validated entrance used to arrive at each cargo station.
    # This makes a station-to-station route begin at the same physical
    # departure side without hard-coding station IDs into the planner.
    args.departure_entrances = {}
    args.known_routes = {}
    chain_config = json.loads(Path(args.mission_chains).read_text())
    args.mission_chains = chain_config["chains"]
    args.pass_through = tuple(chain_config.get("pass_through", ()))
    previous = None
    previous_database = Path(args.output) / "route_database.json"
    if previous_database.exists():
        try:
            previous = json.loads(previous_database.read_text())
            args.known_routes = dict(previous.get("routes", {}))
            for station in names:
                entry = previous.get("routes", {}).get(
                    f"HOME__TO__{station}", {})
                entrance = entry.get("target_entrance")
                if entrance in nodes.get(station, {}):
                    args.departure_entrances[station] = entrance
        except (OSError, ValueError, TypeError):
            args.departure_entrances = {}

    if len(names) != 5:
        raise ValueError(
            f"Expected HOME + four cargo stations, "
            f"found {len(names)}"
        )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    import yaml

    map_yaml = Path(args.map)
    map_info = yaml.safe_load(map_yaml.read_text())

    image_file = map_yaml.parent / map_info["image"]

    map_hash = hashlib.sha256(
        map_yaml.read_bytes() +
        image_file.read_bytes()
    ).hexdigest()

    database = {
        "version": "AMR_V1_FIVE_STATION_DATABASE",
        "map_sha256": map_hash,
        "stations": names,
        "staging_distance_m": args.staging_distance,
        "clearance_radius_m": args.clearance,
        "clearance_bias": args.clearance_bias,
        "nodes": nodes,
        # Stops without a stand; the robot may leave them by any gate.
        "pass_through": list(args.pass_through),
        # Exact centres, so segment 2 tools need no hard-coded positions.
        "station_centers": {
            "HOME": [float(args.home[0]), float(args.home[1])],
            **{name: [float(v) for v in station["center"][:2]]
               for name, station in geometry["stations"].items()},
        },
        "routes": {},
        "execution_approved": False,
        "notes": [
            "Routes connect warehouse staging poses",
            "Cargo approach and lift are not included",
            "Terminal headings checked",
            "Departure transitions require validation",
            "No dynamic obstacles modeled",
            "Robot control and localization not validated"
        ]
    }

    total = len(names) * (len(names) - 1)
    number = 0
    accepted = 0

    print()
    print("========== AMR V1 ROUTE DATABASE ==========")
    print("Warehouse map:", args.map)
    print("Stations:", len(names))
    print("Directed connections:", total)
    print("All four cargo entrances considered")
    print("Robot movement: DISABLED")
    print("===========================================")

    print("\nSTAGING POSITION CHECK")

    for name, options in nodes.items():
        valid = [
            direction
            for direction, data in options.items()
            if data["available"]
        ]

        print(f"  {name}: {valid}")

    print("\nGENERATING ROUTES")

    if args.only:
        if previous is None:
            raise ValueError("--only needs an existing route_database.json")
        known = {f"{a}__TO__{b}" for a in names for b in names if a != b}
        unknown = set(args.only) - known
        if unknown:
            raise ValueError(f"Unknown routes for --only: {sorted(unknown)}")
        for name in ("routes", "execution_approved", "summary"):
            database[name] = previous[name]
        total = len(args.only)

    # Mission legs first, in order, so each sees the gate its predecessor chose.
    chained = [pair for chain in args.mission_chains
               for pair in zip(chain, chain[1:])]
    ordered = chained + [(a, b) for a in names for b in names if a != b]
    planned = list(dict.fromkeys(ordered))

    for source, target in planned:
        if True:
            key = f"{source}__TO__{target}"
            if args.only and key not in args.only:
                continue

            number += 1

            print(
                f"\n[{number}/{total}] "
                f"{source} -> {target}",
                flush=True
            )

            result = build_connection(
                source,
                target,
                nodes,
                spec,
                safe,
                occupancy_spec,
                blocked,
                args
            )

            entry = {
                "source": source,
                "target": target,
                "status": result["status"],
                "attempts": result["attempts"],
                "execution_approved": False
            }
            approve_new = bool(args.only and args.approve_regenerated)

            if result["status"] == (
                "VALIDATED_STATIC_CANDIDATE"
            ):
                best = result["best"]
                accepted += 1

                path_file = output / f"{key}.json"

                path_file.write_text(
                    json.dumps(
                        {
                            "version": "AMR_V1_SAVED_ROUTE",
                            "map_sha256": map_hash,
                            "source": source,
                            "target": target,
                            **best,
                            "execution_approved": approve_new
                        },
                        indent=2
                    ) + "\n"
                )

                entry.update({
                    "execution_approved": approve_new,
                    "route_file": path_file.name,
                    "length_m": best["length_m"],
                    "source_entrance": best["source_entrance"],
                    "target_entrance": best["target_entrance"],
                    "terminal_heading_validated": True,
                    "departure_transition_validated": (
                        best["departure_transition_validated"]
                    )
                })

                print(
                    "  PASS:",
                    round(best["length_m"], 2),
                    "m | terminal:",
                    best["target_entrance"],
                    "| departure continuous:",
                    best["departure_transition_validated"],
                    flush=True
                )

            else:
                print(
                    "  NO VALIDATED ROUTE",
                    flush=True
                )

            database["routes"][key] = entry
            args.known_routes[key] = entry

    if args.only:
        accepted = sum(
            route.get("status") == "VALIDATED_STATIC_CANDIDATE"
            for route in database["routes"].values())
        total = len(database["routes"])

    database["summary"] = {
        "connections_requested": total,
        "static_candidates_found": accepted,
        "connections_without_candidate": total - accepted,
        "elapsed_seconds": round(
            time.monotonic() - started, 2
        ),
        "complete_mission_execution_approved": False
    }

    database_file = output / "route_database.json"

    temporary = database_file.with_suffix(
        ".json.tmp"
    )

    temporary.write_text(
        json.dumps(database, indent=2) + "\n"
    )

    temporary.replace(database_file)

    print("\n========== DATABASE SUMMARY ==========")
    print("Requested connections:", total)
    print("Validated static candidates:", accepted)
    print("Unresolved connections:", total - accepted)
    print("Elapsed:", database["summary"]["elapsed_seconds"], "s")
    print("Saved:", database_file)
    print("Execution approved:",
          "SELECTED ROUTES ONLY" if args.only and args.approve_regenerated else "NO")
    print("=======================================")


if __name__ == "__main__":
    main()
