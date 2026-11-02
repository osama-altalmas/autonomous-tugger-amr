#!/usr/bin/env python3
"""
AMR V1 — Static route-network generator.

Uses the existing Manhattan-Bezier planner.
Validates rectangular swept footprint against the static map.
Generates station-to-station routes and a persistent graph.

OFFLINE ONLY — does not command robot motion.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import plan_routes as planner


# Initial padded robot geometry, metres.
# Must be verified against actual robot and cargo geometry.
HALF_LENGTH = 0.80
HALF_WIDTH = 0.63

# Conservative collision margin in addition to padded dimensions.
EXTRA_MARGIN = 0.025

MAX_TRANSLATION_STEP = 0.025
MAX_YAW_STEP = 0.02


def load_occupancy(map_yaml):
    map_yaml = Path(map_yaml)
    spec = yaml.safe_load(map_yaml.read_text())

    image_file = map_yaml.parent / spec["image"]

    image = np.asarray(
        Image.open(image_file).convert("L"),
        dtype=np.float64
    )

    image = np.flipud(image)

    intensity = image / 255.0

    if spec.get("negate", 0):
        probability = intensity
    else:
        probability = 1.0 - intensity

    # Anything not explicitly classified as free is blocked.
    blocked = probability > float(spec["free_thresh"])

    return spec, blocked


def polygon_at(x, y, yaw):
    length = HALF_LENGTH + EXTRA_MARGIN
    width = HALF_WIDTH + EXTRA_MARGIN

    local = np.array([
        [-length, -width],
        [ length, -width],
        [ length,  width],
        [-length,  width]
    ])

    c = math.cos(yaw)
    s = math.sin(yaw)

    rotation = np.array([
        [c, -s],
        [s,  c]
    ])

    return local @ rotation.T + np.array([x, y])


def world_to_grid(points, spec):
    resolution = float(spec["resolution"])
    origin = spec["origin"]

    ox, oy = origin[:2]
    theta = origin[2] if len(origin) > 2 else 0.0

    delta = points - np.array([ox, oy])

    c = math.cos(theta)
    s = math.sin(theta)

    gx = (c * delta[:, 0] + s * delta[:, 1]) / resolution
    gy = (-s * delta[:, 0] + c * delta[:, 1]) / resolution

    return np.column_stack([gx, gy])



def footprint_is_free(pose, spec, blocked):
    """
    Exact 2D intersection test between the rotated
    rectangular robot footprint and occupied grid cells.

    Uses the Separating Axis Theorem (SAT).
    Unknown cells are treated as occupied.
    """

    polygon = polygon_at(*pose)
    grid_polygon = world_to_grid(polygon, spec)

    height, width = blocked.shape

    # Reject footprints extending beyond the map.
    if (
        np.any(grid_polygon[:, 0] < 0) or
        np.any(grid_polygon[:, 1] < 0) or
        np.any(grid_polygon[:, 0] > width) or
        np.any(grid_polygon[:, 1] > height)
    ):
        return False

    xmin = max(
        0, int(math.floor(grid_polygon[:, 0].min()))
    )
    xmax = min(
        width - 1,
        int(math.ceil(grid_polygon[:, 0].max()))
    )

    ymin = max(
        0, int(math.floor(grid_polygon[:, 1].min()))
    )
    ymax = min(
        height - 1,
        int(math.ceil(grid_polygon[:, 1].max()))
    )

    region = blocked[ymin:ymax + 1, xmin:xmax + 1]
    ys, xs = np.nonzero(region)

    if len(xs) == 0:
        return True

    # Every occupied cell is an axis-aligned unit square
    # in grid coordinates.
    cell_centers = np.column_stack([
        xs + xmin + 0.5,
        ys + ymin + 0.5
    ]).astype(float)

    # SAT axes: X, Y and the two unique rectangle normals.
    edges = np.roll(grid_polygon, -1, axis=0) - grid_polygon

    axes = [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0])
    ]

    for edge in edges[:2]:
        normal = np.array([-edge[1], edge[0]])
        normal /= np.linalg.norm(normal)
        axes.append(normal)

    overlapping = np.ones(
        len(cell_centers), dtype=bool
    )

    for axis in axes:
        projected_polygon = grid_polygon @ axis

        poly_min = projected_polygon.min()
        poly_max = projected_polygon.max()

        projected_centers = cell_centers @ axis

        # Projected half-width of a unit grid square.
        radius = 0.5 * (
            abs(axis[0]) + abs(axis[1])
        )

        cell_min = projected_centers - radius
        cell_max = projected_centers + radius

        separated = (
            (cell_max < poly_min) |
            (cell_min > poly_max)
        )

        overlapping &= ~separated

        if not np.any(overlapping):
            return True

    # Any occupied cell intersecting the footprint
    # invalidates this robot pose.
    return not bool(np.any(overlapping))


def interpolate_trajectory(poses):
    """
    Densely sample positions and orientations.
    Prevent collisions being missed between saved poses.
    """
    if not poses:
        return

    yield poses[0]

    for a, b in zip(poses, poses[1:]):
        dx = b[0] - a[0]
        dy = b[1] - a[1]

        dyaw = math.atan2(
            math.sin(b[2] - a[2]),
            math.cos(b[2] - a[2])
        )

        distance = math.hypot(dx, dy)

        steps = max(
            1,
            math.ceil(distance / MAX_TRANSLATION_STEP),
            math.ceil(abs(dyaw) / MAX_YAW_STEP)
        )

        for i in range(1, steps + 1):
            t = i / steps

            yield [
                a[0] + t * dx,
                a[1] + t * dy,
                a[2] + t * dyaw
            ]


def validate_trajectory(poses, spec, blocked):
    checked = 0

    for pose in interpolate_trajectory(poses):
        checked += 1

        if not footprint_is_free(pose, spec, blocked):
            return {
                "valid": False,
                "checked_poses": checked,
                "failed_pose": [
                    round(float(v), 5) for v in pose
                ]
            }

    return {
        "valid": True,
        "checked_poses": checked,
        "failed_pose": None
    }


def generate_network(args):
    spec, blocked = load_occupancy(args.map)

    stations = json.loads(
        Path(args.stations).read_text()
    )

    if len(stations) < 2:
        raise ValueError("At least two stations required")

    for name, position in stations.items():
        if (
            not isinstance(position, list) or
            len(position) != 2
        ):
            raise ValueError(
                f"Invalid station {name}: expected [x,y]"
            )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    _, safe = planner.load_map(
        Path(args.map),
        args.clearance
    )

    map_yaml = Path(args.map)
    image_path = map_yaml.parent / spec["image"]

    map_hash = hashlib.sha256(
        map_yaml.read_bytes() + image_path.read_bytes()
    ).hexdigest()

    graph = {
        "version": "AMR_V1_PHASE3",
        "mode": "STATIC_MAP_ONLY",
        "validation": "CONSERVATIVE_RECTANGULAR_SWEEP",
        "map_sha256": map_hash,
        "footprint": {
            "half_length_m": HALF_LENGTH,
            "half_width_m": HALF_WIDTH,
            "extra_margin_m": EXTRA_MARGIN
        },
        "stations": stations,
        "routes": {},
        "failures": {}
    }

    names = sorted(stations)

    total = len(names) * (len(names) - 1)

    print(f"Generating {total} directed routes")
    print(f"Stations: {names}")
    print("Static collision checking: ENABLED")
    print("Live sensor dependency: NONE")

    for start in names:
        for goal in names:
            if start == goal:
                continue

            route_id = f"{start}_to_{goal}"

            print(f"\nPlanning {route_id}", flush=True)

            try:
                result = planner.route(
                    stations[start],
                    stations[goal],
                    spec,
                    safe,
                    args.corner_trim,
                    args.max_curvature,
                    args.max_alternatives
                )

                if result.get("status") != \
                        "VALIDATED_STATIC_CLEARANCE":
                    raise RuntimeError(
                        "Unexpected planner validation status"
                    )

                validation = validate_trajectory(
                    result["poses"],
                    spec,
                    blocked
                )

                if not validation["valid"]:
                    raise RuntimeError(
                        "Swept footprint collision at "
                        f"{validation['failed_pose']}"
                    )

                result.update({
                    "start": start,
                    "goal": goal,
                    "map_sha256": map_hash,
                    "swept_footprint": validation,
                    "execution_approved": False,
                    "note": (
                        "Offline geometry validation only. "
                        "Robot execution not yet verified."
                    )
                })

                route_file = output / f"{route_id}.json"

                route_file.write_text(
                    json.dumps(result, indent=2) + "\n"
                )

                graph["routes"][route_id] = {
                    "from": start,
                    "to": goal,
                    "file": route_file.name,
                    "length_m": result["path_length_m"],
                    "poses": len(result["poses"]),
                    "max_curvature": result[
                        "max_curvature_1_per_m"
                    ],
                    "validated": True
                }

                print(
                    f"PASS: {route_id} | "
                    f"{result['path_length_m']} m | "
                    f"{len(result['poses'])} poses | "
                    f"{validation['checked_poses']} "
                    "footprint samples"
                )

            except Exception as exc:
                graph["failures"][route_id] = str(exc)
                print(f"FAIL: {route_id}: {exc}")

    graph_file = output / "route_graph.json"

    graph_file.write_text(
        json.dumps(graph, indent=2) + "\n"
    )

    print("\n========== NETWORK SUMMARY ==========")
    print("Stations:", len(names))
    print("Requested routes:", total)
    print("Validated routes:", len(graph["routes"]))
    print("Failed routes:", len(graph["failures"]))
    print("Database:", graph_file)
    print("=====================================")

    return 0 if not graph["failures"] else 2


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--map", required=True)
    parser.add_argument("--stations", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument(
        "--clearance",
        type=float,
        default=1.05
    )

    parser.add_argument(
        "--corner-trim",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--max-curvature",
        type=float,
        default=2.5
    )

    parser.add_argument(
        "--max-alternatives",
        type=int,
        default=8
    )

    args = parser.parse_args()

    return generate_network(args)


if __name__ == "__main__":
    sys.exit(main())
