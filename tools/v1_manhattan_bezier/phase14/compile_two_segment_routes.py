#!/usr/bin/env python3
"""
AMR Phase 14 — Two-segment station approach compiler.

Segment 1:
    Saved warehouse route -> 1 m predocking checkpoint.

Checkpoint:
    Mandatory stop and alignment verification.

Segment 2:
    Predocking checkpoint -> exact cargo center.

This tool plans and checks geometry ONLY.
It never sends robot commands.

The 2D leg checks are necessary but not sufficient:
3D tabletop/lift clearance and actual steering correction
must also be validated before motion is authorized.
"""

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml


STATIONS = [
    "cargo_station_0",
    "cargo_station_3",
    "cargo_dropoff_northwest",
    "cargo_dropoff_southeast",
]

# Conservative body envelope based on the current URDF.
#
# Chassis: 0.88 x 0.58 m
# Wheels extend approximately to x = +/- 0.345 m,
# y = +/- 0.295 m.
#
# The enlarged rectangle is deliberately conservative.
# It does not replace full articulated-wheel checking.

HALF_LENGTH = 0.465
HALF_WIDTH = 0.315

SAMPLE_STEP = 0.01
PREDOCK_DISTANCE = 1.0
MAX_LATERAL_ERROR = 0.05
MAX_HEADING_ERROR = math.radians(5.0)


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def pose_values(text):
    if not text:
        return [0.0] * 6

    values = [float(x) for x in text.split()]

    if len(values) != 6:
        raise ValueError("Expected six values in SDF pose")

    return values


def compose_2d(parent, child):
    px, py, pyaw = parent
    cx, cy, cyaw = child

    c = math.cos(pyaw)
    s = math.sin(pyaw)

    return (
        px + c * cx - s * cy,
        py + s * cx + c * cy,
        wrap(pyaw + cyaw),
    )


def xy_yaw(element):
    p = pose_values(
        element.findtext("pose")
        if element is not None
        else None
    )

    return p[0], p[1], p[5]


def world_models(world_path):
    root = ET.parse(world_path).getroot()
    return {
        model.get("name"): model
        for model in root.findall(".//model")
        if model.get("name")
    }


def load_station_geometry(models, station):
    name = station + "_legs"

    if name not in models:
        raise ValueError(
            f"Cannot find Gazebo model {name}"
        )

    model = models[name]
    mx, my, myaw = xy_yaw(model)

    legs = []

    for link in model.findall("link"):
        link_pose = xy_yaw(link)

        for collision in link.findall("collision"):
            box = collision.find("geometry/box/size")

            if box is None or not box.text:
                continue

            size = [
                float(v)
                for v in box.text.split()
            ]

            if len(size) != 3:
                raise ValueError(
                    "Invalid stand-leg collision size"
                )

            collision_pose = xy_yaw(collision)

            local = compose_2d(
                link_pose,
                collision_pose
            )

            x, y, yaw = compose_2d(
                (mx, my, myaw),
                local
            )

            legs.append({
                "name": collision.get("name"),
                "x": x,
                "y": y,
                "yaw": yaw,
                "size": size,
            })

    if len(legs) != 4:
        raise ValueError(
            f"{station}: expected four collision legs, "
            f"found {len(legs)}"
        )

    return {
        "center": [mx, my],
        "yaw": myaw,
        "legs": legs,
    }


def rectangle(x, y, yaw, hx, hy):
    c = math.cos(yaw)
    s = math.sin(yaw)

    return [
        (
            x + c * dx - s * dy,
            y + s * dx + c * dy
        )
        for dx, dy in [
            (-hx, -hy),
            (hx, -hy),
            (hx, hy),
            (-hx, hy),
        ]
    ]


def overlaps(a, b):
    # Separating-axis theorem for convex rectangles.

    for polygon in (a, b):
        for i in range(len(polygon)):
            p = polygon[i]
            q = polygon[(i + 1) % len(polygon)]

            axis = (
                -(q[1] - p[1]),
                q[0] - p[0]
            )

            av = [
                x * axis[0] + y * axis[1]
                for x, y in a
            ]

            bv = [
                x * axis[0] + y * axis[1]
                for x, y in b
            ]

            if (
                max(av) < min(bv)
                or max(bv) < min(av)
            ):
                return False

    return True


def check_legs(poses, legs):
    collisions = []

    leg_rectangles = []

    for leg in legs:
        sx, sy, _ = leg["size"]

        leg_rectangles.append(
            rectangle(
                leg["x"],
                leg["y"],
                leg["yaw"],
                sx / 2.0,
                sy / 2.0
            )
        )

    for i, pose in enumerate(poses):
        x, y, yaw = pose

        robot = rectangle(
            x, y, yaw,
            HALF_LENGTH,
            HALF_WIDTH
        )

        for j, leg in enumerate(leg_rectangles):
            if overlaps(robot, leg):
                collisions.append({
                    "pose_index": i,
                    "leg_index": j,
                    "robot_pose": pose,
                })

    return collisions


def linear_poses(start, finish):
    length = distance(start, finish)

    if length < 1e-9:
        return [list(start)]

    count = max(
        1,
        math.ceil(length / SAMPLE_STEP)
    )

    heading = math.atan2(
        finish[1] - start[1],
        finish[0] - start[0]
    )

    result = []

    for i in range(count + 1):
        t = i / count

        result.append([
            start[0] + t * (finish[0] - start[0]),
            start[1] + t * (finish[1] - start[1]),
            heading,
        ])

    return result


def hash_map(yaml_path):
    path = Path(yaml_path)
    info = yaml.safe_load(path.read_text())

    image = path.parent / info["image"]

    return hashlib.sha256(
        path.read_bytes() + image.read_bytes()
    ).hexdigest()


def route_length(poses):
    return sum(
        distance(a, b)
        for a, b in zip(poses, poses[1:])
    )


def compile_station(args, station, models, db):
    key = f"HOME__TO__{station}"

    entry = db["routes"].get(key)

    if entry is None:
        raise ValueError(
            "No saved HOME route: " + key
        )

    if entry.get("status") != "VALIDATED_STATIC_CANDIDATE":
        raise ValueError(
            "Saved warehouse route is not validated"
        )

    route_file = (
        Path(args.database).parent
        / entry["route_file"]
    )

    route = json.loads(route_file.read_text())

    if (
        route.get("source") != "HOME"
        or route.get("target") != station
    ):
        raise ValueError(
            "Saved route identity mismatch"
        )

    if route.get("map_sha256") != args.map_hash:
        raise ValueError(
            "Map changed since route generation"
        )

    if route.get("motion_direction", "FORWARD") != "FORWARD":
        raise ValueError(
            "This first docking compiler requires "
            "a forward-entry warehouse route"
        )

    warehouse = [
        [float(v) for v in pose]
        for pose in route["poses"]
    ]

    if len(warehouse) < 2:
        raise ValueError(
            "Warehouse route has too few poses"
        )

    geometry = load_station_geometry(
        models,
        station
    )

    cx, cy = geometry["center"]

    staging = warehouse[-1]

    dx = cx - staging[0]
    dy = cy - staging[1]

    approach_length = math.hypot(dx, dy)

    if approach_length <= PREDOCK_DISTANCE:
        raise ValueError(
            "Existing staging position is already "
            "inside the proposed 1 m checkpoint"
        )

    if approach_length > 4.0:
        raise ValueError(
            "Unexpected staging-to-center distance "
            f"of {approach_length:.3f} m"
        )

    entry_heading = math.atan2(dy, dx)

    heading_error = abs(
        wrap(staging[2] - entry_heading)
    )

    if heading_error > MAX_HEADING_ERROR:
        raise ValueError(
            "Existing staging heading does not "
            "match the precision-entry direction: "
            f"{math.degrees(heading_error):.2f} deg"
        )

    # Derive a nominal 1 m checkpoint on the
    # centerline of the cargo entry.
    ux = math.cos(entry_heading)
    uy = math.sin(entry_heading)

    predock = [
        cx - PREDOCK_DISTANCE * ux,
        cy - PREDOCK_DISTANCE * uy,
        entry_heading
    ]

    # The old route must connect to the checkpoint.
    # Small lateral differences can be corrected by
    # this planned lead-in, before the stand entry.
    lateral_error = abs(
        -uy * (staging[0] - cx)
        + ux * (staging[1] - cy)
    )

    if lateral_error > MAX_LATERAL_ERROR:
        raise ValueError(
            "Excessive staging lateral error: "
            f"{lateral_error:.3f} m"
        )

    staging_extension = linear_poses(
        staging,
        predock
    )

    # Precision entry is straight and aligned.
    center_pose = [
        cx,
        cy,
        entry_heading
    ]

    precision_entry = linear_poses(
        predock,
        center_pose
    )

    # Ensure that Segment 1 retains all original
    # warehouse samples and appends the new lead-in.
    segment_1 = (
        warehouse
        + staging_extension[1:]
    )

    segment_2 = precision_entry

    complete = (
        segment_1
        + segment_2[1:]
    )

    # Verify continuity at the segment boundary.
    if distance(
        segment_1[-1],
        segment_2[0]
    ) > 1e-6:
        raise ValueError(
            "Segment boundary is discontinuous"
        )

    # Verify final position.
    if distance(
        segment_2[-1],
        center_pose
    ) > 1e-6:
        raise ValueError(
            "Precision entry does not reach center"
        )

    # Check all geometry along the newly generated
    # approach, not only the endpoint.
    approach = (
        staging_extension
        + precision_entry[1:]
    )

    collisions = check_legs(
        approach,
        geometry["legs"]
    )

    status = (
        "2D_LEG_GEOMETRY_PASSED"
        if not collisions
        else "REJECTED_LEG_COLLISION"
    )

    result = {
        "schema": "amr_phase14_two_segment_v1",
        "source": "HOME",
        "destination": station,
        "map_sha256": args.map_hash,
        "world_file": str(args.world),
        "source_route_file": str(route_file),
        "cargo_center": [cx, cy],
        "staging_pose": staging,
        "predocking_pose": predock,
        "final_docking_pose": center_pose,
        "nominal_predock_distance_m":
            PREDOCK_DISTANCE,
        "entry_heading_deg":
            math.degrees(entry_heading),
        "staging_heading_error_deg":
            math.degrees(heading_error),
        "staging_lateral_error_m":
            lateral_error,
        "segment_1": {
            "name": "WAREHOUSE_NAVIGATION",
            "poses": segment_1,
            "length_m": route_length(segment_1),
            "end_action": "MANDATORY_STOP",
        },
        "checkpoint": {
            "required": True,
            "state": "VERIFY_AND_CORRECT",
            "target_pose": predock,
            "position_tolerance_m": 0.02,
            "heading_tolerance_deg": 2.0,
            "steering_alignment_required": True,
            "correction_maneuver":
                "NOT_YET_EXECUTION_VALIDATED",
        },
        "segment_2": {
            "name": "PRECISION_DOCKING",
            "poses": segment_2,
            "length_m": route_length(segment_2),
            "end_action": "STOP_AND_VERIFY",
        },
        "complete_poses": complete,
        "geometry": {
            "robot_half_length_m": HALF_LENGTH,
            "robot_half_width_m": HALF_WIDTH,
            "stand_legs": geometry["legs"],
            "detected_leg_collisions": collisions,
        },
        "validation": {
            "2d_leg_geometry": status,
            "3d_tabletop_clearance":
                "NOT_YET_VALIDATED",
            "articulated_wheel_sweep":
                "NOT_YET_VALIDATED",
            "surrounding_obstacle_clearance":
                "NOT_YET_VALIDATED",
            "predock_correction":
                "NOT_YET_VALIDATED",
            "cargo_lift":
                "NOT_YET_VALIDATED",
        },
        "execution_approved": False,
    }

    outdir = Path(args.output_dir)
    outdir.mkdir(
        parents=True,
        exist_ok=True
    )

    outfile = outdir / (
        f"HOME__TO__{station}__TWO_SEGMENTS.json"
    )

    outfile.write_text(
        json.dumps(result, indent=2) + "\n"
    )

    print()
    print("=" * 60)
    print("STATION:", station)
    print("=" * 60)
    print("Saved warehouse route:", round(
        route_length(warehouse), 3
    ), "m")
    print("Staging:", [
        round(v, 4) for v in staging
    ])
    print("Predock:", [
        round(v, 4) for v in predock
    ])
    print("Cargo center:", [
        round(v, 4) for v in center_pose
    ])
    print("Segment 1:", round(
        result["segment_1"]["length_m"], 3
    ), "m")
    print("Segment 2:", round(
        result["segment_2"]["length_m"], 3
    ), "m")
    print("Complete route:", round(
        route_length(complete), 3
    ), "m")
    print("Leg collisions:", len(collisions))
    print("Geometry status:", status)
    print("Execution approved: NO")
    print("Saved:", outfile)

    if collisions:
        print(
            "First leg collision:",
            collisions[0]
        )

    return len(collisions) == 0


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--database",
        required=True
    )

    parser.add_argument(
        "--map",
        required=True
    )

    parser.add_argument(
        "--world",
        required=True
    )

    parser.add_argument(
        "--output-dir",
        required=True
    )

    parser.add_argument(
        "--station",
        choices=STATIONS + ["all"],
        default="all"
    )

    args = parser.parse_args()
    args.map_hash = hash_map(args.map)

    db = json.loads(
        Path(args.database).read_text()
    )

    if db.get("map_sha256") != args.map_hash:
        raise SystemExit(
            "Map hash differs from saved route database"
        )

    models = world_models(args.world)

    stations = (
        STATIONS
        if args.station == "all"
        else [args.station]
    )

    success = 0
    failures = []

    for station in stations:
        try:
            if compile_station(
                args,
                station,
                models,
                db
            ):
                success += 1
            else:
                failures.append(
                    (station, "Leg collision")
                )

        except Exception as exc:
            failures.append(
                (station, str(exc))
            )
            print()
            print(
                f"REJECTED {station}: {exc}"
            )

    print()
    print("=" * 60)
    print("PHASE 14 COMPILATION SUMMARY")
    print("=" * 60)
    print(
        "2D leg geometry passed:",
        success,
        "/",
        len(stations)
    )
    print(
        "Rejected or not yet compatible:",
        len(failures)
    )
    print(
        "Motion commands published: 0"
    )
    print(
        "Execution approved: NO"
    )
    print("=" * 60)

    for station, reason in failures:
        print(f"  {station}: {reason}")

    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
