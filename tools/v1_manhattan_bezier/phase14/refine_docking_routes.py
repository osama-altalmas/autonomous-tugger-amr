#!/usr/bin/env python3
"""Phase 14B: centerline refinement and conservative static 3D audit.

No ROS node. No motion commands. No lift commands.
Existing routes are never overwritten.
"""

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path


STATIONS = {
    "cargo_station_0": math.pi,
    "cargo_station_3": math.pi / 2,
    "cargo_dropoff_northwest": math.pi,
    "cargo_dropoff_southeast": -math.pi / 2,
}

STEP = 0.01
BODY_X = 0.44
BODY_Y = 0.29

# Conservative 2D envelope used for leg collision checking.
# This includes a margin beyond the nominal chassis footprint.
CHECK_X = 0.465
CHECK_Y = 0.315

# Additional operating margin required beyond the exact nominal
# collision geometry. This is a proposed validation threshold.
VERTICAL_MARGIN = 0.02


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def sdf_pose(element):
    text = element.findtext("pose")
    p = [float(v) for v in text.split()] if text else [0.] * 6
    if len(p) != 6:
        raise ValueError("SDF pose must have six values")
    if any(abs(p[i]) > 1e-8 for i in (3, 4)):
        raise ValueError("Non-planar SDF rotation needs full 3D handling")
    return p


def compose(a, b):
    c, s = math.cos(a[5]), math.sin(a[5])
    return [
        a[0] + c * b[0] - s * b[1],
        a[1] + s * b[0] + c * b[1],
        a[2] + b[2],
        0.,
        0.,
        wrap(a[5] + b[5]),
    ]


def boxes_in_model(model):
    """Extract box collisions, respecting model/link/collision poses."""
    result = []
    mp = sdf_pose(model)
    for link in model.findall("link"):
        lp = compose(mp, sdf_pose(link))
        for collision in link.findall("collision"):
            box = collision.find("geometry/box/size")
            if box is None or not box.text:
                raise ValueError(
                    "Unsupported non-box collision in station model "
                    + str(model.get("name"))
                )
            cp = compose(lp, sdf_pose(collision))
            size = [float(v) for v in box.text.split()]
            if len(size) != 3:
                raise ValueError("Invalid collision-box dimensions")
            result.append({
                "name": collision.get("name", "unnamed"),
                "pose": cp,
                "size": size,
                "z_min": cp[2] - size[2] / 2,
                "z_max": cp[2] + size[2] / 2,
            })
    return result


def load_world(world_file):
    root = ET.parse(world_file).getroot()
    return {
        m.get("name"): m
        for m in root.findall(".//model")
        if m.get("name")
    }


def load_robot_geometry(xacro_file):
    """Read actual body/lift URDF collision dimensions."""
    root = ET.parse(xacro_file).getroot()

    base = root.find("./link[@name='base_link']")
    lift = root.find("./link[@name='lift_platform_link']")
    fixed = root.find("./joint[@name='base_footprint_joint']")
    lift_joint = root.find("./joint[@name='lift_joint']")

    if any(e is None for e in (base, lift, fixed, lift_joint)):
        raise ValueError("Expected robot links/joints are missing")

    def origin(element):
        if element is None:
            return [0.0, 0.0, 0.0]
        o = element.find("origin")
        if o is None:
            return [0.0, 0.0, 0.0]
        return [float(v) for v in o.get("xyz", "0 0 0").split()]

    def box(link):
        c = link.find("collision")
        b = c.find("geometry/box")
        if b is None:
            raise ValueError("Expected robot box collision not found")
        return origin(c), [float(v) for v in b.get("size").split()]

    base_offset = origin(fixed)[2]
    body_origin, body_size = box(base)
    lift_origin, lift_size = box(lift)
    lift_offset = origin(lift_joint)[2]

    body_top = (
        base_offset + body_origin[2] + body_size[2] / 2
    )
    lift_top = (
        base_offset + lift_offset + lift_origin[2]
        + lift_size[2] / 2
    )
    lower_z = min(
        base_offset + body_origin[2] - body_size[2] / 2,
        base_offset + lift_offset + lift_origin[2]
        - lift_size[2] / 2,
    )

    if abs(body_size[0] / 2 - BODY_X) > 1e-6:
        raise ValueError("Robot chassis length changed")
    if abs(body_size[1] / 2 - BODY_Y) > 1e-6:
        raise ValueError("Robot chassis width changed")

    return {
        "body_size_m": body_size,
        "lift_size_m": lift_size,
        "body_top_m": body_top,
        "lowered_lift_top_m": lift_top,
        "collision_bottom_m": lower_z,
    }


def corners(pose, half_x, half_y):
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return [
        (x + c * dx - s * dy, y + s * dx + c * dy)
        for dx, dy in (
            (-half_x, -half_y),
            (half_x, -half_y),
            (half_x, half_y),
            (-half_x, half_y),
        )
    ]


def overlaps(a, b):
    for poly in (a, b):
        for i, p in enumerate(poly):
            q = poly[(i + 1) % len(poly)]
            axis = (p[1] - q[1], q[0] - p[0])
            pa = [x * axis[0] + y * axis[1] for x, y in a]
            pb = [x * axis[0] + y * axis[1] for x, y in b]
            if max(pa) < min(pb) or max(pb) < min(pa):
                return False
    return True


def smooth_extension(start, target, heading):
    """Quintic lateral transition, zero end lateral slopes/curvature."""
    ux, uy = math.cos(heading), math.sin(heading)
    nx, ny = -uy, ux
    dx, dy = start[0] - target[0], start[1] - target[1]

    along = dx * ux + dy * uy
    lateral = dx * nx + dy * ny
    length = -along

    if length < 0.3:
        raise ValueError("Staging position too close to checkpoint")
    if length > 3.5:
        raise ValueError("Staging position too far from checkpoint")
    if abs(lateral) > 0.08:
        raise ValueError("Lateral staging mismatch exceeds 8 cm")

    if abs(wrap(start[2] - heading)) > math.radians(2):
        raise ValueError("Staging heading does not match entry heading")

    n = max(2, math.ceil(length / STEP))
    poses = []

    for i in range(n + 1):
        t = i / n

        # q(0)=1, q(1)=0, q'=q''=0 at both endpoints.
        q = 1 - 10*t**3 + 15*t**4 - 6*t**5
        dq = -30*t*t + 60*t**3 - 30*t**4

        longitudinal = -length * (1 - t)
        lateral_now = lateral * q

        x = target[0] + ux*longitudinal + nx*lateral_now
        y = target[1] + uy*longitudinal + ny*lateral_now

        dxdt = ux*length + nx*lateral*dq
        dydt = uy*length + ny*lateral*dq

        yaw = math.atan2(dydt, dxdt)
        poses.append([x, y, yaw])

    poses[0] = list(start)
    poses[-1] = list(target)
    return poses


def straight(start, target):
    length = dist(start, target)
    count = max(1, math.ceil(length / STEP))
    return [
        [
            start[0] + (target[0]-start[0])*i/count,
            start[1] + (target[1]-start[1])*i/count,
            start[2],
        ]
        for i in range(count + 1)
    ]


def path_length(poses):
    return sum(dist(a, b) for a, b in zip(poses, poses[1:]))


def audit_station(station, original, models, robot):
    heading = STATIONS[station]

    legs_name = station + "_legs"

    # Pickup stands have actual tabletop collisions.
    # Drop-off markers have no tabletop collision in this world.
    table_names = {
        "cargo_station_0": "cargo_station_0_stand",
        "cargo_station_3": "cargo_station_3_stand",
    }
    table_name = table_names.get(station)

    if legs_name not in models:
        raise ValueError(f"Missing leg model: {legs_name}")

    leg_boxes = boxes_in_model(models[legs_name])

    if len(leg_boxes) != 4:
        raise ValueError(f"Expected 4 leg collisions, got {len(leg_boxes)}")

    if table_name is not None:
        if table_name not in models:
            raise ValueError(f"Missing tabletop model: {table_name}")
        table_boxes = boxes_in_model(models[table_name])
        if len(table_boxes) != 1:
            raise ValueError(
                f"Expected 1 tabletop collision, got {len(table_boxes)}"
            )
    else:
        table_boxes = []

    center_pose = sdf_pose(models[legs_name])
    center = center_pose[:2]

    # Opposite direction to arrival, exactly 1 meter from cargo center.
    ux, uy = math.cos(heading), math.sin(heading)
    checkpoint = [center[0]-ux, center[1]-uy, heading]
    destination = [center[0], center[1], heading]

    warehouse_poses = original["segment_1"]["poses"]

    # Original Phase 14A retained the saved warehouse section first.
    # Use the saved route's staging coordinate to trim its extension.
    staging = original["staging_pose"]

    indices = [
        i for i, p in enumerate(warehouse_poses)
        if dist(p, staging) < 1e-7
    ]
    if not indices:
        raise ValueError("Staging pose missing from existing segment")
    staging_index = indices[-1]
    warehouse = warehouse_poses[:staging_index+1]

    extension = smooth_extension(staging, checkpoint, heading)
    dock = straight(checkpoint, destination)

    segment1 = warehouse + extension[1:]
    segment2 = dock
    full = segment1 + segment2[1:]

    if dist(segment1[-1], segment2[0]) > 1e-8:
        raise ValueError("Route discontinuity at predocking")

    # Check actual SDF leg collisions throughout the new approach,
    # with conservative robot envelope. Does not model wheel articulation.
    leg_rects = [
        corners(
            [b["pose"][0], b["pose"][1], b["pose"][5]],
            b["size"][0]/2,
            b["size"][1]/2,
        )
        for b in leg_boxes
    ]

    candidate_poses = extension + dock[1:]
    collisions = []

    for i, p in enumerate(candidate_poses):
        envelope = corners(p, CHECK_X, CHECK_Y)
        for j, leg in enumerate(leg_rects):
            if overlaps(envelope, leg):
                collisions.append((i, j))
                if len(collisions) >= 10:
                    break
        if len(collisions) >= 10:
            break

    if table_boxes:
        tabletop = table_boxes[0]
        table_under = tabletop["z_min"]

        # Real SDF tabletop collision, not visual geometry.
        nominal_gap = table_under - max(
            robot["body_top_m"],
            robot["lowered_lift_top_m"]
        )
        clearance_ok = nominal_gap >= VERTICAL_MARGIN
    else:
        table_under = None
        nominal_gap = None
        # No tabletop collision to test in this station model.
        clearance_ok = True

    result = dict(original)
    result["schema"] = "amr_phase14_two_segment_refined_v1"
    result["predocking_pose"] = checkpoint
    result["final_docking_pose"] = destination
    result["entry_heading_deg"] = math.degrees(heading)

    result["segment_1"] = {
        "name": "WAREHOUSE_NAVIGATION",
        "poses": segment1,
        "length_m": path_length(segment1),
        "end_action": "MANDATORY_STOP",
    }
    result["segment_2"] = {
        "name": "PRECISION_DOCKING",
        "poses": segment2,
        "length_m": path_length(segment2),
        "end_action": "STOP_AND_VERIFY",
    }
    result["complete_poses"] = full

    result["checkpoint"]["target_pose"] = checkpoint

    result["geometry"]["detected_leg_collisions"] = [
        {"approach_index": i, "leg_index": j}
        for i, j in collisions
    ]

    result["robot_collision_geometry"] = robot
    result["tabletop_underside_m"] = table_under
    result["nominal_vertical_clearance_m"] = nominal_gap
    result["minimum_requested_margin_m"] = VERTICAL_MARGIN

    validation = result["validation"]
    validation["2d_leg_geometry"] = (
        "PASSED" if not collisions else "FAILED"
    )
    validation["static_vertical_body_lift_clearance"] = (
        ("PASSED" if clearance_ok else "FAILED")
        if table_boxes else "NOT_APPLICABLE_NO_TABLETOP_COLLISION"
    )
    validation["3d_tabletop_clearance"] = (
        "PARTIAL_STATIC_BOX_CHECK_ONLY"
        if table_boxes else "NOT_APPLICABLE_NO_TABLETOP_COLLISION"
    )
    validation["articulated_wheel_sweep"] = "NOT_YET_VALIDATED"
    validation["surrounding_obstacle_clearance"] = "NOT_YET_VALIDATED"
    validation["predock_correction"] = "NOT_YET_VALIDATED"
    validation["cargo_lift"] = "NOT_YET_VALIDATED"

    result["execution_approved"] = False

    report = {
        "station": station,
        "predocking_pose": checkpoint,
        "final_docking_pose": destination,
        "segment_1_length_m": path_length(segment1),
        "segment_2_length_m": path_length(segment2),
        "leg_collisions": len(collisions),
        "tabletop_underside_m": table_under,
        "lowered_lift_top_m": robot["lowered_lift_top_m"],
        "nominal_vertical_clearance_m": nominal_gap,
        "static_clearance_passed": clearance_ok,
        "execution_approved": False,
    }
    return result, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--world", required=True)
    parser.add_argument("--robot", required=True)
    args = parser.parse_args()

    models = load_world(args.world)
    robot = load_robot_geometry(args.robot)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    failures = []
    reports = []

    for station in STATIONS:
        input_file = (
            Path(args.routes)
            / f"HOME__TO__{station}__TWO_SEGMENTS.json"
        )
        try:
            original = json.loads(input_file.read_text())
            refined, report = audit_station(
                station, original, models, robot
            )

            output_file = (
                output
                / f"HOME__TO__{station}__REFINED.json"
            )
            output_file.write_text(
                json.dumps(refined, indent=2) + "\n"
            )

            reports.append(report)

            print(f"\n{'='*60}")
            print("STATION:", station)
            print("Predock:", [
                round(v, 4) for v in report["predocking_pose"]
            ])
            print("Cargo center:", [
                round(v, 4) for v in report["final_docking_pose"]
            ])
            print(
                "Segment 1:",
                round(report["segment_1_length_m"], 3), "m"
            )
            print(
                "Segment 2:",
                round(report["segment_2_length_m"], 3), "m"
            )
            print("Leg collisions:", report["leg_collisions"])
            gap = report["nominal_vertical_clearance_m"]
            print(
                "Nominal vertical gap:",
                f"{gap*1000:.2f} mm"
                if gap is not None
                else "N/A (no tabletop collision)"
            )
            print(
                "Static box clearance:",
                "PASS" if report["static_clearance_passed"]
                and report["leg_collisions"] == 0 else "FAIL"
            )
            print("Execution approved: NO")
            print("Saved:", output_file)

            if (
                report["leg_collisions"]
                or not report["static_clearance_passed"]
            ):
                failures.append(station)

        except Exception as exc:
            failures.append(station)
            print(f"\nFAILED {station}: {exc}")

    (output / "phase14b_audit.json").write_text(
        json.dumps({
            "reports": reports,
            "failures": failures,
            "execution_approved": False,
        }, indent=2) + "\n"
    )

    print("\n" + "="*60)
    print("PHASE 14B SUMMARY")
    print("Static checks passed:", len(STATIONS)-len(failures),
          "/", len(STATIONS))
    print("Motion commands published: 0")
    print("Execution approved: NO")
    print("="*60)

    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
