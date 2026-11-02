#!/usr/bin/env python3
"""
AMR V1 — Saved-route manager, mission continuity checker,
and ROS 2 path visualizer.

Features:
  * Retrieve saved directed routes
  * Verify database/map fingerprint
  * Revalidate static rectangular footprint
  * Check heading and position continuity
  * Explicitly identify forward/reverse segments
  * Publish nav_msgs/Path for visualization

Never publishes /cmd_vel or controller goals.
"""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

from build_network import load_occupancy, validate_trajectory


POSITION_TOLERANCE = 0.01
HEADING_TOLERANCE = math.radians(5.0)


def angle_error(a, b):
    return abs(math.atan2(
        math.sin(a - b),
        math.cos(a - b)
    ))


def distance(a, b):
    return math.hypot(
        a[0] - b[0],
        a[1] - b[1]
    )


def map_hash(path):
    import yaml

    path = Path(path)
    info = yaml.safe_load(path.read_text())
    image = path.parent / info["image"]

    return hashlib.sha256(
        path.read_bytes() + image.read_bytes()
    ).hexdigest()


def load_route(database_file, database, source, target):
    key = f"{source}__TO__{target}"

    entry = database["routes"].get(key)

    if entry is None:
        raise ValueError(f"Unknown route: {key}")

    if entry["status"] != "VALIDATED_STATIC_CANDIDATE":
        raise ValueError(f"Route not available: {key}")

    filename = entry.get("route_file")

    if not filename:
        raise ValueError(f"No saved route file: {key}")

    file = database_file.parent / filename

    route = json.loads(file.read_text())

    if route["source"] != source or route["target"] != target:
        raise ValueError(f"Route identity mismatch: {key}")

    if route["map_sha256"] != database["map_sha256"]:
        raise ValueError(f"Route map fingerprint mismatch: {key}")

    poses = route.get("poses", [])

    if len(poses) < 2:
        raise ValueError(f"Insufficient route poses: {key}")

    motion = route.get("motion_direction", "FORWARD")

    if motion not in ("FORWARD", "REVERSE"):
        raise ValueError(f"Unknown motion direction: {motion}")

    for i, pose in enumerate(poses):
        if len(pose) != 3:
            raise ValueError(f"{key}: invalid pose {i}")

        if not all(math.isfinite(float(v)) for v in pose):
            raise ValueError(f"{key}: nonfinite pose {i}")

    return {
        "name": key,
        "source": source,
        "target": target,
        "motion_direction": motion,
        "poses": poses,
        "length_m": route["length_m"],
        "departure_transition_validated": route.get(
            "departure_transition_validated", False
        ),
        "execution_approved": False,
    }


def check_motion(segment):
    """
    Check stored body orientations against the direction
    of motion, including explicitly reverse paths.
    """

    poses = segment["poses"]
    motion = segment["motion_direction"]

    maximum_error = 0.0

    for a, b in zip(poses, poses[1:]):
        dx = b[0] - a[0]
        dy = b[1] - a[1]

        if math.hypot(dx, dy) < 1e-8:
            continue

        movement_heading = math.atan2(dy, dx)

        expected = a[2]

        if motion == "REVERSE":
            expected += math.pi

        error = angle_error(movement_heading, expected)
        maximum_error = max(maximum_error, error)

    segment["maximum_motion_heading_error_deg"] = round(
        math.degrees(maximum_error), 3
    )

    # Keep the tolerance explicit. The original routes
    # use discrete sampled headings.
    if maximum_error > math.radians(12):
        raise ValueError(
            f"{segment['name']}: inconsistent body heading "
            f"and {motion.lower()} motion"
        )


def inspect_mission(segments):
    transitions = []

    for earlier, later in zip(segments, segments[1:]):
        previous_pose = earlier["poses"][-1]
        next_pose = later["poses"][0]

        xy_error = distance(previous_pose, next_pose)
        yaw_error = angle_error(
            previous_pose[2], next_pose[2]
        )

        continuous = (
            xy_error <= POSITION_TOLERANCE
            and yaw_error <= HEADING_TOLERANCE
        )

        transitions.append({
            "station": earlier["target"],
            "from_route": earlier["name"],
            "to_route": later["name"],
            "position_error_m": round(xy_error, 4),
            "body_heading_error_deg": round(
                math.degrees(yaw_error), 3
            ),
            "continuous": continuous,
            "transition_planned": continuous,
            "execution_approved": False
        })

    return transitions


def publish_path(poses, frame, topic, seconds):
    import rclpy

    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Path
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, DurabilityPolicy

    rclpy.init()

    node = Node("amr_v1_saved_route_preview")

    qos = QoSProfile(depth=1)
    qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

    publisher = node.create_publisher(Path, topic, qos)

    msg = Path()
    msg.header.frame_id = frame

    for x, y, yaw in poses:
        pose = PoseStamped()
        pose.header.frame_id = frame
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)

        pose.pose.orientation.z = math.sin(float(yaw) / 2)
        pose.pose.orientation.w = math.cos(float(yaw) / 2)

        msg.poses.append(pose)

    print()
    print(f"Publishing {len(poses)} poses to {topic}")
    print("Frame:", frame)
    print("This is VISUALIZATION ONLY; no motion commands.")

    try:
        end = time.monotonic() + seconds

        while rclpy.ok() and time.monotonic() < end:
            stamp = node.get_clock().now().to_msg()

            msg.header.stamp = stamp

            for pose in msg.poses:
                pose.header.stamp = stamp

            publisher.publish(msg)
            rclpy.spin_once(node, timeout_sec=0.1)
            time.sleep(0.4)

    finally:
        node.destroy_node()
        rclpy.shutdown()


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--database", required=True)
    ap.add_argument("--map", required=True)

    ap.add_argument(
        "--stops",
        nargs="+",
        required=True,
        help="Example: HOME cargo_station_0 HOME"
    )

    ap.add_argument("--output")

    ap.add_argument(
        "--publish",
        action="store_true",
        help="Publish visual nav_msgs/Path only"
    )

    ap.add_argument(
        "--topic",
        default="/v1/saved_route"
    )

    ap.add_argument(
        "--frame",
        default="map"
    )

    ap.add_argument(
        "--publish-seconds",
        type=float,
        default=15.0
    )

    args = ap.parse_args()

    if len(args.stops) < 2:
        ap.error("At least two stops are required")

    db_file = Path(args.database)
    db = json.loads(db_file.read_text())

    actual_hash = map_hash(args.map)

    if db["map_sha256"] != actual_hash:
        raise SystemExit(
            "Map/database mismatch. Regenerate routes "
            "for the current occupancy map."
        )

    spec, blocked = load_occupancy(args.map)

    segments = []

    for source, target in zip(
        args.stops, args.stops[1:]
    ):
        segment = load_route(
            db_file, db, source, target
        )

        validation = validate_trajectory(
            segment["poses"], spec, blocked
        )

        if not validation["valid"]:
            raise ValueError(
                f"Static collision validation failed: "
                f"{segment['name']}"
            )

        segment["static_footprint_validated"] = True
        segment["footprint_validation"] = validation

        check_motion(segment)

        segments.append(segment)

    transitions = inspect_mission(segments)

    continuous = all(
        item["continuous"]
        for item in transitions
    )

    combined = []

    for i, segment in enumerate(segments):
        if i == 0:
            combined.extend(segment["poses"])
        elif transitions[i - 1]["continuous"]:
            combined.extend(segment["poses"][1:])
        else:
            # Preserve the discontinuity explicitly.
            # Do not pretend to have a single drivable path.
            combined.extend(segment["poses"])

    report = {
        "version": "AMR_V1_ROUTE_MANAGER",
        "stops": args.stops,
        "map_sha256": actual_hash,
        "segments": segments,
        "transitions": transitions,
        "position_heading_continuous": continuous,
        "all_segments_static_validated": True,
        "controller_validated": False,
        "docking_validated": False,
        "execution_approved": False,
    }

    print()
    print("========== SAVED ROUTE MISSION ==========")

    for segment in segments:
        print(
            f"{segment['source']} -> {segment['target']} | "
            f"{segment['motion_direction']} | "
            f"{segment['length_m']:.2f} m"
        )

    print()
    print("========== ROUTE TRANSITIONS ==========")

    if not transitions:
        print("Single leg: no inter-leg transition.")

    for item in transitions:
        print(
            f"{item['station']}: "
            f"{'CONTINUOUS' if item['continuous'] else 'NOT CONTINUOUS'}"
            f" | XY error {item['position_error_m']:.3f} m"
            f" | yaw error {item['body_heading_error_deg']:.1f}°"
        )

    print()
    print("Static validation: PASSED")
    print("Mission heading continuous:", continuous)
    print("Physical execution approved: NO")

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print("Saved:", output)

    if args.publish:
        if not continuous:
            print(
                "\nWARNING: Mission contains discontinuous "
                "transitions. Publishing ONLY the first "
                "validated route leg for visualization."
            )
            display_poses = segments[0]["poses"]
        else:
            display_poses = combined

        publish_path(
            display_poses,
            args.frame,
            args.topic,
            args.publish_seconds
        )


if __name__ == "__main__":
    main()
