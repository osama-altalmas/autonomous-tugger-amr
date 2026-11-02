#!/usr/bin/env python3
"""
AMR V1 — Complete missing directed routes by reversing
previously validated geometric paths.

IMPORTANT:
    Reversing a path changes the direction of travel,
    NOT the physical robot heading.

    A REVERSE route requires controller support for
    reverse motion, including reverse curved motion.

    OFFLINE ONLY. NO ROBOT COMMANDS.
"""

import argparse
import hashlib
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

from build_network import (
    load_occupancy,
    validate_trajectory,
)


def angular_error(a, b):
    return abs(math.atan2(
        math.sin(a - b),
        math.cos(a - b)
    ))


def position_error(a, b):
    return math.hypot(
        a[0] - b[0],
        a[1] - b[1]
    )


def reverse_route(source, new_source, new_target):
    old = source["poses"]

    if len(old) < 2:
        raise ValueError("Route has fewer than two poses")

    # Keep body headings unchanged.
    # Reverse the order of positions and headings.
    poses = [
        [float(x), float(y), float(yaw)]
        for x, y, yaw in reversed(old)
    ]

    # Check every segment travels behind the robot.
    # The commanded motion is explicitly REVERSE.
    for a, b in zip(poses, poses[1:]):
        dx = b[0] - a[0]
        dy = b[1] - a[1]

        if math.hypot(dx, dy) < 1e-8:
            continue

        movement_yaw = math.atan2(dy, dx)
        expected_yaw = a[2] + math.pi

        if angular_error(
            movement_yaw,
            expected_yaw
        ) > math.radians(12):
            raise ValueError(
                "Reverse motion is inconsistent "
                "with stored robot orientation"
            )

    return {
        "version": "AMR_V1_SAVED_ROUTE",
        "map_sha256": source["map_sha256"],
        "source": new_source,
        "target": new_target,
        "poses": poses,
        "waypoints": list(reversed(
            source.get("waypoints", [])
        )),
        "length_m": source["length_m"],
        "max_curvature_1_per_m": source[
            "max_curvature_1_per_m"
        ],
        "motion_direction": "REVERSE",
        "source_entrance": source["target_entrance"],
        "target_entrance": source["source_entrance"],
        "source_pose": poses[0],
        "target_pose": poses[-1],
        "terminal_heading_validated": True,
        "departure_transition_validated": True,
        "reverse_of": (
            f"{source['source']}__TO__{source['target']}"
        ),
        "static_collision_validated": True,
        "reverse_controller_validated": False,
        "cargo_docking_validated": False,
        "execution_approved": False
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--database", required=True)
    ap.add_argument("--map", required=True)

    args = ap.parse_args()

    db_file = Path(args.database)
    db = json.loads(db_file.read_text())

    spec, blocked = load_occupancy(args.map)

    import yaml

    map_file = Path(args.map)
    map_info = yaml.safe_load(map_file.read_text())

    image = map_file.parent / map_info["image"]

    actual_hash = hashlib.sha256(
        map_file.read_bytes() + image.read_bytes()
    ).hexdigest()

    if actual_hash != db["map_sha256"]:
        raise SystemExit(
            "ERROR: Database does not match the "
            "current warehouse map."
        )

    # Do not overwrite existing successful routes.
    missing = [
        (key, entry)
        for key, entry in db["routes"].items()
        if entry["status"] != "VALIDATED_STATIC_CANDIDATE"
    ]

    print()
    print("========== REVERSE ROUTE COMPLETION ==========")
    print("Total connections:", len(db["routes"]))
    print("Missing:", len(missing))
    print("Original warehouse map: UNCHANGED")
    print("Robot movement: DISABLED")
    print("==============================================")

    additions = []

    for key, entry in missing:
        new_source = entry["source"]
        new_target = entry["target"]

        opposite_key = (
            f"{new_target}__TO__{new_source}"
        )

        opposite_entry = db["routes"].get(
            opposite_key
        )

        print(
            f"\nAttempting: {new_source} -> {new_target}"
        )

        if not opposite_entry:
            print("  REJECTED: opposite route absent")
            continue

        if opposite_entry["status"] != (
            "VALIDATED_STATIC_CANDIDATE"
        ):
            print(
                "  REJECTED: opposite route not validated"
            )
            continue

        opposite_file = (
            db_file.parent /
            opposite_entry["route_file"]
        )

        opposite = json.loads(
            opposite_file.read_text()
        )

        if opposite["map_sha256"] != actual_hash:
            print("  REJECTED: map hash mismatch")
            continue

        try:
            reversed_path = reverse_route(
                opposite,
                new_source,
                new_target
            )

            # Ensure that the positions really match
            # the saved opposite route endpoints.
            assert position_error(
                reversed_path["poses"][0],
                opposite["poses"][-1]
            ) < 1e-6

            assert position_error(
                reversed_path["poses"][-1],
                opposite["poses"][0]
            ) < 1e-6

            # Revalidate the COMPLETE rectangular
            # footprint on the original map.
            validation = validate_trajectory(
                reversed_path["poses"],
                spec,
                blocked
            )

            if not validation["valid"]:
                raise ValueError(
                    "Reversed footprint intersects "
                    "an occupied map cell"
                )

            reversed_path["footprint_validation"] = (
                validation
            )

            additions.append(
                (
                    key,
                    entry,
                    reversed_path
                )
            )

            print(
                "  VALIDATED REVERSE GEOMETRY:",
                reversed_path["length_m"],
                "m"
            )

        except Exception as exc:
            print(
                "  REJECTED:",
                str(exc)
            )

    if not additions:
        print("\nNo additional routes accepted.")
        return

    # Backup the database before changing it.
    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    backup = db_file.with_name(
        f"route_database.before_reverse_{stamp}.json"
    )

    shutil.copy2(db_file, backup)

    for key, entry, route in additions:
        name = f"{key}.json"

        destination = db_file.parent / name

        if destination.exists():
            raise RuntimeError(
                f"Refusing to overwrite: {destination}"
            )

        temporary = destination.with_suffix(
            ".json.tmp"
        )

        temporary.write_text(
            json.dumps(route, indent=2) + "\n"
        )

        temporary.replace(destination)

        entry.update({
            "status": "VALIDATED_STATIC_CANDIDATE",
            "route_file": name,
            "length_m": route["length_m"],
            "source_entrance": route["source_entrance"],
            "target_entrance": route["target_entrance"],
            "motion_direction": "REVERSE",
            "terminal_heading_validated": True,
            "departure_transition_validated": True,
            "reverse_controller_validated": False,
            "execution_approved": False,
        })

        print(
            "SAVED:",
            key,
            "| REVERSE"
        )

    accepted = sum(
        entry["status"] == (
            "VALIDATED_STATIC_CANDIDATE"
        )
        for entry in db["routes"].values()
    )

    db["summary"]["static_candidates_found"] = accepted

    db["summary"][
        "connections_without_candidate"
    ] = len(db["routes"]) - accepted

    db["summary"][
        "reverse_routes_added"
    ] = len(additions)

    db["summary"][
        "complete_mission_execution_approved"
    ] = False

    db["execution_approved"] = False

    temporary = db_file.with_suffix(
        ".json.tmp"
    )

    temporary.write_text(
        json.dumps(db, indent=2) + "\n"
    )

    temporary.replace(db_file)

    print()
    print("========== FINAL DATABASE ==========")
    print("Routes:", accepted, "/", len(db["routes"]))
    print("New reverse routes:", len(additions))
    print("Backup:", backup)
    print("Database:", db_file)
    print("Reverse controller validated: NO")
    print("Robot execution approved: NO")
    print("====================================")


if __name__ == "__main__":
    main()
