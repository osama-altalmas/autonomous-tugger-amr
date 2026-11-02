#!/usr/bin/env python3
"""Launch saved-route follower using editable speed profile."""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

PROFILE = (
    ROOT / "src/tugger_control/config/sim_speed.yaml"
)

FOLLOWER = (
    ROOT / "tools/v1_manhattan_bezier/follow_saved_route_v1.py"
)

DATABASE = (
    ROOT / "tools/v1_manhattan_bezier/routes/route_database.json"
)

MAP = (
    ROOT / "src/tugger_navigation/maps/warehouse_map.yaml"
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--from",
        dest="source",
        default="HOME"
    )

    parser.add_argument(
        "--to",
        dest="target",
        default="cargo_dropoff_southeast"
    )

    parser.add_argument(
        "--resume",
        action="store_true"
    )

    parser.add_argument(
        "--drive",
        action="store_true",
        help="Explicitly enable Gazebo motion"
    )

    args = parser.parse_args()

    with PROFILE.open() as f:
        profile = yaml.safe_load(f)

    control = profile[
        "four_ws_controller"
    ]["ros__parameters"]

    settings = profile["follower"]

    speed = float(settings["cruising_speed"])

    controller_max = float(
        control["max_linear_speed"]
    )

    if not 0 < speed <= controller_max <= 2.5:
        raise SystemExit(
            "ERROR: Invalid speed profile. "
            "Follower speed must not exceed controller maximum."
        )

    lateral = float(
        settings["max_lateral_acceleration"]
    )

    controller_lateral = float(
        control["max_lateral_acceleration"]
    )

    if not 0 < lateral <= controller_lateral:
        raise SystemExit(
            "ERROR: Follower lateral acceleration "
            "must not exceed controller limit."
        )

    command = [
        sys.executable,
        "-u",
        str(FOLLOWER),
        "--database",
        str(DATABASE),
        "--map",
        str(MAP),
        "--from",
        args.source,
        "--to",
        args.target,
        "--speed",
        str(speed),
        "--lookahead",
        str(settings["lookahead"]),
        "--max-lateral-accel",
        str(lateral),
        "--max-tracking-error",
        str(settings["max_tracking_error"]),
        "--max-seconds",
        str(settings["max_seconds"]),
        "--braking-deceleration",
        str(control["max_deceleration"]),
    ]

    if args.resume:
        command.append("--resume")

    if args.drive:
        command.extend([
            "--enable-motion",
            "--gazebo-only",
        ])

    print()
    print("=========================================")
    print("AMR V1 — EDITABLE SPEED PROFILE")
    print("=========================================")
    print("Route:", args.source, "->", args.target)
    print("Cruising speed:", speed, "m/s")
    print("Controller maximum:", controller_max, "m/s")
    print("Lateral acceleration:", lateral, "m/s²")
    print(
        "Mode:",
        "GAZEBO MOTION ENABLED"
        if args.drive
        else "PREFLIGHT ONLY"
    )
    print("=========================================")
    print(flush=True)

    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main())
