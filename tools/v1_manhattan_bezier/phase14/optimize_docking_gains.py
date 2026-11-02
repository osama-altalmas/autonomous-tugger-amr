#!/usr/bin/env python3
"""
Phase 14F — Offline docking gain comparison.

Tests 16 gain combinations against 125 starting poses using
the existing 4WS dynamics simulator.

NO ROS CONNECTION.
NO VELOCITY COMMANDS.
NO GAZEBO MOVEMENT.
NO LIVE CONTROLLER MODIFICATION.
"""

import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent

SOURCE = HERE / "test_4ws_docking_dynamics.py"

if not SOURCE.exists():
    raise SystemExit(
        f"Missing dynamics model: {SOURCE}"
    )

spec = importlib.util.spec_from_file_location(
    "phase14f_dynamics_model",
    SOURCE,
)

model = importlib.util.module_from_spec(spec)

with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(model)

baseline = (
    17.500650636,
    -5.963040895,
    math.radians(-92.82),
)

starting_poses = []

for dx in (-.02, -.01, 0, .01, .02):
    for dy in (-.02, -.01, 0, .01, .02):
        for dh in (-2, -1, 0, 1, 2):
            starting_poses.append((
                f"dx={dx:+.2f} dy={dy:+.2f} dh={dh:+.0f}",
                (
                    baseline[0] + dx,
                    baseline[1] + dy,
                    baseline[2] + math.radians(dh),
                )
            ))

heading_gains = (2.2, 2.8, 3.4, 4.0)
lateral_gains = (4.0, 5.0, 6.0, 7.0)

summaries = []

print("=" * 76)
print("PHASE 14F — 16-CONTROLLER GAIN COMPARISON")
print("=" * 76)
print("Starting conditions per controller:", len(starting_poses))
print("Controllers to test:", len(heading_gains)*len(lateral_gains))
print()
print(
    f"{'Heading':>8} {'Lateral':>8} "
    f"{'Passed':>8} {'Worst pos':>12} "
    f"{'Worst yaw':>12} {'Peak steer':>12}"
)

for hg in heading_gains:
    for lg in lateral_gains:

        model.HEADING_GAIN = hg
        model.LATERAL_GAIN = lg

        results = [
            model.simulate(name, pose)
            for name, pose in starting_poses
        ]

        passed = sum(
            result["passed"]
            for result in results
        )

        worst_position = max(
            result["position_cm"]
            for result in results
        )

        worst_heading = max(
            result["heading_deg"]
            for result in results
        )

        peak_steering = max(
            result["max_steering_deg"]
            for result in results
        )

        summaries.append({
            "heading_gain": hg,
            "lateral_gain": lg,
            "passed": passed,
            "total": len(results),
            "worst_position_cm": worst_position,
            "worst_heading_deg": worst_heading,
            "peak_steering_deg": peak_steering,
            "failed_cases": [
                {
                    "name": r["name"],
                    "reason": r["reason"],
                    "position_cm": r["position_cm"],
                    "heading_deg": r["heading_deg"],
                }
                for r in results
                if not r["passed"]
            ],
        })

        print(
            f"{hg:8.1f} {lg:8.1f} "
            f"{passed:6d}/125 "
            f"{worst_position:10.3f}cm "
            f"{worst_heading:10.3f}deg "
            f"{peak_steering:10.3f}deg",
            flush=True,
        )

summaries.sort(
    key=lambda s: (
        -s["passed"],
        s["worst_heading_deg"],
        s["worst_position_cm"],
    )
)

best = summaries[0]

report_path = (
    HERE / "routes_refined"
    / "phase14f_gain_search.json"
)

report_path.parent.mkdir(parents=True, exist_ok=True)

report_path.write_text(
    json.dumps({
        "schema": "phase14f_gain_search_v1",
        "best": best,
        "all_candidates": summaries,
        "offline_only": True,
        "motion_authorized": False,
        "full_3d_clearance_certified": False,
    }, indent=2) + "\n"
)

print()
print("=" * 76)
print("BEST OFFLINE CANDIDATE")
print("=" * 76)

print(
    "Heading gain:",
    best["heading_gain"]
)
print(
    "Lateral gain:",
    best["lateral_gain"]
)
print(
    "Passed:",
    f'{best["passed"]}/{best["total"]}'
)
print(
    "Worst position error:",
    f'{best["worst_position_cm"]:.3f} cm'
)
print(
    "Worst heading error:",
    f'{best["worst_heading_deg"]:.3f} deg'
)
print(
    "Peak steering:",
    f'{best["peak_steering_deg"]:.3f} deg'
)

print()
print("TOP FIVE CANDIDATES:")

for candidate in summaries[:5]:
    print(
        f'  H={candidate["heading_gain"]:.1f} '
        f'L={candidate["lateral_gain"]:.1f} '
        f'PASS={candidate["passed"]}/125 '
        f'POS={candidate["worst_position_cm"]:.3f}cm '
        f'YAW={candidate["worst_heading_deg"]:.3f}deg'
    )

print()
print("Report:", report_path)
print("Live controller modified: NO")
print("Robot moved: NO")
print("Docking motion authorized: NO")
print("=" * 76)
