#!/usr/bin/env python3
"""
Phase 14F — 125-case dynamic 4WS docking stress test.

Reuses the existing dynamic test's actual motion_math equations,
vehicle limits, collision checks, and Candidate C controller.

This test does not connect to ROS or command the robot.
"""

import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent

DYNAMICS = HERE / "test_4ws_docking_dynamics.py"

if not DYNAMICS.is_file():
    raise SystemExit(
        "ERROR: Missing existing 4WS dynamics test: "
        + str(DYNAMICS)
    )

spec = importlib.util.spec_from_file_location(
    "phase14f_dynamics",
    DYNAMICS
)

model = importlib.util.module_from_spec(spec)

# The existing test prints its own five-case report on import.
# Suppress that output; use only its simulation functions.
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(model)

baseline = (
    17.500650636,
    -5.963040895,
    math.radians(-92.82),
)

offsets_x = (-0.02, -0.01, 0.0, 0.01, 0.02)
offsets_y = (-0.02, -0.01, 0.0, 0.01, 0.02)
offsets_heading = (-2.0, -1.0, 0.0, 1.0, 2.0)

results = []

for dx in offsets_x:
    for dy in offsets_y:
        for dh in offsets_heading:

            start = (
                baseline[0] + dx,
                baseline[1] + dy,
                baseline[2] + math.radians(dh),
            )

            name = (
                f"dx={dx:+.3f}, "
                f"dy={dy:+.3f}, "
                f"dh={dh:+.1f}"
            )

            result = model.simulate(name, start)
            results.append(result)

passed = [r for r in results if r["passed"]]
failed = [r for r in results if not r["passed"]]

worst_position = max(
    results,
    key=lambda r: r["position_cm"]
)

worst_heading = max(
    results,
    key=lambda r: r["heading_deg"]
)

worst_steering = max(
    results,
    key=lambda r: r["max_steering_deg"]
)

print()
print("=" * 74)
print("PHASE 14F — 125-CASE DYNAMIC DOCKING STRESS TEST")
print("=" * 74)

print("Correction law: curvature = 2.2*heading_error - 5.0*lateral")
print("Tests:", len(results))
print("Passed:", len(passed))
print("Failed:", len(failed))

print()
print(
    "Worst final position error:",
    f"{worst_position['position_cm']:.3f} cm"
)
print(
    "Worst final heading error:",
    f"{worst_heading['heading_deg']:.3f} deg"
)
print(
    "Maximum steering angle:",
    f"{worst_steering['max_steering_deg']:.3f} deg"
)

if failed:
    print()
    print("FAILED CASES (first 20):")

    for r in failed[:20]:
        print(
            f"{r['name']} | "
            f"position={r['position_cm']:.3f} cm | "
            f"heading={r['heading_deg']:.3f} deg | "
            f"reason={r['reason']}"
        )

report_path = (
    HERE
    / "routes_refined"
    / "phase14f_stress_report.json"
)

report = {
    "schema": "phase14f_stress_v1",
    "candidate": "C",
    "tests": len(results),
    "passed": len(passed),
    "failed": len(failed),
    "worst_position_cm": worst_position["position_cm"],
    "worst_heading_deg": worst_heading["heading_deg"],
    "max_steering_deg": worst_steering["max_steering_deg"],
    "results": results,
    "full_3d_clearance_validated": False,
    "motion_authorized": False,
}

report_path.write_text(
    json.dumps(report, indent=2) + "\n"
)

print()
print("Report:", report_path)
print("Full articulated-wheel 3D clearance: NOT CERTIFIED")
print("Gazebo movement: NONE")
print("Docking motion authorized: NO")
print("=" * 74)

if failed:
    raise SystemExit(2)
