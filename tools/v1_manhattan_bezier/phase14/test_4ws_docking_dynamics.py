#!/usr/bin/env python3
"""
Phase 14F — Offline 4WS docking dynamics validation.

Uses the project's ACTUAL motion_math functions:
  steering_for_twist()
  speed_limit_for_steering()
  approach()

Tests Candidate C with:
  - steering angle limits
  - steering slew limits
  - acceleration/deceleration
  - controller command processing
  - conservative 2D collision envelope
  - final docking position and heading checks

NO ROS NODE.
NO CMD_VEL.
NO GAZEBO MOVEMENT.
"""

import json
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(
    0,
    str(ROOT / "src/tugger_control")
)

from tugger_control.motion_math import (
    approach,
    steering_for_twist,
    speed_limit_for_steering,
)

ROUTE = ROOT / (
    "tools/v1_manhattan_bezier/phase14/"
    "routes_refined/"
    "HOME__TO__cargo_dropoff_southeast__REFINED.json"
)

PROFILE = ROOT / (
    "src/tugger_control/config/sim_speed.yaml"
)

route = json.loads(ROUTE.read_text())

profile = yaml.safe_load(PROFILE.read_text())

limits = profile[
    "four_ws_controller"
]["ros__parameters"]

GOAL = route["final_docking_pose"]
LEGS = route["geometry"]["stand_legs"]

WHEELBASE = 0.58

MAX_SPEED = 0.08
MAX_STEERING = math.pi / 4

STEERING_RATE = float(
    limits["max_steering_rate"]
)

ACCELERATION = float(
    limits["max_acceleration"]
)

DECELERATION = float(
    limits["max_deceleration"]
)

LATERAL_ACCELERATION = float(
    limits["max_lateral_acceleration"]
)

HEADING_GAIN = 2.2
LATERAL_GAIN = 5.0

DT = 0.05
MAX_TIME = 90.0

# Conservative envelope for preliminary 2D checking.
# Full 3D articulated-wheel clearance is a separate gate.
HALF_LENGTH = 0.49
HALF_WIDTH = 0.355


def wrap(a):
    return math.atan2(
        math.sin(a),
        math.cos(a)
    )


def corners(x, y, yaw, hx, hy):
    c = math.cos(yaw)
    s = math.sin(yaw)

    result = []

    for dx, dy in (
        (-hx, -hy),
        (hx, -hy),
        (hx, hy),
        (-hx, hy),
    ):
        result.append((
            x + c*dx - s*dy,
            y + s*dx + c*dy
        ))

    return result


def overlap(a, b):
    for poly in (a, b):
        for i, p in enumerate(poly):
            q = poly[(i+1) % len(poly)]

            axis = (
                p[1] - q[1],
                q[0] - p[0]
            )

            av = [
                x*axis[0] + y*axis[1]
                for x, y in a
            ]

            bv = [
                x*axis[0] + y*axis[1]
                for x, y in b
            ]

            if (
                max(av) < min(bv)
                or max(bv) < min(av)
            ):
                return False

    return True


LEG_SHAPES = []

for leg in LEGS:
    LEG_SHAPES.append(
        corners(
            leg["x"],
            leg["y"],
            leg["yaw"],
            leg["size"][0]/2 + 0.01,
            leg["size"][1]/2 + 0.01
        )
    )


def simulate(name, initial):
    x, y, yaw = initial

    v = 0.0
    delta = 0.0

    max_delta = 0.0
    max_lateral = 0.0

    collision = False
    reason = "TIMEOUT"

    gx, gy, gh = GOAL

    reached = False
    settled_count = 0

    for step in range(
        int(MAX_TIME / DT)
    ):
        dx = x - gx
        dy = y - gy

        remaining = -(
            dx*math.cos(gh)
            + dy*math.sin(gh)
        )

        lateral = (
            -math.sin(gh)*dx
            + math.cos(gh)*dy
        )

        heading_error = wrap(
            gh - yaw
        )

        position_error = math.hypot(
            dx, dy
        )

        max_lateral = max(
            max_lateral,
            abs(lateral)
        )

        robot = corners(
            x, y, yaw,
            HALF_LENGTH,
            HALF_WIDTH
        )

        if any(
            overlap(robot, leg)
            for leg in LEG_SHAPES
        ):
            collision = True
            reason = "LEG COLLISION"
            break

        if abs(lateral) > 0.09:
            reason = "LATERAL CORRIDOR EXCEEDED"
            break

        pose_ready = (
            position_error <= 0.02
            and abs(heading_error)
                <= math.radians(2)
        )

        if pose_ready:
            reached = True

        if reached:
            target_v = 0.0
            target_delta = 0.0
        else:
            if remaining < 0.015:
                reason = "LONGITUDINAL OVERSHOOT"
                break

            curvature = (
                HEADING_GAIN * heading_error
                - LATERAL_GAIN * lateral
            )

            curvature = max(
                -0.9,
                min(0.9, curvature)
            )

            target_v = min(
                0.08,
                max(
                    0.018,
                    0.20 * remaining
                )
            )

            if remaining < 0.07:
                target_v = min(
                    target_v, 0.025
                )

            target_omega = (
                target_v * curvature
            )

            target_delta = steering_for_twist(
                target_v,
                target_omega,
                WHEELBASE,
                MAX_STEERING
            )

        # Same acceleration/deceleration mechanism
        # as the actual FourWSController.
        rate = (
            ACCELERATION
            if (
                abs(target_v) > abs(v)
                and target_v * v >= 0
            )
            else DECELERATION
        )

        v = approach(
            v,
            target_v,
            rate,
            DT
        )

        delta = approach(
            delta,
            target_delta,
            STEERING_RATE,
            DT
        )

        safe_speed = speed_limit_for_steering(
            delta,
            WHEELBASE,
            LATERAL_ACCELERATION,
            float(limits["max_linear_speed"])
        )

        v = max(
            -safe_speed,
            min(safe_speed, v)
        )

        max_delta = max(
            max_delta,
            abs(delta)
        )

        # Actual counter-phase 4WS curvature equation.
        omega = (
            2 * v * math.tan(delta)
            / WHEELBASE
        )

        # Integrate simulated body motion.
        mid_yaw = yaw + omega*DT/2

        x += v*DT*math.cos(mid_yaw)
        y += v*DT*math.sin(mid_yaw)

        yaw = wrap(
            yaw + omega*DT
        )

        if reached:
            if (
                abs(v) <= 0.01
                and abs(omega) <= 0.02
                and abs(delta)
                    <= math.radians(2)
            ):
                settled_count += 1
            else:
                settled_count = 0

            if settled_count >= 10:
                reason = "SETTLED"
                break

    final_error = math.hypot(
        x-gx,
        y-gy
    )

    final_heading_error = abs(
        wrap(gh-yaw)
    )

    passed = (
        reason == "SETTLED"
        and final_error <= 0.02
        and final_heading_error
            <= math.radians(2)
        and not collision
    )

    return {
        "name": name,
        "passed": passed,
        "reason": reason,
        "x": x,
        "y": y,
        "yaw_deg": math.degrees(yaw),
        "position_cm": final_error*100,
        "heading_deg": math.degrees(
            final_heading_error
        ),
        "max_lateral_cm": max_lateral*100,
        "max_steering_deg": math.degrees(
            max_delta
        ),
        "simulated_seconds": step*DT,
        "collision": collision,
    }


starts = [
    (
        "ACTUAL",
        (
            17.500650636,
            -5.963040895,
            math.radians(-92.82)
        )
    ),
    (
        "LEFT_2CM",
        (
            17.480650636,
            -5.963040895,
            math.radians(-92.82)
        )
    ),
    (
        "RIGHT_2CM",
        (
            17.520650636,
            -5.963040895,
            math.radians(-92.82)
        )
    ),
    (
        "HEADING_PLUS_2",
        (
            17.500650636,
            -5.963040895,
            math.radians(-90.82)
        )
    ),
    (
        "HEADING_MINUS_2",
        (
            17.500650636,
            -5.963040895,
            math.radians(-94.82)
        )
    ),
]


print()
print("=" * 75)
print("PHASE 14F — ACTUAL 4WS EQUATION TEST")
print("=" * 75)

print(
    "Steering rate:",
    STEERING_RATE,
    "rad/s"
)

print(
    "Acceleration:",
    ACCELERATION,
    "m/s²"
)

print(
    "Deceleration:",
    DECELERATION,
    "m/s²"
)

print(
    "Lateral acceleration limit:",
    LATERAL_ACCELERATION,
    "m/s²"
)

print(
    "Correction gains:",
    HEADING_GAIN,
    LATERAL_GAIN
)

print(
    "Maximum docking speed:",
    MAX_SPEED,
    "m/s"
)

print()

results = []

for name, start in starts:
    r = simulate(name, start)
    results.append(r)

    print(
        f"{name:20s} "
        f"{'PASS' if r['passed'] else 'FAIL':4s} "
        f"position={r['position_cm']:.2f}cm "
        f"heading={r['heading_deg']:.2f}deg "
        f"steering={r['max_steering_deg']:.2f}deg "
        f"time={r['simulated_seconds']:.2f}s "
        f"reason={r['reason']}"
    )


passed = sum(
    r["passed"]
    for r in results
)

report_path = ROOT / (
    "tools/v1_manhattan_bezier/"
    "phase14/routes_refined/"
    "phase14f_dynamic_4ws_report.json"
)

report_path.write_text(
    json.dumps({
        "schema": "phase14f_dynamic_4ws_v1",
        "gain_heading": HEADING_GAIN,
        "gain_lateral": LATERAL_GAIN,
        "passed": passed,
        "total": len(results),
        "results": results,
        "full_3d_clearance_validated": False,
        "motion_authorized": False,
    }, indent=2) + "\n"
)

print()
print("=" * 75)
print(
    f"DYNAMIC TESTS PASSED: "
    f"{passed}/{len(results)}"
)
print("Report:", report_path)
print("Full 3D clearance validated: NO")
print("Robot movement: NONE")
print("Motion authorized: NO")
print("=" * 75)

if passed != len(results):
    raise SystemExit(2)
