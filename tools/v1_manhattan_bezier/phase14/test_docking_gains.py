#!/usr/bin/env python3
"""Offline Phase 14E correction-gain robustness test. No ROS commands."""

import json
import math
from pathlib import Path

ROUTE = Path(
    "tools/v1_manhattan_bezier/phase14/routes_refined/"
    "HOME__TO__cargo_dropoff_southeast__REFINED.json"
)

route = json.loads(ROUTE.read_text())
gx, gy, gh = route["final_docking_pose"]
legs = route["geometry"]["stand_legs"]


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def corners(x, y, yaw, hx, hy):
    c, s = math.cos(yaw), math.sin(yaw)
    return [
        (x + c*dx - s*dy, y + s*dx + c*dy)
        for dx, dy in (
            (-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)
        )
    ]


def overlap(a, b):
    for polygon in (a, b):
        for i, p in enumerate(polygon):
            q = polygon[(i + 1) % len(polygon)]
            axis = (p[1]-q[1], q[0]-p[0])

            pa = [u*axis[0]+v*axis[1] for u, v in a]
            pb = [u*axis[0]+v*axis[1] for u, v in b]

            if max(pa) < min(pb) or max(pb) < min(pa):
                return False
    return True


leg_shapes = [
    corners(
        leg["x"], leg["y"], leg["yaw"],
        leg["size"][0]/2,
        leg["size"][1]/2
    )
    for leg in legs
]


def simulate(x, y, yaw, heading_gain, lateral_gain):
    maximum_lateral = 0.0
    collision = False
    passed = False

    for step in range(1500):
        dx, dy = x-gx, y-gy

        remaining = -(dx*math.cos(gh) + dy*math.sin(gh))
        lateral = -math.sin(gh)*dx + math.cos(gh)*dy
        heading_error = wrap(gh-yaw)
        position_error = math.hypot(dx, dy)

        maximum_lateral = max(maximum_lateral, abs(lateral))

        body = corners(x, y, yaw, .49, .355)

        if any(overlap(body, leg) for leg in leg_shapes):
            collision = True
            break

        if position_error <= .02 and abs(heading_error) <= math.radians(2):
            passed = True
            break

        if remaining < .015 or abs(lateral) > .09:
            break

        curvature = (
            heading_gain*heading_error
            - lateral_gain*lateral
        )
        curvature = max(-.9, min(.9, curvature))

        ds = .002
        mid_yaw = yaw + curvature*ds/2

        x += ds*math.cos(mid_yaw)
        y += ds*math.sin(mid_yaw)
        yaw = wrap(yaw + curvature*ds)

    return {
        "passed": passed and not collision,
        "position_cm": math.hypot(x-gx, y-gy)*100,
        "heading_deg": abs(math.degrees(wrap(gh-yaw))),
        "max_lateral_cm": maximum_lateral*100,
        "collision": collision,
    }


# Actual measured starting pose and small nearby perturbations.
starts = [
    ("ACTUAL", 17.500650636, -5.963040895, -92.82),
    ("LEFT_2CM", 17.480650636, -5.963040895, -92.82),
    ("RIGHT_2CM", 17.520650636, -5.963040895, -92.82),
    ("HEADING_PLUS_2", 17.500650636, -5.963040895, -90.82),
    ("HEADING_MINUS_2", 17.500650636, -5.963040895, -94.82),
]

gain_sets = [
    ("ORIGINAL", 1.4, 2.5),
    ("CANDIDATE_A", 1.8, 2.5),
    ("CANDIDATE_B", 2.2, 3.5),
    ("CANDIDATE_C", 2.2, 5.0),
]

print("=" * 75)
print("PHASE 14E — OFFLINE CORRECTION ROBUSTNESS")
print("=" * 75)

for label, heading_gain, lateral_gain in gain_sets:
    print(
        f"\n{label}: heading_gain={heading_gain}, "
        f"lateral_gain={lateral_gain}"
    )

    passes = 0

    for name, x, y, yaw_deg in starts:
        result = simulate(
            x, y, math.radians(yaw_deg),
            heading_gain, lateral_gain
        )

        if result["passed"]:
            passes += 1

        print(
            f"  {name:20s} "
            f"{'PASS' if result['passed'] else 'FAIL':4s} "
            f"position={result['position_cm']:.2f} cm "
            f"heading={result['heading_deg']:.2f} deg "
            f"lateral={result['max_lateral_cm']:.2f} cm "
            f"collision={result['collision']}"
        )

    print(f"  SUMMARY: {passes}/{len(starts)} passed")

print("\nRobot movement: NONE")
print("Articulated-wheel 3D clearance: NOT VALIDATED")
print("Docking motion authorization: NO")
print("=" * 75)
