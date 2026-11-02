#!/usr/bin/env python3
"""Extract actual Gazebo cargo geometry and evaluate docking directions.

Read-only inspection. Does not modify Gazebo, Nav2, or robot controls.
"""

import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path


WORLD = Path("src/tugger_gazebo/worlds/tugger_warehouse.sdf")
ROBOT = Path("src/tugger_description/urdf/tugger.urdf.xacro")
OUTPUT = Path("tools/v1_manhattan_bezier/geometry/cargo_geometry.json")

PRE_DOCK_DISTANCE = 1.0

# Small physical collision tolerance, separate from navigation padding.
DOCK_MARGIN = 0.02


def pose_of(element):
    p = element.findtext("pose", default="0 0 0 0 0 0")
    values = [float(v) for v in p.split()]
    return (values + [0.0] * 6)[:6]


def compose(a, b):
    """Compose planar position and yaw."""
    c = math.cos(a[5])
    s = math.sin(a[5])
    return [
        a[0] + c*b[0] - s*b[1],
        a[1] + s*b[0] + c*b[1],
        a[2] + b[2],
        a[3] + b[3],
        a[4] + b[4],
        a[5] + b[5],
    ]



def box_collisions(model):
    """Read box collisions from a Gazebo SDF model."""
    result = []

    model_pose = pose_of(model)

    for link in model.findall("link"):
        link_pose = pose_of(link)

        for collision in link.findall("collision"):
            box = collision.find("geometry/box")

            if box is None:
                continue

            size_text = box.findtext("size")

            if not size_text:
                continue

            world_pose = compose(
                model_pose,
                compose(link_pose, pose_of(collision))
            )

            result.append({
                "name": collision.get("name", ""),
                "pose": world_pose,
                "size": [
                    float(v) for v in size_text.split()
                ]
            })

    return result


def corners(cx, cy, hx, hy, yaw):
    c = math.cos(yaw)
    s = math.sin(yaw)

    return [
        (
            cx + c*x - s*y,
            cy + s*x + c*y
        )
        for x, y in [
            (-hx, -hy),
            ( hx, -hy),
            ( hx,  hy),
            (-hx,  hy)
        ]
    ]


def intersects(a, b):
    """Separating Axis Theorem for two rectangles."""
    for polygon in (a, b):
        for i in range(4):
            p = polygon[i]
            q = polygon[(i+1) % 4]
            axis = (-(q[1]-p[1]), q[0]-p[0])

            ap = [x*axis[0]+y*axis[1] for x,y in a]
            bp = [x*axis[0]+y*axis[1] for x,y in b]

            if max(ap) < min(bp) or max(bp) < min(ap):
                return False

    return True




def robot_box(urdf_text):
    """Read base_link chassis box from expanded Xacro."""
    import subprocess

    result = subprocess.run(
        ["xacro", str(ROBOT.resolve())],
        capture_output=True,
        text=True,
        check=True
    )

    root = ET.fromstring(result.stdout)

    link = root.find("./link[@name='base_link']")

    if link is None:
        raise RuntimeError(
            "base_link not found in expanded URDF"
        )

    for collision in link.findall("collision"):
        box = collision.find("geometry/box")

        if box is None:
            continue

        # URDF uses <box size="..."/>.
        size_text = box.get("size")

        if not size_text:
            continue

        dimensions = [
            float(v) for v in size_text.split()
        ]

        if len(dimensions) == 3:
            return dimensions

    raise RuntimeError(
        "No chassis collision box found in base_link"
    )


def main():
    world = ET.parse(WORLD).getroot()
    robot_text = ROBOT.read_text()

    body_size = robot_box(robot_text)

    print("Robot chassis collision box:", body_size)

    models = {
        m.get("name"): m
        for m in world.findall(".//world/model")
    }

    report = {
        "source_world": str(WORLD),
        "source_robot": str(ROBOT),
        "robot_chassis_box_m": body_size,
        "navigation_padding_used_for_docking": False,
        "physical_docking_margin_m": DOCK_MARGIN,
        "stations": {},
        "execution_approved": False
    }


    station_bases = sorted(
        name[:-5]
        for name in models
        if name.endswith("_legs")
    )

    for base in station_bases:
        legs_name = base + "_legs"
        stand_name = base + "_stand"

        legs_model = models[legs_name]

        stand = models.get(stand_name)

        if stand is None:
            stand = models.get(base)

        if stand is None:
            print(
                "WARNING: No matching stand model for",
                base
            )
            continue

        center = pose_of(stand)
        leg_collisions = box_collisions(legs_model)
        stand_collisions = box_collisions(stand)

        if not leg_collisions:
            print("WARNING: No leg collisions:", legs_name)
            continue

        print()
        print("=" * 65)
        print("CARGO STATION:", base)
        print(
            "Center:",
            [round(center[0], 3), round(center[1], 3)]
        )
        print("Yaw:", round(math.degrees(center[5]), 2), "deg")
        print("Leg collision boxes:", len(leg_collisions))

        for leg in leg_collisions:
            print(
                "  LEG:",
                leg["name"],
                "center:",
                [round(v, 3) for v in leg["pose"][:3]],
                "size:",
                leg["size"]
            )

        entries = {}

        directions = {
            "east": 0.0,
            "north": math.pi/2,
            "west": math.pi,
            "south": -math.pi/2
        }

        for direction, local_angle in directions.items():

            outward = center[5] + local_angle

            px = center[0] + PRE_DOCK_DISTANCE*math.cos(outward)
            py = center[1] + PRE_DOCK_DISTANCE*math.sin(outward)

            heading = math.atan2(
                math.sin(outward + math.pi),
                math.cos(outward + math.pi)
            )

            # Check the entire straight entry using the actual
            # chassis collision box plus a small physical margin.
            step = 0.02
            count = math.ceil(PRE_DOCK_DISTANCE / step)

            collision = None

            for i in range(count + 1):
                t = i / count

                x = px + t*(center[0]-px)
                y = py + t*(center[1]-py)

                vehicle = corners(
                    x, y,
                    body_size[0]/2 + DOCK_MARGIN,
                    body_size[1]/2 + DOCK_MARGIN,
                    heading
                )

                for leg in leg_collisions:
                    lx, ly, _, _, _, lyaw = leg["pose"]
                    sx, sy, _ = leg["size"]

                    leg_polygon = corners(
                        lx, ly, sx/2, sy/2, lyaw
                    )

                    if intersects(vehicle, leg_polygon):
                        collision = {
                            "leg": leg["name"],
                            "progress": round(t, 3),
                            "robot_center": [
                                round(x, 3),
                                round(y, 3)
                            ]
                        }
                        break

                if collision:
                    break

            entries[direction] = {
                "predock_pose": [
                    round(px, 4),
                    round(py, 4),
                    round(heading, 5)
                ],
                "leg_clearance_pass": collision is None,
                "first_collision": collision,
                "checked_against": "base_link_chassis_and_stand_legs",
                "full_robot_envelope_verified": False,
                "floor_map_verified": False,
                "height_clearance_verified": False,
                "execution_approved": False
            }

            print(
                f"  {direction:5s}: "
                f"{'LEG CLEAR' if collision is None else 'LEG COLLISION'} "
                f"| pre-dock=({px:.3f}, {py:.3f}) "
                f"| heading={math.degrees(heading):.0f} deg"
            )

        report["stations"][base] = {
            "center": center,
            "stand_collisions": stand_collisions,
            "leg_collisions": leg_collisions,
            "entries": entries
        }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")

    print()
    print("=" * 65)
    print("Stations found:", len(report["stations"]))

    for name, station in report["stations"].items():
        valid = [
            direction
            for direction, entry in station["entries"].items()
            if entry["leg_clearance_pass"]
        ]
        print(name, "| chassis-vs-legs clear:", valid)

    print("Saved:", OUTPUT)
    print("ROBOT EXECUTION: NOT APPROVED")


if __name__ == "__main__":
    main()
