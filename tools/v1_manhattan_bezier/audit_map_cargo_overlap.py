#!/usr/bin/env python3
"""
AMR V1 — Cargo geometry versus occupancy map audit.

Purpose:
    Identify occupied map cells coinciding with modeled
    cargo geometry, without changing the original map.

This is a diagnostic, not permission to clear map cells.
"""

import json
import math
from pathlib import Path

import numpy as np

from build_network import (
    load_occupancy,
    world_to_grid,
    footprint_is_free,
)

ROOT = Path("tools/v1_manhattan_bezier")

MAP = Path(
    "src/tugger_navigation/maps/warehouse_map.yaml"
)

GEOMETRY = ROOT / "geometry/cargo_geometry.json"

OUTPUT = ROOT / "diagnostics/map_cargo_overlap.json"


def rectangle_corners(collision):
    x, y, _, _, _, yaw = collision["pose"]
    sx, sy, _ = collision["size"]

    c = math.cos(yaw)
    s = math.sin(yaw)

    points = []

    for px, py in (
        (-sx / 2, -sy / 2),
        ( sx / 2, -sy / 2),
        ( sx / 2,  sy / 2),
        (-sx / 2,  sy / 2),
    ):
        points.append([
            x + c * px - s * py,
            y + s * px + c * py,
        ])

    return np.asarray(points, dtype=float)


def rectangle_cell_overlap(poly, cx, cy):
    """
    SAT intersection between a rotated rectangle and
    a 1x1 occupancy-grid cell.
    """

    cell = np.asarray([
        [cx, cy],
        [cx + 1, cy],
        [cx + 1, cy + 1],
        [cx, cy + 1],
    ], dtype=float)

    for shape in (poly, cell):
        for i in range(4):
            a = shape[i]
            b = shape[(i + 1) % 4]

            edge = b - a
            axis = np.asarray([
                -edge[1],
                edge[0],
            ])

            p1 = poly @ axis
            p2 = cell @ axis

            if (
                p1.max() < p2.min() - 1e-9
                or
                p2.max() < p1.min() - 1e-9
            ):
                return False

    return True


def collision_cells(collision, spec, blocked):
    polygon = rectangle_corners(collision)
    grid_polygon = world_to_grid(polygon, spec)

    h, w = blocked.shape

    x0 = max(
        0, int(math.floor(grid_polygon[:, 0].min()))
    )
    x1 = min(
        w - 1,
        int(math.ceil(grid_polygon[:, 0].max()))
    )

    y0 = max(
        0, int(math.floor(grid_polygon[:, 1].min()))
    )
    y1 = min(
        h - 1,
        int(math.ceil(grid_polygon[:, 1].max()))
    )

    cells = set()

    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            if rectangle_cell_overlap(
                grid_polygon, x, y
            ):
                cells.add((x, y))

    return cells


def main():
    spec, blocked = load_occupancy(MAP)

    geometry = json.loads(
        GEOMETRY.read_text()
    )

    report = {
        "version": "AMR_V1_MAP_CARGO_AUDIT",
        "original_map_unchanged": True,
        "map_cell_removal_permitted": False,
        "stations": {},
        "execution_approved": False,
    }

    print()
    print("========== MAP / CARGO GEOMETRY ==========")
    print("Map:", MAP)
    print("Map resolution:", spec["resolution"])
    print("Map dimensions:", blocked.shape)
    print("Total occupied cells:", int(blocked.sum()))

    for name, station in geometry["stations"].items():

        table_cells = set()
        leg_cells = set()

        for collision in station.get(
            "stand_collisions", []
        ):
            table_cells.update(
                collision_cells(
                    collision, spec, blocked
                )
            )

        for collision in station.get(
            "leg_collisions", []
        ):
            leg_cells.update(
                collision_cells(
                    collision, spec, blocked
                )
            )

        modeled_cells = table_cells | leg_cells

        occupied_modeled = {
            (x, y)
            for x, y in modeled_cells
            if blocked[y, x]
        }

        occupied_table = {
            (x, y)
            for x, y in table_cells
            if blocked[y, x]
        }

        occupied_legs = {
            (x, y)
            for x, y in leg_cells
            if blocked[y, x]
        }

        entry_report = {}

        for direction, entry in station[
            "entries"
        ].items():

            pose = entry["predock_pose"]

            entry_report[direction] = {
                "predock_pose": pose,
                "original_map_footprint_clear": bool(
                    footprint_is_free(
                        pose, spec, blocked
                    )
                ),
                "chassis_vs_legs_clear": bool(
                    entry["leg_clearance_pass"]
                ),
            }

        report["stations"][name] = {
            "center": station["center"],
            "modeled_cells": len(modeled_cells),
            "occupied_model_overlap_cells": len(
                occupied_modeled
            ),
            "occupied_table_overlap_cells": len(
                occupied_table
            ),
            "occupied_leg_overlap_cells": len(
                occupied_legs
            ),
            "occupied_model_cell_coordinates": [
                [int(x), int(y)]
                for x, y in sorted(
                    occupied_modeled
                )
            ],
            "entries": entry_report,
        }

        print()
        print("STATION:", name)

        print(
            "  Modeled cargo cells:",
            len(modeled_cells)
        )

        print(
            "  Occupied cells overlapping cargo geometry:",
            len(occupied_modeled)
        )

        print(
            "  Occupied cells overlapping tabletop:",
            len(occupied_table)
        )

        print(
            "  Occupied cells overlapping legs:",
            len(occupied_legs)
        )

        for direction, result in entry_report.items():
            print(
                f"  {direction:5s} | "
                f"map clear: "
                f"{result['original_map_footprint_clear']} | "
                f"legs clear: "
                f"{result['chassis_vs_legs_clear']}"
            )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT.write_text(
        json.dumps(
            report,
            indent=2
        ) + "\n"
    )

    print()
    print("========== AUDIT COMPLETE ==========")
    print("Report:", OUTPUT)
    print("Original warehouse map: UNCHANGED")
    print("Map cells cleared: ZERO")
    print("Robot execution: NOT APPROVED")
    print("===================================")


if __name__ == "__main__":
    main()
