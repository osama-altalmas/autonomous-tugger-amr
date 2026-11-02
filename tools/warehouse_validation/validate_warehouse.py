#!/usr/bin/env python3

import math
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit(
        "Missing matplotlib. Install python3-matplotlib."
    )

from scipy.ndimage import binary_dilation


ROOT = Path(__file__).resolve().parents[2]

WORLD_FILE = (
    ROOT / "src/tugger_gazebo/worlds/tugger_warehouse.sdf"
)

OUTPUT_DIR = ROOT / "tools/warehouse_validation/results"

RESOLUTION = 0.05

WORLD_X_MIN = -20.0
WORLD_X_MAX = 20.0
WORLD_Y_MIN = -12.5
WORLD_Y_MAX = 12.5

ROBOT_LENGTH = 1.00
ROBOT_WIDTH = 0.66
SAFETY_MARGIN = 0.30

AISLE_Y = [-7.0, 0.0, 7.0]

# Candidate cross-aisle locations. They are NOT automatically
# accepted as navigable routes.
CROSS_AISLE_X = [-6.5, 4.5]

# Height occupied by the AMR body and sensors.
# Conservative initial vertical collision envelope.
ROBOT_Z_MIN = 0.0
ROBOT_Z_MAX = 0.50

# Exclude the floor from horizontal obstacle detection.
FLOOR_NAME_HINTS = ("floor", "ground_plane")

# Ignore decorative models without collision geometry.


def parse_pose(element):
    if element is None or not element.text:
        return np.zeros(6)

    values = [float(x) for x in element.text.split()]

    if len(values) != 6:
        raise ValueError(f"Unsupported pose: {values}")

    return np.array(values)


def rotation_matrix(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr, cr],
    ])

    ry = np.array([
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp],
    ])

    rz = np.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1],
    ])

    return rz @ ry @ rx


def transform(pose):
    result = np.eye(4)
    result[:3, :3] = rotation_matrix(*pose[3:])
    result[:3, 3] = pose[:3]
    return result


def collect_collision_boxes(world):
    boxes = []
    unsupported = []

    for model in world.findall("model"):
        model_name = model.get("name", "")

        if any(
            hint in model_name.lower()
            for hint in FLOOR_NAME_HINTS
        ):
            continue

        model_tf = transform(parse_pose(model.find("pose")))

        for link in model.findall("link"):
            link_tf = model_tf @ transform(
                parse_pose(link.find("pose"))
            )

            for collision in link.findall("collision"):
                collision_tf = link_tf @ transform(
                    parse_pose(collision.find("pose"))
                )

                geometry = collision.find("geometry")

                if geometry is None:
                    continue

                box = geometry.find("box")

                if box is None:
                    unsupported.append(
                        f"{model_name}/{link.get('name')}"
                    )
                    continue

                size_text = box.findtext("size")

                if not size_text:
                    continue

                size = np.array([
                    float(v) for v in size_text.split()
                ])

                if len(size) != 3:
                    continue

                corners = []

                for sx in (-0.5, 0.5):
                    for sy in (-0.5, 0.5):
                        for sz in (-0.5, 0.5):
                            point = np.array([
                                sx * size[0],
                                sy * size[1],
                                sz * size[2],
                                1.0,
                            ])

                            corners.append(
                                (collision_tf @ point)[:3]
                            )

                corners = np.array(corners)

                z_min = float(corners[:, 2].min())
                z_max = float(corners[:, 2].max())

                if (
                    z_max < ROBOT_Z_MIN
                    or z_min > ROBOT_Z_MAX
                ):
                    continue

                boxes.append({
                    "name": model_name,
                    "corners": corners[:, :2],
                    "z_min": z_min,
                    "z_max": z_max,
                })

    return boxes, unsupported


def points_inside_convex_polygon(x, y, vertices):
    """Vectorized convex-polygon occupancy test."""
    inside_pos = np.ones(x.shape, dtype=bool)
    inside_neg = np.ones(x.shape, dtype=bool)

    for i in range(len(vertices)):
        a = vertices[i]
        b = vertices[(i + 1) % len(vertices)]

        cross = (
            (b[0] - a[0]) * (y - a[1])
            - (b[1] - a[1]) * (x - a[0])
        )

        inside_pos &= cross >= -1e-9
        inside_neg &= cross <= 1e-9

    return inside_pos | inside_neg


def rasterize(boxes):
    nx = int(round(
        (WORLD_X_MAX - WORLD_X_MIN) / RESOLUTION
    ))
    ny = int(round(
        (WORLD_Y_MAX - WORLD_Y_MIN) / RESOLUTION
    ))

    occupied = np.zeros((ny, nx), dtype=bool)

    for item in boxes:
        corners = item["corners"]

        # All eight transformed box corners project onto
        # the same convex 2D footprint for upright boxes.
        # A bounding rectangle is conservative for rotated
        # boxes and avoids missing occupied cells.
        xmin = float(corners[:, 0].min())
        xmax = float(corners[:, 0].max())
        ymin = float(corners[:, 1].min())
        ymax = float(corners[:, 1].max())

        ix0 = max(
            0,
            int(math.floor(
                (xmin - WORLD_X_MIN) / RESOLUTION
            ))
        )

        ix1 = min(
            nx,
            int(math.ceil(
                (xmax - WORLD_X_MIN) / RESOLUTION
            ))
        )

        iy0 = max(
            0,
            int(math.floor(
                (ymin - WORLD_Y_MIN) / RESOLUTION
            ))
        )

        iy1 = min(
            ny,
            int(math.ceil(
                (ymax - WORLD_Y_MIN) / RESOLUTION
            ))
        )

        if ix0 < ix1 and iy0 < iy1:
            occupied[iy0:iy1, ix0:ix1] = True

    return occupied


def protected_occupancy(occupied):
    """
    Conservative circular inflation of the occupancy grid.

    Radius is the half-diagonal of the physical footprint
    plus safety margin. This is more conservative than an
    orientation-specific rectangular footprint.
    """
    radius = (
        math.hypot(
            ROBOT_LENGTH / 2,
            ROBOT_WIDTH / 2
        )
        + SAFETY_MARGIN
    )

    cells = int(math.ceil(radius / RESOLUTION))

    yy, xx = np.ogrid[
        -cells:cells + 1,
        -cells:cells + 1
    ]

    kernel = (
        (xx * RESOLUTION) ** 2
        + (yy * RESOLUTION) ** 2
    ) <= radius ** 2

    return binary_dilation(
        occupied,
        structure=kernel
    ), radius


def world_to_cell(x, y, grid):
    ix = int((x - WORLD_X_MIN) / RESOLUTION)
    iy = int((y - WORLD_Y_MIN) / RESOLUTION)

    if (
        ix < 0 or iy < 0
        or iy >= grid.shape[0]
        or ix >= grid.shape[1]
    ):
        return None

    return iy, ix


def is_free(grid, x, y):
    index = world_to_cell(x, y, grid)

    if index is None:
        return False

    return not bool(grid[index])


def analyze_line(grid, axis, value):
    """
    Find contiguous free sections along a proposed
    longitudinal or transverse centerline.
    """
    if axis == "x":
        coordinates = np.arange(
            WORLD_X_MIN + RESOLUTION / 2,
            WORLD_X_MAX,
            RESOLUTION
        )
        samples = [
            is_free(grid, float(x), value)
            for x in coordinates
        ]
    else:
        coordinates = np.arange(
            WORLD_Y_MIN + RESOLUTION / 2,
            WORLD_Y_MAX,
            RESOLUTION
        )
        samples = [
            is_free(grid, value, float(y))
            for y in coordinates
        ]

    segments = []
    start = None

    for i, free in enumerate(samples):
        if free and start is None:
            start = i

        if start is not None and (
            not free or i == len(samples) - 1
        ):
            end = i if not free else i + 1

            if end > start:
                segments.append([
                    round(float(coordinates[start]), 2),
                    round(float(coordinates[end - 1]), 2),
                ])

            start = None

    return segments


def save_plot(occupied, protected, aisles, crosses):
    fig, ax = plt.subplots(figsize=(15, 9))

    ax.imshow(
        occupied.astype(float),
        origin="lower",
        extent=[
            WORLD_X_MIN,
            WORLD_X_MAX,
            WORLD_Y_MIN,
            WORLD_Y_MAX,
        ],
        cmap="Greys",
        vmin=0,
        vmax=1,
        alpha=0.85,
    )

    ax.imshow(
        np.ma.masked_where(
            ~protected,
            protected
        ),
        origin="lower",
        extent=[
            WORLD_X_MIN,
            WORLD_X_MAX,
            WORLD_Y_MIN,
            WORLD_Y_MAX,
        ],
        cmap="Reds",
        alpha=0.22,
    )

    for y in aisles:
        ax.axhline(
            y,
            color="green",
            linestyle="--",
            linewidth=1.5
        )

    for x in crosses:
        ax.axvline(
            x,
            color="orange",
            linestyle="--",
            linewidth=1.2
        )

    ax.set_title(
        "Warehouse geometry and conservative "
        "robot clearance envelope"
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.15)

    fig.tight_layout()

    output = OUTPUT_DIR / "warehouse_clearance.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)

    return output


def main():
    if not WORLD_FILE.exists():
        raise SystemExit(
            f"Warehouse SDF not found: {WORLD_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    root = ET.parse(WORLD_FILE).getroot()
    world = root.find("world")

    if world is None:
        raise SystemExit("No <world> element found.")

    boxes, unsupported = collect_collision_boxes(world)

    occupied = rasterize(boxes)

    protected, radius = protected_occupancy(
        occupied
    )

    longitudinal = {
        str(y): analyze_line(
            protected,
            "x",
            y
        )
        for y in AISLE_Y
    }

    transverse = {
        str(x): analyze_line(
            protected,
            "y",
            x
        )
        for x in CROSS_AISLE_X
    }

    report = {
        "world": world.get("name"),
        "world_file": str(WORLD_FILE),
        "resolution_m": RESOLUTION,
        "robot_length_m": ROBOT_LENGTH,
        "robot_width_m": ROBOT_WIDTH,
        "safety_margin_m": SAFETY_MARGIN,
        "conservative_clearance_radius_m": round(
            radius, 3
        ),
        "collision_boxes": len(boxes),
        "unsupported_collisions": unsupported,
        "longitudinal_free_segments": longitudinal,
        "transverse_free_segments": transverse,
        "note": (
            "Free segments are geometric candidates only. "
            "Turning feasibility and swept-footprint "
            "validation are not yet performed."
        ),
    }

    report_path = OUTPUT_DIR / "warehouse_report.json"

    report_path.write_text(
        json.dumps(report, indent=2)
    )

    image_path = save_plot(
        occupied,
        protected,
        AISLE_Y,
        CROSS_AISLE_X
    )

    print("\n========== WAREHOUSE VALIDATION ==========")
    print("World:", report["world"])
    print("Collision boxes:", len(boxes))
    print("Unsupported collisions:", len(unsupported))
    print("Grid resolution:", RESOLUTION, "m")
    print(
        "Conservative clearance radius:",
        round(radius, 3),
        "m"
    )

    print("\nLONGITUDINAL CENTERLINE CANDIDATES")

    for y, segments in longitudinal.items():
        print(f"  y={y}: {segments}")

    print("\nCROSS-AISLE CANDIDATES")

    for x, segments in transverse.items():
        print(f"  x={x}: {segments}")

    if unsupported:
        print("\nWARNING: Unsupported collision geometry:")
        for item in unsupported[:20]:
            print(" ", item)

    print("\nReport:", report_path)
    print("Image:", image_path)


if __name__ == "__main__":
    main()
