#!/usr/bin/env python3

import argparse
import heapq
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_warehouse import (
    collect_collision_boxes,
    WORLD_X_MIN, WORLD_X_MAX,
    WORLD_Y_MIN, WORLD_Y_MAX,
)

ROOT = Path(__file__).resolve().parents[2]
WORLD = ROOT / "src/tugger_gazebo/worlds/tugger_warehouse.sdf"
RESULTS = Path(__file__).resolve().parent / "results"

RES = 0.05
STEP = 0.025

ROBOT_LENGTH = 1.00
ROBOT_WIDTH = 0.66
SAFETY_MARGIN = 0.30

CLEARANCE = math.hypot(
    ROBOT_LENGTH / 2,
    ROBOT_WIDTH / 2
) + SAFETY_MARGIN

WHEELBASE = 0.58
MAX_STEERING = math.radians(45)
MAX_CURVATURE = 2 * math.tan(MAX_STEERING) / WHEELBASE

HORIZONTAL_Y = [-7.0, 0.0, 7.0]
VERTICAL_X = [-6.5, 4.5]

X_STATIONS = [-18.0, -6.5, 4.5, 18.0]
Y_STATIONS = [-10.5, -7.0, 0.0, 7.0, 10.5]


def build_occupancy():
    world = ET.parse(WORLD).getroot().find("world")
    boxes, unsupported = collect_collision_boxes(world)

    if unsupported:
        raise RuntimeError(
            f"Unsupported collision geometry: {unsupported}"
        )

    nx = round((WORLD_X_MAX - WORLD_X_MIN) / RES)
    ny = round((WORLD_Y_MAX - WORLD_Y_MIN) / RES)

    occupied = np.zeros((ny, nx), dtype=bool)

    for box in boxes:
        corners = box["corners"]

        xmin = float(corners[:, 0].min())
        xmax = float(corners[:, 0].max())
        ymin = float(corners[:, 1].min())
        ymax = float(corners[:, 1].max())

        ix0 = max(0, int(math.floor((xmin-WORLD_X_MIN)/RES)))
        ix1 = min(nx, int(math.ceil((xmax-WORLD_X_MIN)/RES)))
        iy0 = max(0, int(math.floor((ymin-WORLD_Y_MIN)/RES)))
        iy1 = min(ny, int(math.ceil((ymax-WORLD_Y_MIN)/RES)))

        occupied[iy0:iy1, ix0:ix1] = True

    r = int(math.ceil(CLEARANCE / RES))
    yy, xx = np.ogrid[-r:r+1, -r:r+1]

    kernel = (xx*xx + yy*yy) <= (CLEARANCE/RES)**2

    protected = binary_dilation(
        occupied,
        structure=kernel,
        border_value=1
    )

    return occupied, protected


def collision_free(points, protected):
    ny, nx = protected.shape

    for p in points:
        x, y = p[:2]

        ix = int(math.floor((x-WORLD_X_MIN)/RES))
        iy = int(math.floor((y-WORLD_Y_MIN)/RES))

        if ix < 0 or iy < 0 or ix >= nx or iy >= ny:
            return False

        if protected[iy, ix]:
            return False

    return True


def sample_line(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    length = float(np.linalg.norm(b-a))
    n = max(2, math.ceil(length/STEP)+1)

    return [
        (a + (b-a)*t).tolist()
        for t in np.linspace(0, 1, n)
    ]


def build_graph(protected):
    nodes = {}
    adjacency = {}
    edges = {}

    def node(name, x, y):
        nodes[name] = [float(x), float(y)]
        adjacency[name] = []

    def edge(a, b):
        points = sample_line(nodes[a], nodes[b])
        valid = collision_free(points, protected)
        length = math.dist(nodes[a], nodes[b])

        key = f"{a}__{b}"

        edges[key] = {
            "from": a,
            "to": b,
            "length": round(length, 3),
            "valid": bool(valid)
        }

        if valid:
            adjacency[a].append((b, length))
            adjacency[b].append((a, length))

    # Three longitudinal aisles, four stations each.
    for yi, y in enumerate(HORIZONTAL_Y):
        for xi, x in enumerate(X_STATIONS):
            node(f"H{yi}_{xi}", x, y)

    # Cross-aisle endpoints.
    for xi, x in enumerate(VERTICAL_X):
        node(f"V{xi}_south", x, Y_STATIONS[0])
        node(f"V{xi}_north", x, Y_STATIONS[-1])

    # Horizontal graph edges.
    for yi in range(3):
        for xi in range(3):
            edge(f"H{yi}_{xi}", f"H{yi}_{xi+1}")

    # Vertical graph edges.
    for xi in range(2):
        chain = [
            f"V{xi}_south",
            f"H0_{xi+1}",
            f"H1_{xi+1}",
            f"H2_{xi+1}",
            f"V{xi}_north"
        ]

        for a, b in zip(chain[:-1], chain[1:]):
            edge(a, b)

    return nodes, adjacency, edges



def quintic_transition(start, corner, end, u, v):
    """
    Fifth-degree Bezier transition with zero endpoint curvature.

    First three control points are collinear with incoming tangent.
    Last three are collinear with outgoing tangent.
    """

    d_in = np.linalg.norm(corner - start)
    d_out = np.linalg.norm(end - corner)

    a = 0.40 * min(d_in, d_out)

    control = np.array([
        start,
        start + a*u,
        start + 2*a*u,
        end - 2*a*v,
        end - a*v,
        end,
    ])

    estimated_length = (
        np.linalg.norm(corner-start)
        + np.linalg.norm(end-corner)
    )

    n = max(40, math.ceil(estimated_length / STEP))

    samples = []

    for t in np.linspace(0.0, 1.0, n+1):
        omt = 1.0-t

        p = (
            omt**5 * control[0]
            + 5*omt**4*t * control[1]
            + 10*omt**3*t**2 * control[2]
            + 10*omt**2*t**3 * control[3]
            + 5*omt*t**4 * control[4]
            + t**5 * control[5]
        )

        samples.append(p.tolist())

    return samples


def smooth_route(waypoints, distance):
    """
    Smooth corners after graph route selection.

    The generated path has continuous position, tangent,
    and curvature at straight-to-curve transitions.
    """

    points = [
        np.asarray(p, dtype=float)
        for p in waypoints
    ]

    if len(points) < 2:
        return None, "Insufficient waypoints"

    result = [points[0].tolist()]
    cursor = points[0]

    for i in range(1, len(points)-1):
        previous = points[i-1]
        corner = points[i]
        following = points[i+1]

        incoming = corner-previous
        outgoing = following-corner

        len_in = np.linalg.norm(incoming)
        len_out = np.linalg.norm(outgoing)

        if len_in < 1e-8 or len_out < 1e-8:
            return None, "Degenerate route segment"

        u = incoming/len_in
        v = outgoing/len_out

        dot = float(np.clip(np.dot(u, v), -1, 1))
        angle = math.acos(dot)

        if angle < 1e-5:
            continue

        if angle > math.pi-1e-4:
            return None, "U-turn requires a separate maneuver"

        d = min(
            distance,
            0.45*len_in,
            0.45*len_out,
        )

        entry = corner-d*u
        departure = corner+d*v

        result.extend(sample_line(cursor, entry)[1:])

        transition = quintic_transition(
            entry,
            corner,
            departure,
            u,
            v,
        )

        result.extend(transition[1:])
        cursor = departure

    result.extend(sample_line(cursor, points[-1])[1:])

    # Eliminate consecutive duplicate points.
    cleaned = [result[0]]

    for point in result[1:]:
        if math.dist(point, cleaned[-1]) > 1e-7:
            cleaned.append(point)

    return cleaned, None

def check_kinematics(path):
    """
    Calculate curvature and steering from sampled XY geometry.
    This does not validate steering-rate continuity.
    """
    p = np.asarray(path, dtype=float)

    if len(p) < 3:
        return True, 0.0, 0.0

    a = p[1:-1]-p[:-2]
    b = p[2:]-p[1:-1]
    c = p[2:]-p[:-2]

    cross = np.abs(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0])

    denominator = (
        np.linalg.norm(a, axis=1)
        * np.linalg.norm(b, axis=1)
        * np.linalg.norm(c, axis=1)
    )

    curvature = np.divide(
        2*cross,
        denominator,
        out=np.zeros_like(cross),
        where=denominator > 1e-10
    )

    maximum = float(curvature.max())

    steering = math.atan(maximum*WHEELBASE/2)

    valid = maximum <= MAX_CURVATURE + 1e-3

    return valid, maximum, steering


def select_route(nodes, adjacency, protected,
                 start, goal, smoothing, direction):
    """
    Enumerate simple graph routes in increasing geometric length.
    Accept the first whose smoothed path passes validation.
    """
    queue = [(0.0, (start,))]
    attempts = 0

    while queue and attempts < 2000:
        length, sequence = heapq.heappop(queue)
        current = sequence[-1]

        if current == goal:
            attempts += 1

            waypoints = [nodes[n] for n in sequence]

            path, error = smooth_route(
                waypoints, smoothing
            )

            if path is None:
                continue

            if not collision_free(path, protected):
                continue

            feasible, curvature, steering = check_kinematics(
                path
            )

            if not feasible:
                continue

            return {
                "nodes": list(sequence),
                "graph_length_m": round(length, 3),
                "path_length_m": round(
                    sum(
                        math.dist(a, b)
                        for a, b in zip(path[:-1], path[1:])
                    ), 3
                ),
                "max_curvature": round(curvature, 5),
                "max_steering_deg": round(
                    math.degrees(steering), 3
                ),
                "direction": direction,
                "poses": make_poses(path, direction),
                "routes_evaluated": attempts
            }

        else:
            for neighbor, weight in adjacency[current]:
                if neighbor not in sequence:
                    heapq.heappush(
                        queue,
                        (
                            length+weight,
                            sequence+(neighbor,)
                        )
                    )

    return None


def make_poses(path, direction='forward'):
    result = []

    for i, p in enumerate(path):
        if i < len(path)-1:
            dx = path[i+1][0]-p[0]
            dy = path[i+1][1]-p[1]
        else:
            dx = p[0]-path[i-1][0]
            dy = p[1]-path[i-1][1]

        yaw = math.atan2(dy, dx)

        if direction == 'reverse':
            yaw += math.pi

        yaw = math.atan2(math.sin(yaw), math.cos(yaw))

        result.append([
            round(p[0], 4),
            round(p[1], 4),
            round(yaw, 5)
        ])

    return result


def draw(occupied, nodes, edges, route):
    fig, ax = plt.subplots(figsize=(15, 9))

    ax.imshow(
        occupied,
        origin="lower",
        extent=[
            WORLD_X_MIN, WORLD_X_MAX,
            WORLD_Y_MIN, WORLD_Y_MAX
        ],
        cmap="Greys",
        alpha=0.65
    )

    for edge in edges.values():
        a = nodes[edge["from"]]
        b = nodes[edge["to"]]

        ax.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            color="#17844c" if edge["valid"] else "red",
            linewidth=2,
            alpha=0.8
        )

    for name, point in nodes.items():
        ax.scatter(*point, color="black", s=18, zorder=3)

    if route:
        path = np.array(route["poses"])

        ax.plot(
            path[:, 0],
            path[:, 1],
            color="#1677ff",
            linewidth=4,
            label="Selected, interpolated route",
            zorder=5
        )

        ax.scatter(
            path[0, 0], path[0, 1],
            color="limegreen", s=90,
            edgecolors="black", zorder=6,
            label="Start"
        )

        ax.scatter(
            path[-1, 0], path[-1, 1],
            color="orange", s=90,
            edgecolors="black", zorder=6,
            label="Goal"
        )

        ax.legend(loc="lower center")

    ax.set_aspect("equal")
    ax.set_xlim(WORLD_X_MIN, WORLD_X_MAX)
    ax.set_ylim(WORLD_Y_MIN, WORLD_Y_MAX)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(
        "Warehouse graph and selected interpolated route\n"
        "Green: available graph edge | "
        "Red: blocked edge | Blue: selected route"
    )
    ax.grid(alpha=0.15)

    output = RESULTS / "warehouse_route_network.png"
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--start", default="H1_0")
    parser.add_argument("--goal", default="H2_3")
    parser.add_argument(
        "--smoothing",
        type=float,
        default=1.5,
        help="Requested distance before/after a corner (m)"
    )

    parser.add_argument(
        "--direction",
        choices=["forward", "reverse"],
        default="forward"
    )

    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)

    occupied, protected = build_occupancy()
    nodes, adjacency, edges = build_graph(protected)

    if args.start not in nodes or args.goal not in nodes:
        parser.error(
            "Unknown start/goal. Available nodes: "
            + ", ".join(sorted(nodes))
        )

    if args.start == args.goal:
        parser.error("Start and goal must differ")

    if args.smoothing <= 0:
        parser.error("Smoothing distance must be positive")

    route = select_route(
        nodes, adjacency, protected,
        args.start, args.goal, args.smoothing,
        args.direction
    )

    data = {
        "world": "tugger_test",
        "robot_clearance_radius_m": round(CLEARANCE, 4),
        "maximum_curvature": round(MAX_CURVATURE, 5),
        "smoothing_distance_m": args.smoothing,
        "nodes": nodes,
        "edges": edges,
        "selected_start": args.start,
        "selected_goal": args.goal,
        "driving_direction": args.direction,
        "interpolation": "quintic_bezier_zero_endpoint_curvature",
        "selected_route": route
    }

    json_path = RESULTS / "warehouse_route_network.json"
    json_path.write_text(json.dumps(data, indent=2))

    image_path = draw(occupied, nodes, edges, route)

    print("\n========== WAREHOUSE ROUTE PLANNER ==========")
    print("Nodes:", len(nodes))
    print("Graph edges:", len(edges))
    print("Valid edges:", sum(e["valid"] for e in edges.values()))
    print("Start:", args.start)
    print("Goal:", args.goal)
    print("Driving direction:", args.direction)
    print("Clearance radius:", round(CLEARANCE, 3), "m")

    if route:
        print("\nRESULT: GEOMETRIC ROUTE FOUND")
        print("Route:", " -> ".join(route["nodes"]))
        print("Graph length:", route["graph_length_m"], "m")
        print("Smoothed length:", route["path_length_m"], "m")
        print("Maximum steering:", route["max_steering_deg"], "deg")
        print("Routes evaluated:", route["routes_evaluated"])
    else:
        print("\nRESULT: NO VALID ROUTE FOUND")
        print("No path passed all current checks.")

    print("\nJSON:", json_path)
    print("Image:", image_path)


if __name__ == "__main__":
    main()
