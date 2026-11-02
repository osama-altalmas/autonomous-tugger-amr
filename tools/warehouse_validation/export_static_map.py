#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import generate_route_network as planner
from validate_warehouse import (
    WORLD_X_MIN,
    WORLD_Y_MIN,
)

OUTPUT = (
    HERE.parents[1]
    / "src/tugger_navigation/maps"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

occupied, protected = planner.build_occupancy()

height, width = occupied.shape

# ROS occupancy-map convention:
# 0   = occupied (black)
# 254 = free (white)
#
# PGM rows are written top-to-bottom, whereas the
# occupancy array indexes Y from bottom-to-top.

image = np.where(occupied, 0, 254).astype(np.uint8)
image = np.flipud(image)

pgm_path = OUTPUT / "warehouse_map.pgm"
yaml_path = OUTPUT / "warehouse_map.yaml"

with pgm_path.open("wb") as f:
    f.write(
        f"P5\n{width} {height}\n255\n".encode("ascii")
    )
    f.write(image.tobytes())

metadata = {
    "image": "warehouse_map.pgm",
    "mode": "trinary",
    "resolution": float(planner.RES),
    "origin": [
        float(WORLD_X_MIN),
        float(WORLD_Y_MIN),
        0.0,
    ],
    "negate": 0,
    "occupied_thresh": 0.65,
    "free_thresh": 0.196,
}

with yaml_path.open("w") as f:
    yaml.safe_dump(metadata, f, sort_keys=False)

print("=== STATIC WAREHOUSE MAP GENERATED ===")
print("PGM:", pgm_path)
print("YAML:", yaml_path)
print("Dimensions:", width, "x", height)
print("Resolution:", planner.RES)
print("Origin:", WORLD_X_MIN, WORLD_Y_MIN)
print("Physical occupied cells:", int(occupied.sum()))
print("Free cells:", int((~occupied).sum()))
print("Protected cells (NOT exported):", int(protected.sum()))
