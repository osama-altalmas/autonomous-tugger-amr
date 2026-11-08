#!/usr/bin/env python3
"""Export the localization map AMCL matches scans against (Phase B3).

The navigation map is deliberately conservative: a cell is occupied if any
part of an obstacle touches it, which makes walls up to one cell thicker.
That is right for planning but biases laser localization by up to half a
cell. A real site keeps a separate localization map built from the scanners;
here it is generated from the same world geometry, but only from what the
safety scanners can see (boxes crossing their scan plane) and with a cell
occupied only when its centre lies inside an obstacle. Same origin and extent
as the navigation map (one frame), but 2 cm cells: at 5 cm a wall face can
sit a whole cell from the nearest occupied cell centre, which limits laser
localization to a few centimetres.

Usage: python3 tools/warehouse_validation/export_localization_map.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/tugger_control'))
from tugger_control.warehouse_clearance import static_world_boxes  # noqa: E402

WORLD = ROOT / 'src/tugger_gazebo/worlds/tugger_warehouse.sdf'
NAV_MAP = ROOT / 'src/tugger_navigation/maps/warehouse_map.yaml'
OUT = ROOT / 'src/tugger_navigation/maps/localization_map'
RESOLUTION = 0.02
# Scan plane height above the floor (scanner origin z, URDF): base_link at
# 0.100 m plus the 0.0275 m mount.
SCAN_PLANE_Z = 0.1275


def main():
    spec = yaml.safe_load(NAV_MAP.read_text())
    nav_res = float(spec['resolution'])
    ox, oy = float(spec['origin'][0]), float(spec['origin'][1])
    raw = (NAV_MAP.parent / spec['image']).read_bytes().split(b'\n', 3)
    nav_width, nav_height = (int(v) for v in raw[1].split())
    res = RESOLUTION
    width = int(round(nav_width * nav_res / res))
    height = int(round(nav_height * nav_res / res))
    xs = ox + (np.arange(width) + 0.5) * res
    ys = oy + (np.arange(height)[::-1] + 0.5) * res   # row 0 is the top
    gx, gy = np.meshgrid(xs, ys)
    occupied = np.zeros((height, width), dtype=bool)
    seen = 0
    for box in static_world_boxes(WORLD):
        if not box.z_min <= SCAN_PLANE_Z <= box.z_max:
            continue  # the scanners cannot see it
        seen += 1
        c, s = math.cos(box.yaw), math.sin(box.yaw)
        dx, dy = gx - box.x, gy - box.y
        local_x = c * dx + s * dy
        local_y = -s * dx + c * dy
        occupied |= (np.abs(local_x) <= box.half_x) & (np.abs(local_y) <= box.half_y)
    image = np.where(occupied, 0, 254).astype(np.uint8)
    pgm = OUT.with_suffix('.pgm')
    pgm.write_bytes(f'P5\n{width} {height}\n255\n'.encode() + image.tobytes())
    OUT.with_suffix('.yaml').write_text(yaml.safe_dump({
        'image': pgm.name, 'mode': 'trinary', 'resolution': res,
        'origin': [ox, oy, 0.0], 'negate': 0,
        'occupied_thresh': 0.65, 'free_thresh': 0.196}, sort_keys=False))
    print(f'{seen} boxes in the scan plane; {int(occupied.sum())} occupied cells -> {pgm}')


if __name__ == '__main__':
    main()
