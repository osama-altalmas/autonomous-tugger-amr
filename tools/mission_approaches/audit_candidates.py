#!/usr/bin/env python3
"""Sample candidate approaches against the current static SDF collision boxes.

This is an offline, approximate audit, not a live motion or 3D safety proof.
It assumes the modeled cargo is centered and raised; dynamic obstacles, cargo
position uncertainty, full wheel articulation and lift sweeps need separate
checks. No route approval flags are changed.
"""

import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/tugger_control'))
from tugger_control.clearance_2_5d import Box, collision_pairs  # noqa: E402
from generate_candidates import WORLD, load_auditor  # noqa: E402

DATA = Path(__file__).with_name('entrance_candidates.json')
MARGIN = 0.01


def world_boxes(excluded_models=()):
    audit = load_auditor()
    root = ET.parse(WORLD).getroot()
    boxes = []
    for model in root.findall('./world/model'):
        name = model.get('name')
        if name in excluded_models or name == 'ground_plane':
            continue
        for item in audit.boxes_in_model(model):
            pose, size = item['pose'], item['size']
            if item['z_max'] <= 0:
                continue
            boxes.append(Box(f"{name}/{item['name']}", pose[0], pose[1],
                             pose[5], size[0] / 2, size[1] / 2,
                             item['z_min'], item['z_max']))
    return boxes


def robot_parts(pose, loaded):
    x, y, yaw = pose
    parts = [Box('chassis', x, y, yaw, 0.44, 0.29, 0.0725, 0.2225),
             Box('lift', x, y, yaw, 0.25, 0.16, 0.2225, 0.2375)]
    c, s = math.cos(yaw), math.sin(yaw)
    for wheel_x in (-0.29, 0.29):
        for wheel_y in (-0.22, 0.22):
            parts.append(Box('wheel_bound', x + c * wheel_x - s * wheel_y,
                             y + s * wheel_x + c * wheel_y, 0,
                             0.075, 0.075, 0.0, 0.11))
    if loaded:
        # 30 mm horizontal allowance for capture offset, in addition to margin.
        parts.append(Box('raised_cargo', x, y, yaw, 0.68, 0.68, 0.345, 0.375))
    return parts


def audit_candidate(station, candidate):
    loaded_cargo = {
        'cargo_dropoff_northwest': 'cargo_station_0_stand',
        'cargo_dropoff_southeast': 'cargo_station_3_stand',
    }.get(station)
    obstacles = world_boxes([loaded_cargo] if loaded_cargo else [])
    poses = candidate['complete_poses']
    for index, pose in enumerate(poses):
        hits = collision_pairs(robot_parts(pose, bool(loaded_cargo)), obstacles,
                               margin=MARGIN)
        if hits:
            return {'passed': False, 'pose_index': index, 'collisions': hits[:5]}
    return {'passed': True, 'samples': len(poses)}


def audit_all():
    data = json.loads(DATA.read_text())
    return {key: audit_candidate(candidate['destination'], candidate)
            for key, candidate in data['candidates'].items()}


if __name__ == '__main__':
    results = audit_all()
    print(json.dumps(results, indent=2))
    if not all(result['passed'] for result in results.values()):
        raise SystemExit(1)
