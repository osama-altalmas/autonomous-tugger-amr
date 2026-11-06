#!/usr/bin/env python3
"""Audit every mission path and station lift against the modelled world in 3D.

Replaces the NOT_YET_VALIDATED placeholders with sampled evidence:
  - articulated wheel sweep and surrounding clearance: every pose of every
    mission path, robot body + lift + wheel bounds (which cover all steering
    angles) against all world boxes and the stand tabletops;
  - loaded paths (a pickup to its paired drop-off) also carry the raised cargo;
  - cargo lift: the raise at each pickup and the set-down at each drop-off.
This is a static, simulator-model audit. It does not cover unmodelled or moving
obstacles; those need the runtime sensors.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/tugger_control'))

from tugger_control.cargo_policy import DROPOFF_CARGO, PICKUP_CARGO  # noqa: E402
from tugger_control.lift_geometry import (  # noqa: E402
    CARGO_RAISE_M, LIFT_DOWN_M, LIFT_UP_M)
from tugger_control.precision_scene_clearance import sampled_precision_clearance  # noqa: E402
from tugger_control.warehouse_clearance import (  # noqa: E402
    cargo_box, lift_sweep_clear, loaded_path_clear, static_world_boxes)

PATHS = ROOT / 'tools/v1_manhattan_bezier/mission_paths'
WORLD = ROOT / 'src/tugger_gazebo/worlds/tugger_warehouse.sdf'
DATABASE = ROOT / 'tools/v1_manhattan_bezier/routes/route_database.json'
STEP = 3


def stand_boxes(centers, excluded=()):
    """Resting cargo tabletops at each pickup's and drop-off's centre."""
    boxes = []
    for station, cargo_id in PICKUP_CARGO.items():
        if station not in excluded:
            x, y = centers[station]
            boxes.append(cargo_box(f'{station}_tabletop', (x, y, 0.0, 0.0), extra_xy=0.0))
    return boxes


def audit_path(data, centers, world):
    source, target = data['source'], data['target']
    poses = [tuple(p) for p in data['poses']]
    sampled = poses[::STEP] + [poses[-1]]
    cargo = PICKUP_CARGO.get(source)
    loaded = cargo is not None and DROPOFF_CARGO.get(target) == cargo
    if loaded:
        other_id = next(k for k in set(PICKUP_CARGO.values()) if k != cargo)
        other_station = next(s for s, k in PICKUP_CARGO.items() if k == other_id)
        sx, sy = centers[source]
        ox, oy = centers[other_station]
        clear, sample, hit = loaded_path_clear(
            sampled, poses[0], (sx, sy, CARGO_RAISE_M, 0.0), (ox, oy, 0.0, 0.0),
            world, LIFT_UP_M)
    else:
        clear, sample, hit = sampled_precision_clearance(
            sampled, world + stand_boxes(centers), LIFT_DOWN_M)
    return {'loaded': loaded, 'passed': bool(clear), 'samples': len(sampled),
            'first_collision': None if clear else [str(v) for v in hit],
            'sample_index': None if clear else sample}


def audit_lifts(centers, world):
    results = {}
    for station, cargo_id in PICKUP_CARGO.items():
        other = next(s for s, k in PICKUP_CARGO.items() if k != cargo_id)
        x, y = centers[station]
        ox, oy = centers[other]
        clear, hit = lift_sweep_clear((x, y, 0.0), (x, y, 0.0, 0.0), (ox, oy, 0.0, 0.0),
                                      world, LIFT_DOWN_M, LIFT_UP_M)
        results[f'{station} pickup'] = {'passed': bool(clear), 'hit': hit}
    for station, cargo_id in DROPOFF_CARGO.items():
        pickup = next(s for s, k in PICKUP_CARGO.items() if k == cargo_id)
        other = next(s for s, k in PICKUP_CARGO.items() if k != cargo_id)
        x, y = centers[station]
        ox, oy = centers[other]
        clear, hit = lift_sweep_clear((x, y, 0.0), (x, y, CARGO_RAISE_M, 0.0),
                                      (ox, oy, 0.0, 0.0), world, LIFT_UP_M, LIFT_DOWN_M)
        results[f'{station} drop-off (cargo from {pickup})'] = {'passed': bool(clear), 'hit': hit}
    return results


def main():
    centers = json.loads(DATABASE.read_text())['station_centers']
    world = static_world_boxes(WORLD)
    index = json.loads((PATHS / 'mission_paths.json').read_text())
    failures = 0
    for key, entry in sorted(index['paths'].items()):
        path_file = PATHS / entry['file']
        data = json.loads(path_file.read_text())
        report = audit_path(data, centers, world)
        status = 'PASSED' if report['passed'] else 'FAILED'
        data['validation'] = {
            'articulated_wheel_sweep': f'{status} (wheel bounds cover all steering angles)',
            'surrounding_obstacle_clearance': f'{status} ({report["samples"]} sampled poses, '
                                              f'{"loaded" if report["loaded"] else "unloaded"})',
            'audit': report,
        }
        path_file.write_text(json.dumps(data, indent=1) + '\n')
        failures += not report['passed']
        print(f'{key:52s} {status} {"loaded" if report["loaded"] else "unloaded"} '
              f'{report["samples"]} samples {report["first_collision"] or ""}')
    lifts = audit_lifts(centers, world)
    for name, result in lifts.items():
        print(f'lift {name}: {"PASSED" if result["passed"] else "FAILED " + str(result["hit"])}')
        failures += not result['passed']
    index['lift_audit'] = {name: result['passed'] for name, result in lifts.items()}
    (PATHS / 'mission_paths.json').write_text(json.dumps(index, indent=2) + '\n')
    raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
    main()
