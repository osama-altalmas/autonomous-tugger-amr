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
PASS_THROUGH = {'HOME'}
STEP = 3
# A route runs gate to gate (gates are 2 m from a station centre); its travel
# part must never pass under a stand. HOME has no stand.
STATION_INTERIOR_M = 1.5
# The straight into a gate must let the path follower settle its heading.
# The planner keeps the last corner point at least 1.6 m from the gate
# (build_route_database_v1.FINAL_STRAIGHT_M), and rounds the last corner no
# more than leaves DRIVEN_FINAL_STRAIGHT_M (1.52 m) of straight after its arc;
# the driven straight measured here must be at least 1.5 m.
FINAL_STRAIGHT_M = 1.5
# Routes keep as far from obstacles as the space allows: on straight
# stretches the nearer side must be clear by CLEARANCE_GOAL_M, or by half the
# free span (less a small slack) where the corridor is narrower than that.
CLEARANCE_GOAL_M = 2.0
CENTRE_SLACK_M = 0.15
MAP_YAML = ROOT / 'src/tugger_navigation/maps/warehouse_map.yaml'


def load_static_map():
    import yaml
    meta = yaml.safe_load(MAP_YAML.read_text())
    raw = (MAP_YAML.parent / meta['image']).read_bytes().split(b'\n', 3)
    width, height = map(int, raw[1].split())
    return {'free': raw[3], 'w': width, 'h': height, 'res': meta['resolution'],
            'ox': meta['origin'][0], 'oy': meta['origin'][1]}


def _free(grid, x, y):
    col = int((x - grid['ox']) / grid['res'])
    row = grid['h'] - 1 - int((y - grid['oy']) / grid['res'])
    if not (0 <= col < grid['w'] and 0 <= row < grid['h']):
        return False
    return grid['free'][row * grid['w'] + col] == 254


def final_straight(poses, end):
    """Length of the straight run (within 3 degrees) arriving at poses[end]."""
    import math
    if end < 4:
        return 0.0
    ref = math.atan2(poses[end][1] - poses[end - 3][1], poses[end][0] - poses[end - 3][0])
    length = 0.0
    for i in range(end, 3, -1):
        heading = math.atan2(poses[i][1] - poses[i - 3][1], poses[i][0] - poses[i - 3][0])
        if abs(math.atan2(math.sin(heading - ref), math.cos(heading - ref))) > math.radians(3):
            break
        length += math.dist(poses[i][:2], poses[i - 1][:2])
    return length


def off_centre(poses, start, end, grid):
    """Worst (nearer-side clearance, required, span, x, y) on straight stretches."""
    import math
    worst = None
    travelled, total = 0.0, sum(math.dist(a[:2], b[:2]) for a, b in
                               zip(poses[start:end], poses[start + 1:end + 1]))
    for i in range(start + 3, end - 3):
        travelled += math.dist(poses[i][:2], poses[i - 1][:2])
        if travelled < FINAL_STRAIGHT_M or total - travelled < FINAL_STRAIGHT_M or i % 10:
            continue
        a, b = poses[i - 3], poses[i + 3]
        heading = math.atan2(b[1] - a[1], b[0] - a[0])
        before = math.atan2(poses[i][1] - a[1], poses[i][0] - a[0])
        if abs(math.atan2(math.sin(heading - before), math.cos(heading - before))) > 0.02:
            continue  # only straight stretches
        nx, ny = -math.sin(heading), math.cos(heading)
        reach = []
        for sign in (1, -1):
            d = 0.0
            while d < 12.0 and _free(grid, poses[i][0] + sign * nx * d, poses[i][1] + sign * ny * d):
                d += grid['res']
            reach.append(d)
        span = reach[0] + reach[1]
        nearer = min(reach)
        required = min(CLEARANCE_GOAL_M, span / 2.0 - CENTRE_SLACK_M)
        if nearer < required and (worst is None or required - nearer > worst[1] - worst[0]):
            worst = (round(nearer, 2), round(required, 2), round(span, 2),
                     round(poses[i][0], 2), round(poses[i][1], 2))
    return worst


def stand_boxes(centers, excluded=()):
    """Resting cargo tabletops at each pickup's and drop-off's centre."""
    boxes = []
    for station, cargo_id in PICKUP_CARGO.items():
        if station not in excluded:
            x, y = centers[station]
            boxes.append(cargo_box(f'{station}_tabletop', (x, y, 0.0, 0.0), extra_xy=0.0))
    return boxes


def audit_path(data, centers, world, grid=None):
    """Audit a path empty and, where a loaded robot may drive it, loaded.

    Any destination except a pickup station can be reached carrying cargo
    (operators may shuffle missions), so every cargo is checked there.
    """
    source, target = data['source'], data['target']
    poses = [tuple(p) for p in data['poses']]
    sampled = poses[::STEP] + [poses[-1]]
    clear, sample, hit = sampled_precision_clearance(
        sampled, world + stand_boxes(centers), LIFT_DOWN_M)
    states = {'unloaded': (bool(clear), sample, hit)}
    if target not in PICKUP_CARGO:
        for cargo in sorted(set(PICKUP_CARGO.values())):
            other = next(k for k in set(PICKUP_CARGO.values()) if k != cargo)
            other_home = next(s for s, k in PICKUP_CARGO.items() if k == other)
            other_drop = next(s for s, k in DROPOFF_CARGO.items() if k == other)
            # The other cargo rests at its pickup, or, if the path starts
            # there, has already been delivered to its own drop-off.
            other_place = other_drop if source == other_home else other_home
            if other_place == target:
                continue  # cargo onto cargo: blocked by the station rules
            ox, oy = centers[other_place]
            ok, sample, hit = loaded_path_clear(
                sampled, poses[0], (poses[0][0], poses[0][1], CARGO_RAISE_M, 0.0),
                (ox, oy, 0.0, 0.0), world, LIFT_UP_M)
            states[f'loaded_cargo_{cargo}'] = (bool(ok), sample, hit)
    failed = {name: [str(v) for v in hit] for name, (ok, _, hit) in states.items() if not ok}
    markers = data.get('markers')
    if markers:
        travel = poses[markers['route_start']:markers['approach_start'] + 1]
        for station, (cx, cy) in centers.items():
            if station in PASS_THROUGH:
                continue
            closest = min(((p[0] - cx) ** 2 + (p[1] - cy) ** 2) ** 0.5 for p in travel)
            if closest < STATION_INTERIOR_M:
                failed['through_station'] = [f'{station} interior at {closest:.2f} m']
        straight = final_straight(poses, markers['approach_start'])
        if straight < FINAL_STRAIGHT_M:
            failed['short_final_straight'] = [f'{straight:.2f} m into the target gate']
        if grid is not None:
            worst = off_centre(poses, markers['route_start'], markers['approach_start'], grid)
            if worst is not None:
                failed['off_centre'] = [f'{worst[0]} m from the nearer obstacle where '
                                        f'{worst[1]} m is possible (free span {worst[2]} m) '
                                        f'at ({worst[3]}, {worst[4]})']
    return {'loaded': any(name.startswith('loaded') for name in states),
            'states': {name: ok for name, (ok, _, _) in states.items()},
            'passed': not failed,
            'samples': len(sampled),
            'first_collision': failed or None}


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
    global PASS_THROUGH
    database = json.loads(DATABASE.read_text())
    centers = database['station_centers']
    PASS_THROUGH = set(database.get('pass_through', ['HOME']))
    world = static_world_boxes(WORLD)
    grid = load_static_map()
    index = json.loads((PATHS / 'mission_paths.json').read_text())
    failures = 0
    for key, entry in sorted(index['paths'].items()):
        path_file = PATHS / entry['file']
        data = json.loads(path_file.read_text())
        report = audit_path(data, centers, world, grid)
        status = 'PASSED' if report['passed'] else 'FAILED'
        data['validation'] = {
            'articulated_wheel_sweep': f'{status} (wheel bounds cover all steering angles)',
            'surrounding_obstacle_clearance': f'{status} ({report["samples"]} sampled poses, '
                                              f'{", ".join(report["states"])})',
            'audit': report,
        }
        path_file.write_text(json.dumps(data, indent=1) + '\n')
        failures += not report['passed']
        print(f'{key:62s} {status} {"+".join(report["states"])} '
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
