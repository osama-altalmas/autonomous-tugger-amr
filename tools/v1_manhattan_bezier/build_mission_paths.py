#!/usr/bin/env python3
"""Build one continuous path per mission, from the source dock to the target dock.

Every mission path is assembled from generated parts only:
  1. undock: the source's entry approach followed in reverse (the robot leaves
     by the gate it entered by; pass-through stops may use any gate),
  2. route: the route-database path between the two gate staging poses,
  3. approach: the target's segment 2 candidate ending on its exact dock pose.

Joins are checked for continuity, so the result is a single path from the
initial dock state to the final goal. Markers record where the route starts
and where segment 2 starts, so the runtime can drive each part with the
existing controllers. Offline only; no robot commands.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROUTES = ROOT / 'tools/v1_manhattan_bezier/routes'
CANDIDATES = ROOT / 'tools/mission_approaches/entrance_candidates.json'
OUTPUT = ROOT / 'tools/v1_manhattan_bezier/mission_paths'
SCHEMA = 'amr_single_segment_mission_path_v1'
JOIN_TOLERANCE_M = 0.03


def _xy_gap(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _reverse(poses):
    """Reverse travel order; recorded yaw stays the travel direction."""
    return [[x, y, math.atan2(math.sin(yaw + math.pi), math.cos(yaw + math.pi))]
            for x, y, yaw in reversed(poses)]


def _join(first, second, label):
    gap = _xy_gap(first[-1], second[0])
    if gap > JOIN_TOLERANCE_M:
        raise ValueError(f'{label}: parts are {gap:.3f} m apart')
    return first + (second[1:] if gap < 1e-6 else second)


def _length(poses):
    return sum(_xy_gap(a, b) for a, b in zip(poses, poses[1:]))


def build_mission_path(entry, route, candidates):
    """Return one mission path, or raise ValueError with the reason."""
    source, target = entry['source'], entry['target']
    undock_key = f"{source}__{entry['source_entrance']}"
    approach_key = f"{target}__{entry['target_entrance']}"
    if undock_key not in candidates:
        raise ValueError(f'no approach for {undock_key} to undock along')
    if approach_key not in candidates:
        raise ValueError(f'no approach for {approach_key}')
    undock = _reverse(candidates[undock_key]['complete_poses'])
    approach = [list(p) for p in candidates[approach_key]['complete_poses']]
    route_poses = [list(p) for p in route['poses']]
    path = _join(undock, route_poses, f'{source} undock to route')
    route_start = len(undock) - 1
    approach_start = len(path) - 1
    path = _join(path, approach, f'route to {target} approach')
    for a, b in zip(path, path[1:]):
        if _xy_gap(a, b) > 0.06:
            raise ValueError('path has a gap larger than 6 cm')
    approved = (entry.get('execution_approved') is True and
                route.get('execution_approved') is True and
                candidates[undock_key].get('execution_approved') is True and
                candidates[approach_key].get('execution_approved') is True)
    return {
        'schema': SCHEMA,
        'source': source,
        'target': target,
        'source_entrance': entry['source_entrance'],
        'target_entrance': entry['target_entrance'],
        'poses': path,
        'markers': {'route_start': route_start, 'approach_start': approach_start},
        'length_m': round(_length(path), 4),
        'provenance': {
            'undock': f'reverse of generated approach {undock_key}',
            'route': f"generated route {entry['route_file']}",
            'approach': f'generated approach {approach_key}',
        },
        'execution_approved': approved,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--routes', type=Path, default=ROUTES)
    parser.add_argument('--candidates', type=Path, default=CANDIDATES)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    database_bytes = (args.routes / 'route_database.json').read_bytes()
    database = json.loads(database_bytes)
    candidates_bytes = args.candidates.read_bytes()
    candidates = json.loads(candidates_bytes)['candidates']
    args.output.mkdir(parents=True, exist_ok=True)
    index, failures, planned_live = {}, {}, {}
    for key, entry in sorted(database['routes'].items()):
        if entry.get('status') != 'VALIDATED_STATIC_CANDIDATE':
            if entry.get('gate_variant_of'):
                # A gate combination the builder cannot route: the robot
                # still leaves by that gate and the leg is planned live.
                planned_live[key] = 'no validated route; planned live at run time'
            else:
                failures[key] = 'no validated route'
            continue
        try:
            route = json.loads((args.routes / entry['route_file']).read_text())
            mission = build_mission_path(entry, route, candidates)
        except (OSError, ValueError, KeyError) as exc:
            failures[key] = str(exc)
            continue
        (args.output / f'{key}.json').write_text(json.dumps(mission, indent=1) + '\n')
        index[key] = {field: mission[field] for field in (
            'source', 'target', 'source_entrance', 'target_entrance', 'length_m',
            'execution_approved')}
        index[key]['file'] = f'{key}.json'
    summary = {
        'schema': SCHEMA + '_index',
        'source_sha256': {
            'route_database': hashlib.sha256(database_bytes).hexdigest(),
            'candidates': hashlib.sha256(candidates_bytes).hexdigest(),
        },
        'paths': index,
        'failures': failures,
        'planned_live': planned_live,
    }
    (args.output / 'mission_paths.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'{len(index)} mission paths; {len(failures)} failures; '
          f'{len(planned_live)} gate combinations planned live')
    for key, reason in failures.items():
        print(f'  {key}: {reason}')
    raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
    main()
