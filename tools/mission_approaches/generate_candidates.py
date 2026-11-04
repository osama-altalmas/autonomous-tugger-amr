#!/usr/bin/env python3
"""Generate isolated, motion-disabled precision candidates for route entrances.

Reuses the Phase 14 static leg/body audit without modifying its source or route
artifacts. Passing this limited audit does not authorize motion or certify
articulated wheels, payload clearance, dynamic obstacles, or Gazebo behavior.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PHASE14 = ROOT / 'tools/v1_manhattan_bezier/phase14'
ROUTES = ROOT / 'tools/v1_manhattan_bezier/routes'
STATIONS = ROOT / 'src/tugger_control/config/operator_stations.json'
WORLD = ROOT / 'src/tugger_gazebo/worlds/tugger_warehouse.sdf'
ROBOT = ROOT / 'src/tugger_description/urdf/tugger.urdf.xacro'


def load_auditor():
    spec = importlib.util.spec_from_file_location(
        'phase14_refine_read_only', PHASE14 / 'refine_docking_routes.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate():
    """Return per-entrance candidates and any failures; never command motion."""
    audit = load_auditor()
    models = audit.load_world(WORLD)
    robot = audit.load_robot_geometry(ROBOT)
    database = json.loads((ROUTES / 'route_database.json').read_text())
    stations = json.loads(STATIONS.read_text())['stations']
    requested = {(entry['target'], entry['target_entrance'])
                 for entry in database['routes'].values()
                 if entry['target'] != 'HOME'}
    candidates, failures = {}, {}
    for station, entrance in sorted(requested):
        key = f'{station}__{entrance}'
        try:
            entries = [item['pose'] for item in stations[station]['candidates']
                       if item['entrance'] == entrance]
            if len(entries) != 1:
                raise ValueError('Expected a unique configured station entrance')
            staging = entries[0]
            original = json.loads((PHASE14 / 'routes' /
                                  f'HOME__TO__{station}__TWO_SEGMENTS.json').read_text())
            original['staging_pose'] = staging
            original['segment_1']['poses'] = [staging]
            audit.STATIONS[station] = staging[2]
            candidate, report = audit.audit_station(station, original, models, robot)
            candidate['source'] = 'isolated_entrance_candidate'
            candidate['entrance'] = entrance
            candidate['execution_approved'] = False
            candidate['candidate_limitations'] = [
                'Static station leg/body audit only; articulated wheels not checked',
                'Surrounding and dynamic obstacle clearance not checked',
                'Loaded cargo, lift, and 3D swept-volume clearance not checked',
                'Not verified in Gazebo and not approved for execution',
            ]
            if not report['static_clearance_passed'] or report['leg_collisions']:
                raise ValueError('Static leg or lowered-body clearance audit failed')
            candidates[key] = candidate
        except (OSError, KeyError, ValueError, TypeError) as exc:
            failures[key] = str(exc)
    return candidates, failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True,
                        help='New output JSON file; refuses to overwrite an existing file')
    args = parser.parse_args()
    candidates, failures = generate()
    # This sampled SDF audit is additional evidence, not execution approval.
    from audit_candidates import audit_candidate
    for key, candidate in candidates.items():
        report = audit_candidate(candidate['destination'], candidate)
        candidate['static_2_5d_sampled_audit'] = report
        if not report['passed']:
            failures[key] = 'Static sampled 2.5D audit found a collision'
    output = {'schema': 'mission_entrance_candidates_v1',
              'execution_approved': False,
              'source_sha256': {
                  'world': hashlib.sha256(WORLD.read_bytes()).hexdigest(),
                  'robot': hashlib.sha256(ROBOT.read_bytes()).hexdigest(),
                  'database': hashlib.sha256((ROUTES / 'route_database.json').read_bytes()).hexdigest(),
              },
              'candidates': candidates, 'failures': failures}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(output, stream, indent=2)
        stream.write('\n')
    print(f'{len(candidates)} candidates; {len(failures)} failures; {args.output}')
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
