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
CHAINS = ROOT / 'tools/v1_manhattan_bezier/mission_chains.json'
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
    for key, candidate in pass_through_candidates(audit, database).items():
        if isinstance(candidate, str):
            failures[key] = candidate
        else:
            candidates[key] = candidate
    return candidates, failures


def pass_through_candidates(audit, database):
    """Segment 2 for stops without a stand, such as HOME.

    Uses the Phase 14 smooth extension from each gate's staging pose to the
    exact station centre. There are no legs or tabletop to check here; the
    sampled 2.5D world audit in main() still checks every pose.
    """
    pass_through = json.loads(CHAINS.read_text()).get('pass_through', [])
    centers = database.get('station_centers', {})
    result = {}
    for station in pass_through:
        for gate, node in database['nodes'].get(station, {}).items():
            key = f'{station}__{gate}'
            try:
                if not node.get('available'):
                    raise ValueError(f'gate unavailable: {node.get("reason")}')
                if station not in centers:
                    raise ValueError('route database has no station centre')
                staging = [float(value) for value in node['pose']]
                heading = staging[2]
                center = [float(centers[station][0]), float(centers[station][1]), heading]
                poses = audit.smooth_extension(staging, center, heading)
                result[key] = {
                    'schema': 'amr_phase14_two_segment_refined_v1',
                    'source': 'isolated_entrance_candidate',
                    'destination': station,
                    'entrance': gate,
                    'staging_pose': staging,
                    'final_docking_pose': center,
                    'entry_heading_deg': audit.math.degrees(heading),
                    'complete_poses': poses,
                    'validation': {
                        '2d_leg_geometry': 'NOT_APPLICABLE_NO_LEGS',
                        'static_vertical_body_lift_clearance':
                            'NOT_APPLICABLE_NO_TABLETOP_COLLISION',
                        'articulated_wheel_sweep': 'NOT_YET_VALIDATED',
                        'cargo_lift': 'NOT_APPLICABLE',
                    },
                    'execution_approved': False,
                    'candidate_limitations': [
                        'No stand at this stop; sampled 2.5D world audit only',
                        'Articulated wheels and dynamic obstacles not checked',
                        'Not verified in Gazebo and not approved for execution',
                    ],
                }
            except (KeyError, ValueError, TypeError) as exc:
                result[key] = str(exc)
    return result


def main():
    parser = argparse.ArgumentParser()
    global ROUTES
    parser.add_argument('--output', type=Path, required=True,
                        help='New output JSON file; refuses to overwrite an existing file')
    parser.add_argument('--add-missing', type=Path,
                        help='Add only candidates absent from this existing file, keeping '
                             'its candidates and approvals, and refresh its source hashes')
    parser.add_argument('--routes', type=Path, default=ROUTES,
                        help='Route database directory to read')
    args = parser.parse_args()
    ROUTES = args.routes
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
    if args.add_missing is not None:
        existing = json.loads(args.add_missing.read_text())
        added = {key: value for key, value in candidates.items()
                 if key not in existing['candidates']}
        existing['candidates'].update(added)
        existing['source_sha256'] = output['source_sha256']
        existing['failures'] = {key: value for key, value in failures.items()
                                if key in added or key not in existing['candidates']}
        output = existing
        print(f'added {sorted(added)}')
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(output, stream, indent=2)
        stream.write('\n')
    print(f'{len(candidates)} candidates; {len(failures)} failures; {args.output}')
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
