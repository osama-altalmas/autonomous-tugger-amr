"""Single-segment mission paths: generated data, reversal and selection."""
import json
import math
from pathlib import Path

from tugger_control.mission_planning import select_destination_plan
from tugger_control.saved_route_repository import MissionPathRepository

ROOT = Path(__file__).resolve().parents[3]
PATHS = ROOT / 'tools/v1_manhattan_bezier/mission_paths'
CENTERS = json.loads((ROOT / 'tools/v1_manhattan_bezier/routes/route_database.json')
                     .read_text())['station_centers']


def _gap(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def test_every_mission_path_is_one_continuous_dock_to_dock_path():
    index = json.loads((PATHS / 'mission_paths.json').read_text())
    assert len(index['paths']) == 20 and not index['failures']
    for key, entry in index['paths'].items():
        data = json.loads((PATHS / entry['file']).read_text())
        poses = data['poses']
        assert _gap(poses[0], CENTERS[data['source']]) < 1e-6, key
        assert _gap(poses[-1], CENTERS[data['target']]) < 1e-6, key
        assert max(_gap(a, b) for a, b in zip(poses, poses[1:])) < 0.06, key
        markers = data['markers']
        assert 0 < markers['route_start'] < markers['approach_start'] < len(poses) - 1


def test_reversed_path_ends_where_the_original_starts_with_swapped_markers():
    repo = MissionPathRepository(PATHS)
    forward = {k: (m, p) for k, m, p in repo.paths_to('cargo_station_0')}
    meta, poses = forward['HOME__TO__cargo_station_0']
    back_meta, back = {k: (m, p) for k, m, p in repo.paths_to('HOME')}[
        'HOME__TO__cargo_station_0 (reversed)']
    assert back_meta['reversed'] and back_meta['target'] == 'HOME'
    assert _gap(back[-1], CENTERS['HOME']) < 1e-6
    last = len(poses) - 1
    assert back_meta['markers']['approach_start'] == last - meta['markers']['route_start']
    assert back_meta['source_entrance'] == meta['target_entrance']


def test_cancelled_mission_returns_home_by_reversing_the_path_it_is_on():
    repo = MissionPathRepository(PATHS)
    meta, poses = {k: (m, p) for k, m, p in repo.paths_to('cargo_station_0')}[
        'HOME__TO__cargo_station_0']
    middle = (meta['markers']['route_start'] + meta['markers']['approach_start']) // 2
    x, y, yaw = poses[middle]
    decision = select_destination_plan('HOME', repo.paths_to('HOME'), (x, y, yaw))
    assert decision['mode'] == 'SAVED_REJOIN'
    assert decision['route_key'] == 'HOME__TO__cargo_station_0 (reversed)'
    assert decision['gap_m'] < 0.25


def test_rejoin_never_starts_inside_a_stand_zone():
    repo = MissionPathRepository(PATHS)
    candidates = repo.paths_to('cargo_station_0')
    meta, poses = {k: (m, p) for k, m, p in candidates}['HOME__TO__cargo_station_0']
    under_stand = poses[meta['markers']['approach_start'] + 50]
    decision = select_destination_plan('cargo_station_0', candidates, under_stand)
    if decision['mode'] in ('SAVED_REJOIN', 'SHORT_CONNECTOR'):
        chosen = decision['route']['markers']
        assert chosen['route_start'] <= decision['rejoin_index'] < chosen['approach_start']


def test_rejoined_mission_path_splits_at_its_segment_two_marker():
    repo = MissionPathRepository(PATHS)
    candidates = repo.paths_to('cargo_station_3')
    meta, poses = {k: (m, p) for k, m, p in candidates}['HOME__TO__cargo_station_3']
    start = poses[meta['markers']['route_start'] + 5]
    decision = select_destination_plan('cargo_station_3', candidates, start)
    split = decision['route']['markers']['approach_start'] - decision['rejoin_index']
    route_part, approach = decision['poses'][:split + 1], decision['poses'][split:]
    assert route_part[-1] == approach[0]
    assert _gap(approach[-1], CENTERS['cargo_station_3']) < 1e-6


def test_every_mission_path_carries_a_passed_3d_audit():
    index = json.loads((PATHS / 'mission_paths.json').read_text())
    assert index['lift_audit'] and all(index['lift_audit'].values())
    for entry in index['paths'].values():
        data = json.loads((PATHS / entry['file']).read_text())
        assert data['validation']['audit']['passed'] is True
        assert data['validation']['articulated_wheel_sweep'].startswith('PASSED')


def test_the_audit_catches_a_path_through_a_stand():
    import sys
    sys.path.insert(0, str(ROOT / 'tools/v1_manhattan_bezier'))
    from audit_mission_paths import audit_path
    from tugger_control.warehouse_clearance import static_world_boxes
    world = static_world_boxes(ROOT / 'src/tugger_gazebo/worlds/tugger_warehouse.sdf')
    # The unsafe live path from 2026-11-06: south along x = -17.975 past a leg.
    poses = [[-17.975, 11.0 - 0.05 * i, -math.pi / 2] for i in range(90)]
    report = audit_path({'source': 'HOME', 'target': 'cargo_station_3', 'poses': poses},
                        CENTERS, world)
    assert report['passed'] is False
