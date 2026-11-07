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
    database = json.loads((PATHS.parent / 'routes/route_database.json').read_text())
    validated = [e for e in database['routes'].values()
                 if e['status'] == 'VALIDATED_STATIC_CANDIDATE']
    assert len(index['paths']) == len(validated) >= 35 and not index['failures']
    for key, entry in index['paths'].items():
        data = json.loads((PATHS / entry['file']).read_text())
        poses = data['poses']
        assert _gap(poses[0], CENTERS[data['source']]) < 1e-6, key
        assert _gap(poses[-1], CENTERS[data['target']]) < 1e-6, key
        assert max(_gap(a, b) for a, b in zip(poses, poses[1:])) < 0.06, key
        markers = data['markers']
        assert 0 < markers['route_start'] < markers['approach_start'] < len(poses) - 1


def test_no_mission_path_starts_its_route_sideways():
    # A car-like 4WS robot cannot start a 90-degree turn from rest at a gate.
    import math
    index = json.loads((PATHS / 'mission_paths.json').read_text())
    for key, entry in index['paths'].items():
        poses = json.loads((PATHS / entry['file']).read_text())['poses']
        start = json.loads((PATHS / entry['file']).read_text())['markers']['route_start']
        a, b, c = poses[start - 3], poses[start], poses[start + 3]
        before = math.atan2(b[1] - a[1], b[0] - a[0])
        after = math.atan2(c[1] - b[1], c[0] - b[0])
        turn = abs(math.degrees(math.atan2(math.sin(after - before), math.cos(after - before))))
        assert not 30 < turn < 150, (key, round(turn))


def test_no_route_passes_through_a_station():
    # Found 2026-11-07: a stale map let four routes drive under a stand.
    import math
    index = json.loads((PATHS / 'mission_paths.json').read_text())
    database = json.loads((PATHS.parent / 'routes/route_database.json').read_text())
    stands = {k: v for k, v in database['station_centers'].items()
              if k not in database['pass_through']}
    for key, entry in index['paths'].items():
        data = json.loads((PATHS / entry['file']).read_text())
        start, end = data['markers']['route_start'], data['markers']['approach_start']
        for station, (cx, cy) in stands.items():
            closest = min(math.hypot(p[0] - cx, p[1] - cy) for p in data['poses'][start:end + 1])
            assert closest >= 1.5, (key, station, round(closest, 2))


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


def test_static_map_has_no_movable_cargo():
    # Cargo moves between stations; only fixed structure belongs in the map.
    import yaml
    maps = ROOT / 'src/tugger_navigation/maps'
    meta = yaml.safe_load((maps / 'warehouse_map.yaml').read_text())
    raw = (maps / meta['image']).read_bytes().split(b'\n', 3)
    width, height = map(int, raw[1].split())
    pixels = raw[3]
    res, (ox, oy) = meta['resolution'], meta['origin'][:2]
    database = json.loads((PATHS.parent / 'routes/route_database.json').read_text())
    for station, (x, y) in database['station_centers'].items():
        col, row = int((x - ox) / res), height - 1 - int((y - oy) / res)
        assert pixels[row * width + col] == 254, f'{station} centre is not free in the map'
