"""Offline entrance coverage; operator permission does not prove validation."""

import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from tugger_control.saved_route_repository import PrecisionDockingRepository

ROOT = Path(__file__).resolve().parents[3]
ROUTES = ROOT / 'tools/v1_manhattan_bezier/routes'
DATA = ROOT / 'tools/mission_approaches/entrance_candidates.json'


def test_all_saved_station_routes_have_a_matching_static_candidate():
    data = json.loads(DATA.read_text())
    database = json.loads((ROUTES / 'route_database.json').read_text())
    assert data['execution_approved'] is True
    assert not data['failures']
    for entry in database['routes'].values():
        if entry['target'] == 'HOME':
            continue
        key = f"{entry['target']}__{entry['target_entrance']}"
        candidate = data['candidates'][key]
        route = json.loads((ROUTES / entry['route_file']).read_text())
        end = route['poses'][-1]
        start = candidate['staging_pose']
        assert math.hypot(end[0] - start[0], end[1] - start[1]) < 1e-6
        assert candidate['validation']['2d_leg_geometry'] == 'PASSED'
        assert candidate['execution_approved'] is True
        assert candidate['validation']['articulated_wheel_sweep'] == 'NOT_YET_VALIDATED'
        assert candidate['validation']['cargo_lift'] == 'NOT_YET_VALIDATED'


def test_operator_enabled_every_database_route_and_precision_candidate():
    database = json.loads((ROUTES / 'route_database.json').read_text())
    assert len(database['routes']) == 20
    for entry in database['routes'].values():
        assert entry['execution_approved'] is True
        route = json.loads((ROUTES / entry['route_file']).read_text())
        assert route['execution_approved'] is True
    directory = ROOT / 'tools/v1_manhattan_bezier/phase14/routes_refined'
    for station in ('cargo_station_0', 'cargo_station_3',
                    'cargo_dropoff_northwest', 'cargo_dropoff_southeast'):
        route = json.loads((directory / f'HOME__TO__{station}__REFINED.json').read_text())
        assert route['execution_approved'] is True


def test_station_one_dropoff_route_starts_at_loaded_departure_staging_pose():
    database = json.loads((ROUTES / 'route_database.json').read_text())
    entry = database['routes']['cargo_station_0__TO__cargo_dropoff_northwest']
    route = json.loads((ROUTES / entry['route_file']).read_text())
    assert route['source_entrance'] == 'east'
    assert route['departure_transition_validated'] is True
    assert route['poses'][0][:2] == pytest.approx([-15.475, -6.975])
    assert route['source_pose'][:2] == pytest.approx([-15.475, -6.975])
    assert entry['source_entrance'] == 'east'
    assert entry['departure_transition_validated'] is True


def test_station_three_dropoff_route_starts_at_loaded_departure_staging_pose():
    database = json.loads((ROUTES / 'route_database.json').read_text())
    entry = database['routes']['cargo_station_3__TO__cargo_dropoff_southeast']
    route = json.loads((ROUTES / entry['route_file']).read_text())
    assert route['source_entrance'] == 'south'
    assert route['target_entrance'] == 'north'
    assert route['departure_transition_validated'] is True
    assert route['poses'][0][:2] == pytest.approx([17.525, 5.025])
    assert route['source_pose'][:2] == pytest.approx([17.525, 5.025])
    assert entry['source_entrance'] == 'south'
    assert entry['departure_transition_validated'] is True


def test_known_warehouse_candidate_samples_clear_separate_lower_and_upper_envelopes():
    result = subprocess.run(
        [sys.executable, str(ROOT / 'tools/mission_approaches/audit_candidates.py')],
        check=True, capture_output=True, text=True)
    reports = json.loads(result.stdout)
    assert len(reports) == 8
    assert all(report['passed'] for report in reports.values())


def test_repository_selects_the_actual_entrance_and_rejects_changed_world(tmp_path):
    sources = {
        'world': ROOT / 'src/tugger_gazebo/worlds/tugger_warehouse.sdf',
        'robot': ROOT / 'src/tugger_description/urdf/tugger.urdf.xacro',
        'database': ROUTES / 'route_database.json',
    }
    repo = PrecisionDockingRepository(
        ROOT / 'tools/v1_manhattan_bezier/phase14/routes_refined', DATA, sources)
    route = json.loads((ROUTES / 'cargo_dropoff_northwest__TO__cargo_station_0.json').read_text())
    candidate, poses = repo.approach('cargo_station_0', route['poses'][-1])
    assert candidate['entrance'] == 'north'
    assert poses[0][:2] == route['poses'][-1][:2]
    changed_world = tmp_path / 'world.sdf'
    changed_world.write_text('changed', encoding='utf-8')
    repo.source_files['world'] = changed_world
    with pytest.raises(ValueError, match='source changed: world'):
        repo.approach('cargo_station_0', route['poses'][-1])
