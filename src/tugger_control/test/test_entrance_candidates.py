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
        if entry['target'] == 'HOME' or entry['status'] != 'VALIDATED_STATIC_CANDIDATE':
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
    # The 20 mission-chain routes are approved; gate variants (a route from
    # every other gate a station can be left by) wait for operator approval.
    primaries = {k: e for k, e in database['routes'].items() if 'gate_variant_of' not in e}
    assert len(primaries) == 20
    for key, entry in database['routes'].items():
        if 'gate_variant_of' in entry:
            assert entry['gate_variant_of'] in primaries, key
            # A variant without a route is planned live at run time.
            if entry['status'] == 'VALIDATED_STATIC_CANDIDATE':
                assert entry['execution_approved'] is True, key
    for entry in primaries.values():
        assert entry['execution_approved'] is True
        route = json.loads((ROUTES / entry['route_file']).read_text())
        assert route['execution_approved'] is True
    directory = ROOT / 'tools/v1_manhattan_bezier/phase14/routes_refined'
    for station in ('cargo_station_0', 'cargo_station_3',
                    'cargo_dropoff_northwest', 'cargo_dropoff_southeast'):
        route = json.loads((directory / f'HOME__TO__{station}__REFINED.json').read_text())
        assert route['execution_approved'] is True


def test_every_station_gate_has_a_route_to_every_destination():
    # A robot leaves by the gate it entered by, whichever mission it came from.
    database = json.loads((ROUTES / 'route_database.json').read_text())
    approaches = json.loads((ROOT / 'tools/mission_approaches/entrance_candidates.json')
                            .read_text())['candidates']
    pass_through = set(database['pass_through'])
    stations = database['stations']
    for key in approaches:
        station, gate = key.rsplit('__', 1)
        if station in pass_through:
            continue
        for target in stations:
            if target == station:
                continue
            # Every combination is generated; one the builder cannot route
            # is recorded as such and is planned live at run time.
            routed = any(e['source'] == station and e['target'] == target and
                         e.get('source_entrance') == gate and
                         e['status'] == 'VALIDATED_STATIC_CANDIDATE'
                         for e in database['routes'].values())
            recorded = f'{station}__{gate}__TO__{target}' in database['routes']
            assert routed or recorded, (station, gate, target)


def test_station_one_dropoff_route_starts_at_loaded_departure_staging_pose():
    # The chain decides the gate: Station 1 is left by the gate HOME's route
    # arrives at, starting exactly on that gate's staging pose.
    database = json.loads((ROUTES / 'route_database.json').read_text())
    arrive = database['routes']['HOME__TO__cargo_station_0']
    entry = database['routes']['cargo_station_0__TO__cargo_dropoff_northwest']
    route = json.loads((ROUTES / entry['route_file']).read_text())
    gate = arrive['target_entrance']
    staging = json.loads(DATA.read_text())['candidates'][f'cargo_station_0__{gate}']['staging_pose']
    assert route['source_entrance'] == entry['source_entrance'] == gate
    assert route['departure_transition_validated'] is True
    assert route['poses'][0][:2] == pytest.approx(staging[:2], abs=0.03)
    assert entry['departure_transition_validated'] is True


def test_station_three_dropoff_route_starts_at_loaded_departure_staging_pose():
    database = json.loads((ROUTES / 'route_database.json').read_text())
    entry = database['routes']['cargo_station_3__TO__cargo_dropoff_southeast']
    route = json.loads((ROUTES / entry['route_file']).read_text())
    assert route['source_entrance'] == 'south'
    assert route['target_entrance'] == 'north'
    assert route['departure_transition_validated'] is True
    # Exactly on the station axis (2.0 m from the stand centre), not the
    # grid cell centre 2.5 cm beside it (S-bend fix, 2026-11-12).
    assert route['poses'][0][:2] == pytest.approx([17.5, 5.0])
    assert route['source_pose'][:2] == pytest.approx([17.5, 5.0])
    assert entry['source_entrance'] == 'south'
    assert entry['departure_transition_validated'] is True


def test_known_warehouse_candidate_samples_clear_separate_lower_and_upper_envelopes():
    result = subprocess.run(
        [sys.executable, str(ROOT / 'tools/mission_approaches/audit_candidates.py')],
        check=True, capture_output=True, text=True)
    reports = json.loads(result.stdout)
    # Eight station entrances plus the two HOME gates.
    assert len(reports) == 10
    assert {'HOME__east', 'HOME__west'} <= set(reports)
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


def _mission_legs():
    chains = json.loads((ROOT / 'tools/v1_manhattan_bezier/mission_chains.json').read_text())
    return [chain for chain in chains['chains']]


def test_every_mission_chain_leaves_each_station_by_its_entry_gate():
    database = json.loads((ROUTES / 'route_database.json').read_text())['routes']
    chains = _mission_legs()
    assert chains
    for chain in chains:
        for before, station, after in zip(chain, chain[1:], chain[2:]):
            arrived = database[f'{before}__TO__{station}']
            leaving = database[f'{station}__TO__{after}']
            assert arrived['target_entrance'] == leaving['source_entrance'], (before, station, after)
            assert leaving['execution_approved'] is True


def test_every_mission_chain_entry_has_a_precision_candidate_for_that_gate():
    data = json.loads(DATA.read_text())
    database = json.loads((ROUTES / 'route_database.json').read_text())['routes']
    for chain in _mission_legs():
        for before, station in zip(chain, chain[1:]):
            if station == 'HOME':
                continue
            gate = database[f'{before}__TO__{station}']['target_entrance']
            candidate = data['candidates'][f'{station}__{gate}']
            assert candidate['execution_approved'] is True
