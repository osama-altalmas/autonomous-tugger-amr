"""Pure, motion-free regression tests for reusable loaded station departure."""

import json
import math
from pathlib import Path

import pytest

from tugger_control.station_departure import plan_station_departure
from tugger_control.warehouse_clearance import loaded_path_clear, static_world_boxes

ROOT = Path(__file__).resolve().parents[3]
REFINED = ROOT / 'tools/v1_manhattan_bezier/phase14/routes_refined'
WORLD = ROOT / 'src/tugger_gazebo/worlds/tugger_warehouse.sdf'


def staged_approach(station):
    route = json.loads((REFINED / f'HOME__TO__{station}__REFINED.json').read_text())
    positions = route['complete_poses']
    staging = route['staging_pose']
    index = min(range(len(positions)), key=lambda n: math.hypot(
        positions[n][0] - staging[0], positions[n][1] - staging[1]))
    return positions[index:]


@pytest.mark.parametrize('station,body_yaw,leading_end,other', [
    ('cargo_station_0', 0.0, 'FRONT', (17.5, 7.0, 0.0, 0.0)),
    ('cargo_station_3', math.pi / 2, 'REAR', (-17.5, -7.0, 0.0, 0.0)),
])
def test_actual_station_departures_derive_direction_and_clear_loaded_geometry(
        station, body_yaw, leading_end, other):
    approach = staged_approach(station)
    current = (*approach[-1][:2], body_yaw)
    plan = plan_station_departure(current, approach)
    assert plan['leading_end'] == leading_end
    assert plan['length_m'] > 1.9
    assert math.hypot(plan['staging_pose'][0] - approach[0][0],
                      plan['staging_pose'][1] - approach[0][1]) < 1e-8
    cargo_pose = (current[0], current[1], 0.065, body_yaw)
    clear, samples, hit = loaded_path_clear(
        plan['poses'], current, cargo_pose, other, static_world_boxes(WORLD), 0.065)
    assert clear, hit
    assert samples > 100


def test_reusable_across_rotated_and_translated_facility_coordinates():
    # Same algorithm, different facility/station; no cargo-station names or XY constants.
    approach = [(2.0, 3.0, 0), (1.5, 3.0, 0), (1.0, 3.0, 0)]
    plan = plan_station_departure((1.0, 3.0, 0), approach)
    assert plan['leading_end'] == 'FRONT'
    assert plan['staging_pose'][:2] == [2.0, 3.0]
    # Rotate the setup ninety degrees, choose rear-leading departure.
    approach = [(12.0, 22.0, 0), (12.0, 21.5, 0), (12.0, 21.0, 0)]
    plan = plan_station_departure((12.0, 21.0, math.pi / 2), approach)
    assert plan['leading_end'] == 'FRONT'
    assert plan['staging_pose'][:2] == [12.0, 22.0]
    reverse_plan = plan_station_departure((12.0, 21.0, -math.pi / 2), approach)
    assert reverse_plan['leading_end'] == 'REAR'


@pytest.mark.parametrize('current', [
    (2.0, 0, 0), (0, 0, math.pi / 2), (0, 0, float('nan')),
])
def test_departure_rejects_wrong_pose_or_ambiguous_heading(current):
    with pytest.raises(ValueError):
        plan_station_departure(current, [(1.0, 0, 0), (0.0, 0, 0)])


def test_departure_rejects_too_short_approach():
    with pytest.raises(ValueError, match='far enough'):
        plan_station_departure((0, 0, 0), [(0.2, 0, 0), (0, 0, 0)])


def test_departure_does_not_mutate_source_or_assume_stored_yaw_is_body_yaw():
    approach = [(1, 0, math.pi), (0.5, 0, math.pi), (0, 0, math.pi)]
    original = list(approach)
    plan = plan_station_departure((0, 0, 0), approach)
    assert approach == original
    assert plan['leading_end'] == 'FRONT'
    assert max(abs(point[2]) for point in plan['poses']) < 1e-10


def test_departure_rejects_initial_motion_into_station_after_pose_overshoot():
    with pytest.raises(ValueError, match='back into the station'):
        plan_station_departure((0.1, 0, 0), [(0.03, 0, 0), (0, 0, 0)])
