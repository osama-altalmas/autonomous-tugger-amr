"""Offline sampled geometry checks, not live docking validation."""

import json
from pathlib import Path

import pytest

from tugger_control.clearance_2_5d import Box
from tugger_control.direction_policy import prepare_dynamic_path
from tugger_control.precision_scene_clearance import (
    choose_precision_speed, precision_approach_speed_limit,
    precision_stopping_envelope, sampled_precision_clearance,
)
from tugger_control.warehouse_clearance import cargo_box, static_world_boxes


ROOT = Path(__file__).resolve().parents[3]


def test_station_one_nominal_approach_clears_modeled_legs_and_elevated_table():
    route_file = (ROOT / 'tools/v1_manhattan_bezier/phase14/routes_refined' /
                  'HOME__TO__cargo_station_0__REFINED.json')
    route = json.loads(route_file.read_text())
    source_poses = route['complete_poses']
    start = min(range(len(source_poses)), key=lambda index: (
        (source_poses[index][0] - route['staging_pose'][0]) ** 2 +
        (source_poses[index][1] - route['staging_pose'][1]) ** 2))
    source_poses = source_poses[start:]
    headings = prepare_dynamic_path([(p[0], p[1]) for p in source_poses], 0.0)['body_headings']
    poses = [(p[0], p[1], yaw) for p, yaw in zip(source_poses, headings)]
    world = ROOT / 'src/tugger_gazebo/worlds/tugger_warehouse.sdf'
    legs = [box for box in static_world_boxes(world)
            if box.name.startswith('cargo_station_0_legs/')]
    assert len(legs) == 4
    tabletop = cargo_box('station_1_table', (-17.5, -7.0, 0.0, 0.0), extra_xy=0)
    clear, count, hit = sampled_precision_clearance(poses, legs + [tabletop], 0.0)
    assert clear, hit
    assert count >= len(poses)
    # This is an offline worst-case guard check, not a live trajectory test.
    for pose in poses:
        clear, hit = precision_stopping_envelope(pose, 0.0, legs + [tabletop], 0.05, -0.05)
        assert clear, (pose, hit)


def test_interpolation_catches_obstacle_between_sparse_path_poses():
    obstacle = Box('unexpected_floor_obstacle', 0, 0, 0, 0.02, 0.02, 0, 0.15)
    clear, index, hit = sampled_precision_clearance([(-1, 0, 0), (1, 0, 0)], [obstacle], 0.0)
    assert not clear and index > 0
    assert hit[1] == 'unexpected_floor_obstacle'


@pytest.mark.parametrize('lift,margin,obstacles', [
    (0.06, 0.01, [Box('leg', 5, 5, 0, 0.1, 0.1, 0, 0.2)]),
    (0.0, 0.0, [Box('leg', 5, 5, 0, 0.1, 0.1, 0, 0.2)]),
    (0.0, 0.01, []),
])
def test_uncertain_inputs_fail_closed(lift, margin, obstacles):
    with pytest.raises(ValueError):
        sampled_precision_clearance([(0, 0, 0), (1, 0, 0)], obstacles, lift, margin=margin)


def test_stopping_envelope_distinguishes_elevated_cargo_and_low_obstacle():
    table = Box('table', 0, 0, 0, .65, .65, .28, .31)
    assert precision_stopping_envelope((0, 0, 0), 0.0, [table], 0.0, -0.05)[0]
    obstacle = Box('obstacle', .55, 0, 0, .02, .02, 0, .15)
    clear, hit = precision_stopping_envelope((0, 0, 0), 0.0, [table, obstacle], 0.0, 0.05)
    assert not clear and hit[1] == 'obstacle'


def test_precision_speed_steps_down_when_fast_envelope_is_blocked(monkeypatch):
    import tugger_control.precision_scene_clearance as clearance

    def fake_envelope(pose, lift, obstacles, current, requested, **kwargs):
        return (requested <= 0.1, 'near-leg' if requested > 0.1 else None)

    monkeypatch.setattr(clearance, 'precision_stopping_envelope', fake_envelope)
    speed, hit = choose_precision_speed((0, 0, 0), 0.0, [Box(
        'leg', 5, 5, 0, .1, .1, 0, .2)], 0.0, 0.4)
    assert speed == pytest.approx(0.1)
    assert hit is None


def test_precision_approach_tapers_near_fixed_cargo_legs_but_not_at_home():
    leg = Box('leg_1_0', 0.68, 0.0, 0.0, 0.04, 0.04, 0.0, 0.28)
    assert precision_approach_speed_limit((0.0, 0.0, 0.0),
                                           [('cargo_station_0_legs', leg)]) == pytest.approx(.10)
    assert precision_approach_speed_limit((0.0, 0.0, 0.0),
                                           [('cargo_station_0_legs',
                                             Box('leg_1_0', 1.1, 0.0, 0.0, .04, .04, 0, .28))]) == pytest.approx(.2)
    assert precision_approach_speed_limit((0.0, 0.0, 0.0),
                                           [('cargo_station_0_stand',
                                             Box('table', 0.0, 0.0, 0.0, .65, .65, .28, .31))]) == pytest.approx(.4)


@pytest.mark.parametrize('lift,velocity,obstacles', [
    (0.06, 0.03, [Box('far', 5, 5, 0, .1, .1, 0, .2)]),
    (0.0, 0.06, [Box('far', 5, 5, 0, .1, .1, 0, .2)]),
    (0.0, 0.03, []),
])
def test_stopping_envelope_rejects_invalid_operating_conditions(lift, velocity, obstacles):
    with pytest.raises(ValueError):
        precision_stopping_envelope((0, 0, 0), lift, obstacles, 0.0, velocity,
                                    max_speed=0.05)
