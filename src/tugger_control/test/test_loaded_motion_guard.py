import json
import math
from pathlib import Path

import pytest

from tugger_control.clearance_2_5d import Box
from tugger_control.loaded_motion_guard import (
    REQUIRED_MODELS, choose_guarded_speed, decode_scene, loaded_envelope_clear,
    scene_expected_counts,
)
from tugger_control.warehouse_clearance import static_world_boxes


def scene(stamp=10.0):
    return {'valid': True, 'stamp': stamp, 'boxes': [
        {'model': model, 'name': model + '::collision',
         'x': 20.0 + index, 'y': 0, 'z': 0.5,
         'hx': 0.1, 'hy': 0.1, 'hz': 0.1}
        for index, model in enumerate(sorted(REQUIRED_MODELS))]}


def test_scene_requires_complete_fresh_supported_collision_geometry():
    payload = scene()
    assert len(decode_scene(json.dumps(payload), 10.1)) == len(REQUIRED_MODELS)
    for bad in (dict(payload, valid=False), dict(payload, stamp=9.0),
                dict(payload, boxes=payload['boxes'][:-1])):
        with pytest.raises(ValueError):
            decode_scene(json.dumps(bad), 10.1)
    with pytest.raises(ValueError, match='missing modeled collision'):
        decode_scene(json.dumps(payload), 10.1,
                     expected_counts={next(iter(REQUIRED_MODELS)): 2})


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -0.1, 0.0])
def test_scene_rejects_invalid_obstacle_size(value):
    payload = scene()
    payload['boxes'][0]['hx'] = value
    with pytest.raises(ValueError):
        decode_scene(json.dumps(payload), 10.1)


def check(obstacles, speed=0.0, requested=0.0):
    return loaded_envelope_clear((0, 0, 0), (0, 0, 0.065, 0), 0.065, 0,
                                 obstacles, speed, requested)


def test_cargo_can_overhang_station_leg_below_it():
    leg = Box('leg', 0.6, 0.0, 0.0, 0.05, 0.05, 0.0, 0.29)
    assert check([('station_legs', leg)])[0]


def test_obstacle_at_cargo_height_stops_loaded_robot():
    overhead = Box('overhead', 0.6, 0.0, 0.0, 0.05, 0.05, 0.34, 0.40)
    clear, collision = check([('overhead_model', overhead)])
    assert not clear and collision[0] == 'cargo'


def test_stopping_envelope_blocks_obstacle_before_contact():
    wall = Box('wall', 0.84, 0.0, 0.0, 0.03, 0.1, 0.34, 0.40)
    assert check([('wall', wall)], speed=0.0)[0]
    assert not check([('wall', wall)], speed=0.10, requested=0.10)[0]


def test_unexpected_cargo_or_excess_speed_fails_closed():
    with pytest.raises(ValueError):
        loaded_envelope_clear((0, 0, 0), (0.1, 0, 0.065, 0), 0.065, 0, [], 0, 0)
    with pytest.raises(ValueError):
        check([], speed=1.3)
    with pytest.raises(ValueError):
        check([], requested=math.nan)


def test_station_legs_reduce_speed_instead_of_expanding_cargo_to_floor():
    legs = [('station_legs', Box(f'leg_{x}_{y}', x, y, 0.0, 0.04, 0.04, 0, 0.28))
            for x in (-0.56, 0.56) for y in (-0.56, 0.56)]
    velocity, hit = choose_guarded_speed(
        (0, 0, 0), (0, 0, 0.065, 0), 0.065, 0, legs, 0.0, 0.1)
    assert hit is None
    assert 0.0 < velocity < 0.1


def test_known_world_manifest_includes_all_fixed_legs_and_payloads():
    world = (Path(__file__).resolve().parents[2] /
             'tugger_gazebo/worlds/tugger_warehouse.sdf')
    expected = scene_expected_counts(world)
    assert expected['cargo_station_0_legs'] == 4
    assert expected['cargo_dropoff_northwest_legs'] == 4
    boxes = []
    for box in static_world_boxes(world):
        model, name = box.name.split('/', 1)
        boxes.append({'model': model, 'name': model + '::' + name,
                      'x': box.x, 'y': box.y, 'z': (box.z_min + box.z_max) / 2,
                      'hx': box.half_x, 'hy': box.half_y, 'hz': (box.z_max - box.z_min) / 2})
    for station in (0, 3):
        model = f'cargo_station_{station}_stand'
        boxes.append({'model': model, 'name': model + '::tabletop_collision',
                      'x': station, 'y': 0, 'z': .295,
                      'hx': .65, 'hy': .65, 'hz': .015})
    payload = json.dumps({'valid': True, 'stamp': 10.0, 'boxes': boxes})
    assert len(decode_scene(payload, 10.1, expected_counts=expected)) == len(boxes)


def test_station_one_departure_has_nonzero_guarded_speed_in_modeled_world():
    world = (Path(__file__).resolve().parents[2] /
             'tugger_gazebo/worlds/tugger_warehouse.sdf')
    obstacles = [(box.name.split('/', 1)[0], box) for box in static_world_boxes(world)]
    speed, hit = choose_guarded_speed(
        (-17.5, -7, 0), (-17.5, -7, 0.065, 0), 0.065, 0,
        obstacles, 0.0, 0.5)
    assert hit is None and 0 < speed <= 0.5
