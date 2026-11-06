"""Known-world sampled lift checks; these are not Gazebo validation."""

from pathlib import Path
import json
import math
import pytest

from tugger_control.lift_geometry import LIFT_CONTACT_M, LIFT_UP_M

from tugger_control.clearance_2_5d import Box
from tugger_control.warehouse_clearance import (
    lift_sweep_clear, loaded_path_clear, static_world_boxes, validate_modeled_geometry,
)

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize('old,new', [
    ('${x} ${y} -0.045', '${x} ${y} -0.10'),
    ('<axis xyz="0 0 1"/>', '<axis xyz="1 0 0"/>'),
    ('radius="${wheel_radius}"', 'radius="0.5"'),
])
def test_geometry_guard_rejects_changed_wheel_or_lift_model(tmp_path, old, new):
    source = ROOT / 'src/tugger_description/urdf/tugger.urdf.xacro'
    text = source.read_text()
    assert old in text
    # The z axis occurs on steering and lift; mutate the final one for this case.
    if old.startswith('<axis'):
        index = text.rfind(old)
        text = text[:index] + text[index:].replace(old, new, 1)
    else:
        text = text.replace(old, new)
    altered = tmp_path / 'robot.xacro'
    altered.write_text(text)
    with pytest.raises(ValueError):
        validate_modeled_geometry(ROOT / 'src/tugger_gazebo/worlds/tugger_warehouse.sdf',
                                  altered)


def test_dropoff_allows_measured_joint_sag_but_not_real_penetration():
    obstacles = static_world_boxes(ROOT / 'src/tugger_gazebo/worlds/tugger_warehouse.sdf')
    # Values measured in Gazebo while holding cargo at the drop-off.
    measured = lift_sweep_clear((-17.5, 7.0, 0), (-17.5, 7.0, 0.0643, 0),
                                (17.5, 7.0, 0, 0), obstacles, LIFT_UP_M, 0.0)
    assert measured == (True, None)
    clear, hit = lift_sweep_clear((-17.5, 7.0, 0), (-17.5, 7.0, 0.055, 0),
                                  (17.5, 7.0, 0, 0), obstacles, LIFT_UP_M, 0.0)
    assert not clear
    assert hit[0] == 'cargo' and '_legs' in hit[1]


def test_pickup_and_paired_dropoff_allow_station_support_contacts():
    validate_modeled_geometry(ROOT / 'src/tugger_gazebo/worlds/tugger_warehouse.sdf',
                              ROOT / 'src/tugger_description/urdf/tugger.urdf.xacro')
    obstacles = static_world_boxes(ROOT / 'src/tugger_gazebo/worlds/tugger_warehouse.sdf')
    pickup = lift_sweep_clear((-17.5, -7.0, 0), (-17.5, -7.0, 0, 0),
                              (17.5, 7.0, 0, 0), obstacles, 0.0, LIFT_UP_M)
    dropoff = lift_sweep_clear((-17.5, 7.0, 0), (-17.5, 7.0, 0.065, 0),
                               (17.5, 7.0, 0, 0), obstacles, LIFT_UP_M, 0.0)
    assert pickup == (True, None)
    assert dropoff == (True, None)


def test_lift_sweep_rejects_an_overhead_obstacle():
    beam = Box('beam', 0.5, 0, 0, 0.1, 0.1, 0.34, 0.4)
    clear, hit = lift_sweep_clear((0, 0, 0), (0, 0, 0, 0), (5, 5, 0, 0),
                                  [beam], 0.0, LIFT_UP_M)
    assert not clear
    assert hit == ('cargo', 'beam')


def test_robot_collision_with_fixed_leg_is_still_rejected():
    leg = Box('station_legs/leg', 0.4, 0.2, 0, 0.04, 0.04, 0, 0.28)
    clear, hit = lift_sweep_clear((0, 0, 0), (0, 0, 0, 0), (5, 5, 0, 0),
                                  [leg], 0.0, LIFT_UP_M)
    assert not clear
    assert hit == ('chassis', 'station_legs/leg')


def test_loaded_path_uses_upper_cargo_envelope_for_overhead_obstacles():
    leg = Box('station_legs/leg', 0.56, 0.56, 0, 0.04, 0.04, 0, 0.28)
    path = [(0, 0, 0), (2, 0, 0)]
    inputs = (path, (0, 0, 0), (0, 0, 0.065, 0), (5, 5, 0, 0))
    assert loaded_path_clear(*inputs, [leg], LIFT_UP_M)[0]
    beam = Box('beam', 1.5, 0, 0, 0.05, 0.05, 0.34, 0.4)
    clear, _, hit = loaded_path_clear(*inputs, [leg, beam], LIFT_UP_M)
    assert not clear and hit == ('cargo', 'beam')


@pytest.mark.parametrize('station,goal,robot,other', [
    ('cargo_station_0', 'cargo_dropoff_northwest', (-17.5, -7.0, -math.pi / 2), (17.5, 7, 0, 0)),
    ('cargo_station_3', 'cargo_dropoff_southeast', (17.5, 7.0, math.pi / 2), (-17.5, -7, 0, 0)),
])
def test_sampled_loaded_departure_can_reach_saved_route_start(station, goal, robot, other):
    """Static straight-line geometry only; not a planner or Gazebo test."""
    route_file = (ROOT / 'tools/v1_manhattan_bezier/routes' /
                  f'{station}__TO__{goal}.json')
    route = json.loads(route_file.read_text())
    start = route['poses'][0]
    endpoint = (start[0], start[1], robot[2])
    obstacles = static_world_boxes(ROOT / 'src/tugger_gazebo/worlds/tugger_warehouse.sdf')
    cargo = (robot[0], robot[1], .065, robot[2])
    clear, _, hit = loaded_path_clear([robot, endpoint], robot, cargo, other, obstacles, .065)
    assert clear, hit


def test_lift_contact_height_matches_robot_and_world_geometry():
    import xml.etree.ElementTree as ET

    from tugger_control.lift_geometry import CARGO_UNDERSIDE_M, LIFT_LOWERED_TOP_M

    robot = ET.parse(ROOT / 'src/tugger_description/urdf/tugger.urdf.xacro').getroot()
    base = float(robot.find("./joint[@name='base_footprint_joint']/origin").get('xyz').split()[2])
    lift = float(robot.find("./joint[@name='lift_joint']/origin").get('xyz').split()[2])
    platform = robot.find("./link[@name='lift_platform_link']/collision/geometry/box")
    half_platform = float(platform.get('size').split()[2]) / 2
    assert abs(base + lift + half_platform - LIFT_LOWERED_TOP_M) < 1e-9
    upper = float(robot.find("./joint[@name='lift_joint']/limit").get('upper'))
    assert upper >= LIFT_UP_M

    world = ET.parse(ROOT / 'src/tugger_gazebo/worlds/tugger_warehouse.sdf').getroot()
    for station in ('cargo_station_0_stand', 'cargo_station_3_stand'):
        top = world.find(f".//model[@name='{station}']/link/collision")
        z = float(top.findtext('pose').split()[2])
        thickness = float(top.find('geometry/box/size').text.split()[2])
        assert abs(z - thickness / 2 - CARGO_UNDERSIDE_M) < 1e-9
