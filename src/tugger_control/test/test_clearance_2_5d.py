"""Layered clearance behavior with a stationary leg and elevated cargo."""

import math

import pytest

from tugger_control.clearance_2_5d import Box, boxes_overlap, collision_pairs


def test_lifted_cargo_can_pass_above_station_leg_without_ignoring_the_leg():
    leg = Box('fixed leg', 0.56, 0.56, 0, 0.04, 0.04, 0, 0.28)
    robot = Box('robot', 0, 0, 0, 0.44, 0.29, 0.03, 0.23)
    cargo = Box('cargo', 0, 0, 0, 0.65, 0.65, 0.345, 0.375)
    assert collision_pairs([robot, cargo], [leg], margin=0.01) == []
    shifted_robot = Box('robot', 0.2, 0.3, 0, 0.44, 0.29, 0.03, 0.23)
    assert collision_pairs([shifted_robot, cargo], [leg], margin=0.01) == [('robot', 'fixed leg')]


def test_cargo_height_obstacle_is_detected_even_when_lower_robot_is_clear():
    robot = Box('robot', 0, 0, 0, 0.44, 0.29, 0.03, 0.23)
    cargo = Box('cargo', 0, 0, 0, 0.65, 0.65, 0.345, 0.375)
    beam = Box('beam', 0.5, 0, 0, 0.1, 0.1, 0.34, 0.45)
    assert collision_pairs([robot, cargo], [beam]) == [('cargo', 'beam')]


def test_rotated_box_and_clearance_margin_are_respected():
    robot = Box('robot', 0, 0, math.pi / 4, 0.5, 0.2, 0, 0.2)
    obstacle = Box('obstacle', 0.55, 0, 0, 0.05, 0.05, 0, 0.2)
    assert not boxes_overlap(robot, obstacle)
    assert boxes_overlap(robot, obstacle, margin=0.15)


def test_invalid_geometry_is_rejected():
    with pytest.raises(ValueError):
        Box('invalid', 0, 0, 0, 0.5, 0.5, 0.2, 0.1)
