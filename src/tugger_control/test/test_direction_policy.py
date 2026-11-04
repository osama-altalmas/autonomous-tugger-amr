import math

import pytest

from tugger_control.direction_policy import prepare_dynamic_path, validate_planner_motion


def test_front_aligned_dynamic_path_uses_positive_body_motion():
    result = prepare_dynamic_path([(0, 0), (1, 0), (2, 0)], 0.0)
    assert result['leading_end'] == 'FRONT'
    assert result['body_motion_direction'] == 'FORWARD'
    assert result['body_velocity_sign'] == 1
    assert result['body_headings'] == [0.0, 0.0, 0.0]


def test_rear_aligned_dynamic_path_uses_negative_body_motion():
    result = prepare_dynamic_path([(0, 0), (1, 0), (2, 0)], math.pi)
    assert result['leading_end'] == 'REAR'
    assert result['body_motion_direction'] == 'REVERSE'
    assert result['body_velocity_sign'] == -1
    assert all(abs(abs(heading) - math.pi) < 1e-9
               for heading in result['body_headings'])


def test_dynamic_path_rejects_an_ambiguous_start_heading():
    try:
        prepare_dynamic_path([(0, 0), (1, 0)], math.pi / 2)
    except ValueError as exc:
        assert 'AMBIGUOUS START' in str(exc)
    else:
        raise AssertionError('ambiguous dynamic path was accepted')


def test_planner_motion_accepts_single_direction_including_reverse():
    assert validate_planner_motion([(0, 0, 0), (1, 0, 0), (2, 0, 0)]) == 1
    assert validate_planner_motion([(0, 0, math.pi), (1, 0, math.pi)]) == -1


def test_planner_motion_rejects_reeds_shepp_cusp():
    with pytest.raises(ValueError, match='changes direction'):
        validate_planner_motion([(0, 0, 0), (1, 0, 0), (0.5, 0, 0)])


def test_planner_motion_rejects_heading_that_does_not_match_path():
    with pytest.raises(ValueError, match='heading is inconsistent'):
        validate_planner_motion([(0, 0, math.pi / 2), (1, 0, math.pi / 2)])
