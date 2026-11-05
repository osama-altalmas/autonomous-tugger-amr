import math

from tugger_control.mission_planning import (
    assess_route_deviation,
    plan_is_current,
    select_destination_plan,
)


def _route(target='station'):
    return {'source': 'HOME', 'target': target}


def test_selects_an_interior_saved_route_rejoin():
    poses = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    result = select_destination_plan('station', [('HOME__TO__station', _route(), poses)],
                                     (1.05, 0.03, 0.0))
    assert result['mode'] == 'SAVED_REJOIN'
    assert result['rejoin_index'] == 1
    assert result['poses'] == [[1, 0, 0], [2, 0, 0]]
    assert result['direction']['leading_end'] == 'FRONT'


def test_selects_a_short_connector_for_a_rear_aligned_start():
    poses = [[0, 0, 0], [1, 0, 0]]
    result = select_destination_plan('station', [('HOME__TO__station', _route(), poses)],
                                     (-1.0, 0.0, math.pi))
    assert result['mode'] == 'SHORT_CONNECTOR'
    assert result['direction']['leading_end'] == 'REAR'


def test_rejects_a_rejoin_that_is_too_far_from_the_saved_route():
    poses = [[0, 0, 0], [1, 0, 0]]
    result = select_destination_plan('station', [('HOME__TO__station', _route(), poses)],
                                     (10.0, 0.0, 0.0))
    assert result['mode'] == 'LIVE_RECOVERY_REQUIRED'


def test_rejects_a_heading_ambiguous_rejoin():
    poses = [[0, 0, 0], [1, 0, 0]]
    result = select_destination_plan('station', [('HOME__TO__station', _route(), poses)],
                                     (0.0, 0.0, math.pi / 2))
    assert result['mode'] == 'LIVE_RECOVERY_REQUIRED'


def test_recognizes_arrival_instead_of_rejoining_a_route_terminal():
    poses = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    result = select_destination_plan(
        'station', [('HOME__TO__station', _route(), poses)], (2.02, 0.0, 0.0),
        destination_poses=[[2.0, 0.0, 0.0]])
    assert result['mode'] == 'ALREADY_AT_DESTINATION'


def test_does_not_rejoin_the_terminal_section_of_another_route():
    poses = [[0, 0, 0], [1, 0, 0], [1.8, 0, 0], [2, 0, 0]]
    result = select_destination_plan('station', [('HOME__TO__station', _route(), poses)],
                                     (2.01, 0.0, 0.0))
    assert result['mode'] == 'SHORT_CONNECTOR'
    assert result['rejoin_index'] == 1


def test_deviation_monitor_never_moves_route_progress_backwards():
    poses = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]
    result = assess_route_deviation(poses, (1.9, 0.6, 0.0), previous_index=2)
    assert result['state'] == 'REJOIN_REQUIRED'
    assert result['rejoin_index'] == 2
    assert result['remaining_poses'] == [[2, 0, 0], [3, 0, 0]]


def test_plan_must_be_rechecked_when_pose_or_heading_changed_since_preview():
    current = plan_is_current((1.0, 2.0, 0.0), (1.1, 2.0, math.radians(10)))
    moved = plan_is_current((1.0, 2.0, 0.0), (1.3, 2.0, 0.0))
    turned = plan_is_current((1.0, 2.0, 0.0), (1.0, 2.0, math.radians(25)))
    assert current['current'] is True
    assert moved['current'] is False
    assert turned['current'] is False


def test_entrance_gate_is_the_side_the_approach_comes_from():
    from tugger_control.mission_planning import entrance_gate
    assert entrance_gate([(-15.475, -6.975, 0), (-17.5, -7.0, 0)]) == 'east'
    assert entrance_gate([(-17.475, 5.025, 0), (-17.5, 7.0, 0)]) == 'south'
    assert entrance_gate([(17.525, -4.975, 0), (17.5, -7.0, 0)]) == 'north'
    assert entrance_gate([(15.525, 7.025, 0), (17.5, 7.0, 0)]) == 'west'


def test_only_routes_leaving_by_the_entry_gate_are_offered():
    from tugger_control.mission_planning import routes_from_gate
    good = ('a', {'source': 'S2', 'source_entrance': 'south'}, [])
    wrong_gate = ('b', {'source': 'S2', 'source_entrance': 'east'}, [])
    other_station = ('c', {'source': 'S1', 'source_entrance': 'south'}, [])
    assert routes_from_gate([good, wrong_gate, other_station], 'S2', 'south') == [good]
    assert routes_from_gate([wrong_gate], 'S2', 'south') == []


def test_home_accepts_whichever_end_the_robot_arrives_with():
    import math

    import pytest

    from tugger_control.mission_planning import home_alignment_yaw
    assert home_alignment_yaw(0.02) == 0.0
    assert home_alignment_yaw(-0.02) == 0.0
    assert home_alignment_yaw(3.134) == math.pi
    assert home_alignment_yaw(-3.134) == math.pi
    with pytest.raises(ValueError):
        home_alignment_yaw(float('nan'))
