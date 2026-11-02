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


def test_requests_live_recovery_when_no_candidate_is_nearby():
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
