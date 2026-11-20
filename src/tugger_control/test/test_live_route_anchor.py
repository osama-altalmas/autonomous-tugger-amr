"""A live A* path must start on the robot and end exactly on the station entry."""
import math

from tugger_control.live_route_planner import ANCHOR_MAX_M, anchor_ends


def grid_path():
    """A path on cell centres: 2.5 cm off in x and y, along x, ending 2.5 cm past the entry."""
    return [[5.025 + 0.05 * i, 7.025, 0.0] for i in range(211)]      # 5.025 .. 15.525


def test_the_end_lands_exactly_on_the_entry_and_the_start_on_the_robot():
    poses = anchor_ends(grid_path(), (5.0, 7.0, 0.0), (15.5, 7.0, 0.0))
    assert poses[0][:2] == [5.0, 7.0] and poses[-1][:2] == [15.5, 7.0]


def test_the_sideways_shift_fades_along_the_straight_so_the_curvature_stays_tiny():
    poses = anchor_ends(grid_path(), (5.0, 7.0, 0.0), (15.5, 7.0, 0.0))
    steps = [abs(b[1] - a[1]) for a, b in zip(poses, poses[1:])]
    assert max(steps) < 0.001          # 2.5 cm spread over metres, not one jump


def test_a_reversing_start_is_anchored_too():
    path = [[9.875 - 0.05 * i, 7.025, math.pi] for i in range(40)]
    poses = anchor_ends(path, (9.9, 7.0, 0.0), (0.0, 0.0, 0.0))
    assert poses[0][:2] == [9.9, 7.0]


def test_an_end_too_far_or_on_another_heading_is_left_for_the_checks():
    far = anchor_ends(grid_path(), (5.0, 7.0, 0.0), (15.5, 7.0 + ANCHOR_MAX_M + 0.02, 0.0))
    assert far[-1][:2] == [15.525, 7.025]
    turned = anchor_ends(grid_path(), (5.0, 7.0, 0.0), (15.5, 7.0, math.pi / 2))
    assert turned[-1][:2] == [15.525, 7.025]
