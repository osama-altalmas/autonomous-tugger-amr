"""Routes leave a staging pose along its axis (no sideways first move)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'tools/v1_manhattan_bezier'))
from build_route_database_v1 import (  # noqa: E402
    DIRS, search_with_lead_out, start_is_sideways)

EAST, WEST, NORTH, SOUTH = range(4)


def test_sideways_first_move_is_detected():
    assert start_is_sideways([(5, 5), (5, 6)], EAST)          # north move, east-west axis
    assert not start_is_sideways([(5, 5), (6, 5)], EAST)
    assert not start_is_sideways([(5, 5), (4, 5)], EAST)      # reverse along the axis


def test_route_runs_straight_along_the_axis_before_turning():
    # Open 30 x 30 grid; staging faces east-west; goal straight north.
    safe = np.ones((30, 30), dtype=bool)
    cells = search_with_lead_out((15, 5), (15, 25), safe, NORTH, EAST, 6)
    assert not start_is_sideways(cells, EAST)
    first = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(cells[:6], cells[1:7])]
    assert len(set(first)) == 1 and first[0] in (DIRS[EAST], DIRS[WEST])
    assert cells[-1] == (15, 25)


def test_no_lead_out_room_means_no_route():
    safe = np.zeros((30, 30), dtype=bool)
    safe[:, 15] = True                                        # only a north-south corridor
    try:
        search_with_lead_out((15, 5), (15, 25), safe, NORTH, EAST, 6)
    except ValueError:
        return
    raise AssertionError('a sideways start was accepted')


def _runs_of(cells):
    runs = []
    for a, b in zip(cells, cells[1:]):
        move = (b[0] - a[0], b[1] - a[1])
        if runs and runs[-1][0] == move:
            runs[-1][1] += 1
        else:
            runs.append([move, 1])
    return runs


def test_turn_limited_corners_leave_room_to_smooth_both_sides():
    from build_route_database_v1 import turn_limited_astar
    safe = np.ones((60, 60), dtype=bool)
    cells = turn_limited_astar((5, 5), (50, 50), safe, NORTH, 6, axis=EAST)
    runs = _runs_of(cells)
    assert runs[0][1] >= 6 and runs[-1][1] >= 6         # first and last runs
    assert all(run[1] >= 12 for run in runs[1:-1])       # between two corners
    assert runs[0][0] in (DIRS[EAST], DIRS[WEST])


def test_coarse_to_fine_search_matches_the_rules_and_reports_no_route():
    import time
    from build_route_database_v1 import fast_turn_limited_astar
    safe = np.ones((120, 120), dtype=bool)
    safe[40:80, 20:100] = False                          # a block in the middle
    cells = fast_turn_limited_astar((10, 60), (110, 60), safe, EAST, 8, axis=EAST)
    runs = _runs_of(cells)
    assert cells[0] == (10, 60) and cells[-1] == (110, 60)
    assert all(safe[y, x] for x, y in cells)
    assert all(run[1] >= 16 for run in runs[1:-1])
    walled = safe.copy()
    walled[:, 60] = False                                # no way across
    t = time.time()
    try:
        fast_turn_limited_astar((10, 60), (110, 60), walled, EAST, 8, axis=EAST)
    except ValueError:
        assert time.time() - t < 5.0
        return
    raise AssertionError('a route through a wall was returned')


def test_route_ends_sit_exactly_on_the_staging_poses():
    """Cell-centre lanes are moved onto the station axes; corners stay 90 degrees."""
    from build_route_database_v1 import align_route_ends
    # Cell centres 2.5 cm off the axes: start (2, 0) east-west, goal (17.5, 5) north-south.
    points = [[2.025, 0.025], [17.525, 0.025], [17.525, 5.025]]
    out = align_route_ends(points, (2.0, 0.0), (17.5, 5.0))
    assert out == [[2.0, 0.0], [17.5, 0.0], [17.5, 5.0]]
    # Longer route: only the corners next to the ends move.
    points = [[2.025, 0.025], [6.525, 0.025], [6.525, 7.025], [15.525, 7.025]]
    out = align_route_ends(points, (2.0, 0.0), (15.5, 7.0))
    assert out == [[2.0, 0.0], [6.525, 0.0], [6.525, 7.0], [15.5, 7.0]]
    for a, b in zip(out, out[1:]):
        assert a[0] == b[0] or a[1] == b[1]          # every run axis-aligned
    # A single straight run is re-anchored only when both poses lie on one line.
    assert align_route_ends([[2.025, 0.025], [9.025, 0.025]], (2.0, 0.0), (9.0, 0.0)) == \
        [[2.0, 0.0], [9.0, 0.0]]
    assert align_route_ends([[2.025, 0.025], [9.025, 0.025]], (2.0, 0.0), (9.0, 0.05)) == \
        [[2.025, 0.025], [9.025, 0.025]]
    # Without poses nothing changes.
    assert align_route_ends(points) == [[float(v) for v in p] for p in points]
