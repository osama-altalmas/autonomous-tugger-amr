"""Offline tests of the smooth back-out geometry."""

import math

import pytest

from tugger_control.backout_path import (
    MAX_CURVATURE, backout_candidates, max_curvature)

# Straight approach from a dock at the origin back to a gate 2 m along -x.
APPROACH_BACK = [(-0.01 * i, 0.0, math.pi) for i in range(201)]


@pytest.mark.parametrize('offset', [0.08, -0.08, 0.12])
def test_backout_starts_at_the_robot_and_joins_the_approach(offset):
    pose = (0.0, offset, 0.0)
    length, positions = next(backout_candidates(pose, APPROACH_BACK))
    assert positions[0] == (0.0, offset)
    # Leaves parallel to the approach: no sideways kink at the start.
    first = positions[5]
    assert abs(first[1] - offset) < 0.002
    # Back on the approach once the blend is over, and ends at the gate.
    joined = [p for p in positions if -p[0] >= length + 0.01]
    assert joined and all(abs(p[1]) < 1e-9 for p in joined)
    assert math.dist(positions[-1], (-2.0, 0.0)) < 1e-6
    assert max_curvature(positions) <= MAX_CURVATURE


def test_larger_offsets_get_longer_blends_within_the_turning_limit():
    small = next(backout_candidates((0.0, 0.03, 0.0), APPROACH_BACK))[0]
    large = next(backout_candidates((0.0, 0.12, 0.0), APPROACH_BACK))[0]
    assert small <= large


def test_no_candidate_when_the_offset_cannot_be_blended_out_in_time():
    short = APPROACH_BACK[:30]
    assert list(backout_candidates((0.0, 0.3, 0.0), short)) == []
