"""Route corners are circular arcs: constant curvature, tangent to both straights."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'tools/v1_manhattan_bezier'))
import plan_routes as p  # noqa: E402


def _curvature_profile(points):
    q = np.asarray(points, float)
    a, b, c = q[1:-1] - q[:-2], q[2:] - q[1:-1], q[2:] - q[:-2]
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return 2 * cross / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) *
                        np.linalg.norm(c, axis=1))


@pytest.mark.parametrize('radius', [0.75, 1.0, 1.5])
def test_corner_holds_one_curvature(radius):
    arc = p.bezier_arc((10.0 - radius, 0.0), (10.0, radius), (1.0, 0.0), (0.0, 1.0))
    k = _curvature_profile(arc)
    assert k.mean() == pytest.approx(1.0 / radius, rel=0.01)
    assert k.max() - k.min() < 1e-6 / radius            # exact circle (was 21 -> 12 -> 23 deg)
    assert np.allclose(arc[0], (10.0 - radius, 0.0)) and np.allclose(arc[-1], (10.0, radius))


def test_corner_is_tangent_to_both_straights_and_turns_ninety_degrees():
    arc = np.asarray(p.bezier_arc((9.0, 0.0), (10.0, -1.0), (1.0, 0.0), (0.0, -1.0)))
    start = math.atan2(*(arc[1] - arc[0])[::-1])
    end = math.atan2(*(arc[-1] - arc[-2])[::-1])
    assert abs(start) < math.radians(1)                   # leaves heading east
    assert abs(end + math.pi / 2) < math.radians(1)       # arrives heading south
    k = _curvature_profile(arc)
    assert np.all(k < 0)                                  # a right turn throughout


def test_smoothed_route_corner_is_the_arc():
    route = p.smooth([(0.0, 0.0), (5.0, 0.0), (5.0, 5.0)], trim=1.0)
    k = _curvature_profile(route)
    turning = k[np.abs(k) > 0.05]
    # The estimate straddles each join (straight to arc) once and reads about
    # half there; everywhere inside the corner it is the arc's curvature.
    inside = turning[1:-1]
    assert len(inside) > 20
    assert np.allclose(inside, 1.0, atol=1e-6)
    assert 0.0 < turning[0] < 1.0 and 0.0 < turning[-1] < 1.0


def test_non_square_corner_is_refused():
    with pytest.raises(ValueError):
        p.bezier_arc((0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (math.sqrt(0.5), math.sqrt(0.5)))


def test_last_corner_leaves_the_settling_straight():
    # Corner 2.475 m before the end: a full 1.0 m rounding would leave 1.475 m.
    route = np.asarray(p.smooth([(0.0, 0.0), (4.475, 0.0), (4.475, -2.475)], trim=1.0,
                                final_straight=1.52))
    k = np.abs(_curvature_profile(route))
    last_turning = int(np.flatnonzero(k > 0.05)[-1]) + 1
    straight = float(np.sum(np.linalg.norm(np.diff(route[last_turning:], axis=0), axis=1)))
    assert straight >= 1.5
    assert np.allclose(route[-1], (4.475, -2.475))
