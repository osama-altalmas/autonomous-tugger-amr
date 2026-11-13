import math

import numpy as np

from tugger_control.scan_merger import BIN_RAD, bin_points


def _published(angle_min, ranges):
    i = int(np.flatnonzero(np.isfinite(ranges))[0])
    return angle_min + i * BIN_RAD, ranges[i]


def test_a_point_is_published_within_half_a_bin_of_its_angle():
    for angle in np.linspace(-math.pi + 1e-6, math.pi - 1e-6, 997):
        angle_min, ranges = bin_points([3.0 * math.cos(angle)], [3.0 * math.sin(angle)])
        published, r = _published(angle_min, ranges)
        error = math.atan2(math.sin(published - angle), math.cos(published - angle))
        assert abs(error) <= 0.5 * BIN_RAD + 1e-9
        assert abs(r - 3.0) < 1e-9


def test_no_systematic_turn_of_the_scan():
    """Uniformly spread points: the published angles average to the true ones."""
    rng = np.random.default_rng(0)
    angles = rng.uniform(-math.pi, math.pi, 20000)
    errors = []
    for chunk in np.array_split(angles, 200):
        # One point per call so the nearest-per-bin rule does not drop any.
        for angle in chunk[:20]:
            angle_min, ranges = bin_points([math.cos(angle)], [math.sin(angle)])
            published, _ = _published(angle_min, ranges)
            errors.append(math.atan2(math.sin(published - angle), math.cos(published - angle)))
    assert abs(math.degrees(np.mean(errors))) < 0.02     # was -0.25 (lower edge)


def test_nearest_return_wins_in_a_bin():
    angle_min, ranges = bin_points([2.0, 1.0], [0.0001, 0.0001])
    assert abs(np.nanmin(ranges) - 1.0) < 1e-6
    assert np.sum(np.isfinite(ranges)) == 1
    assert len(ranges) == int(round(2 * math.pi / BIN_RAD))
    assert abs(angle_min + math.pi - 0.5 * BIN_RAD) < 1e-12
