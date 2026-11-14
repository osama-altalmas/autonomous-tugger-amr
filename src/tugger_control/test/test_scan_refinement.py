"""Scan-to-map refinement of AMCL's pose (scan_refinement.py)."""
import math

import numpy as np
import pytest

from tugger_control import scan_refinement as sr

RES = 0.02


def _room(width=12.0, height=8.0, features=True):
    """Occupancy grid of a walled room (origin at its lower-left corner)."""
    w, h = int(width / RES), int(height / RES)
    occ = np.zeros((h, w), dtype=bool)
    occ[:2, :] = occ[-2:, :] = occ[:, :2] = occ[:, -2:] = True
    if features:
        occ[150:160, 200:400] = True          # a rack across part of the room
        occ[300:380, 450:460] = True          # a pillar row
    return occ


def _corridor(length=30.0, width=3.0):
    """Two long parallel walls, open at both ends: front-to-back is unknown."""
    w, h = int(length / RES), int((width + 1.0) / RES)
    occ = np.zeros((h, w), dtype=bool)
    occ[:2, :] = occ[-2:, :] = True
    return occ


def _cast(occ, pose, n=720, noise=0.0, rng=None):
    """Ray-cast a 360 degree scan from pose; points in the robot frame."""
    angles = np.linspace(-math.pi, math.pi, n, endpoint=False)
    steps = np.arange(0.1, 30.0, 0.005)
    pts = []
    for a in angles:
        xs = pose[0] + np.cos(pose[2] + a) * steps
        ys = pose[1] + np.sin(pose[2] + a) * steps
        i, j = (xs / RES).astype(int), (ys / RES).astype(int)
        ok = (i >= 0) & (i < occ.shape[1]) & (j >= 0) & (j < occ.shape[0])
        hit = np.zeros(len(steps), dtype=bool)
        hit[ok] = occ[j[ok], i[ok]]
        if hit.any():
            r = steps[hit.argmax()] + (rng.normal(0, noise) if noise else 0.0)
            pts.append((r * math.cos(a), r * math.sin(a)))
    return np.array(pts)


def _odom_points(occ, truth, odom, **kw):
    """Scan at the true pose, expressed in odom (odom pose of the robot given)."""
    return sr.transform_points(odom, _cast(occ, truth, **kw))


def _error(field_transform, odom, truth):
    est = sr.compose(field_transform, odom)
    return math.hypot(est[0] - truth[0], est[1] - truth[1]), abs(est[2] - truth[2])


def test_refinement_removes_an_amcl_sized_error():
    occ = _room()
    field = sr.DistanceField(occ, RES, (0.0, 0.0))
    rng = np.random.default_rng(0)
    truth = (5.3, 4.1, 0.4)
    odom = (1.0, -2.0, 0.1)                       # odom frame anywhere
    pts = _odom_points(occ, truth, odom, noise=0.01, rng=rng)
    exact = sr.compose(truth, sr.inverse(odom))
    # AMCL off by 3 cm front-to-back, 2 cm sideways, 0.4 degrees.
    prior = sr.compose(sr.compose(truth, (0.03, 0.02, math.radians(0.4))), sr.inverse(odom))
    assert _error(prior, odom, truth)[0] > 0.03
    result = sr.refine(field, pts, prior, odom)
    assert result.accepted, result.reason
    assert result.constrained == 3
    shift, turn = _error(result.transform, odom, truth)
    assert shift < 0.005 and turn < math.radians(0.1)
    assert _error(exact, odom, truth)[0] < 1e-9


def test_a_corridor_keeps_amcl_front_to_back_but_fixes_sideways():
    occ = _corridor()
    field = sr.DistanceField(occ, RES, (0.0, 0.0))
    truth = (15.0, 2.0, 0.0)                      # along the corridor
    odom = truth
    pts = _odom_points(occ, truth, odom)
    # AMCL 4 cm ahead and 2 cm to the side.
    prior = sr.compose(sr.compose(truth, (0.04, 0.02, 0.0)), sr.inverse(odom))
    result = sr.refine(field, pts, prior, odom)
    assert result.constrained == 2               # front-to-back not pinned
    est = sr.compose(result.transform, odom)
    assert abs(est[1] - truth[1]) < 0.003        # sideways fixed
    assert est[0] - truth[0] == pytest.approx(0.04, abs=0.003)  # kept AMCL's


def test_points_on_unmapped_objects_are_ignored():
    occ = _room()
    field = sr.DistanceField(occ, RES, (0.0, 0.0))
    truth, odom = (6.0, 5.0, -0.3), (0.0, 0.0, 0.0)
    world = occ.copy()
    world[230:245, 280:330] = True                # a pallet not in the map
    pts = _odom_points(world, truth, odom)
    prior = sr.compose(truth, (0.02, -0.02, 0.003))
    result = sr.refine(field, pts, prior, odom)
    assert result.accepted, result.reason
    assert _error(result.transform, odom, truth)[0] < 0.005


def test_a_prior_far_from_the_truth_is_refused_and_amcl_kept():
    occ = _room()
    field = sr.DistanceField(occ, RES, (0.0, 0.0))
    truth, odom = (6.0, 3.0, 0.2), (0.0, 0.0, 0.0)
    pts = _odom_points(occ, truth, odom)
    # Half a metre and 10 degrees off: the scan does not line up with walls.
    result = sr.refine(field, pts, sr.compose(truth, (0.5, 0.3, math.radians(10))), odom)
    assert not result.accepted


def test_a_correction_larger_than_the_limit_is_refused():
    occ = _room()
    field = sr.DistanceField(occ, RES, (0.0, 0.0))
    truth, odom = (6.0, 5.0, 0.2), (0.0, 0.0, 0.0)
    pts = _odom_points(occ, truth, odom)
    result = sr.refine(field, pts, sr.compose(truth, (0.20, 0.0, 0.0)), odom)
    assert result.shift_m > sr.MAX_SHIFT_M        # it did find the true pose ...
    assert not result.accepted and 'from AMCL' in result.reason  # ... too far to trust


def test_points_pushed_into_a_wall_are_pulled_back_out():
    occ = _room()
    field = sr.DistanceField(occ, RES, (0.0, 0.0))
    truth, odom = (6.0, 5.0, 0.2), (0.0, 0.0, 0.0)
    pts = _odom_points(occ, truth, odom)
    result = sr.refine(field, pts, sr.compose(truth, (0.05, 0.03, 0.01)), odom)
    assert result.accepted and result.constrained == 3
    assert _error(result.transform, odom, truth)[0] < 0.003


def test_a_heading_correction_is_measured_at_the_robot_not_the_map_origin():
    occ = _room()
    field = sr.DistanceField(occ, RES, (0.0, 0.0))
    truth = (9.0, 6.0, 1.0)
    odom = (40.0, 30.0, 0.5)                      # odom origin far from the robot
    pts = _odom_points(occ, truth, odom)
    prior = sr.compose(sr.compose(truth, (0.0, 0.0, math.radians(0.8))), sr.inverse(odom))
    result = sr.refine(field, pts, prior, odom)
    assert result.accepted, result.reason
    assert result.shift_m < 0.01                  # 0.8 deg at the robot, no shift
    assert result.turn_rad == pytest.approx(math.radians(0.8), abs=math.radians(0.05))


def test_distance_field_reads_zero_on_a_wall_surface():
    occ = np.zeros((50, 50), dtype=bool)
    occ[:, 25:] = True                            # wall surface at x = 0.50 m
    field = sr.DistanceField(occ, RES, (0.0, 0.0))
    d, gx, _ = field.lookup(np.array([0.50, 0.40]), np.array([0.5, 0.5]))
    assert d[0] == pytest.approx(0.0, abs=0.002)
    assert d[1] == pytest.approx(0.10, abs=0.002)
    assert gx[1] < 0                              # distance falls toward the wall
