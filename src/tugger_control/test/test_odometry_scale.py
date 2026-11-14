"""Odometry calibration: the scan match's distance over odometry's distance."""
import math

from tugger_control.refined_localization import (OdometryScale, SCALE_SAMPLES,
                                                 SCALE_SEGMENT_M)


def drive(estimator, true_scale, metres, step=0.5, heading=0.3, start=0.0):
    """Feed fits along a straight run; odometry reads distance / true_scale."""
    result = None
    for k in range(int(metres / step) + 1):
        d = start + k * step
        truth = (d * math.cos(heading), d * math.sin(heading), heading)
        odom = (truth[0] / true_scale * estimator.scale, truth[1] / true_scale * estimator.scale,
                heading)
        out = estimator.add(odom, truth)
        result = out if out is not None else result
    return result


def test_wheels_two_percent_small_are_found_and_corrected():
    est = OdometryScale()
    # True wheels 2% smaller: odometry over-counts, the factor is 0.98.
    scale = drive(est, 0.98, SCALE_SEGMENT_M * (SCALE_SAMPLES + 1))
    assert scale is not None and math.isclose(scale, 0.98, abs_tol=0.001)


def test_a_correct_odometry_is_left_alone():
    est = OdometryScale()
    assert drive(est, 1.0, SCALE_SEGMENT_M * (SCALE_SAMPLES + 2)) is None
    assert est.scale == 1.0


def test_short_moves_never_count_and_the_factor_is_bounded():
    est = OdometryScale()
    assert drive(est, 0.98, SCALE_SEGMENT_M * 0.9) is None
    big = OdometryScale()
    assert drive(big, 0.80, 60.0) is None      # implausible ratios are not samples
