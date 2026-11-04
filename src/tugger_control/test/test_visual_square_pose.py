"""Regression tests for the downward-camera square pose estimator."""

import math

import cv2
import numpy as np
import pytest

from tugger_control.visual_square_pose import checked_square_pose


K = np.array([
    [116.470623, 0.0, 320.0],
    [0.0, 116.470627, 240.0],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

HALF = 0.060

OBJECT_POINTS = np.array([
    [-HALF, HALF, 0.0],
    [HALF, HALF, 0.0],
    [HALF, -HALF, 0.0],
    [-HALF, -HALF, 0.0],
], dtype=np.float64)


@pytest.mark.parametrize(
    "offset",
    [-0.015, 0.0, 0.015],
)
def test_square_pose_recovers_lateral_offset(offset):
    """Recover the correct translation from known image geometry."""

    image_points, _ = cv2.projectPoints(
        OBJECT_POINTS,
        np.array([math.pi, 0.0, 0.0]),
        np.array([offset, 0.0, 0.05613]),
        K,
        None,
    )

    rotation, translation, rmse = checked_square_pose(
        image_points,
        K,
        None,
        0.12,
    )

    assert np.all(np.isfinite(rotation))

    assert translation[0, 0] == pytest.approx(
        offset,
        abs=0.0005,
    )

    assert translation[2, 0] == pytest.approx(
        0.05613,
        abs=0.0005,
    )

    assert rmse < 0.01


def test_bad_square_measurements_are_rejected():
    """Reject inconsistent corners and implausible camera height."""

    image_points, _ = cv2.projectPoints(
        OBJECT_POINTS,
        np.array([math.pi, 0.0, 0.0]),
        np.array([0.015, 0.0, 0.05613]),
        K,
        None,
    )

    corrupted = image_points.reshape(4, 2).copy()

    corrupted[0] += [20.0, -25.0]

    with pytest.raises(ValueError):
        checked_square_pose(
            corrupted,
            K,
            None,
            0.12,
        )

    with pytest.raises(ValueError):
        checked_square_pose(
            image_points,
            K,
            None,
            0.12,
            min_depth_m=0.070,
        )
