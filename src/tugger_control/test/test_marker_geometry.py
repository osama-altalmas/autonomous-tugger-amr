"""The red frame fitted from its visible sides (marker_geometry)."""
import math

import cv2
import numpy as np

from tugger_control.marker_geometry import fit_frame, square_corners

SIDE = 140.0


def scene(cx=322.4, cy=238.7, theta=math.radians(17), wipe=(), thickness=12):
    """A grey floor with the red frame; ``wipe`` lists sides painted over
    (0 top, 1 right, 2 bottom, 3 left of the frame's own axes)."""
    image = np.full((480, 640, 3), 110, np.uint8)
    outer = square_corners(cx, cy, theta, SIDE)
    inner = square_corners(cx, cy, theta, SIDE - 2 * thickness)
    cv2.fillPoly(image, [np.round(outer * 16).astype(np.int32)], (20, 20, 170), cv2.LINE_AA, 4)
    cv2.fillPoly(image, [np.round(inner * 16).astype(np.int32)], (200, 200, 200), cv2.LINE_AA, 4)
    for k in wipe:
        a, b = outer[k], outer[(k + 1) % 4]
        cv2.line(image, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)),
                 (110, 110, 110), 2 * thickness + 10)
    return image, outer


def rough(corners, error=4.0, seed=1):
    rng = np.random.default_rng(seed)
    return corners + rng.uniform(-error, error, corners.shape)


def test_a_full_frame_is_found_to_a_fraction_of_a_pixel():
    image, truth = scene()
    result = fit_frame(image, rough(truth), SIDE)
    assert result['ok'] and result['sides_seen'] == 4
    assert np.abs(result['corners'] - truth).max() < 0.8


def test_a_wiped_side_is_rebuilt_from_the_other_three():
    image, truth = scene(wipe=(1,))
    result = fit_frame(image, rough(truth), SIDE)
    assert result['ok'] and result['sides_seen'] == 3
    assert np.abs(result['corners'] - truth).max() < 1.0


def test_two_adjacent_sides_with_the_known_size_are_enough():
    image, truth = scene(wipe=(1, 2))
    result = fit_frame(image, rough(truth), SIDE)
    assert result['ok'] and result['sides_seen'] == 2
    assert np.abs(result['corners'].mean(axis=0) - truth.mean(axis=0)).max() < 1.5


def test_one_side_or_two_opposite_ones_are_not_enough():
    image, truth = scene(wipe=(0, 1, 2))
    assert not fit_frame(image, rough(truth), SIDE)['ok']
    image, truth = scene(wipe=(0, 2))
    assert not fit_frame(image, rough(truth), SIDE)['ok']


def test_a_dim_image_still_fits():
    image, truth = scene()
    dim = (image.astype(np.float32) * 0.4).astype(np.uint8)    # LEDs ageing or one out
    result = fit_frame(dim, rough(truth), SIDE)
    assert result['ok']
    assert np.abs(result['corners'] - truth).max() < 1.0


def test_the_image_border_is_not_taken_for_a_frame_side():
    image, truth = scene(cy=412.0, theta=math.radians(0.5))      # bottom side just outside
    result = fit_frame(image, rough(truth), SIDE)
    assert result['ok']
    assert np.abs(result['corners'] - truth).max() < 1.5
