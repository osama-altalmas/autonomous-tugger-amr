"""Station identity from the four tags, read through the keypoints."""
import math

import cv2
import numpy as np

from tugger_control.docking_markers import (RED_FRAME_SIZE_M, render_floor_marker,
                                            tag_detector, detect_floor_tags, tag_object_corners)
from tugger_control.marker_identity import identify, image_order, to_marker_order

PLATE_PX = 400                      # render_floor_marker: 0.10 m plate


def plate_keypoints():
    h = RED_FRAME_SIZE_M / 2
    frame = np.array([[-h, h], [h, h], [h, -h], [-h, -h]])
    return np.vstack([frame] + [tag_object_corners(q)[:, :2] for q in range(4)])


def camera_view(payload, angle_deg=23.0, scale=2.4, centre=(330.0, 250.0)):
    """The plate texture seen by the camera (rotated, scaled) and its 20 keypoints."""
    plate = render_floor_marker(payload, PLATE_PX)
    a = math.radians(angle_deg)
    # plate pixel -> plate metres -> image
    to_m = np.array([[0.10 / PLATE_PX, 0, -0.05], [0, -0.10 / PLATE_PX, 0.05], [0, 0, 1]])
    k = scale * PLATE_PX / 0.10 / 4.0               # px per metre in the image
    m_to_img = np.array([[k * math.cos(a), k * math.sin(a), centre[0]],
                         [k * math.sin(a), -k * math.cos(a), centre[1]], [0, 0, 1]])
    H = m_to_img @ to_m
    image = cv2.warpPerspective(plate, H, (640, 480), borderValue=(110, 110, 110))
    kp = cv2.perspectiveTransform(plate_keypoints().reshape(-1, 1, 2), m_to_img).reshape(-1, 2)
    return image, kp


def test_every_station_is_read_from_its_keypoints():
    for payload in ('TUGGER:HOME', 'TUGGER:S1', 'TUGGER:D2', 'TUGGER:S3', 'TUGGER:D4'):
        image, kp = camera_view(payload)
        result = identify(image, kp)
        assert result['payload'] == payload and result['tags_read'] == 4, result


def test_a_broken_tag_outline_defeats_the_classic_detector_but_not_this_reader():
    image, kp = camera_view('TUGGER:S3')
    for q in range(4):              # a grey scuff across each tag's top-left border corner
        tl = kp[4 + 4 * q]
        cv2.circle(image, tuple(int(v) for v in tl), 7, (150, 150, 150), -1)
    classic, *_ = detect_floor_tags(tag_detector(), image)
    result = identify(image, kp)
    assert result['payload'] == 'TUGGER:S3' and result['tags_read'] >= 3
    assert classic is None or classic == 'TUGGER:S3'


def test_one_readable_tag_is_enough_and_a_blank_marker_is_a_fault():
    image, kp = camera_view('TUGGER:D2')
    for q in (0, 1, 2):
        cv2.fillPoly(image, [kp[4 + 4 * q:8 + 4 * q].astype(np.int32)], (200, 200, 200))
    result = identify(image, kp)
    assert result['payload'] == 'TUGGER:D2' and result['tags_read'] == 1
    cv2.fillPoly(image, [kp[16:20].astype(np.int32)], (200, 200, 200))
    assert identify(image, kp)['state'] == 'FAULT'


def test_a_few_wrong_bits_are_corrected():
    image, kp = camera_view('TUGGER:S1')
    rng = np.random.default_rng(4)
    tag = kp[4:8]                   # flip 4 inner cells of the top-left tag
    for _ in range(4):
        r, c = rng.integers(1, 7, 2)
        u, v = (c + 0.5) / 8, (r + 0.5) / 8
        p = (tag[0] * (1 - u) * (1 - v) + tag[1] * u * (1 - v) + tag[2] * u * v +
             tag[3] * (1 - u) * v)
        colour = int(255 - image[int(p[1]), int(p[0])].mean())
        cv2.circle(image, tuple(int(x) for x in p), 4, (colour, colour, colour), -1)
    tag0 = identify(image, kp)['tags'][0]
    assert tag0['ok'] and tag0['tag_id'] == 4


def test_image_order_starts_up_left_and_turns_back():
    for angle in range(0, 360, 15):
        _, kp = camera_view('TUGGER:HOME', angle_deg=angle)
        ordered, turn = image_order(kp)
        centre = ordered[:4].mean(axis=0)
        first = ordered[0] - centre
        assert first[0] < 0.5 and first[1] < 0.5 or abs(angle % 90 - 45) < 1, (angle, first)
        assert np.allclose(to_marker_order(ordered, turn), kp)


def test_any_turn_of_the_marker_reads_with_image_ordered_keypoints():
    for payload, angle in [('TUGGER:S1', 10), ('TUGGER:D2', 100), ('TUGGER:S3', 190),
                           ('TUGGER:D4', 280), ('TUGGER:HOME', 315)]:
        image, kp = camera_view(payload, angle_deg=angle)
        ordered, turn = image_order(kp)
        result = identify(image, ordered)
        assert result['payload'] == payload and result['tags_read'] == 4, (angle, result)
        assert result['turn'] == turn
        assert np.allclose(result['marker_keypoints'], kp, atol=0.01)


def test_the_dataset_labels_use_the_same_image_order():
    import importlib.util
    from pathlib import Path
    script = Path(__file__).resolve().parents[3] / 'tools/ai/marker_dataset/augment.py'
    import sys
    sys.path.insert(0, str(script.parent))
    spec = importlib.util.spec_from_file_location('marker_augment', script)
    augment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(augment)
    for angle in range(0, 360, 20):
        _, kp = camera_view('TUGGER:S1', angle_deg=angle)
        ordered, _ = image_order(kp)
        assert np.allclose(np.asarray(kp)[augment.image_order(kp)], ordered)
