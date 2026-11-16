"""YOLO-pose marker reader (v3, one class): output decoding and the OK/ALERT/FAULT rule."""
import numpy as np

from tugger_control.marker_reader import (KEYPOINTS, assess, decode, letterbox, light_check,
                                          marker_ordered_corners)

CLASSES = 1


def raw(boxes):
    """A YOLO-pose style (1, 4 + 1 + 60, N) output from [(cx, cy, w, h, class, score)]."""
    out = np.zeros((1, 4 + CLASSES + 3 * KEYPOINTS, len(boxes)), dtype=np.float32)
    for i, (cx, cy, w, h, c, s) in enumerate(boxes):
        out[0, :4, i] = (cx, cy, w, h)
        out[0, 4 + c, i] = s
        kp = out[0, 4 + CLASSES:, i].reshape(KEYPOINTS, 3)
        kp[:, 0], kp[:, 1], kp[:, 2] = cx, cy, 0.9
        out[0, 4 + CLASSES:, i] = kp.reshape(-1)
    return out


def det(score):
    return [(0, score, 0, 0, 1, 1, np.zeros((KEYPOINTS, 3)))]


def tags(payload='TUGGER:S3', read=4):
    return {'state': 'OK', 'payload': payload, 'tags_read': read, 'reason': ''}


GOOD_FIT = {'ok': True, 'sides_seen': 4}


def test_boxes_and_keypoints_come_back_in_image_pixels_without_duplicates():
    _, scale, pad = letterbox(np.zeros((480, 640, 3), np.uint8))
    assert scale == 1.0 and pad == (0, 80)
    dets = decode(raw([(320, 320, 250, 250, 0, 0.95), (321, 321, 250, 250, 0, 0.90)]), scale, pad)
    assert len(dets) == 1 and dets[0][0] == 0
    assert abs(dets[0][3] - 115) < 1                       # y0 with the padding removed
    assert dets[0][6].shape == (KEYPOINTS, 3) and abs(dets[0][6][0, 1] - 240) < 1


def test_states():
    ok = assess(det(0.95), tags(), frame_fit=GOOD_FIT)
    assert ok['state'] == 'OK' and ok['station'] == 'cargo_station_3'
    alert = assess(det(0.65), tags(), frame_fit=GOOD_FIT)
    assert alert['state'] == 'ALERT' and 'hard to read' in alert['reason']
    one_tag = assess(det(0.95), tags(read=1), frame_fit=GOOD_FIT)
    assert one_tag['state'] == 'ALERT' and 'one of the four tags' in one_tag['reason']
    assert assess(det(0.40), tags(), frame_fit=GOOD_FIT)['state'] == 'FAULT'
    unread = assess(det(0.90), {'state': 'FAULT', 'payload': None, 'tags_read': 0,
                                'reason': 'no tag can be read'}, frame_fit=GOOD_FIT)
    assert unread['state'] == 'FAULT' and 'cannot be read' in unread['reason']
    assert assess([], None)['state'] == 'FAULT'


def test_a_frame_that_cannot_be_fitted_is_a_fault_even_when_the_station_reads():
    result = assess(det(0.95), tags(), frame_fit={'ok': False, 'sides_seen': 1,
                                                  'reason': 'fewer than two adjacent frame sides'})
    assert result['state'] == 'FAULT' and 'measured precisely' in result['reason']


def test_the_wrong_station_is_a_fault():
    result = assess(det(0.95), tags('TUGGER:S1'), expected_station='cargo_station_3')
    assert result['state'] == 'FAULT' and 'wrong station' in result['reason']


def test_fitted_corners_come_back_in_the_markers_own_order():
    marker = np.array([[300, 100], [400, 200], [300, 300], [200, 200]], float)   # turned 45 deg
    fitted = np.roll(marker + 0.5, 2, axis=0)                                  # fit's own start
    assert np.allclose(marker_ordered_corners(fitted, marker), marker + 0.5)


def _ring_lit(level, h=480, w=640):
    """A floor lit by the LED ring: brightest in the middle, about 0.8 at the edges."""
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(xx - w / 2, yy - h / 2) / np.hypot(w / 2, h / 2)
    return np.clip(level * (1 - 0.2 * r), 0, 255).astype(np.uint8)


def test_light_check_tells_a_dark_camera_from_a_lit_one():
    assert light_check(_ring_lit(200))['state'] == 'OK'
    assert light_check(_ring_lit(120))['state'] == 'ALERT'      # weak LEDs
    dark = light_check(_ring_lit(70))                            # LEDs off: only leaking light
    assert dark['state'] == 'FAULT' and 'too dark' in dark['reason']
