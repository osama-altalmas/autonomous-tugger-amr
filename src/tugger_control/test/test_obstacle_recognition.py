"""Naming tracked obstacles from the colour cameras: model output, matching, votes."""
import math

import numpy as np

from tugger_control.obstacle_recognition import (
    CLASSES, Votes, decode, draw_debug, expected_view, focal_px, letterbox, match)

W, H = 640, 480
CAM_HEIGHT = 0.195


def raw_output(boxes):
    """A model output (1, 8, N) with the given (cx, cy, w, h, class, score) in letterbox pixels."""
    out = np.zeros((1, 4 + len(CLASSES), max(1, len(boxes))), np.float32)
    for i, (cx, cy, w, h, c, s) in enumerate(boxes):
        out[0, :4, i] = (cx, cy, w, h)
        out[0, 4 + c, i] = s
    return out


def test_decode_undoes_the_letterbox_and_drops_weak_and_duplicate_boxes():
    _, scale, pad = letterbox(np.zeros((H, W, 3), np.uint8))
    assert scale == 1.0 and pad == (0, 80)
    out = raw_output([(320, 320, 100, 200, 0, 0.9), (322, 321, 100, 200, 0, 0.8),
                      (100, 300, 50, 50, 1, 0.2)])
    boxes = decode(out, scale, pad)
    assert len(boxes) == 1
    x0, y0, x1, y1, c, conf = boxes[0]
    assert (round(x0), round(y0), round(x1), round(y1), c) == (270, 140, 370, 340, 0)
    assert abs(conf - 0.9) < 1e-6


def test_an_object_straight_ahead_is_expected_in_the_middle_standing_on_the_floor_row():
    u0, u1, floor = expected_view((3.0, 0.0), 0.25, W, H, CAM_HEIGHT)
    assert u0 < W / 2 < u1
    f = focal_px(W)
    assert abs(floor - (H / 2 + f * CAM_HEIGHT / 2.75)) < 1e-6
    assert expected_view((3.0, 3.5), 0.25, W, H, CAM_HEIGHT) is None      # outside the 87 degree view
    assert expected_view((-2.0, 0.0), 0.25, W, H, CAM_HEIGHT) is None     # behind
    assert expected_view((12.0, 0.0), 0.25, W, H, CAM_HEIGHT) is None     # beyond the range used


def test_an_object_to_the_left_appears_left_of_centre():
    u0, u1, _ = expected_view((3.0, 1.0), 0.25, W, H, CAM_HEIGHT)
    assert u1 < W / 2


def test_a_detection_matches_by_columns_and_floor_row_not_by_bearing_alone():
    view = expected_view((3.0, 0.0), 0.3, W, H, CAM_HEIGHT)
    floor = view[2]
    person = (300, floor - 200, 340, floor + 3, 0, 0.9)
    far_behind = (300, H / 2 - 30, 330, H / 2 + 8, 3, 0.9)      # same bearing, stands much further away
    beside = (520, floor - 150, 600, floor, 1, 0.9)
    assert match(view, [far_behind, beside, person], H) == person
    assert match(view, [far_behind, beside], H) is None


def test_a_detection_cut_by_the_image_bottom_still_matches():
    view = expected_view((0.9, 0.0), 0.3, W, H, CAM_HEIGHT)
    close = (250, 100, 390, H - 1, 0, 0.8)
    assert match(view, [close], H) == close


def test_a_class_needs_two_votes_and_one_wrong_frame_does_not_relabel():
    v = Votes()
    v.add(0.0, 7, 0, 0.8)
    assert v.classes(0.1) == {}
    v.add(0.4, 7, 0, 0.7)
    v.add(0.8, 7, 3, 0.6)
    named = v.classes(0.9)
    assert named[7]['cls'] == 'person' and named[7]['votes'] == 2
    assert v.classes(5.0) == {}                                   # old votes expire
    v.add(5.0, 8, 1, 0.9)
    v.add(5.2, 8, 1, 0.9)
    assert v.classes(5.3, alive_ids={9}) == {}                    # its object is gone


def test_the_installed_model_runs_and_finds_a_person(tmp_path):
    """The packaged ONNX model end to end on a rendered training-style image, if present."""
    import os
    from pathlib import Path
    model = Path(__file__).resolve().parents[1] / 'models' / 'obstacles_v5n.onnx'
    if not model.exists():
        return
    from tugger_control.obstacle_recognition import ObstacleModel
    m = ObstacleModel(model)
    assert m.detect(np.full((H, W, 3), 128, np.uint8)) == []        # an empty grey image: nothing
    assert m.backend in ('onnxruntime', 'opencv')
    assert math.isfinite(focal_px(W)) and os.path.getsize(model) > 1e6


def test_the_debug_image_draws_detections_and_expected_views_without_changing_the_frame():
    frame = np.zeros((H, W, 3), np.uint8)
    out = draw_debug(frame, [(100, 50, 200, 300, 0, 0.9)], [(7, (120.0, 180.0, 290.0), True),
                                                            (8, (400.0, 420.0, 260.0), False)])
    assert out.shape == frame.shape and out.any()
    assert not frame.any()



def test_the_camera_with_an_object_in_view_looks_else_the_one_facing_the_driving_direction():
    from tugger_control.obstacle_recognition import choose_camera
    assert choose_camera({'front': False, 'rear': True}, 1, 0) == 'rear'
    assert choose_camera({'front': True, 'rear': True}, -1, 0) == 'rear'
    assert choose_camera({'front': False, 'rear': False}, 1, 7) == 'front'
    assert {choose_camera({}, 0, k) for k in range(2)} == {'front', 'rear'}    # standing: in turn


def test_a_single_vote_is_a_recognition_in_progress_until_it_is_named():
    votes = Votes()
    votes.add(10.0, 7, CLASSES.index('person'), 0.8)
    assert votes.pending(10.5, votes.classes(10.5)) == [7]
    votes.add(11.0, 7, CLASSES.index('person'), 0.8)
    named = votes.classes(11.0)
    assert named[7]['cls'] == 'person' and votes.pending(11.0, named) == []
