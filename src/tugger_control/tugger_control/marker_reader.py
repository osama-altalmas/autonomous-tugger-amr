#!/usr/bin/env python3
"""YOLO floor-marker reader: which station's marker, where its red frame is,
and how sure the reader is. Pure functions; qr_docking_vision runs them on
the downward camera.

The YOLO-pose model (tools/ai/marker_dataset, trained on Colab; dataset v3)
has one class, 'marker', and places 20 keypoints in image order (4 frame
corners, then 4 tags of 4 corners; marker_identity.image_order), also where
they are covered. The station is not learnt: it is read from the tags
through the keypoints (marker_identity.identify), so a new station is a line
in docking_markers.STATION_TAG_BASE, not a retraining. The frame is then
fitted precisely from its visible sides (marker_geometry). The state:

  OK     marker found with confidence >= ALERT_BELOW, at least two tags of
         one station read, frame fitted
  ALERT  still reads, but the confidence is below ALERT_BELOW or only one
         tag is left: clean or replace the marker soon
  FAULT  no marker or confidence below FAULT_BELOW, no tag readable (or tags
         of two stations), a wrong station, or a frame that cannot be fitted
         precisely

The camera brings its own light (an LED ring around the lens, so the
floor under the robot and its cargo is lit the same everywhere). The light is
checked on every image, so a dark camera is reported as a light fault, not
mistaken for a worn marker: FAULT when the floor is too dark to read (LEDs
off or failed), ALERT when it is weak (LEDs ageing).

qr_docking_vision publishes the state on /tugger/marker_reader (JSON)
while a marker is in view, and docks with the reader's frame and station.
"""
from pathlib import Path

import cv2
import numpy as np

from tugger_control.docking_markers import STATION_MARKER_IDS
from tugger_control.marker_geometry import fit_frame
from tugger_control.marker_identity import identify

KEYPOINTS = 20                 # 4 frame corners, then 4 corners of each of the 4 tags
# Camera above the RED FRAME's plane: its size in the image, focal 116.5 px x 0.12 m
# / height, is what the frame fit holds. Measured 2026-11-16: the red frame is a
# 248.3 px square (fit residual 0.3 px). In the simulation the tag plate lies 1 mm
# above the frame, so the tags (5.52 cm away) look 2 % larger; heights taken from
# the tags gave 0.055 and every fit was refused. On a real robot: measure the frame.
CAMERA_HEIGHT_M = 0.0563
FRAME_SIZE_M = 0.12
STATION_OF_PAYLOAD = {payload: station for station, payload in STATION_MARKER_IDS.items()}
ALERT_TAGS_BELOW = 2           # one readable tag left: still names the station, but alert
INPUT_SIZE = 640
DETECT_MIN = 0.10          # below this a box is not a detection at all
ALERT_BELOW = 0.80         # user plan 2026-11-15: alert before the marker fails
FAULT_BELOW = 0.50
NMS_IOU = 0.5
# Light check on the 90th grey percentile (the lit floor and plate; not the
# black print). Measured in simulation 2026-11-15: LED ring on 177-184,
# LEDs off (only light leaking past the shield) 68.
LIGHT_FAULT_BELOW = 90.0
LIGHT_ALERT_BELOW = 130.0


def letterbox(image, size=INPUT_SIZE):
    """Resize keeping the aspect ratio, pad to size x size; returns the
    padded image, the scale and the (x, y) padding."""
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = cv2.resize(image, (nw, nh))
    return canvas, scale, (pad_x, pad_y)


def decode(output, scale, pad, min_score=DETECT_MIN, keypoints=KEYPOINTS):
    """YOLO-pose (v8/11) output (1, 4 + classes + 3 * keypoints, N) ->
    [(class, score, x0, y0, x1, y1, kp)], kp = keypoints x 3 (x, y, confidence)
    in original image pixels, after non-maximum suppression."""
    pred = np.asarray(output)[0].T
    classes = pred.shape[1] - 4 - 3 * keypoints
    boxes, scores = pred[:, :4], pred[:, 4:4 + classes]
    kps = pred[:, 4 + classes:].reshape(len(pred), keypoints, 3)
    cls = scores.argmax(axis=1)
    best = scores[np.arange(len(scores)), cls]
    keep = best >= min_score
    boxes, cls, best, kps = boxes[keep], cls[keep], best[keep], kps[keep]
    if not len(best):
        return []
    cx, cy, bw, bh = boxes.T
    x0, y0 = (cx - bw / 2 - pad[0]) / scale, (cy - bh / 2 - pad[1]) / scale
    x1, y1 = (cx + bw / 2 - pad[0]) / scale, (cy + bh / 2 - pad[1]) / scale
    rects = [[float(x0[i]), float(y0[i]), float(x1[i] - x0[i]), float(y1[i] - y0[i])]
             for i in range(len(best))]
    chosen = cv2.dnn.NMSBoxes(rects, [float(b) for b in best], min_score, NMS_IOU)
    out = []
    for i in np.asarray(chosen).reshape(-1):
        kp = kps[i].copy()
        kp[:, 0] = (kp[:, 0] - pad[0]) / scale
        kp[:, 1] = (kp[:, 1] - pad[1]) / scale
        out.append((int(cls[i]), float(best[i]), float(x0[i]), float(y0[i]),
                    float(x1[i]), float(y1[i]), kp))
    return sorted(out, key=lambda d: -d[1])


def light_check(image):
    """State of the camera's LED light from the image: dict(state, level, reason).

    Uses the 90th grey percentile: with a marker in view that is its white
    plate (about 15 % of the image), so the floor colour hardly matters.
    An uneven ring (an LED out) is not judged here: a dark floor mat looks
    the same; the model is trained to read through it instead.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    level = float(np.percentile(gray, 90))
    result = {'level': round(level, 1)}
    if level < LIGHT_FAULT_BELOW:
        return dict(result, state='FAULT', reason='camera image too dark: marker light '
                    'off or failed; check the LED ring')
    if level < LIGHT_ALERT_BELOW:
        return dict(result, state='ALERT', reason='marker light weak: check the LED ring')
    return dict(result, state='OK', reason='')


def assess(detections, identity=None, expected_station=None, frame_fit=None):
    """Reader state from the best detection, its tags and the frame fit (module doc)."""
    if not detections:
        return {'state': 'FAULT', 'station': None, 'confidence': 0.0,
                'reason': 'no marker found'}
    score = detections[0][1]
    payload = identity.get('payload') if identity else None
    station = STATION_OF_PAYLOAD.get(payload)
    result = {'station': station, 'confidence': round(score, 3),
              'tags_read': identity.get('tags_read', 0) if identity else 0,
              'frame_fitted': bool(frame_fit and frame_fit.get('ok')),
              'frame_sides_seen': frame_fit.get('sides_seen') if frame_fit else None}
    if score < FAULT_BELOW:
        return dict(result, state='FAULT', reason=f'marker not recognised (confidence '
                    f'{score:.0%}): worn, dirty or covered; clean or replace it')
    if identity is None or identity.get('state') != 'OK':
        why = identity.get('reason') if identity else 'tags not read'
        return dict(result, state='FAULT', reason=f'marker found but its station cannot be '
                    f'read ({why}): the marker is worn or covered; clean or replace it')
    if station is None:
        return dict(result, state='FAULT', reason=f'{payload} is not a known station')
    if expected_station is not None and station != expected_station:
        return dict(result, state='FAULT',
                    reason=f'reads {station}, expected {expected_station} (wrong station?)')
    if frame_fit is not None and not frame_fit.get('ok'):
        return dict(result, state='FAULT', reason='red frame cannot be measured precisely: '
                    f'{frame_fit.get("reason")}')
    if score < ALERT_BELOW:
        return dict(result, state='ALERT', reason=f'marker getting hard to read (confidence '
                    f'{score:.0%}): clean or replace it soon')
    if result['tags_read'] < ALERT_TAGS_BELOW:
        return dict(result, state='ALERT', reason='only one of the four tags still reads: '
                    'clean or replace the marker soon')
    return dict(result, state='OK', reason='')


def marker_ordered_corners(fit_corners, marker_keypoints):
    """The fitted frame corners in the marker's own order (TL, TR, BR, BL), using
    the turn the tags gave: each fitted corner goes to the nearest keypoint."""
    fit_corners = np.asarray(fit_corners, dtype=np.float64)
    marker = np.asarray(marker_keypoints, dtype=np.float64)[:4]
    errors = [np.linalg.norm(np.roll(fit_corners, -k, axis=0) - marker, axis=1).sum()
              for k in range(4)]
    shift = int(np.argmin(errors))
    return np.roll(fit_corners, -shift, axis=0)


def default_model_path():
    """The trained model installed with the package (models/marker_reader_v3.onnx)."""
    from ament_index_python.packages import get_package_share_directory
    return str(Path(get_package_share_directory('tugger_control'))
               / 'models' / 'marker_reader_v3.onnx')


def read_marker(model, image, side_px, expected_station=None):
    """One image through the whole v3 reader: model, tags, frame fit, state.

    Returns the assess() result plus 'payload', 'detections' and, when the
    frame is fitted and the tags read, 'frame_corners' (4 x 2, the marker's
    own order TL, TR, BR, BL), 'marker_turn' and 'frame_residual_px'.
    """
    detections = model.detect(image)
    fit, identity = None, None
    if detections:
        keypoints = detections[0][6][:, :2]
        identity = identify(image, keypoints)
        if side_px is not None:
            fit = fit_frame(image, keypoints[:4], side_px)
    result = assess(detections, identity, expected_station, fit)
    result['detections'] = len(detections)
    result['payload'] = identity.get('payload') if identity else None
    if fit is not None and fit.get('ok') and result['payload']:
        result['frame_corners'] = marker_ordered_corners(
            fit['corners'], identity['marker_keypoints'])
        result['marker_turn'] = identity['turn']
        result['frame_residual_px'] = round(fit['residual_px'], 2)
    return result


class MarkerModel:
    """The exported ONNX model run with OpenCV's DNN module (CPU)."""

    def __init__(self, path):
        self.net = cv2.dnn.readNetFromONNX(str(path))

    def detect(self, image):
        padded, scale, pad = letterbox(image)
        blob = cv2.dnn.blobFromImage(padded, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        return decode(self.net.forward(), scale, pad)
