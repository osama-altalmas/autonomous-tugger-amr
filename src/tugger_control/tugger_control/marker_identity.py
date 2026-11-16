"""Which station a floor marker names, read from its four tags.

The YOLO-pose model only finds "a marker" and places its keypoints (the 4
frame corners, then the 4 corners of each tag); it knows no station, so a
new station needs no retraining, only a line in the station table
(docking_markers.STATION_TAG_BASE). Here each tag is straightened with its
four keypoints, its 6 x 6 bits are read and matched against the whole
AprilTag 36h11 code list (587 codes, any two differ in at least 11 bits, so
up to 5 wrong bits still give the right code). The classic detector fails
mostly when damage breaks the black outline it searches for; here the
keypoints say where the tag is, so its bits can still be read.

Keypoint order (2026-11-15): the marker looks the same turned by 90 degrees
(a square frame, four tags), so a model cannot see which frame corner is the
marker's own top-left; model v2, trained on marker-ordered corners, mixed
them up. Labels are therefore in IMAGE order (image_order): the frame corner
nearest the image's up-left direction first, then clockwise; tags and each
tag's corners follow the same turn. identify() tries the four turns and keeps
the one where the tags' IDs match their quarters; that turn also says how the
marker lies (to_marker_order gives the marker-ordered keypoints back).

identify(gray, keypoints) -> dict(payload, turn, tags_read, votes, state, reason).
"""
import math

import cv2
import numpy as np

from tugger_control.docking_markers import TAG_DICTIONARY, TAG_PAYLOAD

CELLS = 8                       # 6 x 6 bits inside a one-cell black border
CELL_PX = 12
MAX_BIT_ERRORS = 5              # 36h11: minimum distance 11 -> 5 correctable
MIN_MARGIN_BITS = 3             # the next code must be at least this much worse
MIN_BORDER_BLACK = 0.75         # share of border cells that must read black
MIN_CELL_CONTRAST = 18.0        # grey levels between the dark and light cell groups


def _codes():
    dictionary = cv2.aruco.getPredefinedDictionary(TAG_DICTIONARY)
    to_bits = cv2.aruco.Dictionary.getBitsFromByteList
    return np.array([to_bits(row.reshape(1, -1, 4), 6).reshape(-1)
                     for row in dictionary.bytesList], dtype=np.uint8)


CODES = _codes()


def read_cells(gray, corners):
    """Mean grey level of each of the 8 x 8 cells of a tag (TL, TR, BR, BL corners)."""
    side = CELLS * CELL_PX
    target = np.array([[0, 0], [side, 0], [side, side], [0, side]], dtype=np.float32)
    warp = cv2.getPerspectiveTransform(np.asarray(corners, dtype=np.float32), target)
    flat = cv2.warpPerspective(gray, warp, (side, side), flags=cv2.INTER_LINEAR)
    m = CELL_PX // 4                    # the inner half of each cell: away from its edges
    return np.array([[flat[r * CELL_PX + m:(r + 1) * CELL_PX - m,
                           c * CELL_PX + m:(c + 1) * CELL_PX - m].mean()
                      for c in range(CELLS)] for r in range(CELLS)])


def read_tag(gray, corners):
    """Best matching code for one tag: dict(tag_id, errors, margin, ok, reason)."""
    cells = read_cells(gray, corners)
    values = np.sort(cells.reshape(-1))
    # Split dark from light where the gap between sorted cell values is widest.
    gaps = np.diff(values)
    cut = int(np.argmax(gaps[8:-8]) + 8) if len(gaps) > 16 else len(values) // 2
    threshold = (values[cut] + values[cut + 1]) / 2.0
    contrast = float(values[cut + 1:].mean() - values[:cut + 1].mean())
    bits = (cells > threshold).astype(np.uint8)
    border = np.concatenate([bits[0], bits[-1], bits[1:-1, 0], bits[1:-1, -1]])
    border_black = float(np.mean(border == 0))
    inner = bits[1:-1, 1:-1].reshape(-1)
    distances = np.count_nonzero(CODES != inner, axis=1)
    order = np.argsort(distances)
    best, second = int(order[0]), int(order[1])
    errors, margin = int(distances[best]), int(distances[second] - distances[best])
    result = {'tag_id': best, 'errors': errors, 'margin': margin,
              'border_black': round(border_black, 2), 'contrast': round(contrast, 1)}
    if contrast < MIN_CELL_CONTRAST:
        return dict(result, ok=False, reason='no contrast left')
    if border_black < MIN_BORDER_BLACK:
        return dict(result, ok=False, reason='no tag border here')
    if errors > MAX_BIT_ERRORS or margin < MIN_MARGIN_BITS:
        return dict(result, ok=False, reason=f'{errors} bits wrong')
    return dict(result, ok=True, reason='')


UP_LEFT = math.atan2(-1.0, -1.0)        # image direction (y down) of the first corner


def _turn_indices(turn):
    """Index list: keypoints in one order -> the same points turned by `turn` quarters."""
    order = [(turn + i) % 4 for i in range(4)]
    return order + [4 + 4 * q + (turn + j) % 4 for q in order for j in range(4)]


def image_order(keypoints):
    """Marker-ordered keypoints (20 x 2+) -> (image-ordered keypoints, turn).

    turn = which marker corner (0 TL, 1 TR, 2 BR, 3 BL) lies nearest the
    image's up-left direction from the frame centre; it comes first.
    """
    kp = np.asarray(keypoints, dtype=np.float64)
    frame = kp[:4, :2]
    centre = frame.mean(axis=0)
    angles = np.arctan2(frame[:, 1] - centre[1], frame[:, 0] - centre[0])
    off = np.abs((angles - UP_LEFT + math.pi) % (2 * math.pi) - math.pi)
    turn = int(np.argmin(off))
    return kp[_turn_indices(turn)], turn


def to_marker_order(keypoints, turn):
    """Undo image_order: image-ordered keypoints -> marker-ordered."""
    kp = np.asarray(keypoints, dtype=np.float64)
    out = np.empty_like(kp)
    out[_turn_indices(turn)] = kp
    return out


def _read_turn(gray, keypoints, turn, tag_payload):
    """Read the four tags assuming the marker is turned by `turn` (see image_order)."""
    marker_kp = to_marker_order(keypoints, turn)
    tags, votes = [], {}
    for quarter in range(4):
        tag = read_tag(gray, marker_kp[4 + 4 * quarter:8 + 4 * quarter])
        tag['quarter'] = quarter
        if tag['ok']:
            payload = tag_payload.get(tag['tag_id'])
            if payload is None:
                tag.update(ok=False, reason=f'tag {tag["tag_id"]} belongs to no station')
            elif tag['tag_id'] % 4 != quarter:
                tag.update(ok=False, reason=f'tag {tag["tag_id"]} in the wrong corner')
            else:
                votes[payload] = votes.get(payload, 0) + 1
        tags.append(tag)
    return tags, votes


def identify(gray, keypoints, tag_payload=TAG_PAYLOAD):
    """Station from the four tags of one marker.

    keypoints: 20 x 2 image points, 4 frame corners then 4 tags of 4 corners,
    in image order (image_order) or marker order: every turn is tried and the
    one whose tags read and sit in their own quarters wins.
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    keypoints = np.asarray(keypoints, dtype=np.float64).reshape(20, 2)
    best = None
    for turn in range(4):
        tags, votes = _read_turn(gray, keypoints, turn, tag_payload)
        score = (max(votes.values()) if votes else 0) - (len(votes) > 1)
        if best is None or score > best[0]:
            best = (score, turn, tags, votes)
    _, turn, tags, votes = best
    result = {'tags': tags, 'votes': votes, 'tags_read': sum(votes.values()), 'turn': turn,
              'marker_keypoints': to_marker_order(keypoints, turn).round(2).tolist()}
    if len(votes) > 1:
        return dict(result, payload=None, state='FAULT',
                    reason=f'tags of different stations: {sorted(votes)}')
    if not votes:
        return dict(result, payload=None, state='FAULT', reason='no tag can be read')
    payload = next(iter(votes))
    return dict(result, payload=payload, state='OK', reason='')
