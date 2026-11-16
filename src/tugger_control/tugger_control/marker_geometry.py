"""The red docking frame fitted from the sides that are still visible.

The YOLO-pose model gives the frame's four corners even where they are worn
or covered, to a few pixels. Docking needs better (10 mm, 1 degree), so the
known square is fitted to the frame's visible outer edge: every outer-edge
pixel pulls on the side it belongs to. The square's size in the image is
known from the camera height, so two adjacent sides fix the whole frame;
with three or four the fit is over-determined and checks itself.

Pure functions, no ROS: fit_frame(image, corners_guess, side_px).
"""
import math

import cv2
import numpy as np

EDGE_BAND_PX = 8.0          # outer-edge pixels this close to a guessed side belong to it
MIN_SIDE_POINTS = 25        # a side counts as seen with at least this many edge pixels
MAX_RESIDUAL_PX = 1.5       # rms distance of the edge pixels to the fitted square
MAX_SHIFT_FROM_GUESS_PX = 12.0
SCALE_WEIGHT = 50.0         # how firmly the known size holds (in edge-pixel units)


LIGHT_REFERENCE = 180.0     # 90th grey percentile under the full LED ring (2026-11-16)
MAX_LIGHT_GAIN = 4.0
IMAGE_BORDER_PX = 2         # contour pixels this near the image border: the cut, not the frame


def normalise_light(image):
    """Brighten a dim image to the full-LED level (as an auto-exposure would), so
    the red threshold holds when the LEDs age or one fails."""
    level = float(np.percentile(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 90))
    gain = min(MAX_LIGHT_GAIN, LIGHT_REFERENCE / max(level, 1.0))
    if gain <= 1.0:
        return image
    return np.clip(image.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def red_mask(image):
    hsv = cv2.cvtColor(normalise_light(image), cv2.COLOR_BGR2HSV)
    lower = cv2.inRange(hsv, (0, 120, 80), (10, 255, 255))
    upper = cv2.inRange(hsv, (170, 120, 80), (180, 255, 255))
    return cv2.bitwise_or(lower, upper)


def outer_edge_points(mask):
    """Pixels on the outside boundary of the red region (n x 2, x/y)."""
    filled = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((0, 2))
    points = np.vstack([c.reshape(-1, 2) for c in contours]).astype(np.float64)
    h, w = mask.shape[:2]
    inside = ((points[:, 0] >= IMAGE_BORDER_PX) & (points[:, 0] < w - IMAGE_BORDER_PX)
              & (points[:, 1] >= IMAGE_BORDER_PX) & (points[:, 1] < h - IMAGE_BORDER_PX))
    return points[inside]


def square_corners(cx, cy, theta, side):
    """Corners of a square (image px) in the order TL, TR, BR, BL of its own frame."""
    h = side / 2.0
    c, s = math.cos(theta), math.sin(theta)
    local = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])
    return np.column_stack([cx + c * local[:, 0] - s * local[:, 1],
                            cy + s * local[:, 0] + c * local[:, 1]])


def _side_distances(points, cx, cy, theta, side):
    """Signed distance of each point to each of the four side lines (n x 4)."""
    c, s = math.cos(theta), math.sin(theta)
    u = c * (points[:, 0] - cx) + s * (points[:, 1] - cy)
    v = -s * (points[:, 0] - cx) + c * (points[:, 1] - cy)
    h = side / 2.0
    return np.column_stack([v + h, u - h, v - h, u + h])   # top, right, bottom, left


def _assign(points, cx, cy, theta, side):
    """Nearest side for each point within the band (-1 = none) and its distance."""
    d = _side_distances(points, cx, cy, theta, side)
    c, s = math.cos(theta), math.sin(theta)
    u = c * (points[:, 0] - cx) + s * (points[:, 1] - cy)
    v = -s * (points[:, 0] - cx) + c * (points[:, 1] - cy)
    h = side / 2.0
    along = np.column_stack([np.abs(u), np.abs(v), np.abs(u), np.abs(v)]) <= h + EDGE_BAND_PX
    d_abs = np.where(along, np.abs(d), np.inf)
    nearest = np.argmin(d_abs, axis=1)
    dist = d_abs[np.arange(len(points)), nearest]
    nearest[dist > EDGE_BAND_PX] = -1
    return nearest, d


def guess_from_corners(corners):
    """Centre, angle and side length of the square through four rough corners."""
    corners = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    cx, cy = corners.mean(axis=0)
    top = corners[1] - corners[0]
    bottom = corners[2] - corners[3]
    theta = math.atan2(top[1] + bottom[1], top[0] + bottom[0])
    side = float(np.mean([np.linalg.norm(corners[(i + 1) % 4] - corners[i]) for i in range(4)]))
    return cx, cy, theta, side


def fit_frame(image, corners_guess, side_px, iterations=12):
    """Fit the known square to the visible outer edge of the red frame.

    corners_guess: 4 x 2 rough corners (TL, TR, BR, BL), e.g. from YOLO-pose.
    side_px: the frame's outer side length in the image (from the camera height).
    Returns a dict: corners (4 x 2) or None, sides_seen, residual_px, ok, reason.
    """
    points = outer_edge_points(red_mask(image))
    cx, cy, theta, _ = guess_from_corners(corners_guess)
    side = float(side_px)
    start = (cx, cy)
    if len(points) < 2 * MIN_SIDE_POINTS:
        return {'corners': None, 'sides_seen': 0, 'residual_px': None, 'ok': False,
                'reason': 'too little of the red frame is visible'}
    for _ in range(iterations):
        nearest, d = _assign(points, cx, cy, theta, side)
        used = nearest >= 0
        if used.sum() < 2 * MIN_SIDE_POINTS:
            break
        rows, residuals = [], []
        c, s = math.cos(theta), math.sin(theta)
        for i in np.where(used)[0]:
            k = nearest[i]
            px, py = points[i, 0] - cx, points[i, 1] - cy
            u, v = c * px + s * py, -s * px + c * py
            # d/d(cx, cy, theta, side) of the signed distance to side k
            if k in (0, 2):     # v +- h
                grad = [s, -c, -u, 0.5 if k == 0 else -0.5]
            else:               # u -+ h
                grad = [-c, -s, v, -0.5 if k == 1 else 0.5]
            rows.append(grad)
            residuals.append(d[i, k])
        J, r = np.array(rows), np.array(residuals)
        # The known size, as a soft extra equation.
        J = np.vstack([J, [0, 0, 0, SCALE_WEIGHT]])
        r = np.append(r, SCALE_WEIGHT * (side - side_px))
        step, *_ = np.linalg.lstsq(J, -r, rcond=None)
        cx, cy, theta, side = cx + step[0], cy + step[1], theta + step[2], side + step[3]
        if np.abs(step[:2]).max() < 1e-3 and abs(step[2]) < 1e-5:
            break
    nearest, d = _assign(points, cx, cy, theta, side)
    counts = [int((nearest == k).sum()) for k in range(4)]
    seen = [n >= MIN_SIDE_POINTS for n in counts]
    used = nearest >= 0
    residual = float(np.sqrt(np.mean(d[used, nearest[used]] ** 2))) if used.any() else None
    adjacent = any(seen[k] and seen[(k + 1) % 4] for k in range(4))
    shift = math.hypot(cx - start[0], cy - start[1])
    corners = square_corners(cx, cy, theta, side)
    result = {'corners': corners, 'sides_seen': int(sum(seen)), 'side_points': counts,
              'residual_px': residual, 'shift_from_guess_px': shift, 'ok': False}
    if not adjacent:
        result['reason'] = 'fewer than two adjacent frame sides are visible'
    elif residual is None or residual > MAX_RESIDUAL_PX:
        result['reason'] = f'the frame edge does not fit a square ({residual:.1f} px)'
    elif shift > MAX_SHIFT_FROM_GUESS_PX:
        result['reason'] = f"fitted frame is {shift:.0f} px from the model's guess"
    else:
        result.update(ok=True, reason='')
    return result
