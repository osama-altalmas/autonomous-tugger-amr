"""Shared identities and dimensions for the floor docking markers.

Each station's marker is a 120 mm red frame around a 100 mm white plate with
four AprilTags (36h11), one per quarter, all naming the same station.
"""

from pathlib import Path

import cv2
import numpy as np


RED_FRAME_SIZE_M = 0.12

MARKER_IDS = frozenset({
    'TUGGER:HOME',
    'TUGGER:S1',
    'TUGGER:D2',
    'TUGGER:S3',
    'TUGGER:D4',
})

STATION_MARKER_IDS = {
    'HOME': 'TUGGER:HOME',
    'cargo_station_0': 'TUGGER:S1',
    'cargo_dropoff_northwest': 'TUGGER:D2',
    'cargo_station_3': 'TUGGER:S3',
    'cargo_dropoff_southeast': 'TUGGER:D4',
}


def expected_marker(station):
    """Return the unique floor-marker identity for a known station."""
    try:
        return STATION_MARKER_IDS[station]
    except KeyError as exc:
        raise ValueError(f'No QR marker registered for station: {station}') from exc


def marker_asset_directory() -> Path:
    """Return the source asset directory for offline QR decode tests."""
    return Path(__file__).resolve().parents[2] / 'tugger_gazebo' / 'media' / 'markers'


# Floor markers since 2026-11-15: four AprilTags (36h11) on the white plate
# inside the red frame, one per quarter, all naming the same station. A QR's
# three finder corners are the same on every code and only one quarter told
# the stations apart; with four tags any single quarter that survives wear
# identifies the station (and which corner it is). 36h11 codes differ in at
# least 11 of 36 bits. Plate axes as the QR before: texture right = +x,
# texture up = +y, centre at the station marker.
TAG_DICTIONARY = cv2.aruco.DICT_APRILTAG_36h11
MARKER_PLATE_M = 0.10
TAG_SIZE_M = 0.04
TAG_OFFSET_M = 0.025          # tag centres at (+-0.025, +-0.025)
# Quarter order: top-left, top-right, bottom-right, bottom-left.
TAG_QUARTERS = ((-1, 1), (1, 1), (1, -1), (-1, -1))
STATION_TAG_BASE = {
    'TUGGER:HOME': 0,
    'TUGGER:S1': 4,
    'TUGGER:D2': 8,
    'TUGGER:S3': 12,
    'TUGGER:D4': 16,
}
TAG_PAYLOAD = {base + q: payload for payload, base in STATION_TAG_BASE.items() for q in range(4)}


def tag_object_corners(tag_id):
    """The tag's corners on the plate (m), in OpenCV's order TL, TR, BR, BL."""
    sx, sy = TAG_QUARTERS[tag_id % 4]
    cx, cy, h = sx * TAG_OFFSET_M, sy * TAG_OFFSET_M, TAG_SIZE_M / 2
    return np.array([[cx - h, cy + h, 0.0], [cx + h, cy + h, 0.0],
                     [cx + h, cy - h, 0.0], [cx - h, cy - h, 0.0]])


def render_floor_marker(payload, pixels=400):
    """The plate texture for one station: white, four tags (one per quarter)."""
    dictionary = cv2.aruco.getPredefinedDictionary(TAG_DICTIONARY)
    image = np.full((pixels, pixels), 255, dtype=np.uint8)
    scale = pixels / MARKER_PLATE_M
    side = int(round(TAG_SIZE_M * scale))
    for q in range(4):
        tag = cv2.aruco.generateImageMarker(dictionary, STATION_TAG_BASE[payload] + q, side)
        corner = tag_object_corners(q)[0]                  # top-left on the plate
        col = int(round((corner[0] + MARKER_PLATE_M / 2) * scale))
        row = int(round((MARKER_PLATE_M / 2 - corner[1]) * scale))
        image[row:row + side, col:col + side] = tag
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def tag_detector():
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(TAG_DICTIONARY), parameters)


def detect_floor_tags(detector, image):
    """Station identity and matched corners from the visible tags.

    Returns (payload, image_points n x 2, object_points n x 3, tag ids) or
    (None, None, None, ids) when no registered tag is seen or tags of two
    stations are (an impossible scene, refused).
    """
    corners, ids, _ = detector.detectMarkers(image)
    if ids is None:
        return None, None, None, []
    found = [(int(i), c.reshape(4, 2)) for i, c in zip(ids.flatten(), corners)
             if int(i) in TAG_PAYLOAD]
    payloads = {TAG_PAYLOAD[i] for i, _ in found}
    seen = [i for i, _ in found]
    if len(payloads) != 1:
        return None, None, None, seen
    image_points = np.vstack([c for _, c in found]).astype(np.float64)
    object_points = np.vstack([tag_object_corners(i) for i, _ in found])
    return payloads.pop(), image_points, object_points, seen


def detect_red_alignment_frame(image):
    """Return ordered image corners of the docking red frame, if present.

    The red frame is the final-position target.  Its enclosing square is
    deliberately larger than the QR quiet zone, so colour segmentation can
    locate the docking centre without altering the QR payload.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Red wraps around the OpenCV hue range, so both endpoints are required.
    lower = cv2.inRange(hsv, (0, 120, 80), (10, 255, 255))
    upper = cv2.inRange(hsv, (170, 120, 80), (180, 255, 255))
    mask = cv2.bitwise_or(lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 400.0:
        return None
    rectangle = cv2.minAreaRect(contour)
    width, height = rectangle[1]
    if min(width, height) < 20.0 or max(width, height) / min(width, height) > 1.25:
        return None
    corners = cv2.boxPoints(rectangle)
    # A clipped frame cannot provide reliable square dimensions or centre.
    # Reject it instead of allowing a partial contour to steer the robot.
    image_height, image_width = image.shape[:2]
    if (np.any(corners[:, 0] < 3) or np.any(corners[:, 1] < 3) or
            np.any(corners[:, 0] >= image_width - 3) or
            np.any(corners[:, 1] >= image_height - 3)):
        return None
    # Stable clockwise ordering: top-left, top-right, bottom-right, bottom-left.
    by_y = corners[corners[:, 1].argsort()]
    top = by_y[:2][by_y[:2, 0].argsort()]
    bottom = by_y[2:][by_y[2:, 0].argsort()]
    return np.asarray(
        [top[0], top[1], bottom[1], bottom[0]], dtype=np.float64)


def downward_camera_planar_error(translation):
    """Convert OpenCV camera pose into forward/lateral base-plane error.

    The downward camera link's X axis points toward the floor.  OpenCV reports
    depth on its Z axis, which is camera height and must never become a motion
    target. Gazebo's camera link uses X forward, Y left, Z up; OpenCV optical
    axes are X right, Y down, Z forward. With the +90 degree pitch mounting,
    optical X is base -Y and optical Y is base -X.
    """
    camera_x, camera_y, camera_z = (float(value) for value in translation)
    del camera_z
    return -camera_y, -camera_x
