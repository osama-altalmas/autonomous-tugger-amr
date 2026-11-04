"""Shared identities and dimensions for the floor docking markers.

The visual backing plate is 220 mm square, but QRCodeDetector returns the
corners of the encoded grid *inside* its white quiet zone.  Pose estimation
must therefore use the 140 mm grid dimension, not the backing-plate size.
"""

from pathlib import Path

import cv2
import numpy as np


QR_GRID_SIZE_M = 0.064
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
    return Path(__file__).resolve().parents[2] / 'tugger_gazebo' / 'media' / 'qr'


def decode_floor_marker(detector, image):
    """Decode a high-contrast floor marker, retrying a thresholded image.

    Gazebo's anti-aliasing can soften a black/white QR edge.  The normal BGR
    frame remains the primary source; the threshold retry only makes the
    decoder tolerant of that rendering artefact.
    """
    payload, corners, _ = detector.detectAndDecode(image)
    if payload and corners is not None:
        return payload, corners
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    payload, corners, _ = detector.detectAndDecode(thresholded)
    return payload, corners


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
    target.  The visible-image axes map to link Z (robot forward) and link Y
    (robot lateral), respectively.
    """
    camera_x, camera_y, camera_z = (float(value) for value in translation)
    del camera_z
    return camera_y, camera_x
