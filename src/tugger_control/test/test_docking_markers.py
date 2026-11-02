import cv2
import numpy as np

from tugger_control.docking_markers import (
    MARKER_IDS,
    QR_GRID_SIZE_M,
    RED_FRAME_SIZE_M,
    decode_floor_marker,
    detect_red_alignment_frame,
    downward_camera_planar_error,
    expected_marker,
    marker_asset_directory,
)


def test_every_floor_marker_decodes_to_a_registered_unique_identity():
    assets = marker_asset_directory()
    identities = set()
    detector = cv2.QRCodeDetector()
    for image_path in sorted(assets.glob('*.png')):
        image = cv2.imread(str(image_path))
        identity, corners = decode_floor_marker(detector, image)
        assert corners is not None
        assert corners.reshape(4, 2)[0][0] == 60.0
        assert QR_GRID_SIZE_M == 0.064
        identities.add(identity)
    assert identities == MARKER_IDS


def test_station_markers_are_unique_and_match_the_registered_identities():
    identities = {
        expected_marker('cargo_station_0'),
        expected_marker('cargo_dropoff_northwest'),
        expected_marker('cargo_station_3'),
        expected_marker('cargo_dropoff_southeast'),
    }
    assert identities == MARKER_IDS


def test_red_alignment_frame_returns_the_outer_square_corners():
    image = np.full((320, 320, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (50, 50), (270, 270), (0, 0, 230), 12)
    corners = detect_red_alignment_frame(image)
    assert corners is not None
    centre = corners.mean(axis=0)
    assert np.allclose(centre, (160.0, 160.0), atol=2.0)
    assert RED_FRAME_SIZE_M == 0.12


def test_red_alignment_frame_rejects_a_scene_without_red_geometry():
    image = np.full((320, 320, 3), 255, dtype=np.uint8)
    assert detect_red_alignment_frame(image) is None


def test_downward_camera_depth_is_not_used_as_a_planar_motion_error():
    forward, lateral = downward_camera_planar_error((0.012, -0.018, 0.060))
    assert forward == -0.018
    assert lateral == 0.012
