import cv2
import json
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET

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
        expected_marker('HOME'),
        expected_marker('cargo_station_0'),
        expected_marker('cargo_dropoff_northwest'),
        expected_marker('cargo_station_3'),
        expected_marker('cargo_dropoff_southeast'),
    }
    assert identities == MARKER_IDS


def test_home_marker_matches_station_geometry_and_configured_home_pose():
    gazebo = marker_asset_directory().parents[1]
    world = ET.parse(gazebo / 'worlds' / 'tugger_warehouse.sdf').getroot()
    home = world.find("./world/model[@name='qr_home']")
    station = world.find("./world/model[@name='qr_station_1']")
    assert home is not None and station is not None
    home_xy = [float(v) for v in home.findtext('pose').split()[:2]]
    stations = json.loads((Path(__file__).resolve().parents[1] /
                           'config/operator_stations.json').read_text())['stations']
    assert home_xy == [0.0, 0.0]
    database = json.loads((Path(__file__).resolve().parents[3] /
                           'tools/v1_manhattan_bezier/routes/route_database.json').read_text())
    # HOME segment 2 ends on the marker; its entries are the gate staging poses.
    assert database['station_centers']['HOME'] == home_xy
    assert [entry['pose'] for entry in stations['HOME']['candidates']] == [
        node['pose'] for node in database['nodes']['HOME'].values() if node['available']]
    for name in ('red_alignment_frame', 'qr'):
        visual = home.find(f"./link/visual[@name='{name}']")
        local_pose = visual.findtext('pose', '0 0 0 0 0 0')
        assert [float(v) for v in local_pose.split()[:2]] == [0.0, 0.0]
        selector = f"./link/visual[@name='{name}']/geometry/plane/size"
        assert home.findtext(selector) == station.findtext(selector)
    assert home.findtext('./link/visual[@name="qr"]/material/pbr/metal/albedo_map').endswith('/home.png')


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


def test_red_alignment_frame_rejects_a_square_touching_image_edge():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(image, (0, 40), (100, 140), (0, 0, 255), 8)
    assert detect_red_alignment_frame(image) is None


def test_downward_camera_depth_is_not_used_as_a_planar_motion_error():
    forward, lateral = downward_camera_planar_error((0.012, -0.018, 0.060))
    assert forward == 0.018
    assert lateral == -0.012


def test_optical_to_base_mapping_matches_the_actual_camera_mount():
    import math
    urdf = Path(__file__).resolve().parents[2] / 'tugger_description/urdf/tugger.urdf.xacro'
    root = ET.parse(urdf).getroot()
    origin = root.find("./joint[@name='downward_camera_joint']/origin")
    roll, pitch, yaw = map(float, origin.attrib['rpy'].split())
    assert abs(roll) < 1e-9 and abs(yaw) < 1e-9
    rotation = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                         [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
    for optical in ((0.01, 0., 0.06), (-0.01, 0., 0.06),
                    (0., 0.01, 0.06), (0., -0.01, 0.06)):
        ox, oy, oz = optical
        base = rotation @ np.array((oz, -ox, -oy))
        assert np.allclose(downward_camera_planar_error(optical), base[:2], atol=1e-9)
