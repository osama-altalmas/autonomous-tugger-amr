#!/usr/bin/env python3
"""Read unique floor QR markers for the tugger's final docking measurement.

This node is deliberately observation-only.  It publishes the identified QR
payload and the marker pose relative to the downward camera; a controller may
consume it only after the image path is verified in Gazebo.

Two readers (parameter marker_reader):
  v3       (default) the trained YOLO-pose model finds the marker, its four
           tags name the station and the red frame is fitted from its visible
           sides (marker_reader.read_marker): reads worn markers too. The
           heading comes from the fitted frame. The classic tag reader runs
           beside it as a check: if both name a station and they differ,
           nothing is published (the docking holds).
  classic  OpenCV AprilTag detection and the red frame's colour outline.
"""

import json
import math
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float64MultiArray, String

from tugger_control.marker_geometry import red_mask
from tugger_control.marker_reader import (
    CAMERA_HEIGHT_M, FAULT_BELOW, MarkerModel, default_model_path, light_check, read_marker)
from tugger_control.visual_square_pose import checked_square_pose
from tugger_control.docking_markers import (
    MARKER_IDS,
    detect_floor_tags,
    detect_red_alignment_frame,
    tag_detector,
    downward_camera_planar_error,
    RED_FRAME_SIZE_M,
)


# The v3 model runs only when this much red frame is in view (about 2 ms to
# count). Measured 2026-11-16: every frame the fit measured had >= 7,800 red
# pixels, the plain warehouse floor 0. Without the gate the model ran on every
# image in the aisles and slowed the simulation by about 20 %.
MIN_RED_PIXELS = 3000


class QrDockingVision(Node):
    """Decode the one uniquely identified floor marker beneath the robot."""

    def __init__(self):
        super().__init__('qr_docking_vision')
        self._bridge = CvBridge()
        # Four AprilTags per floor marker (docking_markers); the topic and
        # message names keep "qr" for the consumers.
        self._detector = tag_detector()
        self.declare_parameter('marker_reader', 'v3')
        self.declare_parameter('model', '')
        self.declare_parameter('camera_height_m', CAMERA_HEIGHT_M)
        self._model = None
        if self.get_parameter('marker_reader').value == 'v3':
            path = self.get_parameter('model').value or default_model_path()
            try:
                self._model = MarkerModel(path)
            except (cv2.error, OSError) as exc:
                self.get_logger().error(
                    f'Marker model {path} not loaded ({exc}); using the classic reader.')
        self._side_px = None
        self._last_reader_warning_at = 0.0
        self._camera_matrix = None
        self._distortion = None
        self._last_frame_pose_warning_at = 0.0
        self._pose_pub = self.create_publisher(PoseStamped, '/qr_docking/pose', 10)
        self._frame_pose_pub = self.create_publisher(
            PoseStamped, '/qr_docking/frame_pose', 10)
        self._state_pub = self.create_publisher(Float64MultiArray, '/qr_docking/error', 10)
        self._id_pub = self.create_publisher(String, '/qr_docking/id', 10)
        # The v3 reader's own state (OK / ALERT / FAULT with the reason), so a
        # worn, dirty or badly lit marker is reported before it fails.
        self._report_pub = self.create_publisher(String, '/tugger/marker_reader', 10)
        self._debug_pub = self.create_publisher(
            Image, '/qr_docking/debug_image', 10)
        self.create_subscription(CameraInfo, '/downward_camera/camera_info', self._info, 10)
        # The v3 model needs about 0.15 s per image: keep only the newest one,
        # so a queue of old images never builds up.
        self.create_subscription(Image, '/downward_camera/image_raw', self._image,
                                 1 if self._model is not None else 10)
        reader = 'v3 (model + tags + frame fit)' if self._model is not None else 'classic'
        self.get_logger().info(f'QR docking vision ready ({reader} reader); observation-only.')

    def _info(self, message):
        if self._camera_matrix is not None:
            return
        self._camera_matrix = np.array(message.k, dtype=np.float64).reshape(3, 3)
        self._distortion = np.array(message.d, dtype=np.float64)
        self._side_px = self._camera_matrix[0, 0] * RED_FRAME_SIZE_M / float(
            self.get_parameter('camera_height_m').value)
        self.get_logger().info('Downward-camera calibration received.')

    def _publish_debug(self, image, header, payload='', corners=None,
                       frame_corners=None):
        """Publish an annotated frame so docking can be inspected live."""
        debug = image.copy()
        if corners is not None:
            cv2.polylines(debug, [np.asarray(corners, dtype=np.int32)],
                          True, (0, 255, 0), 2)
        if frame_corners is not None:
            cv2.polylines(debug, [np.asarray(frame_corners, dtype=np.int32)],
                          True, (0, 0, 255), 2)
        label = payload if payload else 'NO REGISTERED QR MARKER'
        color = (0, 255, 0) if payload else (0, 0, 255)
        cv2.putText(debug, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2, cv2.LINE_AA)
        debug_message = self._bridge.cv2_to_imgmsg(debug, encoding='bgr8')
        debug_message.header = header
        self._debug_pub.publish(debug_message)

    def _publish_frame_pose(self, corners, header):
        """Publish the red-frame centre in the camera frame for alignment;
        returns the checked (rotation, translation), or None."""
        try:
            rotation, translation, _rmse = checked_square_pose(
                corners,
                self._camera_matrix,
                self._distortion,
                RED_FRAME_SIZE_M,
            )

        except (ValueError, cv2.error) as exc:
            # Never publish an invalid red-frame pose.
            now = time.monotonic()

            if now - self._last_frame_pose_warning_at >= 2.0:
                self.get_logger().warning(
                    f'Red-frame pose rejected: {exc}'
                )
                self._last_frame_pose_warning_at = now

            return None
        pose = PoseStamped()
        forward, lateral = downward_camera_planar_error(translation)
        pose.header = header
        # The pose is already expressed in the robot base plane.  Gazebo's
        # scoped sensor frame is not present in ROS TF and OpenCV Z is height.
        pose.header.frame_id = 'base_footprint'
        pose.pose.position.x = forward
        pose.pose.position.y = lateral
        pose.pose.position.z = float(translation[2])
        pose.pose.orientation.w = 1.0
        self._frame_pose_pub.publish(pose)
        return rotation, translation

    def _warn_reader(self, text):
        now = time.monotonic()
        if now - self._last_reader_warning_at >= 2.0:
            self.get_logger().warning(text)
            self._last_reader_warning_at = now

    def _image(self, message):
        if self._camera_matrix is None:
            return
        if self._model is not None:
            self._image_v3(message)
            return
        try:
            image = self._bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
            payload, image_points, object_points, _ = detect_floor_tags(self._detector, image)
            corners = (cv2.convexHull(image_points.astype(np.float32)).reshape(-1, 2)
                       if image_points is not None else None)
            frame_corners = detect_red_alignment_frame(image)
        except Exception as exc:
            self.get_logger().warning(f'QR image conversion failed: {exc}')
            return
        if frame_corners is not None:
            self._publish_frame_pose(frame_corners, message.header)
        if not payload or corners is None:
            self._publish_debug(image, message.header, frame_corners=frame_corners)
            return
        if payload not in MARKER_IDS:
            self.get_logger().warning(
                f'Ignoring unregistered floor-marker identity: {payload}')
            self._publish_debug(image, message.header, payload, corners,
                                frame_corners)
            return
        # Every visible tag corner at once: one tag (4 points) is enough, more
        # tags make the pose steadier. IPPE handles any planar point set.
        ok, rotation, translation = cv2.solvePnP(
            object_points, image_points, self._camera_matrix, self._distortion,
            flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            return
        rotation_matrix, _ = cv2.Rodrigues(rotation)
        yaw = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        pose = PoseStamped()
        pose.header = message.header
        pose.header.frame_id = 'downward_camera_link'
        pose.pose.position.x = float(translation[0])
        pose.pose.position.y = float(translation[1])
        pose.pose.position.z = float(translation[2])
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        self._pose_pub.publish(pose)
        identity = String()
        identity.data = payload
        self._id_pub.publish(identity)
        # Camera-frame x/y error, range and QR-plane yaw.  Consumers must use
        # the fixed downward-camera transform before commanding base motion.
        error = Float64MultiArray()
        error.data = [float(translation[0]), float(translation[1]),
                      float(translation[2]), yaw]
        self._state_pub.publish(error)
        self._publish_debug(image, message.header, payload, corners,
                            frame_corners)

    def _image_v3(self, message):
        """Marker read by the trained model, tags and frame fit (module doc)."""
        try:
            image = self._bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
            if np.count_nonzero(red_mask(image)) < MIN_RED_PIXELS:
                self._publish_debug(image, message.header)
                return
            result = read_marker(self._model, image, self._side_px)
            check = detect_floor_tags(self._detector, image)[0]
        except Exception as exc:
            self.get_logger().warning(f'Marker image processing failed: {exc}')
            return
        self._report(message, image, result)
        payload = result['payload']
        corners = result.get('frame_corners')
        if result['confidence'] < FAULT_BELOW or payload not in MARKER_IDS:
            payload, corners = None, None
        if payload and check and check != payload:
            self._warn_reader(f'Marker readers disagree: model {payload}, tags {check}; '
                              'nothing published.')
            payload, corners = None, None
        if payload is None:
            self._publish_debug(image, message.header)
            return
        identity = String()
        identity.data = payload
        self._id_pub.publish(identity)
        if corners is None:
            self._warn_reader(f'{payload} read, but its red frame is not measured: '
                              f'{result.get("reason") or "frame not fitted"}')
            self._publish_debug(image, message.header, payload)
            return
        checked = self._publish_frame_pose(corners, message.header)
        # Heading from the fitted frame: its corners are in the marker's own
        # order, so the plate axes are the same as the tags' (docking_markers).
        # Not IPPE: the fitted frame is an exact square, the face-on case in
        # which IPPE returned the wrong one of its two solutions (2026-11-16);
        # the SQPNP pose matched the tags' heading within 0.3 degrees.
        if checked is not None:
            self._publish_marker_pose(message.header, *checked)
        self._publish_debug(image, message.header, payload, frame_corners=corners)

    def _report(self, message, image, result):
        """Publish the reader's state; a light fault outranks the marker's."""
        light = light_check(image)
        report = {k: v for k, v in result.items() if k not in ('frame_corners', 'detections')}
        report['light'] = light
        if light['state'] == 'FAULT' or (light['state'] == 'ALERT' and result['state'] == 'OK'):
            report.update(state=light['state'], reason=light['reason'])
        report['stamp'] = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        out = String()
        out.data = json.dumps(report)
        self._report_pub.publish(out)

    def _publish_marker_pose(self, header, rotation, translation):
        rotation_matrix, _ = cv2.Rodrigues(rotation)
        yaw = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        pose = PoseStamped()
        pose.header = header
        pose.header.frame_id = 'downward_camera_link'
        pose.pose.position.x = float(translation[0])
        pose.pose.position.y = float(translation[1])
        pose.pose.position.z = float(translation[2])
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        self._pose_pub.publish(pose)
        error = Float64MultiArray()
        error.data = [float(translation[0]), float(translation[1]),
                      float(translation[2]), yaw]
        self._state_pub.publish(error)


def main(args=None):
    rclpy.init(args=args)
    node = QrDockingVision()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
