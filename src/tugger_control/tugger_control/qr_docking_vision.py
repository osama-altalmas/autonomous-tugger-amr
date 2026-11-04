#!/usr/bin/env python3
"""Read unique floor QR markers for the tugger's final docking measurement.

This node is deliberately observation-only.  It publishes the identified QR
payload and the marker pose relative to the downward camera; a controller may
consume it only after the image path is verified in Gazebo.
"""

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

from tugger_control.visual_square_pose import checked_square_pose
from tugger_control.docking_markers import (
    MARKER_IDS,
    QR_GRID_SIZE_M,
    decode_floor_marker,
    detect_red_alignment_frame,
    downward_camera_planar_error,
    RED_FRAME_SIZE_M,
)


class QrDockingVision(Node):
    """Decode the one uniquely identified floor marker beneath the robot."""

    def __init__(self):
        super().__init__('qr_docking_vision')
        self.declare_parameter('qr_grid_size_m', QR_GRID_SIZE_M)
        self.qr_grid_size_m = float(
            self.get_parameter('qr_grid_size_m').value)
        self._bridge = CvBridge()
        self._detector = cv2.QRCodeDetector()
        self._camera_matrix = None
        self._distortion = None
        self._last_frame_pose_warning_at = 0.0
        self._pose_pub = self.create_publisher(PoseStamped, '/qr_docking/pose', 10)
        self._frame_pose_pub = self.create_publisher(
            PoseStamped, '/qr_docking/frame_pose', 10)
        self._state_pub = self.create_publisher(Float64MultiArray, '/qr_docking/error', 10)
        self._id_pub = self.create_publisher(String, '/qr_docking/id', 10)
        self._debug_pub = self.create_publisher(
            Image, '/qr_docking/debug_image', 10)
        self.create_subscription(CameraInfo, '/downward_camera/camera_info', self._info, 10)
        self.create_subscription(Image, '/downward_camera/image_raw', self._image, 10)
        self.get_logger().info('QR docking vision ready; observation-only mode.')

    def _info(self, message):
        if self._camera_matrix is not None:
            return
        self._camera_matrix = np.array(message.k, dtype=np.float64).reshape(3, 3)
        self._distortion = np.array(message.d, dtype=np.float64)
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
        """Publish the red-frame centre in the camera frame for alignment."""
        try:
            _rotation, translation, _rmse = checked_square_pose(
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

            return
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

    def _image(self, message):
        if self._camera_matrix is None:
            return
        try:
            image = self._bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
            payload, corners = decode_floor_marker(self._detector, image)
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
        corners = np.asarray(corners, dtype=np.float64).reshape(4, 2)
        half = self.qr_grid_size_m / 2.0
        object_points = np.array([
            [-half, half, 0.0], [half, half, 0.0],
            [half, -half, 0.0], [-half, -half, 0.0],
        ], dtype=np.float64)
        ok, rotation, translation = cv2.solvePnP(
            object_points, corners, self._camera_matrix, self._distortion,
            flags=cv2.SOLVEPNP_IPPE_SQUARE)
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
