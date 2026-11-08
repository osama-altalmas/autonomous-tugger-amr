#!/usr/bin/env python3
"""Top QR reader: identifies the cargo above the lift (as on the KMP 600P).

An upward camera in the lift platform reads the QR tag on the cargo's
underside ("TUGGER:CARGO:<id>") while the robot is docked under it, before
lifting. Publishes /tugger/cargo/top_qr as JSON {"id": <id or null>}.
"""
import json

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from std_msgs.msg import String

PREFIX = 'TUGGER:CARGO:'


# A frame without a code in view (open space above the robot, plus sensor
# noise) has almost no brightness spread; the QR detector can take very long
# on such noise, so only frames with real contrast are decoded.
MIN_CONTRAST = 15.0


def decode_cargo_tag(detector, image):
    """Cargo id from an image, or None (also tries the mirrored view)."""
    if float(image.std()) < MIN_CONTRAST:
        return None
    for candidate in (image, cv2.flip(image, 1)):
        payload, _points, _ = detector.detectAndDecode(candidate)
        if payload.startswith(PREFIX):
            try:
                return int(payload[len(PREFIX):])
            except ValueError:
                return None
    return None


class CargoIdReader(Node):
    def __init__(self):
        super().__init__('tugger_cargo_id_reader')
        self.set_parameters([Parameter('use_sim_time', value=True)])
        self._detector = cv2.QRCodeDetector()
        self._pub = self.create_publisher(String, '/tugger/cargo/top_qr', 10)
        self.create_subscription(Image, '/top_qr_camera/image_raw', self._on_image, 2)

    def _on_image(self, msg):
        if msg.encoding not in ('rgb8', 'bgr8', 'mono8'):
            return
        channels = 1 if msg.encoding == 'mono8' else 3
        image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, channels)
        gray = image[:, :, 0] if channels == 1 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        out = String()
        out.data = json.dumps({'id': decode_cargo_tag(self._detector, gray)})
        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = CargoIdReader()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
