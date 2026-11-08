#!/usr/bin/env python3
"""Viewing aid for recording a mission; not part of the robot software.

Opens three small camera windows (QR detector, front, rear) with only a close
button, and locks Gazebo's view to the robot like a game camera: 2 m behind
its direction of travel and 1 m up, moving at the robot's own speed.

Run it next to a live simulation. Use --no-spectator to show only the cameras.
"""
import argparse
import math
import os
import threading
import time

import cv2
from geometry_msgs.msg import PoseStamped
from gz.msgs.cameratrack_pb2 import CameraTrack
from gz.msgs.image_pb2 import Image as GzImage
from gz.transport import Node as GzNode
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

BACK_M, UP_M = 2.0, 1.0
ROBOT = 'autonomous_tugger'
MIN_SPEED, HOLD_S = 0.1, 2.0   # a change of travel direction must last this long
WIDTH, HEIGHT = 320, 240
# Window title -> screen position (x, y): a column at the right of a 1920 px screen.
WINDOWS = {'QR detector': (1590, 60), 'Front camera': (1590, 340), 'Rear camera': (1590, 620)}


class MissionView(Node):
    """Keep the latest camera images and the robot's pose."""

    def __init__(self):
        super().__init__('mission_view')
        self.images, self.counts = {}, {name: 0 for name in WINDOWS}
        self.pose, self.sign, self.candidate, self.since = None, 1.0, None, 0.0
        self.create_subscription(Image, '/qr_docking/debug_image', self._on_qr, 5)
        self.create_subscription(PoseStamped, '/tugger/ground_truth_pose', self._on_pose,
                                 qos_profile_sensor_data)
        self.create_subscription(Odometry, '/wheel/odom', self._on_odom, 10)
        self.gz = GzNode()
        self.gz.subscribe(GzImage, '/front_depth/image', lambda m: self._on_gz('Front camera', m))
        self.gz.subscribe(GzImage, '/rear_depth/image', lambda m: self._on_gz('Rear camera', m))

    def _store(self, name, image):
        self.images[name] = image
        self.counts[name] += 1

    def _on_qr(self, msg):
        shape = (msg.height, msg.width, 3)
        self._store('QR detector', np.frombuffer(msg.data, np.uint8).reshape(shape).copy())

    def _on_gz(self, name, msg):
        rgb = np.frombuffer(msg.data, np.uint8).reshape((msg.height, msg.width, 3))
        self._store(name, rgb[:, :, ::-1].copy())

    def _on_pose(self, msg):
        q = msg.pose.orientation
        self.pose = (msg.pose.position.x, msg.pose.position.y,
                     math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))

    def _on_odom(self, msg):
        speed = msg.twist.twist.linear.x
        wanted = 1.0 if speed > 0 else -1.0
        if abs(speed) < MIN_SPEED or wanted == self.sign:
            self.candidate = None
        elif self.candidate != wanted:
            self.candidate, self.since = wanted, time.monotonic()
        elif time.monotonic() - self.since >= HOLD_S:
            self.sign, self.candidate = wanted, None

    def spectator_message(self):
        """Gazebo's own follow mode: the view is locked to the robot every frame."""
        message = CameraTrack()
        message.track_mode = CameraTrack.FOLLOW_LOOK_AT
        message.follow_target.name = ROBOT
        message.track_target.name = ROBOT
        message.follow_offset.x = -self.sign * BACK_M
        message.follow_offset.z = UP_M
        message.follow_pgain = 1.0     # 1 = rigidly at the robot's own speed
        message.track_pgain = 1.0
        return message


def spectator_loop(view, stop):
    """Set the follow camera; swap ends when the travel direction changes."""
    publisher = view.gz.advertise('/gui/track', CameraTrack)
    last_sign, last_sent = None, 0.0
    while not stop.is_set():
        # Repeated now and then, so a message lost at startup cannot leave it off.
        if view.sign != last_sign or time.monotonic() - last_sent > 3.0:
            publisher.publish(view.spectator_message())
            last_sign, last_sent = view.sign, time.monotonic()
        time.sleep(0.05)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-spectator', action='store_true', help='cameras only')
    parser.add_argument('--seconds', type=float, default=0.0, help='close after this long')
    args = parser.parse_args()
    rclpy.init()
    view = MissionView()
    stop = threading.Event()
    if not args.no_spectator:
        threading.Thread(target=spectator_loop, args=(view, stop), daemon=True).start()
    flags = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL
    blank = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    for name, (x, y) in WINDOWS.items():
        cv2.namedWindow(name, flags)
        cv2.resizeWindow(name, WIDTH, HEIGHT)
        cv2.moveWindow(name, x, y)
        cv2.imshow(name, blank)
    open_windows, started = set(WINDOWS), time.monotonic()
    try:
        while open_windows and rclpy.ok():
            rclpy.spin_once(view, timeout_sec=0.01)
            for name in list(open_windows):
                if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
                    open_windows.discard(name)     # closed with its X
                elif name in view.images:
                    cv2.imshow(name, cv2.resize(view.images[name], (WIDTH, HEIGHT)))
            cv2.waitKey(15)
            if args.seconds and time.monotonic() - started > args.seconds:
                break
    except KeyboardInterrupt:
        pass
    stop.set()
    print('images received:', view.counts, flush=True)
    cv2.destroyAllWindows()
    # Gazebo's transport threads can still deliver images; leave without
    # interpreter teardown, which crashes if a callback arrives during it.
    os._exit(0)


if __name__ == '__main__':
    main()
