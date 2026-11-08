#!/usr/bin/env python3
"""Merge the two corner safety scanners into one 360-degree scan.

Each scanner sees 275 degrees from its corner; together they cover the whole
robot. AMCL takes one scan, so both are expressed in base_footprint and
binned into one LaserScan (nearest return per 0.5-degree bin) on /scan.
Returns on the robot's own body are dropped. A merged scan is published only
when both inputs are recent, so a stopped scanner never looks like free space.

The two scanners fire up to 40 ms apart; at 1 m/s the robot moves 4 cm in
between. Each scan is therefore motion-compensated: its points are moved by
the odometry displacement (odom frame) between its own stamp and the merged
scan's stamp, as a real scanner fusion does.
"""
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener

INPUTS = ('/front_left/scan', '/rear_right/scan')
BIN_RAD = math.radians(0.5)
MAX_AGE_S = 0.15
BODY_HALF = (0.46, 0.31)


def _yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class ScanMerger(Node):
    def __init__(self):
        super().__init__('tugger_scan_merger')
        self.set_parameters([Parameter('use_sim_time', value=True)])
        self._tf = Buffer()
        self._listener = TransformListener(self._tf, self)
        self._mounts = {}
        self._latest = {}
        self._odom = {}
        for topic in INPUTS:
            self.create_subscription(LaserScan, topic,
                                     lambda msg, t=topic: self._on_scan(t, msg), 10)
        self._pub = self.create_publisher(LaserScan, '/scan', 10)
        # Scans wait until TF covers their stamps (usually 10-30 ms), so the
        # motion compensation always has the robot's pose at each scan.
        self._pending = None
        self.create_timer(0.01, self._retry)

    def _mount(self, frame):
        if frame not in self._mounts:
            t = self._tf.lookup_transform('base_footprint', frame, rclpy.time.Time())
            self._mounts[frame] = (t.transform.translation.x, t.transform.translation.y,
                                   _yaw(t.transform.rotation))
        return self._mounts[frame]

    def _on_scan(self, topic, msg):
        try:
            mx, my, myaw = self._mount(msg.header.frame_id)
        except Exception:
            return
        ranges = np.asarray(msg.ranges, dtype=float)
        angles = msg.angle_min + msg.angle_increment * np.arange(len(ranges)) + myaw
        ok = np.isfinite(ranges) & (ranges >= msg.range_min) & (ranges <= msg.range_max)
        x = mx + ranges[ok] * np.cos(angles[ok])
        y = my + ranges[ok] * np.sin(angles[ok])
        off_body = (np.abs(x) > BODY_HALF[0]) | (np.abs(y) > BODY_HALF[1])
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self._latest[topic] = (stamp, x[off_body], y[off_body], msg)
        self._odom[topic] = self._odom_pose(msg.header.stamp)
        if topic == INPUTS[0]:
            self._pending = (msg, self.get_clock().now().nanoseconds * 1e-9)
            self._retry()

    def _retry(self):
        """Publish the pending merge once both scans have their odometry pose."""
        if self._pending is None:
            return
        msg, since = self._pending
        for topic, value in self._latest.items():
            if self._odom.get(topic) is None:
                self._odom[topic] = self._odom_pose(value[3].header.stamp)
        ready = all(self._odom.get(topic) is not None for topic in self._latest)
        waited = self.get_clock().now().nanoseconds * 1e-9 - since
        if ready or waited > 0.1:
            self._pending = None
            if ready:
                self._publish(msg)  # never merge without motion compensation

    def _odom_pose(self, stamp):
        """Robot pose in odom at ``stamp`` (x, y, yaw), or None."""
        try:
            t = self._tf.lookup_transform('odom', 'base_footprint', rclpy.time.Time.from_msg(stamp))
        except Exception:
            return None
        return (t.transform.translation.x, t.transform.translation.y, _yaw(t.transform.rotation))

    @staticmethod
    def _compensate(x, y, at_scan, at_reference):
        """Points seen from the pose at_scan, expressed in the pose at_reference."""
        if at_scan is None or at_reference is None:
            return x, y
        c, s = math.cos(at_scan[2]), math.sin(at_scan[2])
        wx, wy = at_scan[0] + c * x - s * y, at_scan[1] + s * x + c * y
        dx, dy = wx - at_reference[0], wy - at_reference[1]
        c, s = math.cos(-at_reference[2]), math.sin(-at_reference[2])
        return c * dx - s * dy, s * dx + c * dy

    def _publish(self, reference):
        if len(self._latest) < len(INPUTS):
            return
        stamps = [value[0] for value in self._latest.values()]
        if max(stamps) - min(stamps) > MAX_AGE_S:
            return
        at_reference = self._odom.get(INPUTS[0])
        parts = [self._compensate(value[1], value[2], self._odom.get(topic), at_reference)
                 for topic, value in self._latest.items()]
        x = np.concatenate([p[0] for p in parts])
        y = np.concatenate([p[1] for p in parts])
        bins = int(round(2 * math.pi / BIN_RAD))
        ranges = np.full(bins, np.inf)
        if len(x):
            r = np.hypot(x, y)
            index = ((np.arctan2(y, x) + math.pi) / BIN_RAD).astype(int) % bins
            np.minimum.at(ranges, index, r)
        out = LaserScan()
        out.header.stamp = reference.header.stamp
        out.header.frame_id = 'base_footprint'
        out.angle_min = -math.pi
        out.angle_max = math.pi - BIN_RAD
        out.angle_increment = BIN_RAD
        out.time_increment = 0.0
        out.scan_time = reference.scan_time
        out.range_min = 0.05
        out.range_max = reference.range_max
        out.ranges = [float(v) for v in ranges]
        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ScanMerger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
